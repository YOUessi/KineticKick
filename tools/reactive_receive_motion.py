"""Measured-ball interception from a grounded foot, without launch-time cues."""
import numpy as np
from scipy.spatial.transform import Rotation
from cushion_motion import hermite, smooth_step


class ReactiveReceiveMotion:
    def __init__(self,plan):
        self.plan=plan;self.cfg=plan['cushion'];self.reaction=plan['reaction']
        self.initial_foot=np.array(plan['feet']['right'],float)
        self.initial_root=np.array(plan['root_pose'],float)
        self.contact_time=None;self.move_start=None;self.hit_position=None;self.hit_velocity=None
        self.normal=np.array([0.,1.,0.]);self.rotation=self.initial_foot[:3,:3].copy()
        self.predictions=[];self.events=[];self.observations=0;self.absorbed_displacement=0.
        self.angular_velocity=np.zeros(3);self.aerial=False
        self.first_touch=None;self.brake_start=None;self.brake_from=None;self.ball=None;self.ball_velocity=None

    def update_prediction(self,now,ball,velocity,touched=False):
        self.observations+=1
        self.ball=np.array(ball,float);self.ball_velocity=np.array(velocity,float)
        if touched:
            if self.first_touch is None:self.first_touch=now
            if (self.reaction.get('ground_brake',False) and self.brake_start is None
                    and now-self.first_touch>=.10 and self.ball[2]<.145
                    and abs(self.ball_velocity[2])<.2 and np.linalg.norm(self.ball_velocity)<.7):
                self.brake_from=(self.actual_foot if hasattr(self,'actual_foot') else self.targets(now)[0]).copy()
                self.brake_start=now
                self.events.append(dict(kind='brake_from_observed_landing',time_s=now,
                    ball_position=self.ball.tolist(),ball_velocity=self.ball_velocity.tolist(),physical_state_write=False))
            return
        if touched or (self.contact_time is not None and now>=self.contact_time-self.reaction['freeze_s']):return
        ball=np.array(ball,float);v=np.array(velocity,float);vv=float(v[:2]@v[:2])
        if vv<.04:return
        aerial=bool(ball[2]>.15 or abs(v[2])>.25)
        anchor=self.reaction.get('anchor_aerial' if aerial else 'anchor_ground',self.reaction['anchor'])
        dt=float((np.array(anchor)-ball[:2])@v[:2]/vv)
        if dt<.04 or dt>1.5:return
        aerial=bool(ball[2]>.15 or abs(v[2])>.25)
        fraction=self.reaction['aerial_fraction' if aerial else 'ground_fraction']
        if aerial:
            speed_error=np.sqrt(vv)-self.reaction.get('aerial_reference_speed_mps',np.sqrt(vv))
            fraction=float(np.clip(fraction+self.reaction.get('aerial_speed_gain',0.)*speed_error,.04,.20))
        dt+=.5*fraction*.16
        acceleration=np.array([0.,0.,-9.81 if aerial else 0.])
        impact=ball+v*dt+.5*acceleration*dt*dt
        incoming=v+acceleration*dt
        if not aerial:impact[2]=.11;incoming[2]=0.
        if impact[2]<.105 or impact[2]>.55 or (aerial and incoming[2]>=0):return
        normal=-incoming/np.linalg.norm(incoming)
        yaw=np.arctan2(normal[1],normal[0])-np.pi/2
        # The right foot's medial-face lag grows on the positive-incidence side.
        # Use the observed horizontal heading, with a bounded calibrated trim.
        yaw_trim=0.
        if aerial:
            incidence=np.arctan2(-v[0],v[1])
            start=self.reaction.get('aerial_yaw_trim_start_angle_rad',0.)
            end=self.reaction.get('aerial_yaw_trim_full_angle_rad',.13)
            amount=np.clip((incidence-start)/max(1e-6,end-start),0.,1.)
            yaw_trim=float(self.reaction.get('aerial_yaw_trim_rad',0.)*amount)
            yaw+=yaw_trim
        roll=np.arctan2(normal[2],np.linalg.norm(normal[:2]))
        rotation=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('x',roll).as_matrix()
        normal=rotation[:,1]
        offset=rotation@np.array(self.plan['contact_center_local'])+.12*normal
        hit=impact-offset
        if np.linalg.norm(hit[:2]-self.initial_root[:2])>.65 or hit[2]<.04 or hit[2]>.50:return
        self.contact_time=now+dt;self.hit_position=hit;self.hit_velocity=incoming*fraction
        self.rotation=rotation;self.normal=normal;self.aerial=aerial
        if self.move_start is None and dt<=self.reaction.get('aerial_lead_s' if aerial else 'lead_s',self.reaction['lead_s']) and self.observations>=3:
            self.move_start=now
            self.events.append(dict(kind='react_to_observed_ball',time_s=now,ball_height_m=float(ball[2]),
                predicted_contact_height_m=float(impact[2]),expected_contact_time_s=self.contact_time,
                planned_foot_height_m=float(hit[2]),physical_state_write=False))
        self.predictions.append(dict(time_s=now,ball_position=ball.tolist(),ball_velocity=v.tolist(),
            expected_contact_time_s=self.contact_time,incoming_velocity=incoming.tolist(),
            foot_hit_position=hit.tolist(),foot_hit_velocity=self.hit_velocity.tolist(),normal=normal.tolist(),
            observed_aerial=aerial,velocity_fraction=fraction,yaw_trim_rad=yaw_trim,move_start_s=self.move_start))

    def targets(self,now):
        foot=self.initial_foot.copy();root=np.eye(4);root[:3,3]=self.initial_root[:3]
        root[:3,:3]=Rotation.from_quat(self.initial_root[[4,5,6,3]]).as_matrix()
        self.angular_velocity[:]=0.
        if self.move_start is None:return foot,np.zeros(3),root,'observe'
        if self.brake_start is not None:
            duration=self.reaction.get('brake_seconds',.16)
            u=np.clip((now-self.brake_start)/duration,0.,1.);w=smooth_step(u)
            yaw=Rotation.from_matrix(self.rotation).as_euler('xyz')[2]
            flat=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('x',.15 if self.aerial else 0.).as_matrix()
            delta=Rotation.from_matrix(flat@self.brake_from[:3,:3].T).as_rotvec()
            foot[:3,:3]=Rotation.from_rotvec(delta*w).as_matrix()@self.brake_from[:3,:3]
            rate=30*u*u*(1-u)**2/duration;self.angular_velocity=delta*rate
            normal=flat[:,1]
            penetration=np.clip(-float(self.ball_velocity@normal)*.02,0.,.003)
            goal=self.ball-flat@np.array(self.plan['contact_center_local'])-.12*normal+penetration*normal
            goal[2]=.11-(flat@np.array(self.plan['contact_center_local'])+.12*normal)[2]
            foot[:3,3]=(1-w)*self.brake_from[:3,3]+w*goal
            speed=rate*(goal-self.brake_from[:3,3])-.5*w*self.ball_velocity
            speed[2]=rate*(goal[2]-self.brake_from[2,3])
            return foot,speed,root,'brake_ground'

        duration=max(.06,self.contact_time-self.move_start)
        ru=np.clip((now-self.move_start)/max(.06,duration-.06),0.,1.)
        rotation_delta=Rotation.from_matrix(self.rotation@self.initial_foot[:3,:3].T).as_rotvec()
        foot[:3,:3]=Rotation.from_rotvec(rotation_delta*smooth_step(ru)).as_matrix()@self.initial_foot[:3,:3]
        self.angular_velocity=rotation_delta*30*ru*ru*(1-ru)**2/max(.06,duration-.06)
        if now<=self.contact_time:
            foot[:3,3],speed=hermite(self.initial_foot[:3,3],self.hit_position,np.zeros(3),self.hit_velocity,duration,now-self.move_start)
            # Early clearance avoids rotating the foot capsules through the ground.
            u=np.clip((now-self.move_start)/duration,0.,1.)
            foot[2,3]+=.035*np.sin(np.pi*u)**2
            speed[2]+=.035*np.pi*np.sin(2*np.pi*u)/duration
            phase='intercept'
        else:
            follow=self.reaction.get('aerial_follow_s',self.reaction['follow_s']) if self.aerial else self.reaction['follow_s'];dt=min(now-self.contact_time,follow)
            foot[:3,3]=self.hit_position+self.hit_velocity*(dt-.5*dt*dt/follow)
            speed=self.hit_velocity*(1-dt/follow);phase='cushion' if dt<follow else 'settle'
        foot[:3,3]-=self.normal*self.absorbed_displacement
        return foot,speed,root,phase

    def yield_to_contact(self,target,actual_position,force):
        if self.brake_start is not None:return target
        if not self.aerial and force<=.2:return target
        away=-self.normal;cap=self.cfg.get('absorption_limit_m',.12) if self.aerial else .04
        displacement=max(0.,float((actual_position-target[:3,3])@away))
        increment=min(displacement,max(0.,cap-self.absorbed_displacement))
        self.absorbed_displacement+=increment
        result=target.copy();result[:3,3]+=increment*away
        return result
