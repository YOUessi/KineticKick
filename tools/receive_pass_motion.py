"""Measured stopped-ball footwork followed by a directional inside-foot pass.

Outputs are motor targets only. Timers never replace observed foot landing.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from cushion_motion import smooth_step


class ReceivePassMotion:
    def __init__(self,cfg):
        self.cfg=cfg
        d=np.asarray(cfg.get('direction',[0.,-1.]),float)
        if d.shape!=(2,) or not np.isfinite(d).all() or np.linalg.norm(d)<1e-8:
            raise ValueError('A finite nonzero horizontal pass direction is required')
        target=cfg.get('target_xy')
        if target is not None and (np.asarray(target).shape!=(2,) or not np.isfinite(target).all()):
            raise ValueError('Pass target must contain two finite coordinates')
        self.direction=np.r_[d/np.linalg.norm(d),0.]
        self.active=False;self.phase='wait_receive';self.events=[]
        self.rest_since=None;self.stable_since=None;self.last_time=None
        self.lifted=False;self.stage_start=0.;self.abort_reason=None
        self.contact_center=np.array([.075,.026,-.025])

    def transition(self,phase,now,foot):
        self.phase=phase;self.stage_start=now;self.stage_from=foot.copy()
        self.events.append(dict(kind=phase,time_s=now,physical_state_write=False))

    def strike_geometry(self,ball):
        yaw=np.arctan2(self.direction[1],self.direction[0])-np.pi/2
        self.strike_rotation=Rotation.from_euler('z',yaw).as_matrix()
        self.hit=np.asarray(ball,float)-self.strike_rotation@self.contact_center-.12*self.direction
        self.hit[2]=.11-(self.strike_rotation@self.contact_center)[2]
        self.back=self.hit-self.cfg.get('backswing_m',.07)*self.direction
        self.end=self.hit+self.cfg.get('followthrough_m',.08)*self.direction

    def update(self,now,ball,velocity,spin,root,feet,forces,root_speed,root_tilt,received):
        ball=np.asarray(ball,float);foot=np.asarray(feet['right'],float)
        slow=bool(received and ball[2]<.145 and np.linalg.norm(velocity)<.02 and np.linalg.norm(spin)<.3)
        stable=bool(root[2,3]>.65 and root_speed<.15 and root_tilt<12. and min(forces)>20.)
        if not self.active:
            self.rest_since=(now if self.rest_since is None else self.rest_since) if slow else None
            self.stable_since=(now if self.stable_since is None else self.stable_since) if stable else None
            if (self.rest_since is None or self.stable_since is None
                    or now-self.rest_since<self.cfg.get('stopped_seconds',.85)
                    or now-self.stable_since<self.cfg.get('standing_seconds',.6)):return
            if np.linalg.norm(ball[:2]-root[:2,3])>.65:return
            if self.cfg.get('target_xy') is not None:
                delta=np.asarray(self.cfg['target_xy'],float)-ball[:2]
                if np.linalg.norm(delta)<.2:raise ValueError('Pass target is too close to the stopped ball')
                self.direction=np.r_[delta/np.linalg.norm(delta),0.]
            self.initial_foot=foot.copy();self.stopped_ball=ball.copy()
            self.strike_geometry(ball)
            self.step_goal=np.eye(4);self.step_goal[:3,:3]=foot[:3,:3]
            self.step_goal[:3,3]=self.hit-self.cfg.get('step_behind_m',.10)*self.direction
            self.step_goal[2,3]=.033
            if np.linalg.norm(self.step_goal[:2,3]-foot[:2,3])>self.cfg.get('maximum_step_m',.30):
                self.abort_reason='step_out_of_range';return
            shift=self.step_goal[:2,3]-foot[:2,3]
            length=float(np.linalg.norm(shift));limit=self.cfg.get('local_step_limit_m',.13)
            if length>limit:self.step_goal[:2,3]=foot[:2,3]+shift*(limit/length)
            self.active=True
            self.events.append(dict(kind='plan_pass_from_stopped_ball',time_s=now,ball_position=ball.tolist(),
                ball_velocity=np.asarray(velocity).tolist(),direction=self.direction.tolist(),
                step_goal=self.step_goal.tolist(),initial_foot=foot.tolist(),physical_state_write=False))
            self.transition('step_unload',now,foot)
            return
        elapsed=now-self.stage_start
        unloaded=bool(forces[0]>250. and forces[1]<8. and root_speed<.04 and root_tilt<8.)
        if self.phase=='step_unload' and elapsed>=self.cfg.get('unload_s',.35) and unloaded:
            self.transition('step_move',now,foot)
        elif self.phase=='step_move':
            self.lifted |= bool(foot[2,3]>self.initial_foot[2,3]+.025 and forces[1]<5.)
            if self.lifted and elapsed>.5 and forces[1]>10. and foot[2,3]<.07:
                self.planted=foot.copy()
                self.events.append(dict(kind='step_landed',time_s=now,foot_position=foot[:3,3].tolist(),
                    displacement_m=float(np.linalg.norm(foot[:2,3]-self.initial_foot[:2,3])),ground_force_N=float(forces[1]),physical_state_write=False))
                self.transition('step_load',now,foot)
            elif elapsed>2.5:
                self.abort_reason='step_did_not_land';self.transition('pass_abort',now,foot)
        elif (self.phase=='step_load' and elapsed>=self.cfg.get('step_load_s',.4) and root_speed<.035 and root_tilt<8.
                and (forces[1]>10. or (elapsed>=.8 and forces[0]>300. and forces[1]<8. and root_speed<.01 and foot[2,3]<.06))):
            if np.linalg.norm(velocity)>.05 or np.linalg.norm(foot[:2,3]-self.initial_foot[:2,3])<.03:
                self.abort_reason='ball_moved_or_no_step';self.transition('pass_abort',now,foot);return
            self.events.append(dict(kind='step_ready',time_s=now,ground_force_N=float(forces[1]),physical_state_write=False))
            self.strike_geometry(ball);self.pass_ball=ball.copy()
            self.transition('pass_unload',now,foot)
        elif self.phase=='pass_unload' and elapsed>=self.cfg.get('unload_s',.35) and unloaded:
            self.transition('pass_lift',now,foot)
        elif self.phase=='pass_lift':
            if elapsed>=self.cfg.get('lift_s',.55) and np.linalg.norm(foot[:3,3]-self.back)<.05:
                self.transition('pass_strike',now,foot)
            elif elapsed>2.:
                self.abort_reason='backswing_unreachable';self.transition('pass_abort',now,foot)
        elif self.phase=='pass_strike' and elapsed>=self.cfg.get('strike_s',.26):
            self.transition('pass_recover',now,foot)
        elif self.phase=='pass_recover':
            if elapsed>.4 and forces[1]>10. and foot[2,3]<.07:
                self.planted=foot.copy();self.transition('pass_load',now,foot)
        elif self.phase=='pass_load' and elapsed>=.6:
            self.transition('pass_stand',now,foot)

    def targets(self,now):
        if not self.active:return None
        elapsed=now-self.stage_start;foot=self.stage_from.copy();v=np.zeros(3);omega=np.zeros(3);load=0.
        phase=self.phase;goal=None;duration=1.;arch=0.
        if phase in ('step_unload','pass_unload'):
            load=.1*(1-smooth_step(elapsed/self.cfg.get('unload_s',.35)))
        elif phase=='step_move':goal=self.step_goal;duration=self.cfg.get('step_s',.8);arch=.075
        elif phase in ('step_load','pass_load','pass_stand'):
            foot=self.planted.copy()
            load=.1 if phase=='pass_stand' else .1*smooth_step(elapsed/(.6 if phase=='pass_load' else self.cfg.get('step_load_s',.4)))
        elif phase=='pass_lift':
            goal=np.eye(4);goal[:3,:3]=self.strike_rotation;goal[:3,3]=self.back;duration=self.cfg.get('lift_s',.55);arch=.035
        elif phase=='pass_strike':
            goal=np.eye(4);goal[:3,:3]=self.strike_rotation;goal[:3,3]=self.end;duration=self.cfg.get('strike_s',.26)
        elif phase=='pass_recover':goal=self.initial_foot;duration=self.cfg.get('recover_s',.75);arch=.04
        elif phase=='pass_abort':load=0.
        if goal is not None:
            u=np.clip(elapsed/duration,0.,1.);w=smooth_step(u);rate=30*u*u*(1-u)**2/duration
            displacement=goal[:3,3]-self.stage_from[:3,3]
            foot[:3,3]=self.stage_from[:3,3]+w*displacement;v=rate*displacement
            foot[2,3]+=arch*np.sin(np.pi*u)**2;v[2]+=arch*np.pi*np.sin(2*np.pi*u)/duration
            dr=Rotation.from_matrix(goal[:3,:3]@self.stage_from[:3,:3].T).as_rotvec()
            foot[:3,:3]=Rotation.from_rotvec(w*dr).as_matrix()@self.stage_from[:3,:3];omega=rate*dr
        return dict(foot=foot,velocity=v,angular_velocity=omega,right_load=load,phase=phase)
