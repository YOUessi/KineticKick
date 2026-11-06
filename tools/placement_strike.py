"""Measured-state placement/phase controller over the frozen feedback experts.

A small fitted gain vector changes commands, never physical robot/ball state.
Swing IK preserves world foot orientation, not merely the ankle joint angles.
"""
import torch
from closed_loop_strike import ClosedLoopStrikeTask
from closed_loop_expert import cosine_window,FrozenRecurrent

PARAMETER_NAMES=['placement_x','placement_y','root_p_x','root_p_y','root_d_x','root_d_y',
                 'phase_gain','foot_x','foot_y','yaw_x','yaw_y','yaw_v','yaw_gate',
                 'stance_pitch_p','stance_pitch_d','stance_roll_p','stance_roll_d']
DEFAULT_PARAMETERS=[1.,1.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.]


class BatchedLeg:
    def __init__(self,path,device,maximum_delta=.25):
        self.maximum_delta=maximum_delta
        from cartesian_strike import LegKinematics
        leg=LegKinematics(path)
        self.origins=torch.tensor(__import__('numpy').array(leg.origins),dtype=torch.float32,device=device)
        self.axes=torch.tensor(__import__('numpy').array(leg.axes),dtype=torch.float32,device=device)
        self.limits=torch.tensor(leg.limits,dtype=torch.float32,device=device)

    def forward(self,q,root_rotation,root_position):
        rotation=root_rotation;position=root_position;axes=[];centers=[]
        for j in range(6):
            origin=self.origins[j]
            position=position+(rotation@origin[:3,3]).squeeze(-1)
            rotation=rotation@origin[:3,:3]
            axis=self.axes[j];axes.append(rotation@axis);centers.append(position)
            x,y,z=axis;zero=x*0
            skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
            angle=q[:,j,None,None]
            turn=torch.eye(3,device=q.device)+torch.sin(angle)*skew+(1.-torch.cos(angle))*(skew@skew)
            rotation=rotation@turn
        axes=torch.stack(axes,dim=-1);centers=torch.stack(centers,dim=1)
        linear=torch.linalg.cross(axes.transpose(1,2),position[:,None]-centers,dim=-1).transpose(1,2)
        return position,rotation,torch.cat([linear,axes],dim=1)

    def translate(self,q0,rotation,position,delta_xy,yaw,pitch=None,height=None,reference=None,maximum_delta=None,normal_only=False,contact_point_local=None):
        normal_mask=torch.as_tensor(normal_only,device=q0.device,dtype=torch.bool).expand(len(q0))
        has_normal=bool(normal_mask.any())
        target,target_rotation,_=self.forward(q0,rotation,position)
        if reference is not None:target,target_rotation=reference
        target=target.clone();target[:,:2]+=delta_xy
        if height is not None:target[:,2]+=height
        c=yaw.cos();s=yaw.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
        turn=torch.stack([c,-s,z,s,c,z,z,z,o],-1).reshape(-1,3,3)
        target_rotation=turn@target_rotation
        if pitch is not None:
            c=pitch.cos();s=pitch.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
            pitch_turn=torch.stack([c,z,s,z,o,z,-s,z,c],-1).reshape(-1,3,3)
            target_rotation=target_rotation@pitch_turn
        point=None
        if contact_point_local is not None and has_normal:
            point=torch.as_tensor(contact_point_local,device=q0.device,dtype=q0.dtype)[None].expand(len(q0),3)*normal_mask[:,None]
            target=target+(target_rotation@point[:,:,None]).squeeze(-1)
        maximum=self.maximum_delta if maximum_delta is None else maximum_delta[:,None]
        q=q0.clone();low=torch.maximum(q0-maximum,self.limits[:,0]);high=torch.minimum(q0+maximum,self.limits[:,1])
        eye=torch.eye(6,device=q.device)[None]
        for _ in range(10):
            p,r,j=self.forward(q,rotation,position)
            if point is not None:
                lever=(r@point[:,:,None]).squeeze(-1)
                p=p+lever;j=j.clone()
                j[:,:3]+=torch.linalg.cross(j[:,3:].transpose(1,2),lever[:,None].expand(-1,6,-1),dim=-1).transpose(1,2)
            mismatch=target_rotation@r.transpose(1,2)
            ori=.5*torch.stack([mismatch[:,2,1]-mismatch[:,1,2],mismatch[:,0,2]-mismatch[:,2,0],mismatch[:,1,0]-mismatch[:,0,1]],-1)
            j=j.clone()
            if has_normal:
                normal_error=target_rotation[:,:,2]-r[:,:,2]
                normal_jacobian=torch.linalg.cross(j[:,3:].transpose(1,2),r[:,:,2][:,None].expand(-1,6,-1),dim=-1).transpose(1,2)
                ori=torch.where(normal_mask[:,None],normal_error,ori)
                j[:,3:]=torch.where(normal_mask[:,None,None],normal_jacobian,j[:,3:])
            error=torch.cat([target-p,ori*.15],-1)
            j[:,3:]*=.15
            dq=torch.linalg.solve(j.transpose(1,2)@j+eye*1e-5,(j.transpose(1,2)@error[:,:,None])).squeeze(-1)
            q=(q+dq.clamp(-.1,.1)).clamp(min=low,max=high)
        p,r,_=self.forward(q,rotation,position)
        if point is not None:p=p+(r@point[:,:,None]).squeeze(-1)
        angle=((torch.diagonal(target_rotation@r.transpose(1,2),dim1=1,dim2=2).sum(-1)-1.)*.5).clamp(-1,1).acos()
        if has_normal:angle=torch.where(normal_mask,(target_rotation[:,:,2]*r[:,:,2]).sum(-1).clamp(-1,1).acos(),angle)
        return q,(target-p).norm(dim=-1),angle


    def translate_feasible(self,q0,rotation,position,delta_xy,yaw,*,pitch=None,height=None):
        """Backtrack an unreachable task target instead of discarding all reach.

        Preserve feasible original commands exactly. The bounded local search
        finds an accepted fraction; it does not certify global IK optimality.
        """
        q,error,angle=self.translate(q0,rotation,position,delta_xy,yaw,pitch,height)
        valid=(error<.025)&(angle<.12)
        fraction=torch.ones_like(error)
        ids=(~valid).nonzero(as_tuple=False).flatten()
        if not len(ids):return q,error,angle,fraction
        q[ids]=q0[ids];error[ids]=0.;angle[ids]=0.;fraction[ids]=0.
        low=torch.zeros(len(ids),device=q0.device);high=torch.ones_like(low)
        for _ in range(6):
            mid=(low+high)*.5
            candidate,e,a=self.translate(q0[ids],rotation[ids],position[ids],delta_xy[ids]*mid[:,None],yaw[ids]*mid,
                None if pitch is None else pitch[ids]*mid,None if height is None else height[ids]*mid)
            feasible=(e<.025)&(a<.12)
            chosen=ids[feasible]
            q[chosen]=candidate[feasible];error[chosen]=e[feasible];angle[chosen]=a[feasible];fraction[chosen]=mid[feasible]
            low=torch.where(feasible,mid,low);high=torch.where(feasible,high,mid)
        return q,error,angle,fraction


class PlacementStrikeTask(ClosedLoopStrikeTask):
    def __init__(self,env,config):
        super().__init__(env,config)
        from pathlib import Path
        self.swing_max=float(config.get('maximum_swing_correction_rad',.25))
        if config.get('project_swing_target') and any(config.get(k) for k in ['fast_global_retarget','fast_normal_only','fast_soft_ik','fast_project_ik']):raise ValueError('Feasible reach projection uses the original full-orientation IK only')
        if not .1<=self.swing_max<=.4:raise ValueError('Swing correction bound outside0.1..0.4rad')
        self.leg=BatchedLeg(Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf',self.device,self.swing_max)
        values=list(config.get('placement_parameters',DEFAULT_PARAMETERS))
        if len(values)==13:values+=4*[0.]
        if len(values)!=17 or not all(__import__('math').isfinite(v) for v in values):raise ValueError('17 finite fitted gains required')
        self.parameters=torch.tensor(values,device=self.device).expand(self.n,-1).clone()
        self.stance_ids=[[self.robot.joint_names.index(side+'_'+name+'_joint') for name in ['hip_pitch','hip_roll','ankle_pitch','ankle_roll']] for side in ['left','right']]
        prefix_names=[side+'_'+joint+'_joint' for side in ['left','right'] for joint in ['hip_pitch','hip_roll','hip_yaw','ankle_pitch','ankle_roll']]
        self.prefix_ids=[self.robot.joint_names.index(name) for name in prefix_names]
        coefficients=config.get('fast_prefix_coefficients',[[0.]*10 for _ in range(3)])
        prefix=torch.tensor(coefficients,device=self.device)
        if prefix.shape!=(3,10) or not bool(torch.isfinite(prefix).all()) or bool((prefix.abs()>.20).any()):raise ValueError('Finite3x10 prefix coefficients bounded by0.20rad required')
        self.prefix_parameters=prefix.expand(self.n,3,10).clone()
        self.approach_cfg=config.get('approach_adapter')
        approach=torch.tensor((self.approach_cfg or {}).get('coefficients',[[0.]*10 for _ in range(3)]),device=self.device)
        if approach.shape!=(3,10) or not bool(torch.isfinite(approach).all()) or bool((approach.abs()>.20).any()):raise ValueError('Finite3x10 approach coefficients bounded by0.20rad required')
        if self.approach_cfg:
            import math
            ac=self.approach_cfg
            if not all(math.isfinite(ac[k]) and ac[k]>0 for k in ['lookahead_s','scale_m','maximum_rad','maximum_step_rad']) or ac['maximum_rad']>.2 or ac['maximum_step_rad']>.04:raise ValueError('Invalid approach adapter bounds')
        self.approach_parameters=approach.expand(self.n,3,10).clone()
        self.approach_previous=torch.zeros(self.n,10,device=self.device)
        self.clearance_cfg=config.get('follow_through_clearance')
        self.hip_clearance=None
        if self.clearance_cfg:
            import math
            values=[self.clearance_cfg[k] for k in ['margin_m','deceleration_rad_s2','gain','maximum_correction_rad','latest_s']]
            if not all(math.isfinite(v) and v>0 for v in values) or self.clearance_cfg['maximum_correction_rad']>.4:raise ValueError('Finite positive clearance settings and correction<=0.4rad required')
            from hip_clearance import HipClearance
            self.hip_clearance=HipClearance(Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf',self.device)
            self.pitch_id=self.robot.joint_names.index('right_hip_pitch_joint')
            self.roll_id=self.robot.joint_names.index('right_hip_roll_joint')
        self.shank_cfg=config.get('shank_clearance');self.shank_geometry=None
        self.shank_delta=torch.zeros(self.n,device=self.device);self.shank_ankle_delta=self.shank_delta.clone()
        if self.shank_cfg:
            from shank_clearance import ShankClearance
            import math
            values=[self.shank_cfg[k] for k in ['margin_m','horizon_s','maximum_correction_rad','maximum_step_rad','start_s','end_s','support_min_N','swing_max_N']]
            if not all(math.isfinite(x) and x>0 for x in values) or self.shank_cfg['maximum_correction_rad']>.2 or self.shank_cfg['horizon_s']>.1 or self.shank_cfg['start_s']>=self.shank_cfg['end_s']:raise ValueError('Invalid bounded shank-clearance settings')
            self.shank_geometry=ShankClearance(Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf',self.robot.joint_names,self.device)
            self.shank_roll_id=self.robot.joint_names.index('right_hip_roll_joint');self.shank_ankle_id=self.robot.joint_names.index('right_ankle_roll_joint')
        self.anchor_shift=torch.zeros(self.n,3,device=self.device)
        self.phase_shift=torch.zeros(self.n,device=self.device)
        self.early_running=None
        if config.get('early_running_checkpoint'):
            import hashlib
            checkpoint=Path(__file__).resolve().parents[1]/config['early_running_checkpoint']
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=config['early_running_checkpoint_sha256']:raise ValueError('Early running checkpoint changed')
            self.early_running=FrozenRecurrent('running',self.n,self.device,checkpoint)
        native=self.expert.running.act
        def running(observations,mask,frames):
            from isaaclab.utils.math import quat_apply_inverse
            observations=observations.clone()
            observations[:,160:163]+=quat_apply_inverse(self.expert.motion.robot_anchor_quat_w,self.anchor_shift)
            base=native(observations,mask,frames)
            if self.early_running is not None:
                window=self.cfg.get('early_running_window_s',[0.,.10,.60,.90])
                now=self.elapsed.double()*.02
                fast=(-self.velocity_error[:,1]/.2).clamp(0,1)
                active=mask&(now<window[-1])&(fast>0)
                early=self.early_running.act(observations,active,frames)
                alpha=cosine_window(now,window).float()*fast
                base=base*(1-alpha[:,None])+early*alpha[:,None]
            return base
        self.expert.running.act=running

    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.anchor_shift[ids]=0.;self.phase_shift[ids]=0.;self.shank_delta[ids]=0.;self.shank_ankle_delta[ids]=0.
        self.approach_previous[ids]=0.
        if self.early_running is not None:self.early_running.reset(ids)

    def commands(self,actions=None):
        from isaaclab.utils.math import matrix_from_quat,quat_apply_inverse
        from running_balance import ground_force
        from joint_strike_core import residual_action
        self.capture_context();r=self.robot.data;p=self.parameters;now=self.elapsed*.02
        speed_gate=(self.velocity_error[:,1].abs()/.2).clamp(0,1)
        gait_gate=torch.maximum(speed_gate,(self.shift[:,1].abs()/.1).clamp(0,1)) if self.cfg.get('context_gait_gate') else torch.ones_like(speed_gate)
        near_gate=(self.shift[:,1]/.1).clamp(0,1)*(1.-speed_gate)
        gait_gate*=1.-near_gate*(1.-float(self.cfg.get('near_gait_scale',1.)))
        prefix_only=(self.velocity_error[:,1]<-1e-4)&bool(self.cfg.get('fast_prefix_only'))
        gait_gate=torch.where(prefix_only,0.,gait_gate)
        nominal=self.sample('root_pose',self.elapsed)[:,:2]
        velocity=self.sample('root_velocity',self.elapsed)[:,:2]
        root=r.root_pos_w[:,:2]-self.env.scene.env_origins[:,:2]
        ball=self.ball.data.root_pos_w[:,:2]-self.env.scene.env_origins[:,:2]
        observed_shift=ball-torch.tensor(self.cfg['ball_xy'],device=self.device)
        shift=torch.where(self.contact[:,None],self.shift,observed_shift).clamp(-.15,.15)
        ramp=(now/1.2).clamp(0,1);ramp=ramp*ramp*(3-2*ramp)
        fade=((3.-now)/.9).clamp(0,1)
        command=p[:,:2]*shift*ramp[:,None]+p[:,2:4]*(nominal-root)+p[:,4:6]*(velocity-r.root_lin_vel_w[:,:2])*.15
        self.anchor_shift[:,:2]=command.clamp(-.4,.4)*(self.gate*fade*gait_gate*(1.-(-self.velocity_error[:,1]/.2).clamp(0,1)*(1.-float(self.cfg.get('fast_anchor_scale',1.)))))[:,None]
        desired=((shift[:,1]-(root-nominal)[:,1])/(-r.root_lin_vel_w[:,1]).clamp_min(.8)*p[:,6]).clamp(-.12,.12)
        desired*=((now-.6)/.6).clamp(0,1)*self.gate*gait_gate
        desired=torch.where(desired>0,desired*float(self.cfg.get('positive_phase_gain',1.)),desired)
        if self.cfg.get('phase_advance_after_s') is not None:
            support_ready=ground_force(self.env)[:,0]>=80.
            advance=((now-float(self.cfg['phase_advance_after_s']))/.08).clamp(0,1)*support_ready
            desired=torch.where(desired>0,desired*advance,desired)
        active=(~self.contact)&(now<2.1)
        self.phase_shift+=torch.where(active,(desired-self.phase_shift).clamp(-.008,.008),0.)
        self.frame=self.elapsed+self.phase_shift/.02
        with torch.no_grad():base,self.step_velocity=self.expert.act(self.frame,self.elapsed,self.contact)
        term=self.env.action_manager.get_term('joint_pos')
        prefix_delta=torch.zeros_like(base)
        if self.cfg.get('fast_prefix_only'):
            windows=self.cfg.get('fast_prefix_windows_s',[[0.,.05,.15,.35],[.2,.3,.5,.65],[.5,.65,.8,1.]])
            weights=torch.stack([cosine_window(now.double(),window).float() for window in windows],-1)
            correction=(weights[:,:,None]*self.prefix_parameters).sum(1).clamp(-.20,.20)*((-self.velocity_error[:,1]/.2).clamp(0,1)*prefix_only)[:,None]
            prefix_delta[:,self.prefix_ids]=correction
            base=residual_action(base,torch.atanh((prefix_delta/.20).clamp(-.999999,.999999)),term._scale,term._offset,self.robot.data.soft_joint_pos_limits,.20)
        approach_delta=torch.zeros_like(base);approach_feature=torch.zeros(self.n,device=self.device)
        approach_request=torch.zeros_like(self.approach_previous);approach_before=approach_request.clone()
        approach_limits=torch.zeros(self.n,10,2,device=self.device)
        if self.approach_cfg:
            from approach_adapter import approach_features,approach_command
            cfg=self.approach_cfg
            approach_feature=approach_features(nominal,velocity,root,r.root_lin_vel_w[:,:2],shift,ramp,self.velocity_error[:,1],cfg['lookahead_s'],cfg['scale_m'])
            windows=self.cfg['fast_prefix_windows_s']
            weights=torch.stack([cosine_window(now.double(),w).float() for w in windows],-1)
            approach_request=approach_command(weights,self.approach_parameters,approach_feature,self.approach_previous,cfg['maximum_rad'],cfg['maximum_step_rad'])
            approach_before=(base*term._scale+term._offset)[:,self.prefix_ids]
            approach_limits=r.soft_joint_pos_limits[:,self.prefix_ids]
            delta=torch.zeros_like(base);delta[:,self.prefix_ids]=approach_request
            maximum=cfg['maximum_rad']
            adapted=residual_action(base,torch.atanh((delta/maximum).clamp(-.999999,.999999)),term._scale,term._offset,r.soft_joint_pos_limits,maximum)
            approach_delta=(adapted-base)*term._scale;self.approach_previous=approach_delta[:,self.prefix_ids];base=adapted
        error=torch.cat([nominal+shift*ramp[:,None]-root,torch.zeros(self.n,1,device=self.device)],-1)
        verror=torch.cat([velocity-r.root_lin_vel_w[:,:2],torch.zeros(self.n,1,device=self.device)],-1)
        body_error=quat_apply_inverse(r.root_quat_w,error);body_velocity=quat_apply_inverse(r.root_quat_w,verror)
        pitch=(p[:,13]*body_error[:,0]+p[:,14]*body_velocity[:,0]).clamp(-.15,.15)
        fast_gate=(-self.velocity_error[:,1]/.2).clamp(0,1)
        roll_d=p[:,16]+fast_gate*float(self.cfg.get('fast_roll_d_add',0.))
        roll=(-p[:,15]*body_error[:,1]-roll_d*body_velocity[:,1]).clamp(-.15,.15)
        speed_gate=(self.velocity_error[:,1].abs()/.2).clamp(0,1)
        support=ground_force(self.env).clamp_min(0);support=support/support.sum(-1,keepdim=True).clamp_min(80.)
        stance_gate=speed_gate*(1.-fast_gate*(1.-float(self.cfg.get('fast_stance_scale',1.))))
        stance_gate=torch.where(prefix_only,0.,stance_gate)
        fade_stance=((1.42-now)/.22).clamp(0,1)*stance_gate
        roll_end=1.42+fast_gate*(float(self.cfg.get('fast_stance_release_s',1.42))-1.42)
        fade_roll=((roll_end-now)/.22).clamp(0,1)*stance_gate
        direct=torch.zeros_like(base)
        for side,joints in enumerate(self.stance_ids):
            weight_stance=support[:,side]*fade_stance
            roll_weight=support[:,side]*fade_roll
            direct[:,joints]=torch.stack([pitch*weight_stance,roll*roll_weight,-pitch*weight_stance,-roll*roll_weight],-1)
        base=residual_action(base,torch.atanh((direct/.15).clamp(-.999999,.999999)),term._scale,term._offset,self.robot.data.soft_joint_pos_limits,.15)
        window=list(self.expert.plan['strike']['window'])
        if self.cfg.get('late_near_retarget'):
            near_gate=(self.shift[:,1]/.1).clamp(0,1).double()
            window[0]=window[0]+6.*near_gate;window[1]=window[1]+2.*near_gate
        weight=cosine_window(self.frame.double()+1.,window).float()*self.gate
        ids=(weight>0).nonzero(as_tuple=False).flatten();out=base.clone()
        errors=torch.zeros(self.n,device=self.device);angles=torch.zeros_like(errors);reach_fraction=torch.zeros_like(errors)
        if len(ids):
            joints=self.expert.ids
            q0=(base*term._scale+term._offset)[ids][:,joints]
            nominal_phase=self.sample('root_pose',self.frame)[:,:2]
            reach=float(self.cfg.get('foot_reach_limit_m',.12))
            delta=(shift-(root-nominal_phase)).clamp(-reach,reach)*p[:,7:9]
            delta=torch.where(prefix_only[:,None],shift*p[:,7:9],delta)
            yaw=(p[:,9]*self.shift[:,0]/.05+p[:,10]*self.shift[:,1]/.1+p[:,11]*self.velocity_error[:,1]/.2+p[:,12]*self.gate).clamp(-.25,.25)*gait_gate
            height_gate=speed_gate if self.cfg.get('height_condition')=='initial_velocity' else self.gate
            if self.cfg.get('height_condition')=='velocity_or_far_ball':height_gate=torch.maximum(speed_gate,(-self.shift[:,1]/.1).clamp(0,1))
            height_gate=torch.where(prefix_only,0.,height_gate)
            reference=None;maximum=None;pitch=self.gate[ids]*float(self.cfg.get('foot_pitch_gain_rad',0.));height=height_gate[ids]*float(self.cfg.get('swing_height_offset_m',0.))
            if self.cfg.get('fast_global_retarget') and bool((self.velocity_error[ids,1]<-1e-4).any()):
                fast=self.velocity_error[ids,1]<-1e-4
                recorded_root=self.sample('root_pose',self.frame)[ids].clone();recorded_root[:,:3]+=self.env.scene.env_origins[ids]
                recorded_root[:,3:]/=recorded_root[:,3:].norm(dim=-1,keepdim=True).clamp_min(1e-8)
                recorded_q=(self.sample('action',self.frame)*term._scale+term._offset)[ids][:,joints]
                nominal_p,nominal_r,_=self.leg.forward(recorded_q,matrix_from_quat(recorded_root[:,3:]),recorded_root[:,:3])
                current_p,current_r,_=self.leg.forward(q0,matrix_from_quat(r.root_quat_w[ids]),r.root_pos_w[ids])
                reference=(torch.where(fast[:,None],nominal_p,current_p),torch.where(fast[:,None,None],nominal_r,current_r))
                delta[ids]=torch.where(fast[:,None],shift[ids],delta[ids]);yaw[ids]=torch.where(fast,0.,yaw[ids])
                height=torch.where(fast,0.,height);pitch=torch.where(fast,0.,pitch)
                maximum=torch.where(fast,float(self.cfg.get('fast_swing_correction_rad',self.swing_max)),self.swing_max)
            if self.cfg.get('project_swing_target'):
                q,error,angle,fraction=self.leg.translate_feasible(q0,matrix_from_quat(r.root_quat_w[ids]),r.root_pos_w[ids],delta[ids],yaw[ids],pitch=pitch,height=height)
                reach_fraction[ids]=fraction
            else:
                q,error,angle=self.leg.translate(q0,matrix_from_quat(r.root_quat_w[ids]),r.root_pos_w[ids],delta[ids],yaw[ids],pitch,height,reference,maximum,(self.velocity_error[ids,1]<-1e-4) if self.cfg.get('fast_normal_only') else False,self.cfg.get('fast_contact_point_local'))
                reach_fraction[ids]=((error<.025)&(angle<.12)).float()
            valid=((error<.025)&(angle<.12)).float()
            if self.cfg.get('fast_soft_ik'):
                feasible=((.08-error)/.055).clamp(0,1)*((.30-angle)/.18).clamp(0,1)
                valid=torch.where(self.velocity_error[ids,1]<-1e-4,feasible,valid)
            if self.cfg.get('fast_project_ik'):valid=torch.where(self.velocity_error[ids,1]<-1e-4,torch.ones_like(valid),valid)
            errors[ids]=error;angles[ids]=angle
            correction=(q-q0)*weight[ids,None]*valid[:,None]
            scale=torch.as_tensor(term._scale,device=self.device).expand(self.n,-1)
            out[ids[:,None],torch.tensor(joints,device=self.device)[None,:]]+=correction/scale[ids][:,joints]
        clearance_delta=torch.zeros(self.n,device=self.device)
        clearance_gap=torch.zeros_like(clearance_delta);clearance_stop=torch.zeros_like(clearance_delta)
        clearance_floor=torch.zeros_like(clearance_delta);clearance_active=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        if self.hip_clearance is not None:
            from hip_clearance import braking_correction,pulse_has_finished
            force=self.env.scene['right_ball_contact'].data.force_matrix_w
            separated=torch.zeros(self.n,device=self.device,dtype=torch.bool) if force is None else force.reshape(self.n,-1,3).sum(1).norm(dim=-1)<=2.
            cfg=self.clearance_cfg
            clearance_active=self.contact&separated&(self.gate>0.)&(now<float(cfg['latest_s']))&(r.joint_vel[:,self.pitch_id]<0.)&(r.joint_pos[:,self.pitch_id]<-.6)&pulse_has_finished(self.frame+1.,self.step_velocity[:,self.pitch_id],self.expert.plan['strike_velocity']['window'][-1])
            selected=clearance_active.nonzero(as_tuple=False).flatten()
            if len(selected):
                position=r.joint_pos[selected,self.pitch_id];velocity=r.joint_vel[selected,self.pitch_id];roll=r.joint_pos[selected,self.roll_id]
                floor,feasible=self.hip_clearance.pitch_floor(roll,float(cfg['margin_m']))
                requested,stopping=braking_correction(position,velocity,floor,feasible,
                    deceleration=float(cfg['deceleration_rad_s2']),gain=float(cfg['gain']),maximum=float(cfg['maximum_correction_rad']))
                delta=torch.zeros_like(out);delta[selected,self.pitch_id]=requested
                maximum=float(cfg['maximum_correction_rad'])
                corrected=residual_action(out,torch.atanh((delta/maximum).clamp(-.999999,.999999)),term._scale,term._offset,r.soft_joint_pos_limits,maximum)
                clearance_delta=((corrected-out)*term._scale)[:,self.pitch_id];out=corrected
                clearance_gap[selected]=self.hip_clearance.gap(position,roll);clearance_stop[selected]=stopping;clearance_floor[selected]=floor
        shank_active=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        shank_gap=torch.zeros_like(self.shank_delta);shank_pred=shank_gap.clone();shank_time=shank_gap.clone();shank_ankle=shank_gap.clone()
        shank_requested=torch.zeros(self.n,2,device=self.device);shank_before=shank_requested.clone();shank_limits=torch.zeros(self.n,2,2,device=self.device)
        if self.shank_geometry is not None:
            from shank_clearance import ankle_compensation
            cfg=self.shank_cfg;ground=ground_force(self.env)
            shank_active=(self.gate>0)&(~self.contact)&(now>=cfg['start_s'])&(now<cfg['end_s'])&(ground[:,0]>=cfg['support_min_N'])&(ground[:,1]<=cfg['swing_max_N'])
            requested,evidence=self.shank_geometry.command(r.joint_pos,r.joint_vel,shank_active,self.shank_delta,
                margin=cfg['margin_m'],horizon=cfg['horizon_s'],maximum=cfg['maximum_correction_rad'],maximum_step=cfg['maximum_step_rad'])
            ankle_request=ankle_compensation(requested,self.shank_ankle_delta,shank_active,cfg['maximum_step_rad'])
            shank_requested=torch.stack([requested,ankle_request],-1)
            indices=[self.shank_roll_id,self.shank_ankle_id]
            shank_before=(out*term._scale+term._offset)[:,indices];shank_limits=r.soft_joint_pos_limits[:,indices]
            delta=torch.zeros_like(out);delta[:,self.shank_roll_id]=requested;delta[:,self.shank_ankle_id]=ankle_request
            maximum=cfg['maximum_correction_rad']
            corrected=residual_action(out,torch.atanh((delta/maximum).clamp(-.999999,.999999)),term._scale,term._offset,r.soft_joint_pos_limits,maximum)
            applied=(corrected-out)*term._scale;self.shank_delta=applied[:,self.shank_roll_id];self.shank_ankle_delta=applied[:,self.shank_ankle_id];shank_ankle=self.shank_ankle_delta;out=corrected
            shank_gap=self.shank_geometry.gap(r.joint_pos);shank_pred=evidence['predicted_gap'];shank_time=evidence['closest_time']
        if self.n==1:
            self.debug.append(dict(time_s=float(now[0]),phase_s=float(self.frame[0])*.02,context_gate=float(self.gate[0]),
                measured_root_xy=root[0].tolist(),measured_ball_xyz=(self.ball.data.root_pos_w[0]-self.env.scene.env_origins[0]).tolist(),
                measured_initial_velocity_error=self.velocity_error[0].tolist(),contact_latched=bool(self.contact[0]),
                approach_feature=float(approach_feature[0]),approach_correction_rad=approach_delta[0].tolist(),approach_requested_rad=approach_request[0].tolist(),approach_target_before_rad=approach_before[0].tolist(),approach_soft_limits_rad=approach_limits[0].tolist(),
                placement_command_xyz=self.anchor_shift[0].tolist(),stance_correction_rad=direct[0].tolist(),prefix_correction_rad=prefix_delta[0].tolist(),ik_error_m=float(errors[0]),ik_angle_error_rad=float(angles[0]),swing_reach_fraction=float(reach_fraction[0]),
                residual_rad=((out-base)*term._scale)[0].tolist(),phase_action=float(self.phase_shift[0]),
                shank_requested_rad=shank_requested[0].tolist(),shank_target_before_rad=shank_before[0].tolist(),shank_soft_limits_rad=shank_limits[0].tolist(),
                shank_active=bool(shank_active[0]),shank_correction_rad=float(self.shank_delta[0]),shank_ankle_correction_rad=float(shank_ankle[0]),shank_gap_m=float(shank_gap[0]),shank_predicted_gap_m=float(shank_pred[0]),shank_closest_time_s=float(shank_time[0]),
                follow_through_active=bool(clearance_active[0]),follow_through_correction_rad=float(clearance_delta[0]),
                thigh_pelvis_gap_m=float(clearance_gap[0]),hip_stopping_angle_rad=float(clearance_stop[0]),hip_clearance_floor_rad=float(clearance_floor[0])))
        self.elapsed+=1
        return out


def get_task(env):
    if not hasattr(env,'_placement_strike_task'):env._placement_strike_task=PlacementStrikeTask(env,env.cfg.joint_strike)
    return env._placement_strike_task
