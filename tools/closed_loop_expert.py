"""Frozen feedback experts, masked recurrent state and original shot envelopes.

No physical state writes. Import in simulator processes only after AppLauncher.
"""
import json
from pathlib import Path
import numpy as np
import torch
from soccer_onnx_policy import OnnxSoccerPolicy as NativeSoccerPolicy
ROOT=Path(__file__).resolve().parents[1]


def cosine_window(frames,window):
    start,rise,fall,end=window
    phase=torch.where(frames<=rise,(frames-start)/(rise-start),(end-frames)/(end-fall)).clamp(0.,1.)
    return .5-.5*torch.cos(torch.pi*phase)


def measured_context_gate(shift,velocity_error):
    # Fixed numerical dead zone, not a simulator case id or desired outcome.
    scale=torch.tensor([.05,.10],device=shift.device,dtype=shift.dtype)
    xy=torch.where(shift.abs()<=1e-4,0.,shift)/scale
    dv=torch.where(velocity_error.abs()<=1e-4,0.,velocity_error)/.20
    return torch.cat([xy,dv],-1).norm(dim=-1).clamp(0.,1.)


class FrozenRecurrent:
    def __init__(self,kind,n,device,checkpoint,backend='torch'):
        self.n=n;self.device=device;self.native=None
        if kind!='running' and backend=='onnx':
            if n!=1:raise ValueError('Exact ONNX backend is a single-environment audit')
            self.native=NativeSoccerPolicy(checkpoint,1);return
        if kind=='running':
            from running_balance import RunningBalanceActor
            self.model=RunningBalanceActor(176,286,29,warmstart_path=checkpoint).to(device).eval()
        else:
            from paid_actor_warmstart import PaidActorCriticRecurrent
            self.model=PaidActorCriticRecurrent(160,286,29,onnx_path=checkpoint).to(device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.h=torch.zeros(2,n,128,device=device);self.c=torch.zeros_like(self.h)

    def reset(self,ids):
        if self.native is not None:
            if len(ids):self.native.reset()
        else:self.h[:,ids]=0.;self.c[:,ids]=0.

    def act(self,observations,mask,frames):
        output=torch.zeros(self.n,29,device=self.device)
        ids=mask.nonzero(as_tuple=False).flatten()
        if not len(ids):return output
        if self.native is not None:
            output[:]=torch.from_numpy(self.native.step(observations.cpu().numpy(),frames.cpu().numpy())).to(self.device)
        else:
            x=(observations[ids]-self.model.paid_obs_mean)/self.model.paid_obs_denominator
            y,(h,c)=self.model.memory_a.rnn(x.unsqueeze(0),(self.h[:,ids].contiguous(),self.c[:,ids].contiguous()))
            self.h[:,ids]=h;self.c[:,ids]=c
            output[ids]=self.model.actor(y.squeeze(0))
        return output


class FrozenStrideExpert:
    def __init__(self,env,config):
        self.env=env;self.robot=env.scene['robot'];self.ball=env.scene['soccer_ball'];self.motion=env.command_manager.get_term('motion')
        self.n=env.num_envs;self.device=self.robot.device;self.config=config
        self.plan=json.loads((ROOT/config['expert_plan']).read_text())
        backend=config.get('teacher_backend','torch')
        if backend=='auto':backend='onnx' if self.n==1 else 'torch'
        self.backend=backend
        self.running=FrozenRecurrent('running',self.n,self.device,ROOT/self.plan['running_checkpoint'])
        self.swing=FrozenRecurrent('soccer',self.n,self.device,ROOT/config['soccer_checkpoint'],backend)
        self.recovery=FrozenRecurrent('soccer',self.n,self.device,ROOT/config['soccer_checkpoint'],backend)
        data=np.load(ROOT/self.plan['reference'])
        self.ref={k:torch.as_tensor(data[k],device=self.device) for k in ['joint_pos','joint_vel','body_ang_vel_w']}
        self.ids=[self.robot.joint_names.index('right_'+x+'_joint') for x in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
        self.start=torch.full((self.n,),-1.,device=self.device);self.ready=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.last_running=torch.zeros(self.n,29,device=self.device);self.visible=torch.zeros(self.n,3,device=self.device)
        self.cue_initialized=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.ball_cue=torch.zeros(self.n,3,device=self.device);self.goal_cue=torch.zeros_like(self.ball_cue)
        self.goal=torch.tensor([*self.plan['goal_xy'],.11],device=self.device)+env.scene.env_origins
        self.strike_target=torch.full((29,),float('nan'),device=self.device,dtype=torch.float64)
        for name,value in self.plan['strike']['targets'].items():self.strike_target[self.robot.joint_names.index(name)]=value
        self.velocity_peak=torch.zeros(29,device=self.device,dtype=torch.float64)
        for name,value in self.plan['strike_velocity']['velocities'].items():self.velocity_peak[self.robot.joint_names.index(name)]=value
        self.events=[];self.last_base_observations=None

    def reset(self,ids):
        self.running.reset(ids);self.swing.reset(ids);self.recovery.reset(ids)
        self.start[ids]=-1.;self.ready[ids]=0;self.last_running[ids]=0.;self.visible[ids]=0.
        self.cue_initialized[ids]=False;self.ball_cue[ids]=0.;self.goal_cue[ids]=0.

    def observations(self):
        from isaaclab.utils.math import quat_apply_inverse
        r=self.robot.data
        ball=quat_apply_inverse(r.root_quat_w,self.ball.data.root_pos_w-r.root_pos_w)
        goal=quat_apply_inverse(r.root_quat_w,self.goal-r.root_pos_w)
        return torch.cat([self.motion.command,r.projected_gravity_b,self.motion.anchor_ang_vel_w,
            r.root_ang_vel_b,r.joint_pos-r.default_joint_pos,r.joint_vel-r.default_joint_vel,
            self.env.action_manager.action,ball,goal],-1)

    def act(self,frame,elapsed,kicked,*,landing_retention=None,release_mode="all"):
        from isaaclab.utils.math import quat_apply_inverse,quat_apply
        from soccer.tasks.tracking import mdp
        from running_balance import ground_force,support_observation,reference_contact
        r=self.robot.data;now=elapsed.double()*.02;phase=frame.double()*.02
        recovering=self.start>=0
        desired=torch.where(recovering,self.plan['shot_start_frame']+self.plan['recovery_entry_frame']+elapsed-self.start,torch.round(frame)+1.)
        self.motion.time_steps[:]=torch.minimum(desired.long(),self.motion.motion_length-1)
        support=ground_force(self.env);distance=(self.ball.data.root_pos_w[:,:2]-r.root_pos_w[:,:2]).norm(dim=-1)
        gate=self.plan.get('recovery_gate',{})
        ready=(~recovering)&(now>=gate.get('earliest_s',3.))&(self.motion.anchor_lin_vel_w[:,:2].norm(dim=-1)<.1)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.45)&(-r.projected_gravity_b[:,2]>np.cos(np.deg2rad(15.)))&(support.amin(-1)>=30.)&(r.root_pos_w[:,2]-self.env.scene.env_origins[:,2]>=gate.get('minimum_root_height_m',0.))
        self.ready=torch.where(ready,self.ready+1,torch.zeros_like(self.ready))
        entering=ready&(self.ready>=gate.get('consecutive_policy_steps',1))
        self.start[entering]=elapsed[entering]
        self.motion.time_steps[entering]=self.plan['shot_start_frame']+self.plan['recovery_entry_frame']
        if self.n==1 and bool(entering[0]):self.events.append(dict(time_s=float(now[0]),kind='post_shot_standing_recovery',physical_state_write=False))
        recovering=self.start>=0;obs=self.observations();self.last_base_observations=obs
        visible=(~kicked)|(distance<=2.);self.visible[visible]=obs[visible,-6:-3]
        running_obs=obs.clone()
        for sl,y in [(slice(-6,-3),-8.),(slice(-3,None),-10.)]:
            point=torch.tensor([0.,y,.11],device=self.device)+self.env.scene.env_origins
            running_obs[:,sl]=quat_apply_inverse(r.root_quat_w,point-r.root_pos_w)
        running_obs=torch.cat([running_obs,mdp.motion_anchor_pos_b(self.env,'motion'),mdp.motion_anchor_ori_b(self.env,'motion'),r.root_lin_vel_b,support_observation(self.env),reference_contact(self.env)],-1)
        running=self.running.act(running_obs,~recovering,self.motion.time_steps)
        self.last_running[~recovering]=running[~recovering];action=running.clone()
        start,full,release,end=self.plan['stride']['window_s']
        swinging=(~recovering)&(phase>=start)&(phase<end)
        shot_frame=(self.plan['shot_start_frame']+self.plan['stride']['entry_frame']+torch.round((phase-start)/.02)).long().clamp(0,len(self.ref['joint_pos'])-1)
        shot_obs=obs.clone();shot_obs[:,:29]=self.ref['joint_pos'][shot_frame];shot_obs[:,29:58]=self.ref['joint_vel'][shot_frame]
        shot_obs[:,61:64]=self.ref['body_ang_vel_w'][shot_frame,self.motion.robot_anchor_body_index]
        swing=self.swing.act(shot_obs,swinging,shot_frame-self.plan['shot_start_frame'])
        swing_original=cosine_window(phase,[start,full,release,end]).float()[:,None]
        weight=swing_original if landing_retention is None or release_mode=='pose' else swing_original*landing_retention[:,None]
        swing_effective=weight
        mixed=(1-weight)*running[:,self.ids]+weight*swing[:,self.ids]
        action[:,self.ids]=torch.where(swinging[:,None],mixed,action[:,self.ids])
        recovery_obs=obs.clone();hold=recovering&kicked&(distance>2.)
        recovery_obs[hold,-6:-3]=self.visible[hold]
        create=hold&~self.cue_initialized
        cues=self.plan.get('recovery_cues_local')
        if cues:
            for key,buffer in [('ball',self.ball_cue),('goal',self.goal_cue)]:
                point=torch.tensor(cues[key],device=self.device).expand(self.n,3)
                world=r.root_pos_w+quat_apply(r.root_quat_w,point);world[:,2]=self.env.scene.env_origins[:,2]+.11
                buffer[create]=world[create]
            self.cue_initialized |= create
            recovery_obs[hold,-6:-3]=quat_apply_inverse(r.root_quat_w[hold],self.ball_cue[hold]-r.root_pos_w[hold])
            recovery_obs[hold,-3:]=quat_apply_inverse(r.root_quat_w[hold],self.goal_cue[hold]-r.root_pos_w[hold])
        recovery=self.recovery.act(recovery_obs,recovering,(self.motion.time_steps-self.plan['shot_start_frame']).clamp_min(0))
        blend=(.5-.5*torch.cos(torch.pi*((elapsed-self.start).double()*.02/.1).clamp(0.,1.))).float()[:,None]
        action=torch.where(recovering[:,None],(1-blend)*self.last_running+blend*recovery,action)
        # External frozen strike/pulse use the pre-action motion frame; initial
        # command is frame0 and all later approach commands have the +1 lead.
        external_frame=torch.where(elapsed==0,torch.zeros_like(frame),frame+1.).double()
        external_frame=torch.where(recovering,self.motion.time_steps.double(),external_frame)
        term=self.env.action_manager.get_term('joint_pos');scale=torch.as_tensor(term._scale,device=self.device).double();offset=torch.as_tensor(term._offset,device=self.device).double()
        absolute_original=cosine_window(external_frame,self.plan['strike']['window'])[:,None]
        weight=absolute_original if landing_retention is None else absolute_original*landing_retention[:,None]
        self.release_weights=dict(swing_weight_original=swing_original[:,0],swing_weight_effective=swing_effective[:,0],absolute_weight_original=absolute_original[:,0],absolute_weight_effective=weight[:,0])
        target=action.double()*scale+offset;absolute=torch.isfinite(self.strike_target)[None,:]&(absolute_original>0.)
        modified=target*(1-absolute_original)+torch.nan_to_num(self.strike_target)[None,:]*absolute_original
        limits=self.robot.data.soft_joint_pos_limits
        projected=modified.clamp(min=limits[:,:,0],max=limits[:,:,1])
        modified=projected
        if landing_retention is not None:
            from landing_release import retain_operator
            modified=retain_operator(target,projected,landing_retention)
            self.release_targets=dict(absolute_base_target_rad=target[:,self.ids],
                absolute_projected_target_rad=projected[:,self.ids],absolute_blended_target_rad=modified[:,self.ids],
                absolute_soft_limits_rad=limits[:,self.ids])
        original_action=action
        action=torch.where(absolute,((modified-offset)/scale).float(),action)
        if landing_retention is not None:
            action=torch.where((landing_retention==0.)[:,None],original_action,action)
        pulse=self.plan['strike_velocity'];w=cosine_window(external_frame,pulse['window'])
        velocity=(w[:,None]*self.velocity_peak).float()*(support[:,0]>=pulse['minimum_support_N'])[:,None]
        return action,velocity.clamp(-r.joint_vel_limits,r.joint_vel_limits)
