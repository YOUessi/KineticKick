"""Shared learned residual/phase execution for real approach, strike and recovery.

Import after AppLauncher in simulator processes. Pure tensor functions are also
usable in CPU tests. Physical state writes are confined to trainer reset code.
"""
from pathlib import Path
import json
import numpy as np
import torch
from rsl_rl.modules import ActorCritic


def bounded_phase_step(frame, elapsed, action, max_shift_frames=7.5, rate_limit=.30):
    rate=torch.where(elapsed<100.,rate_limit*torch.tanh(action),torch.zeros_like(action))
    target=frame+1.+rate
    return torch.maximum(frame,torch.clamp(target,min=elapsed+1.-max_shift_frames,max=elapsed+1.+max_shift_frames))


def residual_action(base, raw, scale, offset, limits, maximum):
    delta=maximum*torch.tanh(raw)
    target=base*scale+offset
    # A frozen PD target may lie outside the kinematic interval. Preserve that
    # actuator command, and permit only inward corrections there. Clipping the
    # whole target would turn infinitesimal residuals into large discontinuities.
    lower=torch.minimum(limits[...,0]-target,torch.zeros_like(target))
    upper=torch.maximum(limits[...,1]-target,torch.zeros_like(target))
    delta=torch.clamp(delta,min=lower,max=upper)
    return base+delta/scale


class JointStrikeActor(ActorCritic):
    def __init__(self,num_actor_obs,num_critic_obs,num_actions,init_noise_std=.15,**kwargs):
        if num_actions!=30:raise ValueError('29 joint residuals plus one phase-rate action required')
        super().__init__(num_actor_obs,num_critic_obs,num_actions,actor_hidden_dims=[256,128,64],
                         critic_hidden_dims=[256,128,64],activation='elu',init_noise_std=init_noise_std,**kwargs)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)


class JointStrikeTask:
    def __init__(self,env,config):
        self.env=env;self.cfg=config;self.robot=env.scene['robot'];self.ball=env.scene['soccer_ball']
        self.device=self.robot.device;self.n=env.num_envs
        data=np.load(Path(__file__).resolve().parents[1]/config['template'])
        self.meta=json.loads(str(data['metadata']))
        if self.robot.joint_names!=self.meta['joint_names']:raise ValueError('Nominal actuator joint order mismatch')
        self.ref={k:torch.as_tensor(data[k],device=self.device,dtype=torch.float32) for k in data.files if k!='metadata'}
        self.frame=torch.zeros(self.n,device=self.device);self.elapsed=torch.zeros_like(self.frame)
        self.last_action=torch.zeros(self.n,30,device=self.device);self.shift=torch.zeros(self.n,2,device=self.device)
        self.contact=torch.zeros(self.n,device=self.device,dtype=torch.bool);self.scored=self.contact.clone()
        self.hit_event=torch.zeros(self.n,device=self.device);self.score_event=self.hit_event.clone()
        self.contact_quality=self.hit_event.clone();self.impulse=self.hit_event.clone();self.stable_seconds=self.hit_event.clone()
        self.previous_ball=self.ball.data.root_pos_w.clone()
        self.goal=torch.tensor([*config['goal_xy'],.11],device=self.device)+env.scene.env_origins
        self.feet=[self.robot.body_names.index(s+'_ankle_roll_link') for s in ['left','right']]
        self.nominal_mask=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.step_velocity=torch.zeros(self.n,29,device=self.device)
        self.debug=[]

    def sample(self,key,frame=None):
        f=self.frame if frame is None else frame
        table=self.ref[key];f=f.clamp(0,len(table)-1);lo=f.long();hi=(lo+1).clamp_max(len(table)-1)
        w=(f-lo).reshape((self.n,)+(1,)*(table.ndim-1))
        return table[lo]+w*(table[hi]-table[lo])

    def reset_fields(self,ids):
        self.frame[ids]=0.;self.elapsed[ids]=0.;self.last_action[ids]=0.;self.shift[ids]=0.
        self.contact[ids]=False;self.scored[ids]=False;self.hit_event[ids]=0.;self.score_event[ids]=0.
        self.contact_quality[ids]=0.;self.impulse[ids]=0.;self.step_velocity[ids]=0.;self.stable_seconds[ids]=0.
        self.previous_ball[ids]=self.ball.data.root_pos_w[ids]

    def root_target(self):
        target=self.sample('root_pose')[:,:3].clone()
        ramp=(self.frame*.02/1.2).clamp(0,1)
        target[:,:2]+=self.shift*ramp[:,None]
        return target

    def features(self):
        from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_conjugate,matrix_from_quat
        from running_balance import ground_force
        r=self.robot.data;origin=self.env.scene.env_origins
        root=r.root_pos_w-origin;qref=self.sample('joint_pos')
        refquat=self.sample('root_pose')[:,3:];refquat=refquat/refquat.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        ori=matrix_from_quat(quat_mul(quat_conjugate(r.root_quat_w),refquat))[:,:,:2].reshape(self.n,-1)
        ball_pos=quat_apply_inverse(r.root_quat_w,self.ball.data.root_pos_w-r.root_pos_w)/3.
        ball_vel=quat_apply_inverse(r.root_quat_w,self.ball.data.root_lin_vel_w)/5.
        goal=quat_apply_inverse(r.root_quat_w,self.goal-r.root_pos_w)/6.
        support=ground_force(self.env);support=support/(support+100.)
        foot_pos=(r.body_pos_w[:,self.feet]-r.root_pos_w[:,None]).reshape(self.n,6)
        foot_vel=r.body_lin_vel_w[:,self.feet].reshape(self.n,6)/5.
        phase=torch.stack([self.frame/700.,(self.frame-self.elapsed)/7.5,self.contact.float(),self.scored.float()],-1)
        values=[r.joint_pos-qref,r.joint_vel/15.,qref,self.sample('action')/5.,
                (root-self.root_target())/.2,r.root_lin_vel_w/3.,r.root_ang_vel_w/5.,
                (r.root_lin_vel_w-self.sample('root_velocity'))/3.,r.projected_gravity_b,ori,
                ball_pos,ball_vel,goal,foot_pos,foot_vel,support,phase,self.last_action]
        return torch.cat(values,-1).clamp(-5.,5.)

    def commands(self,actions):
        if actions.shape!=(self.n,30):raise ValueError('Residual action shape mismatch')
        term=self.env.action_manager.get_term('joint_pos')
        base=self.sample('action')
        out=residual_action(base,actions[:,:29],term._scale,term._offset,self.robot.data.soft_joint_pos_limits,self.cfg['maximum_joint_residual_rad'])
        self.step_velocity=torch.clamp(self.sample('velocity_command'),-self.robot.data.joint_vel_limits,self.robot.data.joint_vel_limits)
        if self.n==1:
            self.debug.append(dict(time_s=float(self.elapsed[0])*.02,phase_s=float(self.frame[0])*.02,
                residual_rad=((out-base)*term._scale)[0].detach().cpu().tolist(),
                phase_action=float(actions[0,29]),contact_latched=bool(self.contact[0]),
                measured_ball_xyz=(self.ball.data.root_pos_w[0]-self.env.scene.env_origins[0]).cpu().tolist()))
        self.frame=bounded_phase_step(self.frame,self.elapsed,actions[:,29],self.cfg['maximum_phase_shift_s']/.02,self.cfg['phase_rate_limit']);self.elapsed+=1
        self.last_action=torch.tanh(actions).detach()
        return out

    def after_physics(self,dt):
        from isaaclab.utils.math import quat_apply_inverse
        sensor=self.env.scene['right_ball_contact'];force=sensor.data.force_matrix_w
        if force is not None:
            vector=force.reshape(self.n,-1,3).sum(1);magnitude=vector.norm(dim=-1)
            normal=quat_apply_inverse(self.robot.data.body_quat_w[:,self.feet[1]],-vector/magnitude[:,None].clamp_min(1e-6))
            quality=normal[:,2].clamp(0.,1.)
            new=(magnitude>2.)&~self.contact
            self.hit_event+=new.float()*quality
            self.contact_quality=torch.where(new,quality,self.contact_quality)
            self.contact |= magnitude>2.
            self.impulse+=magnitude*dt
        current=self.ball.data.root_pos_w
        plane=self.goal[:,1]-.11
        crossing=(self.previous_ball[:,1]>plane)&(current[:,1]<=plane)
        fraction=((self.previous_ball[:,1]-plane)/(self.previous_ball[:,1]-current[:,1]).clamp_min(1e-8)).clamp(0.,1.)
        at=self.previous_ball+(current-self.previous_ball)*fraction[:,None]
        inside=(at[:,0]-self.goal[:,0]).abs()<=.8-.11
        inside &= (at[:,2]-self.env.scene.env_origins[:,2]>=.11-1e-4)&(at[:,2]-self.env.scene.env_origins[:,2]<=1.1-.11)
        goal=crossing&inside&self.contact&~self.scored
        self.score_event+=goal.float();self.scored |= goal
        self.previous_ball=current.clone()

    def reward(self):
        from running_balance import ground_force
        r=self.robot.data;root=r.root_pos_w-self.env.scene.env_origins
        root_error=(root-self.root_target()).square().sum(-1)
        q_error=(r.joint_pos-self.sample('joint_pos')).square().mean(-1)
        refq=self.sample('root_pose')[:,3:];refq=refq/refq.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        alignment=(r.root_quat_w*refq).sum(-1).square().clamp(0.,1.)
        speed_error=(r.root_lin_vel_w-self.sample('root_velocity')).square().sum(-1)
        track=4.*torch.exp(-root_error/.12**2)+2.*torch.exp(-q_error/.18**2)+2.*torch.exp(-(1.-alignment)/.03)+torch.exp(-speed_error/.8**2)
        direction=self.goal[:,:2]-self.ball.data.root_pos_w[:,:2];direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        forward=(self.ball.data.root_lin_vel_w[:,:2]*direction).sum(-1).clamp(0.,6.)
        impulse_bonus=self.hit_event*(40.+15.*forward)
        support=ground_force(self.env)
        stable=(r.root_pos_w[:,2]-self.env.scene.env_origins[:,2]>.55)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.3)&(-r.projected_gravity_b[:,2]>.9063)&(support.sum(-1)>80.)
        stable &= ~self.env.termination_manager.terminated
        self.stable_seconds=torch.where(stable&self.contact,self.stable_seconds+self.env.step_dt,0.)
        recovery=4.*stable.float()*self.contact.float()*(self.elapsed>150).float()
        regularization=.04*self.last_action[:,:29].square().sum(-1)*(1.+3.*self.nominal_mask.float())
        # Event bonuses are consumed exactly once per policy reward calculation.
        reward=track+1.+impulse_bonus+250.*self.score_event+recovery-regularization
        reward-=100.*self.env.termination_manager.terminated.float()
        self.hit_event.zero_();self.score_event.zero_()
        return reward


def get_task(env):
    if not hasattr(env,'_joint_strike_task'):env._joint_strike_task=JointStrikeTask(env,env.cfg.joint_strike)
    return env._joint_strike_task


def task_features(env):return get_task(env).features()
def task_reward(env):return get_task(env).reward()
