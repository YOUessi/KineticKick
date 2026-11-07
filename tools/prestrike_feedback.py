"""Bounded learned pre-strike references over the complete frozen v32 expert.

The new action never replaces the learned post-shot policy. Initial 0.6 s and
original numerical nominal-context dead zone are structurally protected. No
phase action, PD-target projection, ball force, reset or physical-limit change.
"""
import hashlib
import math
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic

from post_shot_recovery import PostShotRecoveryTask, FEATURE_LAYOUT as RECOVERY_LAYOUT, _rotation, _quintic

ROOT=Path(__file__).resolve().parents[1]
FEATURE_LAYOUT=RECOVERY_LAYOUT+(
    ('ball_position_in_root_over_3',3),('ball_minus_root_velocity_in_root_over_5',3),
    ('ball_to_goal_direction_in_root',3),('ball_goal_distance_over_6',1),
    ('training_speed_center_over_5',1),('eligible_touched_touch_age_over_point2',3),
    ('previous_prestrike_delta_over_point04',29),('original_context_gate',1))
FEATURE_WIDTH=sum(width for _,width in FEATURE_LAYOUT)


class PreStrikeActor(ActorCritic):
    def __init__(self,num_actor_obs,num_critic_obs,num_actions,init_noise_std=.25,**kwargs):
        if num_actor_obs!=FEATURE_WIDTH or num_actions!=29:
            raise ValueError(f'Expected {FEATURE_WIDTH} state features and 29 pre-strike actions')
        super().__init__(num_actor_obs,num_critic_obs,num_actions,
            actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64],activation='elu',
            init_noise_std=init_noise_std,**kwargs)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)


class PreStrikeEnvelope:
    def __init__(self,n,device):
        self.actual=torch.zeros(n,29,device=device);self.exit_delta=self.actual.clone()
        self.exit_time=torch.full((n,),-1.,device=device)
        self.eligible=torch.zeros(n,device=device,dtype=torch.bool)

    def latch(self,now,touched):
        entering=touched & (self.exit_time<0)
        self.exit_time=torch.where(entering,now,self.exit_time)
        self.exit_delta=torch.where(entering[:,None],self.actual,self.exit_delta)

    def update(self,now,raw,context,touched):
        n=len(self.actual)
        if now.shape!=(n,) or raw.shape!=(n,29) or context.shape!=(n,) or touched.shape!=(n,):
            raise ValueError('Batched pre-strike envelope shape mismatch')
        if not all(bool(torch.isfinite(x).all()) for x in (now,raw,context)) or bool(((context<0)|(context>1)).any()):
            raise ValueError('Finite state/actions and original context in [0,1] required')
        self.eligible |= now>=.6
        self.latch(now,touched | (now>=2.1))
        desired=.04*torch.tanh(raw)*(_quintic((now-.6)/.12)*context)[:,None]
        fading=self.exit_time>=0
        desired=torch.where(fading[:,None],self.exit_delta*(1.-_quintic((now-self.exit_time)/.20))[:,None],desired)
        change=desired-self.actual;peak=change.abs().amax(-1,keepdim=True)
        self.actual=self.actual+change*torch.minimum(torch.ones_like(peak),.008/peak.clamp_min(1e-12))
        finished=fading & (now+1e-7>=self.exit_time+.20)
        self.actual=torch.where(finished[:,None],torch.zeros_like(self.actual),self.actual)
        return self.actual

    def reset(self,ids):
        self.actual[ids]=0.;self.exit_delta[ids]=0.;self.exit_time[ids]=-1.;self.eligible[ids]=False


def actual_speed_excess(velocity,limits):
    # Match the independent evaluator's absolute 1e-6 rad/s tolerance.
    return (velocity.double().abs()>limits.double()+1e-6).any(-1)


class OnceEvents:
    def __init__(self,n,device):
        self.pending=torch.zeros(n,device=device);self.paid={}

    def add(self,name,mask,value):
        if name not in self.paid:self.paid[name]=torch.zeros_like(mask,dtype=torch.bool)
        new=mask & ~self.paid[name]
        self.pending+=torch.where(new,torch.as_tensor(value,device=self.pending.device,dtype=self.pending.dtype),0.)
        self.paid[name] |= mask

    def consume(self):
        value=self.pending.clone();self.pending.zero_();return value

    def reset(self,ids):
        self.pending[ids]=0.
        for flag in self.paid.values():flag[ids]=False


class PreStrikeTask(PostShotRecoveryTask):
    def __init__(self,env,config):
        if config.get('early_path_feedback') or config.get('support_reference_frame'):
            raise ValueError('v35 uses frozen v32 without v34 lateral or v31 spatial corrections')
        if config.get('post_shot_recovery',{}).get('checkpoint_sha256')!='ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326':
            raise ValueError('Pinned v32 learned recovery required')
        super().__init__(env,config)
        self.pre_cfg=config['prestrike_feedback']
        expected={'maximum_rad':.04,'maximum_step_rad':.008,'earliest_s':.6,'onset_s':.12,'fade_s':.2,'latest_s':2.1,'shot_sample_delay_s':.08}
        if any(not math.isfinite(self.pre_cfg[k]) or abs(self.pre_cfg[k]-v)>1e-10 for k,v in expected.items()):
            raise ValueError('Fixed v35 first-trial envelope and timing required')
        self.pre_gate=PreStrikeEnvelope(self.n,self.device)
        self.pre_events=OnceEvents(self.n,self.device)
        self.pre_first_touch=torch.full((self.n,),-1.,device=self.device)
        self.pre_first_legal=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.pre_illegal=self.pre_first_legal.clone()
        self.pre_previous=torch.zeros(self.n,29,device=self.device)
        self.pre_last_base=self.pre_previous.clone()
        self.pre_integral=torch.zeros(self.n,device=self.device);self.pre_duration=self.pre_integral.clone()
        self.last_pre_reward_components={};self.pre_actor=None
        sensor=env.scene['contact_forces']
        self._pre_ball_view=sensor._physics_sim_view.create_rigid_contact_view(
            '/World/envs/env_*/SoccerBall',
            ['/World/envs/env_*/Robot/'+name for name in self.robot.body_names],
            max_contact_data_count=256*self.n)
        checkpoint,digest=self.pre_cfg.get('checkpoint'),self.pre_cfg.get('checkpoint_sha256')
        if bool(checkpoint)!=bool(digest):raise ValueError('Pre-strike checkpoint and SHA256 must be paired')
        if checkpoint:
            path=ROOT/checkpoint
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Pre-strike checkpoint SHA256 mismatch')
            saved=torch.load(path,map_location=self.device,weights_only=False)
            self.pre_actor=PreStrikeActor(FEATURE_WIDTH,FEATURE_WIDTH,29).to(self.device)
            self.pre_actor.load_state_dict(saved['model_state_dict'],strict=True);self.pre_actor.eval()
            for parameter in self.pre_actor.parameters():parameter.requires_grad_(False)

    def pre_features(self):
        r=self.robot.data;rotation=_rotation(r.root_quat_w)
        def local(vector):return torch.einsum('nji,nj->ni',rotation,vector)
        direction=self.goal-self.ball.data.root_pos_w;direction=direction.clone();direction[:,2]=0.
        distance=direction.norm(dim=-1,keepdim=True)
        direction/=distance.clamp_min(1e-6)
        touched=self.pre_first_touch>=0
        phase=torch.stack((self.pre_gate.eligible.float(),touched.float(),
            torch.where(touched,(self._recovery_time_s-self.pre_first_touch).clamp_min(0.)/.2,0.)),-1)
        values=(self.recovery_features(),local(self.ball.data.root_pos_w-r.root_pos_w)/3.,
            local(self.ball.data.root_lin_vel_w-r.root_lin_vel_w)/5.,local(direction),distance/6.,
            torch.full((self.n,1),4.75/5.,device=self.device),phase,self.pre_gate.actual/.04,self.gate[:,None])
        features=torch.cat(values,-1)
        if features.shape!=(self.n,FEATURE_WIDTH) or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite complete pre-strike observation required')
        return features

    def commands(self,actions=None):
        if actions is not None and (actions.shape!=(self.n,29) or not bool(torch.isfinite(actions).all())):
            raise ValueError('Finite N x 29 pre-strike raw actions required')
        features=self.pre_features() if actions is None and self.pre_actor is not None else None
        now=self.elapsed.clone()*.02
        base=super().commands(None)  # Retain the frozen learned post-shot actor.
        if actions is None:
            with torch.no_grad():
                actions=self.pre_actor.act_inference(features) if self.pre_actor is not None else torch.zeros_like(base)
        if actions.shape!=(self.n,29) or not bool(torch.isfinite(actions).all()):raise ValueError('Invalid pre-strike actor output')
        self.pre_previous=self.pre_gate.actual.clone()
        delta=self.pre_gate.update(now,actions.detach(),self.gate,self.pre_first_touch>=0)
        scale=torch.as_tensor(self.env.action_manager.get_term('joint_pos')._scale,device=base.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):raise ValueError('Finite nonzero original action scale required')
        changed=delta!=0.
        out=torch.where(changed,base+delta/scale,base) if bool(changed.any()) else base
        self.pre_gate.actual=(out-base)*scale
        self.pre_last_base=base.detach().clone()
        if self.n==1 and self.debug:
            self.debug[-1]['prestrike_feedback']=dict(actual_delta_rad=self.pre_gate.actual[0].tolist(),
                previous_delta_rad=self.pre_previous[0].tolist(),baseline_action=base[0].tolist(),final_action=out[0].tolist(),
                first_touch_relative_s=float(self.pre_first_touch[0]),first_touch_right_instep_proxy=bool(self.pre_first_legal[0]),
                illegal_contact_proxy=bool(self.pre_illegal[0]),exit_time_s=float(self.pre_gate.exit_time[0]),
                context_gate=float(self.gate[0]),eligible=bool(self.pre_gate.eligible[0]),
                post_recovery_overlap=bool((self.actual_delta[0]!=0).any() and (self.pre_gate.actual[0]!=0).any()),
                ball_state_write=False)
        return out

    def after_physics(self,dt):
        super().after_physics(dt)
        if dt<=0:return
        from prestrike_objective import prestrike_rate,shot_event
        now=self._recovery_time_s;r=self.robot.data
        self.pre_gate.eligible |= now>=.6
        matrix=self._pre_ball_view.get_contact_force_matrix(dt).reshape(self.n,-1,3)
        if matrix.shape[1]!=len(self.robot.body_names) or not bool(torch.isfinite(matrix).all()):
            raise ValueError('Ordered finite ball-to-robot contact coverage required')
        magnitude=matrix.norm(dim=-1)
        touched=magnitude.amax(-1)>2.
        new=touched & (self.pre_first_touch<0)
        wrong=magnitude.clone();wrong[:,self.feet[1]]=0.
        wrong=wrong.amax(-1)>2.
        right=(magnitude[:,self.feet[1]]>2.) & ~wrong & (self.contact_quality>=.75)
        self.pre_first_legal=torch.where(new,right,self.pre_first_legal)
        self.pre_first_touch=torch.where(new,now,self.pre_first_touch)
        self.pre_illegal |= wrong | (new & ~right)
        self.pre_gate.latch(now,touched)
        eligible=self.pre_gate.eligible
        qd_ratio=(r.joint_vel/r.joint_vel_limits).abs().amax(-1)
        pre_control=eligible & ((self.pre_gate.exit_time<0)|(now<self.pre_gate.exit_time+.2))
        state=dict(eligible=eligible,pre_control=pre_control,
            root_height_m=r.root_pos_w[:,2]-self.env.scene.env_origins[:,2],
            reference_height_m=self.sample('root_pose')[:,2],upright_cos=-r.projected_gravity_b[:,2],
            root_velocity_w=r.root_lin_vel_w,reference_velocity_w=self.sample('root_velocity'),
            joint_position_error_rad=r.joint_pos-self.sample('joint_pos'),qd_ratio=qd_ratio,
            actual_delta=self.pre_gate.actual,previous_delta=self.pre_previous)
        rate,components=prestrike_rate(state)
        self.pre_integral+=rate.detach()*dt;self.pre_duration+=dt
        self.last_pre_reward_components={key:value.detach().clone() for key,value in components.items()}
        self.pre_events.add('overspeed',eligible & actual_speed_excess(r.joint_vel,r.joint_vel_limits),-20.)
        self.pre_events.add('illegal',eligible & self.pre_illegal,-20.)
        self.pre_events.add('no_touch',eligible & (now>=2.4) & (self.pre_first_touch<0),-20.)
        self.pre_events.add('goal',eligible & self.scored,20.)
        due=eligible & (self.pre_first_touch>=.6) & (now+1e-7>=self.pre_first_touch+.08)
        unpaid=due & ~self.pre_events.paid.get('shot',torch.zeros_like(due))
        ids=unpaid.nonzero(as_tuple=False).flatten()
        if len(ids):
            direction=self.goal[ids,:2]-self.ball.data.root_pos_w[ids,:2]
            values,shot_components=shot_event(self.ball.data.root_lin_vel_w[ids],direction,
                torch.ones(len(ids),device=self.device,dtype=torch.bool),(self.pre_first_legal & ~self.pre_illegal)[ids])
            bonus=torch.zeros(self.n,device=self.device);bonus[ids]=values
            self.pre_events.add('shot',unpaid,bonus)
            for key,value in shot_components.items():
                full=torch.zeros(self.n,device=self.device);full[ids]=value
                self.last_pre_reward_components[key]=full

    def reward(self):
        post=super().reward()
        available=self.pre_duration>0
        manager=getattr(self.env,'termination_manager',None)
        terminated=torch.zeros_like(available) if manager is None else manager.terminated
        # Original post reward already owns terminal cost once post-active.
        self.pre_events.add('terminal',self.pre_gate.eligible & terminated & (self.activation_time<0),-20.)
        events=self.pre_events.consume()
        pre=torch.where(available,self.pre_integral/self.pre_duration.clamp_min(.005),0.)
        result=post+pre+events/self.env.step_dt
        self.pre_integral.zero_();self.pre_duration.zero_()
        return result

    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.pre_gate.reset(ids);self.pre_events.reset(ids);self.pre_first_touch[ids]=-1.
        self.pre_first_legal[ids]=False;self.pre_illegal[ids]=False
        for name in ('pre_previous','pre_last_base','pre_integral','pre_duration'):getattr(self,name)[ids]=0.
        for value in self.last_pre_reward_components.values():value[ids]=0.


def get_task(env):
    if not hasattr(env,'_prestrike_feedback_task'):
        env._prestrike_feedback_task=PreStrikeTask(env,env.cfg.joint_strike)
    return env._prestrike_feedback_task


def task_features(env):return get_task(env).pre_features()
def task_reward(env):return get_task(env).reward()
