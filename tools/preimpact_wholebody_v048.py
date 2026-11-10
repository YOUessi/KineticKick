"""v48: genuine PRE-CONTACT learned 29DoF soccer whole-body coordination.

A new PPO actor drives bounded joint POSITION references BEFORE real ball
contact. The existing v32 PPO actor separately retains its trained post-shot
recovery. Reward uses real G1/PhysX 5ms qdot, right-foot-ball first contact,
ball outgoing velocity direction and nonfoot collisions. A small reference
from ONE 3DSP broadcast-estimated TRAIN clip is a weak pose prior: it is NOT
verified human high-speed instep kinetics, aesthetics or naturalness.
"""
from pathlib import Path
import hashlib,math
import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from post_shot_recovery import PostShotRecoveryTask,FEATURE_WIDTH,_quintic
from running_balance import ground_force

ROOT=Path(__file__).resolve().parents[1]
FEATURE_DIM=FEATURE_WIDTH+6

class PreImpactCoordinationActor(ActorCritic):
    def __init__(self,num_actor_obs,num_critic_obs,num_actions,init_noise_std=.2,**kwargs):
        if num_actor_obs!=FEATURE_DIM or num_critic_obs!=FEATURE_DIM or num_actions!=29:
            raise ValueError("G1 preimpact actor expects 160 present-physics features and 29 joint actions")
        super().__init__(num_actor_obs,num_critic_obs,num_actions,
            actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64],
            activation="elu",init_noise_std=init_noise_std,**kwargs)
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def phase_envelope(now,settings):
    """Smooth anticipatory whole-body action with exact fade before impact."""
    up=_quintic((now-settings["start_s"])/settings["ramp_s"])
    down=1.-_quintic((now-settings["end_s"]-(-settings["fade_s"]))/settings["fade_s"])
    active=(now>=settings["start_s"])&(now<=settings["end_s"])
    return torch.where(active,up*down,torch.zeros_like(now))

class PreImpactWholeBodyTask(PostShotRecoveryTask):
    def __init__(self,env,config):
        cfg=config.get("preimpact_wholebody_v048")
        if not isinstance(cfg,dict) or type(cfg.get("enabled")) is not bool:
            raise ValueError("Pre-impact research actor requires a literal enabled boolean")
        fixed=dict(start_s=1.02,end_s=1.54,ramp_s=.14,fade_s=.12,
          maximum_joint_delta_rad=.018,maximum_20ms_slew_rad=.003,
          left_support_min_N=80.,right_swing_max_N=40.,
          maximum_root_ball_distance_m=1.25,max_qdot_ratio=.90,
          cue_gain=.10,
          physical_reward_weight=.8,impact_direction_reward_weight=1.,
          human_proxy_cue_reward_weight=.05,
          preimpact_speed_penalty_weight=2.)
        for k,v in fixed.items():
            if not isinstance(cfg.get(k),(float,int)) or abs(float(cfg[k])-v)>1e-10:
                raise ValueError("Frozen 29DoF pre-impact research safety/reward contract changed: "+k)
        if config.get("human_coordination_reference") or config.get("coordination_style_v041"):
            raise ValueError("Do not silently combine hand-coded v41 coordination style and learned v48 fullbody PPO")
        if config.get("contact_landing_transfer_v046") or config.get("contact_recovery_governor_v045"):
            raise ValueError("Rejected fullbody safety patches are not used in independent training")
        if config.get("post_shot_recovery",{}).get("checkpoint_sha256")!="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326":
            raise ValueError("Frozen original v32 PPO lower-body recovery teacher required")
        super().__init__(env,config)
        if self.recovery_actor is None:
            raise ValueError("Separate original PPO recovery actor must be loaded and frozen")
        self.pre_cfg=cfg
        path=ROOT/cfg["one_human_train_clip_precomputed_proxy"]
        if _sha(path)!=cfg["one_human_train_clip_precomputed_proxy_sha256"]:
            raise ValueError("Single weak human-3DSP pose-prior reference changed")
        with np.load(path,allow_pickle=False) as data:
            time=data["controller_time_s"]
            labels=data["joint_names"].tolist()
            delta=data["delta_joint_position_rad"]
            if time.shape!=(700,) or delta.shape!=(700,29) or list(self.robot.joint_names)!=labels:
                raise ValueError("G1 700x29 original joint semantic order mismatch")
            if bool(data["source_is_verified_instep"]):
                raise ValueError("Cannot pretend human pose-estimated data certifies real instep physics")
        self.human_cue=torch.tensor(delta,device=self.device,dtype=torch.float32)
        self.pre_actor=None
        if cfg.get("checkpoint"):
            chk=ROOT/cfg["checkpoint"]
            if _sha(chk)!=cfg["checkpoint_sha256"]:
                raise ValueError("Unknown new pre-impact learned Actor checkpoint")
            saved=torch.load(chk,map_location=self.device,weights_only=False)
            self.pre_actor=PreImpactCoordinationActor(FEATURE_DIM,FEATURE_DIM,29).to(self.device)
            self.pre_actor.load_state_dict(saved["model_state_dict"],strict=True)
            self.pre_actor.eval()
            for w in self.pre_actor.parameters():w.requires_grad_(False)
        elif cfg.get("checkpoint_sha256") is not None:
            raise ValueError("Unpaired checkpoint SHA")
        self.pre_delta=torch.zeros(self.n,29,device=self.device)
        self.previous_pre_delta=self.pre_delta.clone()
        self.pre_current_target=self.pre_delta.clone()
        self._pre_reward_integral=torch.zeros(self.n,device=self.device)
        self._pre_reward_duration=torch.zeros(self.n,device=self.device)
        self._pre_first_hit=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self._pre_contact_age=torch.full((self.n,),-1.,device=self.device)
        self._pre_active_events=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self._pre_nonzero_events=self._pre_active_events.clone()
        self._pre_last_ball_speed=self._pre_reward_integral.clone()

    def reset_fields(self,ids):
        super().reset_fields(ids)
        for name in ("pre_delta","previous_pre_delta","pre_current_target",
                     "_pre_reward_integral","_pre_reward_duration","_pre_active_events",
                     "_pre_nonzero_events","_pre_last_ball_speed"):
            getattr(self,name)[ids]=0.
        self._pre_first_hit[ids]=False
        self._pre_contact_age[ids]=-1.

    def pre_features(self):
        old=self.recovery_features()
        root=self.robot.data.root_pos_w
        position=self.ball.data.root_pos_w
        relative=(position-root)/3.
        velocity=self.ball.data.root_lin_vel_w/5.
        x=torch.cat([old,relative,velocity],-1)
        if x.shape!=(self.n,FEATURE_DIM) or not bool(torch.isfinite(x).all()):
            raise ValueError("G1 preimpact state and ball state must be finite 160-d")
        return x

    def commands(self,actions=None):
        if actions is not None and (actions.shape!=(self.n,29) or not bool(torch.isfinite(actions).all())):
            raise ValueError("PPO preimpact action is original batchx29 finite tensor")
        if not self.pre_cfg["enabled"]:
            if actions is not None:raise ValueError("Disabled action must not train or inject references")
            base=PostShotRecoveryTask.commands(self,None)
            if self.n==1 and self.debug:
                self.debug[-1]["preimpact_wholebody_v048"]=dict(
                    enabled=False,phase=0.,eligible=False,
                    applied_delta_rad=[0.]*29,previous_delta_rad=[0.]*29,
                    baseline_action=base[0].tolist(),final_action=base[0].tolist(),
                    ball_state_write=False)
            return base
        state=self.pre_features() if actions is None else None
        # Freeze v32 post-shot expert: no new PPO action is sent to its
        # recovery residual, it uses its original SHA-locked trained actor.
        base=PostShotRecoveryTask.commands(self,None)
        if actions is None:
            if self.pre_actor is None:
                raise ValueError("Only a trained PPO actor may execute model inference")
            with torch.no_grad():actions=self.pre_actor.act_inference(state)
        actions=actions.detach()
        now=self.elapsed*.02
        index=self.elapsed.long().clamp(0,699)
        envelope=phase_envelope(now,self.pre_cfg)
        stance=ground_force(self.env)
        qd_ratio=(self.robot.data.joint_vel/self.robot.data.joint_vel_limits).abs().amax(-1)
        ball_distance=(self.ball.data.root_pos_w[:,:2]-self.robot.data.root_pos_w[:,:2]).norm(dim=-1)
        eligible=(~self.contact)&(envelope>0)&(stance[:,0]>=self.pre_cfg["left_support_min_N"])&(
            stance[:,1]<=self.pre_cfg["right_swing_max_N"])&(qd_ratio<=self.pre_cfg["max_qdot_ratio"])&(
            ball_distance<=self.pre_cfg["maximum_root_ball_distance_m"])
        # All 29 original fullbody G1 joints share one bounded actor; lower
        # authority on support-leg and arms avoids arbitrary high impulses.
        ratio=torch.as_tensor([
            1. if name.startswith("right_") else .45 if name.startswith("left_")
            else .3 if name.startswith("waist_") else .25
            for name in self.robot.joint_names],device=self.device,dtype=base.dtype)
        raw=self.pre_cfg["maximum_joint_delta_rad"]*torch.tanh(actions)*envelope[:,None]*ratio[None]
        raw=torch.where(eligible[:,None],raw,torch.zeros_like(raw))
        self.previous_pre_delta=self.pre_delta.clone()
        maxstep=self.pre_cfg["maximum_20ms_slew_rad"]
        self.pre_delta=(self.pre_delta+(raw-self.pre_delta).clamp(-maxstep,maxstep)).clamp(
             -self.pre_cfg["maximum_joint_delta_rad"],self.pre_cfg["maximum_joint_delta_rad"])
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
                               device=self.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all() and (scale!=0.).all()):
            raise ValueError("Existing G1 original actuator reference scaling not available")
        out=torch.where(self.pre_delta!=0,base+self.pre_delta/scale,base) if bool((self.pre_delta!=0).any()) else base
        self.pre_delta=(out-base)*scale
        self.pre_current_target=self.human_cue[index]
        self._pre_active_events+=eligible.long()
        self._pre_nonzero_events+=(self.pre_delta.abs().amax(-1)>1e-7).long()
        if self.n==1 and self.debug:
            self.debug[-1]["preimpact_wholebody_v048"]=dict(
                enabled=True,phase=float(envelope[0]),eligible=bool(eligible[0]),
                left_right_stance_N=stance[0].tolist(),
                measured_qdot_ratio=float(qd_ratio[0]),
                ball_distance_m=float(ball_distance[0]),
                applied_delta_rad=self.pre_delta[0].tolist(),
                previous_delta_rad=self.previous_pre_delta[0].tolist(),
                baseline_action=base[0].tolist(),final_action=out[0].tolist(),
                human_proxy_only_not_real_professional_instep=True,
                ball_state_write=False)
        return out

    def after_physics(self,dt):
        super().after_physics(dt)
        if dt<=0 or not self.pre_cfg["enabled"]:return
        now=self.elapsed*.02
        ratio=(self.robot.data.joint_vel/self.robot.data.joint_vel_limits).abs().amax(-1)
        if not bool(torch.isfinite(ratio).all()):
            raise ValueError("Original G1 5ms joint-speed state is not finite")
        phase=phase_envelope(now,self.pre_cfg)
        pre=(phase>0)&(~self.contact)
        # G1 foot-body velocity is an angular-lever-ignorant PROXY only.
        # Measured outgoing soccer ball velocity determines actual task result.
        root=self.robot.data.root_pos_w
        ball=self.ball.data.root_pos_w
        bvelocity=self.ball.data.root_lin_vel_w
        direction=torch.as_tensor(self.env.cfg.joint_strike["goal_xy"],device=self.device)
        desired=torch.nn.functional.normalize(
            direction[None]+self.env.scene.env_origins[:,:2]-ball[:,:2],dim=-1)
        ahead=(bvelocity[:,:2]*desired).sum(-1)
        actual_speed=bvelocity[:,:2].norm(dim=-1)
        # Event phase measured by original collision, not inferred from
        # any time annotation in 3DSP broadcast video.
        just=self.contact&(~self._pre_first_hit)
        self._pre_first_hit|=self.contact
        self._pre_contact_age=torch.where(just,torch.zeros_like(now),self._pre_contact_age)
        self._pre_contact_age=torch.where(self._pre_contact_age>=0,
              self._pre_contact_age+dt,self._pre_contact_age)
        scoring=(self._pre_contact_age>=.015)&(self._pre_contact_age<=.085)
        # Actual ball direction and speed: positive if soccer ball exits
        # toward the fixed physical goal. Never write ball velocity.
        ball_reward=self.pre_cfg["impact_direction_reward_weight"]*torch.where(
             scoring,ahead.clamp_min(0.)/5.,torch.zeros_like(ahead))
        cue_reward= -self.pre_cfg["human_proxy_cue_reward_weight"]*(
            self.pre_delta-self.pre_cfg["cue_gain"]*self.pre_current_target).square().mean(-1)*500.
        speed_cost=-self.pre_cfg["preimpact_speed_penalty_weight"]*(ratio-.80).clamp_min(0.).square()
        action_cost=-.12*(self.pre_delta/.018).square().mean(-1)
        sample=(ball_reward + torch.where(pre,cue_reward+speed_cost+action_cost,torch.zeros_like(ball_reward)))
        if not bool(torch.isfinite(sample).all()):raise ValueError("Nonfinite G1 physical preimpact reward")
        self._pre_reward_integral+=sample.detach()*dt
        self._pre_reward_duration+=dt
        self._pre_last_ball_speed=actual_speed.detach()

    def reward(self):
        base=PostShotRecoveryTask.reward(self)
        mean=torch.where(self._pre_reward_duration>0,
                self._pre_reward_integral/self._pre_reward_duration.clamp_min(1e-8),
                torch.zeros_like(self._pre_reward_integral))
        self._pre_reward_integral.zero_()
        self._pre_reward_duration.zero_()
        return base+self.pre_cfg["physical_reward_weight"]*mean

def get_task(env):
    if not hasattr(env,"_preimpact_wholebody_v048_task"):
        env._preimpact_wholebody_v048_task=PreImpactWholeBodyTask(env,env.cfg.joint_strike)
    return env._preimpact_wholebody_v048_task

def task_features(env):return get_task(env).pre_features()
def task_reward(env):return get_task(env).reward()
