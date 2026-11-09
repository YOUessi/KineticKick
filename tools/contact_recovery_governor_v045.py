"""v45 distinct late CONTACT-RECOVERY whole-body velocity governor.

The rejected v44 forecaster is NOT used in robot control because it made
dangerous false-safe predictions. This is a separate constrained experiment:
a 29-joint velocity opposing POSITION-reference residual only after verified
real foot-ball contact and original separation. It cannot guarantee actual
qdot limits or preserve the motion, and must pass full independent PhysX.
Not new human style, not automatic body-brain trained with professional data.
"""
from __future__ import annotations
import math
import torch
from coordination_style_v041 import CoordinationStyleReferenceTask

def braking_reference(joint_qdot,original_joint_limits,eligible,params):
    """Return bounded opposite-qdot *position-target* correction.

    The returned reference is a command, NOT a measured velocity guarantee.
    It is driven by original actual speed ratios and ONLY after an observed
    contact event. No force/ball velocity write or added physical friction.
    """
    if joint_qdot.ndim!=2 or joint_qdot.shape[1]!=29 or original_joint_limits.shape!=joint_qdot.shape:
        raise ValueError("Original G1 qdot/limits must be Bx29")
    if eligible.shape!=(len(joint_qdot),):
        raise ValueError("Batched contact event eligibility missing")
    if not all(bool(torch.isfinite(v).all()) for v in (joint_qdot,original_joint_limits)):
        raise ValueError("Measured current G1 physics must be finite")
    if not bool((original_joint_limits>0).all()):raise ValueError("Original actuator limits must stay positive")
    ratio=(joint_qdot/original_joint_limits).abs()
    fraction=((ratio-params["warning_ratio"])/(params["maximum_ratio"]-params["warning_ratio"])).clamp(0,1)
    desired=(-params["maximum_position_brake_rad"]*fraction.square()*joint_qdot.sign())
    return desired*eligible[:,None],dict(actual_qdot_ratio=ratio,
        danger_actuator_count=(fraction>0).sum(dim=-1))

class ContactRecoveryGovernorTask(CoordinationStyleReferenceTask):
    def __init__(self,env,config):
        cfg=config.get("contact_recovery_governor_v045")
        if not isinstance(cfg,dict) or type(cfg.get("enabled")) is not bool:
            raise ValueError("Explicit v45 governor component and boolean required")
        fixed={
          "warning_ratio":.68,"maximum_ratio":.98,
          "maximum_position_brake_rad":.025,
          "maximum_position_step_rad":.004,
          "after_contact_min_s":.10,
          "after_contact_max_s":1.25,
          "separation_confirm_s":.015
        }
        if any(k not in cfg or not math.isfinite(float(cfg[k]))
               or abs(float(cfg[k])-v)>1e-9 for k,v in fixed.items()):
            raise ValueError("Only predeclared v45 measured velocity shield settings allowed")
        if not config.get("coordination_style_v041",{}).get("enabled"):
            raise ValueError("v45 is only a post-contact extension of the frozen v41 style controller")
        super().__init__(env,config)
        self.guard_cfg=cfg
        self.guard_delta=torch.zeros(self.n,29,device=self.device)
        self.guard_first_contact=torch.full((self.n,),-1.,device=self.device)
        self.guard_last_output=self.guard_delta.clone()
        self.guard_enabled_counter=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.guard_nonzero_counter=self.guard_enabled_counter.clone()

    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.guard_delta[ids]=0.
        self.guard_first_contact[ids]=-1.
        self.guard_last_output[ids]=0.
        self.guard_enabled_counter[ids]=0
        self.guard_nonzero_counter[ids]=0

    def commands(self,actions=None):
        if actions is not None:raise ValueError("Research governor has no externally submitted actions")
        # The zero path must call exactly the original v41 once, and return
        # that very action tensor without new force or joint math.
        if self.guard_cfg["enabled"] is False:
            base=super().commands(None)
            if self.n==1 and self.debug:
                zero=[0.]*29
                self.debug[-1]["contact_recovery_governor_v045"]=dict(
                    enabled=False,previous_brake_rad=zero,actual_brake_rad=zero,
                    original_action=base[0].tolist(),final_action=base[0].tolist(),
                    qualified_contact=False,postcontact_eligible=False,
                    physical_qdot_ratio_max=0.,braking_actuator_count=0,
                    no_state_writes=True)
            return base
        base=super().commands(None)
        now=self.elapsed*.02
        # First original G1 right-foot-ball filtered contact is physically
        # observed. This is NOT the human 3DSP shot-annotation proxy.
        new_hit=self.contact&(self.guard_first_contact<0)
        self.guard_first_contact=torch.where(new_hit,now,self.guard_first_contact)
        since=now-self.guard_first_contact
        separated=self.landing.separated_s>=self.guard_cfg["separation_confirm_s"]
        ready=(self.guard_first_contact>=0)&self.contact&separated
        ready&=(since>=self.guard_cfg["after_contact_min_s"])&(since<=self.guard_cfg["after_contact_max_s"])
        qd=self.robot.data.joint_vel
        limit=self.robot.data.joint_vel_limits
        requested,metrics=braking_reference(qd,limit,ready,self.guard_cfg)
        self.guard_last_output=self.guard_delta.clone()
        step=self.guard_cfg["maximum_position_step_rad"]
        maxrad=self.guard_cfg["maximum_position_brake_rad"]
        self.guard_delta=(self.guard_delta+(requested-self.guard_delta).clamp(-step,step)).clamp(-maxrad,maxrad)
        # Once faded to near zero after the frozen contact horizon, eliminate
        # numerical residue WITHOUT allowing a large discontinuous target jump.
        done=(self.guard_first_contact>=0)&(since>=self.guard_cfg["after_contact_max_s"]+.20)
        self.guard_delta=torch.where(done[:,None],torch.zeros_like(self.guard_delta),self.guard_delta)
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,device=base.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):
            raise ValueError("Original G1 PD action scale invalid")
        corrected=torch.where(self.guard_delta!=0,base+self.guard_delta/scale,base) if bool((self.guard_delta!=0).any()) else base
        self.guard_delta=(corrected-base)*scale
        self.guard_enabled_counter+=ready.long()
        self.guard_nonzero_counter+=(self.guard_delta.abs().amax(-1)>1e-7).long()
        if self.n==1 and self.debug:
            self.debug[-1]["contact_recovery_governor_v045"]=dict(
                enabled=True,previous_brake_rad=self.guard_last_output[0].tolist(),
                actual_brake_rad=self.guard_delta[0].tolist(),
                original_action=base[0].tolist(),final_action=corrected[0].tolist(),
                qualified_contact=bool(self.contact[0]),
                postcontact_eligible=bool(ready[0]),
                time_since_first_foot_ball_contact_s=float(since[0]),
                foot_ball_separation_time_s=float(self.landing.separated_s[0]),
                physical_qdot_ratio_max=float(metrics["actual_qdot_ratio"][0].amax()),
                braking_actuator_count=int(metrics["danger_actuator_count"][0]),
                guard_eligible_count=int(self.guard_enabled_counter[0]),
                guard_actual_nonzero_control_steps=int(self.guard_nonzero_counter[0]),
                no_state_writes=True)
        return corrected

def get_task(env):
    if not hasattr(env,"_contact_recovery_v045_task"):
        env._contact_recovery_v045_task=ContactRecoveryGovernorTask(env,env.cfg.joint_strike)
    return env._contact_recovery_v045_task
