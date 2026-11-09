"""v46: contact-phase ANTICIPATORY whole-body landing-transition reference.

The v45 reactive alarm never engaged: its 50Hz observations of qdot in the
qualified window were always below the fixed .68 threshold, despite isolated
5ms delayed overspeed. A second frozen hypothesis now acts after real ball
contact but BEFORE first landing: uses original PhysX support, capsule-foot
height/descent and root/29-joint velocities to produce a globally bounded,
phase-continuous 29DoF reference. No simulator state/ball impulses, no new
hardware torque limits or claimed human kinetics. Needs independent full-task
PhysX evaluation, with all failures retained.
"""
from __future__ import annotations
import math,torch
from coordination_style_v041 import CoordinationStyleReferenceTask
from running_balance import ground_force
from post_shot_recovery import _quintic

def prepare_landing_weights(names,device):
    if len(names)!=29 or len(set(names))!=29:
        raise ValueError("Requires original 29 G1 joint names exactly once")
    weights=[]
    for n in names:
        if n.startswith("left_") and "ankle" in n:
            weight=1.0
        elif n.startswith("left_") and "knee" in n:
            weight=.78
        elif n.startswith("left_") and "hip" in n:
            weight=.50
        elif n.startswith("right_") and "knee" in n:
            weight=.75
        elif n.startswith("right_") and "ankle" in n:
            weight=.62
        elif n.startswith("right_") and "hip" in n:
            weight=.50
        elif n.startswith("waist_"):
            weight=.32
        elif any(k in n for k in ("shoulder","elbow","wrist")):
            weight=.16
        else:
            raise ValueError("Unexpected joint outside original pelvis/legs/arms: "+n)
        weights.append(weight)
    return torch.tensor(weights,dtype=torch.float32,device=device)

def landing_phase_reference(since_contact_s,qd,qd_limits,forces,right_lowest_height,descent_speed,cfg,weights):
    """Anticipate landing with an actual measured *before-landing* state.

    Unlike v45, the controller does not wait for a high speed to occur. It
    opposes the currently scheduled high-frequency body motion during a
    phase in which the kicking leg is unloaded or about to touch down.
    Upper body counter-motion is bounded and designed into one whole-body
    reference; no guaranteed stability from the reference amplitude.
    """
    n=len(since_contact_s)
    if (since_contact_s.shape!=(n,) or qd.shape!=(n,29) or qd_limits.shape!=(n,29)
        or forces.shape!=(n,2) or right_lowest_height.shape!=(n,) or descent_speed.shape!=(n,)
        or weights.shape!=(29,)):
        raise ValueError("Original 29DoF landing inputs shape invalid")
    if not all(bool(torch.isfinite(a).all()) for a in (qd,qd_limits,forces,right_lowest_height,descent_speed,since_contact_s,weights)):
        raise ValueError("Nonfinite physical landing state")
    if not bool((qd_limits>0).all()):raise ValueError("Original positive G1 velocity limits required")
    tau=cfg["early_contact_s"]
    late=cfg["late_contact_s"]
    phase=_quintic((since_contact_s-tau)/cfg["ramp_s"])
    phase*=1.-_quintic((since_contact_s-(late-cfg["fade_s"]))/cfg["fade_s"])
    phase=torch.where((since_contact_s>=tau)&(since_contact_s<=late),phase,torch.zeros_like(phase))
    # Original measured stance. A descending unloaded swing foot can also be
    # protected when the other foot briefly loses contact in flight.
    stance=(forces[:,0]>=cfg["left_support_min_N"]) & (forces[:,1]<=cfg["right_unloaded_max_N"])
    predicted_right_landing=(right_lowest_height<=cfg["right_capsule_max_height_m"]) & (
        descent_speed>=cfg["right_descent_min_mps"]) & (forces[:,1]<=cfg["right_unloaded_max_N"])
    qualified=stance|predicted_right_landing
    actual_ratio=(qd/qd_limits)
    # No qdot danger THRESHOLD: actuate proactively in the phase if actual
    # body motion exists. qd is measured, not generated from a nominal step.
    raw=-cfg["reference_gain_rad"]*actual_ratio*weights[None]*phase[:,None]
    raw=torch.where(qualified[:,None],raw,torch.zeros_like(raw))
    raw=raw.clamp(-cfg["maximum_delta_rad"],cfg["maximum_delta_rad"])
    return raw,dict(qualified=qualified,stance=stance,
        anticipated_right_landing=predicted_right_landing,
        raw_phase_gain=phase,
        actual_joint_speed_ratio_max=actual_ratio.abs().amax(-1))

class ContactLandingTransferTask(CoordinationStyleReferenceTask):
    def __init__(self,env,config):
        c=config.get("contact_landing_transfer_v046")
        if not isinstance(c,dict) or type(c.get("enabled")) is not bool:
            raise ValueError("One explicit v46 landing-transfer control contract required")
        fixed=dict(early_contact_s=.06,late_contact_s=.62,ramp_s=.08,fade_s=.20,
            left_support_min_N=50.,right_unloaded_max_N=90.,
            right_capsule_max_height_m=.35,right_descent_min_mps=.04,
            reference_gain_rad=.028,maximum_delta_rad=.016,
            maximum_step_rad=.002)
        if any(k not in c or not isinstance(c[k],(float,int)) or not math.isfinite(float(c[k]))
               or abs(float(c[k])-v)>1e-9 for k,v in fixed.items()):
            raise ValueError("The frozen v46 physics or control gain cannot be changed")
        if config.get("contact_recovery_governor_v045"):
            raise ValueError("Do not combine this experiment with rejected v45 non-engaging alarm")
        if config.get("coordination_style_v041",{}).get("enabled") is not True:
            raise ValueError("Reference source must remain the frozen v41 coordinated G1 control")
        super().__init__(env,config)
        self.transfer_cfg=c
        self.landing_first_contact=torch.full((self.n,),-1.,device=self.device)
        self.landing_delta=torch.zeros(self.n,29,device=self.device)
        self.landing_previous=self.landing_delta.clone()
        self.joint_weights=prepare_landing_weights(self.robot.joint_names,self.device)

    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.landing_first_contact[ids]=-1.
        self.landing_delta[ids]=0.
        self.landing_previous[ids]=0.

    def commands(self,actions=None):
        if actions is not None:
            raise ValueError("Do not externally write a landing trial action")
        if self.transfer_cfg["enabled"] is False:
            original=super().commands(None)
            if self.n==1 and self.debug:
                z=[0.]*29
                self.debug[-1]["contact_landing_transfer_v046"]=dict(
                    enabled=False,actual_joint_delta_rad=z,previous_joint_delta_rad=z,
                    baseline_action=original[0].tolist(),
                    final_action=original[0].tolist(),physical_contact_latched=False,
                    qualified=False,measured_foot_descent_mps=0.,
                    actual_joint_speed_ratio_max=0.,ball_state_write=False)
            return original

        original=super().commands(None)
        now=self.elapsed*.02
        new_contact=self.contact&(self.landing_first_contact<0)
        self.landing_first_contact=torch.where(new_contact,now,self.landing_first_contact)
        since=torch.where(self.landing_first_contact>=0,now-self.landing_first_contact,
                          torch.full_like(now,-10.))
        # Parent _feet_state computes lowest actual capsule surface, accounting
        # for link orientation, measured link angular velocity, and radius.
        _,_,_,capsule_height,lowest_velocity=self._feet_state()
        right_height=capsule_height[:,1]
        right_descent=(-lowest_velocity[:,1,2]).clamp_min(0.)
        forces=ground_force(self.env)
        qd=self.robot.data.joint_vel
        limit=self.robot.data.joint_vel_limits
        desired,info=landing_phase_reference(since,qd,limit,forces,right_height,right_descent,
                                           self.transfer_cfg,self.joint_weights)
        self.landing_previous=self.landing_delta.clone()
        max_step=self.transfer_cfg["maximum_step_rad"]
        delta=(desired-self.landing_delta).clamp(-max_step,max_step)
        self.landing_delta=(self.landing_delta+delta).clamp(
            -self.transfer_cfg["maximum_delta_rad"],self.transfer_cfg["maximum_delta_rad"])
        # Allow gradual fade after the defined landing phase, never snap a
        # non-negligible q target to zero at a single 20ms sample.
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
                             device=original.device,dtype=original.dtype)
        if not bool(torch.isfinite(scale).all() and (scale!=0).all()):
            raise ValueError("Original physical action scale invalid")
        updated=torch.where(self.landing_delta!=0,
                            original+self.landing_delta/scale,original) if bool((self.landing_delta!=0).any()) else original
        self.landing_delta=(updated-original)*scale
        if self.n==1 and self.debug:
            self.debug[-1]["contact_landing_transfer_v046"]=dict(
                enabled=True,actual_joint_delta_rad=self.landing_delta[0].tolist(),
                previous_joint_delta_rad=self.landing_previous[0].tolist(),
                baseline_action=original[0].tolist(),final_action=updated[0].tolist(),
                physical_contact_latched=bool(self.contact[0]),
                time_since_foot_ball_contact_s=float(since[0]),
                qualified=bool(info["qualified"][0]),
                left_stance_evidence=bool(info["stance"][0]),
                right_anticipated_landing=bool(info["anticipated_right_landing"][0]),
                raw_phase_gain=float(info["raw_phase_gain"][0]),
                right_capsule_lowest_height_m=float(right_height[0]),
                measured_foot_descent_mps=float(right_descent[0]),
                measured_left_right_support_N=forces[0].tolist(),
                actual_joint_speed_ratio_max=float(info["actual_joint_speed_ratio_max"][0]),
                ball_state_write=False)
        return updated

def get_task(env):
    if not hasattr(env,"_contact_landing_transfer_v046_task"):
        env._contact_landing_transfer_v046_task=ContactLandingTransferTask(env,env.cfg.joint_strike)
    return env._contact_landing_transfer_v046_task
