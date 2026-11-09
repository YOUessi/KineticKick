"""v41 separate human-coordination STYLE reference channel from soccer ERROR feedback.

v40's original error feedback term self.gate is deliberately zero on the
nominal center ball and silently nullified the entire human-coordination
reference. This independent, prespecified research channel uses an explicit
style gate (active in nominal tasks) AND the same measured-support/qdot/contact
safety restrictions as v40. It does NOT rewrite original robot or ball state.
The style motion is still only a video-estimated-human-POSE proxy; no true
explosive instep biomechanics or contact time is available.
"""
from __future__ import annotations
import math,torch
from post_shot_recovery import PostShotRecoveryTask
from coordination_reference_v040 import CoordinationReferenceTask
from running_balance import ground_force

def prespecified_style_safety_gate(now,support,qd_ratio,root_ball_distance,contact,cfg):
    """The STYLE gate is intentionally independent of soccer error gate.

    Requiring a legacy context gain here would make the motion generator a
    no-op on nominal shooting, as the v40 independent experiment established.
    """
    n=len(now)
    if (now.shape!=(n,) or support.shape!=(n,2) or qd_ratio.shape!=(n,)
        or root_ball_distance.shape!=(n,) or contact.shape!=(n,)):
        raise ValueError("Invalid v41 whole-body safety input")
    if not all(bool(torch.isfinite(x).all()) for x in (now,support,qd_ratio,root_ball_distance)):
        raise ValueError("Untrusted nonfinite physical state")
    time_ok=(now>=cfg["first_s"])&(now<=cfg["last_s"])
    loaded=(support[:,0]>=cfg["support_min_N"]) & (support[:,1]<=cfg["swing_max_N"])
    velocity_ok=qd_ratio<=cfg["max_qdot_ratio"]
    ball_ok=(root_ball_distance>=cfg["minimum_ball_distance_m"]) & (root_ball_distance<=cfg["maximum_ball_distance_m"])
    eligible=time_ok&loaded&velocity_ok&ball_ok&~contact
    return eligible,dict(time=time_ok,stance=loaded,qdot=velocity_ok,ball=ball_ok)


class CoordinationStyleReferenceTask(CoordinationReferenceTask):
    def __init__(self,env,config):
        cfg=config.get("coordination_style_v041")
        if not isinstance(cfg,dict):raise ValueError("Must explicitly register independent style gate")
        fixed={
          "nominal_style_activation":1.0,
          "max_joint_rad":.006,
          "max_step_rad":.0012,
          "first_s":1.05,"last_s":1.56,
          "support_min_N":80.,
          "swing_max_N":40.,
          "max_qdot_ratio":.90,
          "minimum_ball_distance_m":0.,
          "maximum_ball_distance_m":1.15}
        if type(cfg.get("enabled")) is not bool:raise ValueError("Style gate enabled is boolean")
        if any(k not in cfg or not math.isfinite(float(cfg[k])) or abs(float(cfg[k])-v)>1e-10
               for k,v in fixed.items()):
            raise ValueError("Style gate physical bounds/timing differ from preregistration")
        if cfg["enabled"] is not config["human_coordination_reference"]["enabled"]:
            raise ValueError("Separate style and G1 reference must be enabled or disabled together")
        super().__init__(env,config)
        self.style_config=cfg

    def commands(self,actions=None):
        if actions is not None:raise ValueError("v41 style reference accepts no separate policy action")
        if self.style_config["enabled"] is False:
            # v40 disabled adapter directly uses the frozen v32 original:
            # keeps byte-identical actions, no additional sensor access/IK.
            a=super().commands(None)
            if self.n==1 and self.debug:
                self.debug[-1]["coordination_style_v041"]=dict(
                   enabled=False,style_activation=0.,
                   nominal_task_error_gate=float(self.gate[0]),
                   physical_reference_applied=False,
                   parent_action_identical=True,ball_state_write=False)
            return a
        # CRITICAL: bypass the v40 ERROR-gated method entirely.
        # The existing v32 expert gets exactly one original forward call.
        base=PostShotRecoveryTask.commands(self,None)
        now=self.elapsed*.02
        frame=self.elapsed.long().clamp(0,699)
        reference=self.reference[frame]
        self.previous=self.applied.clone()
        self.coord_first_touch=torch.where(
            self.contact&(self.coord_first_touch<0),now,self.coord_first_touch)
        support=ground_force(self.env)
        r=self.robot.data
        root_to_ball=torch.linalg.vector_norm(self.ball.data.root_pos_w[:,:2]-r.root_pos_w[:,:2],dim=-1)
        qd_ratio=(r.joint_vel/r.joint_vel_limits).abs().amax(dim=-1)
        cfg=self.style_config
        eligible,gates=prespecified_style_safety_gate(
            now,support,qd_ratio,root_to_ball,self.contact,cfg)
        loaded=gates["stance"]
        qd_ok=gates["qdot"]
        ball_ok=gates["ball"]
        # This is human-style/coordination activation, not the legacy ball
        # position/velocity ERROR gate; remains enabled at center ball.
        style_gain=cfg["nominal_style_activation"]
        desired=torch.where(eligible[:,None],reference*style_gain,torch.zeros_like(reference))
        changed=(desired-self.applied).clamp(-cfg["max_step_rad"],cfg["max_step_rad"])
        self.applied=(self.applied+changed).clamp(-cfg["max_joint_rad"],cfg["max_joint_rad"])
        expired=(now>=cfg["last_s"]+.25)&(self.applied.abs().amax(dim=-1)<1e-6)
        self.applied=torch.where(expired[:,None],torch.zeros_like(self.applied),self.applied)
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
            device=self.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):
            raise ValueError("Original G1 actuator scale absent/invalid")
        modified=self.applied!=0.
        command=torch.where(modified,base+self.applied/scale,base) if bool(modified.any()) else base
        self.applied=(command-base)*scale
        if self.n==1 and self.debug:
            applied=bool((self.applied[0].abs()>1e-7).any())
            self.debug[-1]["human_coordination_reference"]=dict(
                enabled=True,applied_delta_rad=self.applied[0].tolist(),
                previous_delta_rad=self.previous[0].tolist(),
                original_reference_delta_rad=reference[0].tolist(),
                reference_nonzero=bool((reference[0].abs()>1e-7).any()),
                eligible=bool(eligible[0]),safety_gate=bool(loaded[0]&qd_ok[0]&ball_ok[0]),
                support_normal_N=support[0].tolist(),
                maximum_measured_qdot_ratio=float(qd_ratio[0]),
                ball_distance_m=float(root_to_ball[0]),
                original_context_gate=float(self.gate[0]),
                original_post_shot_recovery_active=bool((self.actual_delta[0].abs()>1e-7).any()),
                baseline_action=base[0].tolist(),final_action=command[0].tolist(),
                ball_state_write=False)
            self.debug[-1]["coordination_style_v041"]=dict(
                enabled=True,style_activation=style_gain,
                nominal_task_error_gate=float(self.gate[0]),
                physical_reference_applied=applied,
                independent_of_nominal_error_gate=True,
                measured_support_gate=bool(loaded[0]),
                qdot_gate=bool(qd_ok[0]),
                ball_distance_gate=bool(ball_ok[0]),
                before_contact=bool(~self.contact[0]),
                eligible=bool(eligible[0]),ball_state_write=False)
        return command

def get_task(env):
    if not hasattr(env,"_coordination_style_v041_task"):
        env._coordination_style_v041_task=CoordinationStyleReferenceTask(env,env.cfg.joint_strike)
    return env._coordination_style_v041_task
