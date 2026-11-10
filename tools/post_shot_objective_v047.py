"""v47 safety-aware POST-CONTACT learned G1 recovery reward: 5ms actual qdot.

Adds a smooth penalty on *measured* G1 velocity margin, not just samples
already over original 100% limit. Original reward (upright, landing approach,
support/slip/shank, terminal and actuator) remains present and unchanged.
This is an engineer-designed training shaping hypothesis, NOT a CBF safety
certificate and not a verified human explosive football coordination objective.
"""
import math
import torch
from post_shot_objective import recovery_objective as original_recovery_objective

FIXED=dict(warning_ratio=.70,limit_ratio=1.,weight=2.5,
           maximum_penalty_per_second=20.)

def recovery_objective_v047(state,base):
    r,components=original_recovery_objective(state,base)
    qdot=state["substep_qd_ratio_peak"]
    if qdot.shape!=r.shape or not bool(torch.isfinite(qdot).all()):
        raise ValueError("PhysX measured qdot ratio must be finite [num_envs]")
    danger=((qdot-FIXED["warning_ratio"])/
        (FIXED["limit_ratio"]-FIXED["warning_ratio"])).clamp_min(0)
    penalty=-(FIXED["weight"]*danger.square()).clamp(
        max=FIXED["maximum_penalty_per_second"])
    penalty=torch.where(state["post_active"],penalty,torch.zeros_like(penalty))
    components=dict(components,preemptive_real_joint_speed_margin_cost=penalty)
    return r+penalty,components
