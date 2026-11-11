#!/usr/bin/env python3
"""Frozen, SYNTHETIC (not learned human biomechanics) multi-joint temporal coordination hypothesis.

Real G1 29DoF executable pre-contact action adapter. The old v32 balance expert
remains responsible for recovery. No source data or learned high-speed soccer
ground-reaction/contact labels exist for the proposed phase waveform.
All safety guards here are REJECTION PROXIES, not certified collider clearance.
Original PhysX limits, nonfoot terminal logic and ball physics are untouched.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from post_shot_recovery import PostShotRecoveryTask
from running_balance import ground_force
from wholebody_safety_interface_v055 import G1_JOINT_ORDER, phase_coupled_intent

WINDOW=(.86,1.56)
AMPLITUDE=.075
SLEW=.006
def phase_trajectory():
    """Frozen two-stage temporal intent, 50Hz, intended to test nontrivial 29DoF coordination."""
    out=np.zeros((700,29),dtype=np.float32)
    for i in range(700):
        t=i*.02
        if WINDOW[0]<=t<=WINDOW[1]:
            # A continuous, explicitly NOT human-ground-truth phase curve.
            phase=(t-WINDOW[0])/(WINDOW[1]-WINDOW[0])
            out[i]=phase_coupled_intent(phase,AMPLITUDE).astype(np.float32)
    assert np.isfinite(out).all() and np.abs(out).max()<=AMPLITUDE+1e-6
    return out

def admissibility(t,support,ratio,ball_distance,wrist_hip,contact,nonfoot,q,limits):
    if len(t)!=len(ratio) or support.shape!=(len(t),2) or q.shape!=(len(t),29) or limits.shape!=(len(t),29,2):
        raise ValueError("Original G1 29DoF batched safety shape mismatch")
    numeric=(t, support,ratio,ball_distance,wrist_hip,nonfoot,q,limits)
    if not all(bool(torch.isfinite(x).all()) for x in numeric):
        raise ValueError("Nonfinite G1 safety input: fail closed")
    flags={
        "window":(t>=WINDOW[0])&(t<=WINDOW[1]),
        "stance":(support[:,0]>=80.)&(support[:,1]<=65.),
        "qdot":ratio<=.94,
        "ball":(ball_distance<=1.2)&(ball_distance>=0.),
        "wrist_hip_origin_proxy":wrist_hip>=.18,
        "nonfoot_force":nonfoot<5.,
        "pre_contact":~contact,
        "joint_position_limits":((q>limits[:,:,0]+.025)&(q<limits[:,:,1]-.025)).all(-1)
    }
    ok=torch.ones_like(t,dtype=torch.bool)
    for x in flags.values():ok &= x
    return ok,flags

class PhaseCoordinatedTask(PostShotRecoveryTask):
    def __init__(self,env,config):
        cfg=config.get("phase_coordinated_v063")
        if not isinstance(cfg,dict) or cfg != {
            "enabled":True,"source":"synthetic_temporal_ablation_not_human_learned",
            "max_joint_rad":.075,"max_step_rad":.006,
            "preimpact_start_s":.86,"preimpact_end_s":1.56}:
            raise ValueError("Frozen synthetic-phase protocol mismatch")
        if (config.get('human_coordination_reference',{}).get('enabled') or
            config.get('coordination_style_v041',{}).get('enabled')):
            raise ValueError("Do not mix weak v40/v41 human proxy into independent v63 candidate")
        super().__init__(env,config)
        if tuple(self.robot.joint_names)!=G1_JOINT_ORDER:
            raise ValueError("Original G1 joint order not verified")
        self.phase_curve=torch.tensor(phase_trajectory(),device=self.device)
        self.phase_applied=torch.zeros(self.n,29,device=self.device)
        self.phase_last=self.phase_applied.clone()
        self.phase_nonzero_steps=0
        self.phase_max_rad=0.
        self.phase_gate_counts={key:0 for key in (
            "window","stance","qdot","ball","wrist_hip_origin_proxy",
            "nonfoot_force","pre_contact","joint_position_limits")}
        self.phase_eligible_steps=0
    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.phase_applied[ids]=0.
        self.phase_last[ids]=0.
        self.phase_nonzero_steps=0
        self.phase_max_rad=0.
        self.phase_eligible_steps=0
        for k in self.phase_gate_counts:self.phase_gate_counts[k]=0
    def commands(self,actions=None):
        if actions is not None:raise ValueError("Synthetic coordination has no external tunable actions")
        base=super().commands(None)  # one and only one original expert action call
        self.phase_last=self.phase_applied.clone()
        now=self.elapsed*.02
        r=self.robot.data
        support=ground_force(self.env)
        ratio=(r.joint_vel/r.joint_vel_limits).abs().amax(-1)
        distance=torch.linalg.vector_norm(
            self.ball.data.root_pos_w[:,:2]-r.root_pos_w[:,:2],dim=-1)
        wrist=r.body_pos_w[:,self.robot.body_names.index("left_wrist_yaw_link")]
        hip=r.body_pos_w[:,self.robot.body_names.index("left_hip_roll_link")]
        link_origin_distance=torch.linalg.vector_norm(wrist-hip,dim=-1)
        sensor=self.env.scene["contact_forces"]
        nonfoot_indices=[i for i,n in enumerate(sensor.body_names)
            if n not in ("left_ankle_roll_link","right_ankle_roll_link")]
        nonfoot=sensor.data.net_forces_w[:,nonfoot_indices].norm(dim=-1).amax(-1)
        limits=r.soft_joint_pos_limits
        if limits.shape!=(self.n,29,2):
            raise ValueError("G1 original 29DoF limit tensor absent")
        eligible,gates=admissibility(now,support,ratio,distance,link_origin_distance,
            self.contact,nonfoot,r.joint_pos,limits)
        for key,flag in gates.items():
            self.phase_gate_counts[key]+=int(flag.sum().item())
        self.phase_eligible_steps+=int(eligible.sum().item())
        frame=self.elapsed.long().clamp(0,699)
        target=torch.where(eligible[:,None],self.phase_curve[frame],torch.zeros_like(self.phase_applied))
        # Fail closed on ineligible inputs; then gently decay any outstanding
        # commanded deviation instead of creating an instantaneous discontinuity.
        self.phase_applied+=torch.clamp(target-self.phase_applied,-SLEW,SLEW)
        self.phase_applied.clamp_(-AMPLITUDE,AMPLITUDE)
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
            device=self.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):
            raise ValueError("Original motor scale invalid")
        command=base+self.phase_applied/scale
        self.phase_applied=(command-base)*scale
        active=bool((self.phase_applied.abs()>1e-5).any())
        if active:self.phase_nonzero_steps+=1
        self.phase_max_rad=max(self.phase_max_rad,float(self.phase_applied.abs().amax()))
        if self.n==1 and self.debug:
            self.debug[-1]["phase_coordinated_v063"]=dict(
                enabled=True,synthetic_not_learned=True,
                eligible=bool(eligible[0]),gates={k:bool(x[0]) for k,x in gates.items()},
                maximum_delta_rad=float(self.phase_applied[0].abs().amax()),
                applied_delta_rad=self.phase_applied[0].tolist(),
                baseline_action=base[0].tolist(),final_action=command[0].tolist(),
                actual_measured_qdot_ratio=float(ratio[0]),
                left_wrist_hip_link_origin_distance_m=float(link_origin_distance[0]),
                nonfoot_force_N=float(nonfoot[0]),
                actuator_state_write=False,ball_state_write=False)
        return command

def get_task(env):
    if not hasattr(env,"_phase_coordinated_v063_task"):
        env._phase_coordinated_v063_task=PhaseCoordinatedTask(env,env.cfg.joint_strike)
    return env._phase_coordinated_v063_task
