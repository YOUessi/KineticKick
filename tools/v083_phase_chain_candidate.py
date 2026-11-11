#!/usr/bin/env python3
"""Phase-conditioned 29DoF *candidate* kinetic-chain generator, with fail-closed interface.

Research hypothesis only. Phase envelopes are fixed and deliberately not fit to
the six unqualified CMU clips, which lack synchronized GRF/instep impulse.
Offline tests quantify coherent timing, actuator speed targets, support gating,
and full-body ablations; they DO NOT certify PhysX or hardware collision safety.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from stance_safe_temporal_policy_v067 import RealG1State,stage_29dof_action

N=29
JOINT=G1_JOINT_ORDER
INDEX={s:i for i,s in enumerate(JOINT)}
assert len(JOINT)==N==len(INDEX)
# Exact set of future controller inputs, not an observational annotation.
PHASES=("approach","support_plant","hip_drive","torso_transfer","swing_drive","instep","follow_through","landing")
KNOTS=np.array([0.,.18,.31,.42,.53,.61,.70,.90,1.],float)
# Human sequencing is only a testable hypothesis. Angles are synthetic
# control-intent amplitude, not inferred strength or verified human biomechanics.
MOTIONS={
    "support_plant": {"left_hip_roll_joint":-.08,"left_hip_pitch_joint":.07,
                      "left_knee_joint":.065,"left_ankle_roll_joint":-.04},
    "hip_drive": {"right_hip_pitch_joint":-.15,"right_hip_yaw_joint":.105,
                  "left_hip_yaw_joint":-.07,"waist_yaw_joint":.125},
    "torso_transfer":{"waist_yaw_joint":-.12,"waist_pitch_joint":.065,
                      "waist_roll_joint":-.06},
    "swing_drive":{"right_hip_pitch_joint":.155,"right_knee_joint":-.18,
                   "right_ankle_pitch_joint":-.11,"right_ankle_roll_joint":.045},
    "instep":{"right_knee_joint":.09,"right_ankle_pitch_joint":.065},
    "follow_through":{"right_hip_pitch_joint":.075,"right_knee_joint":.07,
                      "left_hip_roll_joint":.05,"left_knee_joint":-.045},
    "landing":{"right_hip_pitch_joint":-.03,"right_knee_joint":-.025,
               "left_hip_pitch_joint":-.035,"waist_pitch_joint":-.04},
    "counter_arm":{"left_shoulder_pitch_joint":-.115,"right_shoulder_pitch_joint":.12,
                   "left_shoulder_roll_joint":.08,"right_shoulder_roll_joint":-.08,
                   "left_shoulder_yaw_joint":.095,"right_shoulder_yaw_joint":-.095,
                   "left_elbow_joint":-.08,"right_elbow_joint":.08,
                   "left_wrist_roll_joint":.025,"right_wrist_roll_joint":-.025},
}

def _smooth(t,start,end):
    if not (np.isfinite(t) and np.isfinite(start) and np.isfinite(end) and end>start):
        raise ValueError("Finite ordered support-phase interval required")
    s=np.clip((t-start)/(end-start),0.,1.)
    return float(s*s*(3.-2.*s)),float(6*s*(1-s)/(end-start) if 0<s<1 else 0.)

def _pulse(t,left,top,right):
    a,da=_smooth(t,left,top);b,db=_smooth(t,top,right)
    return a*(1-b),da*(1-b)-a*db

@dataclass(frozen=True)
class CoupledIntent:
    """Reference and derivative at one 50Hz timestep; untested intended kinematics."""
    q_offset_rad:np.ndarray
    dq_offset_rad_s:np.ndarray
    phase:str
    segment_amplitudes:dict[str,float]
    source:str="synthetic_multiphase_29dof_not_human_supervision"

def generate(t_s:float,duration_s:float=1.05,ablation:str="full",power:float=1.):
    """Smooth plant->hip->torso->leg->opposite arm->follow; 0 angle endpoints.

    Times are relative to START of candidate motion, NOT measured ball impact;
    requires externally verified support, phase and collision geometry in G1.
    """
    if not (math.isfinite(t_s) and t_s>=0 and math.isfinite(duration_s) and duration_s>0
            and math.isfinite(power) and 0<=power<=1):
        raise ValueError("Invalid time/amplitude")
    if ablation not in ("full","no_counter_arm","phase_reversed","weak_foot_only"):
        raise ValueError("Pre-registered ablation required")
    u=np.clip(t_s/duration_s,0,1)
    if ablation=="phase_reversed":u=1-u
    sign=-1. if ablation=="phase_reversed" else 1.
    q=np.zeros(N);dq=np.zeros(N);amps={}
    windows={
        "support_plant":(.12,.30,.50),
        "hip_drive":(.27,.43,.64),
        "torso_transfer":(.36,.51,.70),
        "swing_drive":(.42,.64,.80),
        "instep":(.55,.64,.74),
        "follow_through":(.65,.78,.95),
        "landing":(.79,.91,.995),
        "counter_arm":(.30,.52,.79)
    }
    for group,(start,peak,end) in windows.items():
        if group=="counter_arm" and ablation=="no_counter_arm":continue
        if ablation=="weak_foot_only" and group not in ("instep",):continue
        a,der=_pulse(float(u),start,peak,end)
        if ablation=="weak_foot_only":a*=.0125;der*=.0125
        amps[group]=power*a
        for name,magnitude in MOTIONS[group].items():
            idx=INDEX[name]
            q[idx]+=power*a*magnitude
            dq[idx]+=power*der*sign*magnitude/duration_s
    if not (np.isfinite(q).all() and np.isfinite(dq).all()):
        raise ValueError("Nonfinite output")
    seg=min(len(PHASES)-1,int(np.searchsorted(KNOTS,u,side="right")-1))
    return CoupledIntent(q,dq,PHASES[seg],amps)

def gated_original_controller(original_action,previous_delta_rad,motor_scale,
                              real:RealG1State,relative_motion_time_s,
                              ablation="full"):
    """No matching in-frame native geometry witness -> unchanged baseline.

    stage_29dof_action also checks real support loads, margin, true q/qdot,
    in-frame shape clearance and nonfoot. Never infer safety from net force.
    """
    proposed=generate(relative_motion_time_s,ablation=ablation)
    decision=stage_29dof_action(
        original_action,proposed.q_offset_rad,previous_delta_rad,motor_scale,real)
    return proposed,decision

def offline_plan():
    """Strict CPU plan for design review; no GPU activation or fit to holdout."""
    clock=np.arange(0,1.05+1e-9,.02)
    rows={}
    for ablation in ("full","no_counter_arm","phase_reversed","weak_foot_only"):
        ref=[generate(float(t),ablation=ablation) for t in clock]
        q=np.stack([a.q_offset_rad for a in ref])
        dq=np.stack([a.dq_offset_rad_s for a in ref])
        groups={}
        for name,names in {
            "support_leg":["left_hip_pitch_joint","left_knee_joint","left_hip_roll_joint"],
            "torso":["waist_yaw_joint","waist_pitch_joint","waist_roll_joint"],
            "swing_leg":["right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"],
            "arms":["left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_elbow_joint","right_elbow_joint"],
        }.items():
            ids=[INDEX[n] for n in names]
            energy=(dq[:,ids]**2).sum(axis=1)
            groups[name]={"peak_rate_s":float(clock[np.argmax(energy)]),
                           "integrated_squared_reference_velocity":float(np.trapezoid(energy,clock))}
        rows[ablation]={"peak_joint_offset_rad":float(np.abs(q).max()),
                        "peak_intended_joint_rate_rad_s":float(np.abs(dq).max()),
                        "active_joint_count":int((np.max(np.abs(q),axis=0)>1e-4).sum()),
                        "group_velocity_proxy":groups,
                        "terminal_offset_rad":float(np.abs(q[-1]).max())}
    return {"classification":"UNTRAINED_CONTROL_CANDIDATE_NOT_PHYSX_EVIDENCE",
            "human_GRF_available":False,"direct_G1_collider_manifold":False,
            "original_G1_action_deployment":"BLOCKED_FAIL_CLOSED_UNTIL_NATIVE_WITNESS",
            "ablation_controls":rows}
