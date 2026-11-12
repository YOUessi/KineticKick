#!/usr/bin/env python3
"""Real CMU 120Hz C3D whole-body observable feature eligibility, no impact labels.

The local files are not qualified elite athlete instep force supervision.
Provides SHA-locked marker-based kinematic observables useful for later
temporal-model hypotheses, not a foot-ball contact timestamp or GRF.
"""
from __future__ import annotations
import json
from pathlib import Path

import ezc3d
import numpy as np
from v116_human_qualification import ROOT,REL,TRIALS,digest

REQUIRED=("RANK","RTOE","LANK","LTOE","RSHO","LSHO",
          "RWRA","LWRA","LBWT","RBWT","RTHI","LTHI")
RATE=120.0


def derive_marker_kinematics(c3d) -> dict:
    units=c3d["parameters"]["POINT"]["UNITS"]["value"]
    if list(units)!=["mm"]:
        raise ValueError("C3D_UNITS_NOT_MM_OR_UNQUALIFIED")
    rate=float(c3d["header"]["points"]["frame_rate"])
    if rate!=RATE:raise ValueError("NOT_ORIGINAL_120HZ")
    labels=c3d["parameters"]["POINT"]["LABELS"]["value"]
    pts=np.asarray(c3d["data"]["points"][:3],dtype=float)
    if pts.ndim!=3 or pts.shape[0]!=3:raise ValueError("INVALID_C3D_POINTS")
    picked={}
    for name in REQUIRED:
        matches=[i for i,v in enumerate(labels) if v.split(":")[-1]==name]
        if len(matches)!=1:raise ValueError("AMBIGUOUS_OR_MISSING_REAL_MARKER:"+name)
        picked[name]=pts[:,matches[0],:].T/1000.0
    all_valid=np.array([np.isfinite(picked[k]).all(axis=1) for k in REQUIRED])
    valid_all=all_valid.all(axis=0)
    n=pts.shape[-1]
    if n<120 or valid_all.mean()<0.9:raise ValueError("INSUFFICIENT_HUMAN_MARKER_COVERAGE")

    # Source measured positions; finite differences only on valid adjacent frames.
    def speed(name):
        p=picked[name]
        v=np.linalg.norm(np.diff(p,axis=0)*RATE,axis=1)
        return np.where(np.isfinite(v),v,np.nan)
    right_foot=speed("RTOE")
    left_foot=speed("LTOE")
    pelvis=(picked["RBWT"]+picked["LBWT"])*.5
    torso=(picked["RSHO"]+picked["LSHO"])*.5
    pelvis_speed=np.linalg.norm(np.diff(pelvis,axis=0)*RATE,axis=1)
    torso_speed=np.linalg.norm(np.diff(torso,axis=0)*RATE,axis=1)
    left_wrist_speed=speed("LWRA")
    right_wrist_speed=speed("RWRA")
    # Yaw-like shoulder transverse vector angle is only in raw lab x/y axes,
    # NOT calibrated to the ball/goals and not inferred kinetic power.
    shoulder=picked["RSHO"]-picked["LSHO"]
    shoulder_angle=np.unwrap(np.arctan2(shoulder[:,1],shoulder[:,0]))
    shoulder_angular_rate=np.diff(shoulder_angle)*RATE
    measured=dict(right_toe_speed_mps=right_foot,
                  left_toe_speed_mps=left_foot,
                  pelvis_center_speed_mps=pelvis_speed,
                  shoulder_center_speed_mps=torso_speed,
                  left_wrist_speed_mps=left_wrist_speed,
                  right_wrist_speed_mps=right_wrist_speed,
                  transverse_shoulder_axis_rate_radps=shoulder_angular_rate)
    def summarize(signal):
        good=np.isfinite(signal)
        if good.mean()<.9:raise ValueError("INSUFFICIENT_VALID_DERIVED_FEATURE")
        arg=np.nanargmax(np.abs(signal))
        return dict(valid_adjacent_intervals=int(good.sum()),
                    median_absolute=float(np.nanmedian(np.abs(signal))),
                    peak_absolute=float(np.nanmax(np.abs(signal))),
                    time_of_peak_absolute_s=float((arg+1)/RATE))
    return dict(n_frames=n,marker_xyz_sample_rate_hz=RATE,
        distinct_required_markers=len(REQUIRED),
        fraction_frames_with_all_required_markers=float(valid_all.mean()),
        derived_kinematics={k:summarize(v) for k,v in measured.items()},
        impact_time_known=False,kicking_foot_certified=False,
        biological_force_cop_or_ball_impulse_available=False,
        actual_ball_goal_coordinate_system_verified=False)


def inspect(root:Path=ROOT):
    out=[]
    for subject,stem in TRIALS:
        path=root/REL/subject/(stem+".c3d")
        if not path.is_file():raise FileNotFoundError(path)
        obj=ezc3d.c3d(str(path))
        row=derive_marker_kinematics(obj)
        out.append(dict(subject=subject,trial=stem,
            source_sha256=digest(path),**row))
    if len({r["source_sha256"] for r in out})!=6:
        raise ValueError("SIX_UNIQUE_SOURCE_TRIALS_REQUIRED")
    return dict(version="v0.1.117",subjects=len({r["subject"] for r in out}),
        unique_real_c3d_trials=len(out),
        source_marker_frames=sum(r["n_frames"] for r in out),
        true_kinematic_markers_in_qualification=tuple(REQUIRED),
        across_subject_heldout_training_not_approved=True,
        elite_instep_collision_impulses_certified=0,
        synchronized_grf_cop_trials=0,
        supervised_real_foot_ball_contact_labels=0,
        professional_player_skill_identity_certified=False,
        no_muscle_or_ball_force_labels_invented=True,
        per_trial=out)


if __name__=="__main__":
    result=inspect()
    out=ROOT/"reports/v117_human_marker_temporal_eligibility.audit.json"
    with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V117_HUMAN_MARKER_TEMPORAL",json.dumps({
        k:v for k,v in result.items() if k!="per_trial"
    }),flush=True)
