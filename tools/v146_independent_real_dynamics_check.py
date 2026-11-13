#!/usr/bin/env python3
"""Independent historical v54 200Hz whole-body audit of v146 derived data.

This implementation reads raw SHA-known JSON directly and DOES NOT import
v146_real_200hz_coupled_audit or use its indexing/ratio helper functions.
Counts all 29 original joint velocities and compares every stored raw state.
Fails closed on a missing or wrong original source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from v142_native_pair_ledger import safe_write_once

SOURCE = Path("/home/you/projects/kinetickick")
ORIGINAL_SHA = {
    "frozen_v32":"f7a130707fdb761d7f7185d6805f4887d4ca929fec8965bfd35ae13d6666e01c",
    "cmu_10_02":"e197cf6807709e077fa5cfd2b2582bf988dd1437f249bd61addf7835a7d2b255",
    "cmu_10_05":"d46a48f84dbaa7a367eac3e5dc399d4378007485c369c2b57c1c67775ccbfac1",
}
ORIGINAL_COUNT = {"frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as file:
        while True:
            part=file.read(1<<20)
            if not part:break
            h.update(part)
    return h.hexdigest()


def independent_check(output, *, originals=SOURCE):
    recorded=json.loads(Path(output).read_text())
    assert recorded["schema"]=="v146_original_physics_cpu_reread_not_new_rollout"
    cases=recorded["cases"]
    findings={}
    for label in ORIGINAL_SHA:
        source=Path(originals)/"reports"/("v054_"+label+"_contact_diagnostics.json")
        if sha(source)!=ORIGINAL_SHA[label]:
            raise ValueError("INDEPENDENT_ORIGINAL_REAL_PHYSX_SHA_DRIFT")
        data=json.loads(source.read_text())
        limit=data["joint_velocity_limits"][0]
        original=data["traces"][0]
        derived=cases[label]["full_real_200hz_coupled_rows"]
        expected=ORIGINAL_COUNT[label]
        if len(derived)!=expected:
            raise ValueError("DERIVED_FRAME_COUNT_INCONSISTENT")
        # Independent real per-tick timestamp index and *all 29* qdot checks.
        old_peak=0.
        old_exceed=0
        missing=0
        for i in range(expected):
            src=original[i]
            dst=derived[i]
            native_step=round(float(src["time_s"])*200.)
            if native_step!=i+1 or dst["physics_step"]!=native_step:
                raise ValueError("INDEPENDENT_CLOCK_MISMATCH")
            if len(src["joint_vel"])!=29 or len(dst["qdot29"])!=29:
                raise ValueError("INDEPENDENT_NOT_ORIGINAL29")
            if src["joint_pos"]!=dst["q29"] or src["joint_vel"]!=dst["qdot29"]:
                raise ValueError("REAL_Q_DQ_MUTATED")
            if src["joint_torque_estimate"]!=dst["sim_torque29"]:
                raise ValueError("REAL_SIM_TORQUES_MUTATED")
            if src["whole_body_com_xyz"]!=dst["com_world"]:
                raise ValueError("REAL_COM_MUTATED")
            if src["ball_velocity"]!=dst["ball_velocity"]:
                raise ValueError("REAL_BALL_VELOCITY_MUTATED")
            if (src["support_foot_force_N"]["left"]!=dst["left_foot_net_contact_N"]
                    or src["support_foot_force_N"]["right"]!=dst["right_foot_net_contact_N"]):
                raise ValueError("REAL_FOOT_NET_FORCES_MUTATED")
            ratio=max(abs(v)/L for v,L in zip(src["joint_vel"],limit))
            old_peak=max(old_peak,ratio)
            old_exceed+=int(ratio>1.)
            if not math.isclose(ratio,max(dst["qdot_ratio29"]),rel_tol=0,abs_tol=1e-12):
                raise ValueError("INDEPENDENT_QDOT_RATIO_WRONG")
            # Do not invent arm-to-hip distance from missing 200Hz link poses.
            missing+=int(dst["arm_hip_true_collider_pair"] is None)
        summary=cases[label]
        if (old_exceed!=summary["qdot_limit_violating_steps"]
                or abs(old_peak-summary["peak_original29_qdot_limit_ratio"])>1e-12
                or missing!=expected):
            raise ValueError("INDEPENDENT_SUMMARY_CONFLICT")
        center=float(data["records"][0]["contact_time_s"])
        manual_pre=[r for r in original[:expected]
                    if center-.400000001 <=r["time_s"] < center-1e-9]
        manual_post=[r for r in original[:expected]
                     if center+1e-9 < r["time_s"] <=center+1.250000001]
        if (len(manual_pre)!=80 or len(manual_post)!=250
                or summary["window"]["before_steps"]!=80
                or summary["window"]["after_steps"]!=250
                or abs(manual_pre[0]["time_s"]-summary["window"]["first_before_time_s"])>1e-7):
            raise ValueError("INDEPENDENT_WINDOW_MISMATCH")
        tail=original[expected:]
        if label=="cmu_10_05":
            if len(tail)!=1 or not tail[0]["terminated"] or abs(tail[0]["time_s"]-5.14)>0.0001:
                raise ValueError("INDEPENDENT_TERMINAL_DUPLICATE_MISMATCH")
        elif tail:
            raise ValueError("UNEXPECTED_EXTRA_PHYSICS_TAIL")
        findings[label]={
            "independent_sha_checked_source":ORIGINAL_SHA[label],
            "original_29dof_5ms_states_independently_compared":expected,
            "independent_peak_qdot_limit_ratio":old_peak,
            "independent_qdot_exceeding_steps":old_exceed,
            "whole_body_q_dq_tau_com_ball_foot_net_exact_per_row":True,
            "pre_real_200hz_steps":len(manual_pre),
            "post_real_200hz_steps":len(manual_post),
            "true_native_wrist_hip_pair":None,
            "gpu_simulation_run_here":False,
        }
    # Independent source comparison, no previously derived precomputed diffs.
    originals_by_name={
        n:json.loads((Path(originals)/"reports"/
            ("v054_"+n+"_contact_diagnostics.json")).read_text())["traces"][0]
        for n in ORIGINAL_SHA}
    pairs={}
    for n,m in (("frozen_v32","cmu_10_02"),("frozen_v32","cmu_10_05"),
                ("cmu_10_02","cmu_10_05")):
        a,b=originals_by_name[n],originals_by_name[m]
        end=min(ORIGINAL_COUNT[n],ORIGINAL_COUNT[m])
        first=next((i+1 for i in range(end)
                    if a[i]["joint_pos"]!=b[i]["joint_pos"]),None)
        got=recorded["interpolicy_original_center_ball_comparison"][n+"__"+m][
            "first_different_original29_q_time_s"]
        if first is None or abs(first*.005-got)>1e-8:
            raise ValueError("INDEPENDENT_PRECONTACT_DIVERGENCE_MISMATCH")
        pairs[n+"__"+m]={"first_divergent_true_200hz_sample":first,
                          "first_divergent_time_s":first*.005}
    return {
        "version":"v0.1.146",
        "independent_actual_original_200hz_rows":sum(x["original_29dof_5ms_states_independently_compared"]
                                                     for x in findings.values()),
        "independent_no_gpu_reruns":True,
        "no_real_contact_manifold_claimed":True,
        "source_conditions":findings,
        "independent_original_center_pair_differences":pairs,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=independent_check(a.input)
    checksum=safe_write_once(a.output,r)
    print(json.dumps({"output":str(a.output),"sha256":checksum,**r},ensure_ascii=False))


if __name__=="__main__":
    main()
