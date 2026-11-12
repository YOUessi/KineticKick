#!/usr/bin/env python3
"""SHA-pinned, post-hoc *measured* coordination diagnostics for frozen v112 PhysX.

Only preregistered OFF runs are independent controller comparisons.
OFF/ON is a sensor read-only parity control, not an independent success trial.
Peaks are observational kinematic proxies, never human coordination labels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "left_leg": ("left_hip_", "left_knee_", "left_ankle_"),
    "right_leg": ("right_hip_", "right_knee_", "right_ankle_"),
    "waist": ("waist_",),
    "left_arm": ("left_shoulder_", "left_elbow_", "left_wrist_"),
    "right_arm": ("right_shoulder_", "right_elbow_", "right_wrist_"),
}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def measured(raw: dict, impact: float) -> dict:
    if raw.get("root_fixed") is not False or len(raw["joint_names"]) != 29:
        raise ValueError("NOT_ORIGINAL_FREE_ROOT_29DOF")
    tr = raw["traces"][0]
    ts = np.array([r["time_s"] for r in tr], dtype=float)
    if len(ts) < 2700 or not np.all(np.isclose(np.diff(ts), .005, atol=1e-6)):
        raise ValueError("NO_COMPLETE_REAL_200HZ_TRACE")
    vel = np.asarray([r["joint_vel"] for r in tr], dtype=float)
    com = np.asarray([r["whole_body_com_xyz"] for r in tr], dtype=float)
    support = np.asarray([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in tr],float)
    if vel.shape != (len(tr),29) or com.shape != (len(tr),3) or support.shape != (len(tr),2):
        raise ValueError("INVALID_PHYSICS_SHAPE")
    if not all(np.isfinite(a).all() for a in (ts,vel,com,support)):
        raise ValueError("INVALID_NONFINITE_PHYSICS")
    pre = (ts >= impact - .4 - 1e-8) & (ts < impact)
    post = (ts >= impact) & (ts <= impact + 1.25 + 1e-8)
    if pre.sum()<79 or post.sum()<249:
        raise ValueError("PRE_POST_WINDOW_RIGHT_CENSORED")
    indices = {key:[i for i,name in enumerate(raw["joint_names"])
                    if any(name.startswith(s) for s in prefixes)] for key,prefixes in GROUPS.items()}
    if sorted(i for ix in indices.values() for i in ix) != list(range(29)):
        raise ValueError("UNCATEGORIZED_SOURCE_JOINT")
    result = {"actual_steps_200hz":len(tr),"impact_s":impact,"full_14s":bool(ts[-1]>=13.995),
              "pre_0p4_s_rows":int(pre.sum()),"post_1p25_s_rows":int(post.sum()),
              "pre_contact_COM_horizontal_delta_m":float(np.linalg.norm(com[pre][-1,:2]-com[pre][0,:2])),
              "pre_contact_force_proxy_impulse_Ns":{"left":float(support[pre,0].sum()*.005),
                    "right":float(support[pre,1].sum()*.005)},
              "group_speed_peak_before_contact":{}}
    for group,ix in indices.items():
        speed = np.linalg.norm(vel[:,ix],axis=1)
        preidx = np.flatnonzero(pre)
        peak_idx = int(preidx[np.argmax(speed[pre])])
        result["group_speed_peak_before_contact"][group] = {
            "peak_joint_group_l2_radps":float(speed[peak_idx]),
            "peak_time_s":float(ts[peak_idx]),
            "lead_to_impact_s":float(impact-ts[peak_idx])}
    return result

def pairwise(a: dict, b: dict, impact: float) -> dict:
    if a["joint_names"] != b["joint_names"]:
        raise ValueError("CONTROLLER_JOINT_ORDER_DIFFERS")
    t1=np.array([r["time_s"] for r in a["traces"][0]],float)
    t2=np.array([r["time_s"] for r in b["traces"][0]],float)
    if t1.shape != t2.shape or not np.array_equal(t1,t2):
        raise ValueError("DIFFERENT_PHYSICAL_TIME_GRID")
    qa=np.asarray([r["joint_vel"] for r in a["traces"][0]],float)
    qb=np.asarray([r["joint_vel"] for r in b["traces"][0]],float)
    if qa.shape!=(len(t1),29) or qb.shape!=qa.shape:
        raise ValueError("INVALID_PAIRED_29DOF")
    delta=np.max(np.abs(qb-qa),axis=1)
    pre=(t1>=impact-.4-1e-8)&(t1<impact)
    pre_delta=delta[pre]
    which=np.flatnonzero(delta>1e-7)
    onset=float(t1[which[0]]) if len(which) else None
    return {"first_actual_qdot_difference_gt_1e-7_s":onset,
            "difference_onset_lead_to_impact_s":float(impact-onset) if onset is not None else None,
            "whole_episode_joint_speed_l2_rms_radps":float(np.sqrt(np.mean((qb-qa)**2))),
            "pre_0p4_s_max_joint_speed_abs_delta_radps":float(pre_delta.max()),
            "pre_0p4_s_joint_speed_rms_delta_radps":float(np.sqrt(np.mean((qb[pre]-qa[pre])**2))),
            "pre_0p4_s_distinct_physical_frames":int(np.count_nonzero(pre_delta>1e-7))}

def audit():
    batch_path=ROOT/"reports/v112_original_g1_batch.audit.json"
    batch=json.loads(batch_path.read_text())
    if batch["source_commit"]!="f9bbdb1af416b302d144d7ff3eb42c47ed8f50d2":
        raise ValueError("DIFFERENT_FROZEN_SOURCE")
    if batch["true_completed_scored_original_G1_runs"]!=10 or batch["all_shared_physical_state_parity_pass"]!=4:
        raise ValueError("INCOMPLETE_FROZEN_BATCH")
    if any(not p["passed"] for p in batch["parity"]):raise ValueError("SENSOR_CHANGED_PHYSICS")
    outputs={}
    for row in batch["runs"]:
        fn=ROOT/"reports"/("v112_"+row["name"]+".json")
        if digest(fn)!=row["raw_sha256"]:raise ValueError("SOURCE_SHA_MISMATCH:"+row["name"])
        if row["observer"]:continue
        if not row["certified_original_G1_cuda_200Hz"]:
            raise ValueError("UNQUALIFIED_200HZ:"+row["name"])
        raw=json.loads(fn.read_text())
        impact=float(row["hard_gates"]["observed_ball_contact_s"])
        key=row["scene"]+"/"+row["control"]
        outputs[key]={"source_run":row["name"],"source_sha256":row["raw_sha256"],
                      "hard_gate_pass":row["hard_gates"]["hard_gate_pass"],
                      "failed_gates":row["hard_gates"]["failed_gates"],
                      "instep_fraction":row["hard_gates"]["measured_instep_fraction"],
                      "ball_speed_proxy_mps":row["hard_gates"]["ball_speed_peak_mps_post_5_to_110ms_proxy"],
                      "dynamics":measured(raw,impact)}
    pairs=[]
    for scene in ("W","X","Y"):
        rows=[r for r in batch["runs"] if r["scene"]==scene and not r["observer"]]
        if [r["control"] for r in rows]!=["v32","cmu02"]:raise ValueError("MISSING_REGISTERED_CONTROLLER")
        data=[]
        for row in rows:
            raw=json.loads((ROOT/"reports"/("v112_"+row["name"]+".json")).read_text())
            data.append(raw)
        impact=float(rows[0]["hard_gates"]["observed_ball_contact_s"])
        if float(rows[1]["hard_gates"]["observed_ball_contact_s"])!=impact:raise ValueError("DIFFERENT_CONTACT_TIME")
        pairs.append({"scene":scene,"independent_controllers":2,
                      "cmu_minus_v32_ball_speed_proxy_mps":float(rows[1]["hard_gates"]["ball_speed_peak_mps_post_5_to_110ms_proxy"]-rows[0]["hard_gates"]["ball_speed_peak_mps_post_5_to_110ms_proxy"]),
                      "cmu_minus_v32_instep_fraction":float(rows[1]["hard_gates"]["measured_instep_fraction"]-rows[0]["hard_gates"]["measured_instep_fraction"]),
                      "measured_controller_difference":pairwise(*data,impact)})
    return {"version":"v0.1.112","source_master_sha256":digest(batch_path),
            "n_original_physx_repeats":10,"independent_controller_scene_combinations":6,
            "independent_scenes":3,"parity_pairs":4,
            "strict_pass_repeats":sum(r["hard_gates"]["hard_gate_pass"] for r in batch["runs"]),
            "strict_pass_independent":sum(v["hard_gate_pass"] for v in outputs.values()),
            "native_wrist_hip_manifold_certified":False,"human_grf_cop_impact_truth_certified":False,
            "no_new_learned_controller_actuated":True,
            "independent_runs":outputs,"preregistered_scene_pairs":pairs}

def main():
    dst=ROOT/"reports/v112_coordination_falsification.audit.json"
    if dst.exists() or dst.is_symlink():raise FileExistsError("WRITE_ONCE_COORDINATION_AUDIT")
    result=audit()
    with dst.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V112_COORDINATION_FALSIFICATION",json.dumps({k:v for k,v in result.items() if k not in ("independent_runs","preregistered_scene_pairs")},ensure_ascii=False))
    for row in result["preregistered_scene_pairs"]:
        print(row["scene"],row["measured_controller_difference"],"speed_diff",row["cmu_minus_v32_ball_speed_proxy_mps"])
if __name__=="__main__":main()
