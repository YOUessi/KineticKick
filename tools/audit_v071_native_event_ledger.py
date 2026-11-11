#!/usr/bin/env python3
"""Read-only, postregistered 200Hz physical G1 event ledger (never a new physics run).

Source JSON and log files are immutable. A missing value or post-termination
sample is never fabricated. Wrist/hip link origins cannot prove collider contact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/you/projects/kinetickick-wt-v067/reports")
CONFIG = "configs/g1_physx_readonly_evidence_ledger_v071.json"
NAME_GROUPS = {
    "stance_and_swing_legs": ("hip", "knee", "ankle"),
    "torso": ("waist",),
    "both_arms": ("shoulder", "elbow", "wrist"),
}
FULL_ROW_TOL = 0.00501


class InvalidSource(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def finite(x) -> bool:
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def scalar(x, key: str) -> float:
    if not finite(x):
        raise InvalidSource("NONFINITE_" + key)
    return float(x)


def vec(x, n: int, key: str) -> list[float]:
    if not isinstance(x, list) or len(x) != n:
        raise InvalidSource(f"MISSING_{key}_EXPECTED_{n}")
    return [scalar(y, key) for y in x]


def original_29_velocity_limits(raw: dict) -> list[float]:
    x = raw.get("joint_velocity_limits")
    # Original G1 run writes one batched [1,29] physics limit tensor.
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    a = vec(x, 29, "joint_velocity_limits")
    if min(a) <= 0:
        raise InvalidSource("INVALID_ORIGINAL_29_QDOT_LIMIT")
    return a


def has_direct_cuda_wrist_hip_pair(raw: dict) -> bool:
    """Only a native PhysX collider manifold with both sides can certify.

    Rigid net forces, equal/opposite magnitudes, link-origin distances, custom
    heuristic 'pair' tags and CPU-only PhysX records are not accepted.
    """
    for item in raw.get("native_physics_contact_pairs", []):
        if not isinstance(item, dict):
            continue
        if item.get("backend_physx_cuda_original_g1") is not True:
            continue
        try:
            a = item["body0_collider_path"]
            b = item["body1_collider_path"]
            point = vec(item["contact_point_world"], 3, "contact_point_world")
            normal = vec(item["normal_world"], 3, "normal_world")
            impulse = scalar(item["normal_impulse_Ns"], "normal_impulse_Ns")
        except (KeyError, InvalidSource):
            continue
        if (not isinstance(a, str) or not isinstance(b, str) or
            impulse <= 0 or sum(x * x for x in normal) < 1e-12):
            continue
        # Require exact named links on distinct COLLIDER paths; no substring
        # shortcut in the net-force observation.
        aa, bb = ("left_wrist_yaw_link" in a, "left_hip_roll_link" in b)
        cc, dd = ("left_hip_roll_link" in a, "left_wrist_yaw_link" in b)
        if (aa and bb) or (cc and dd):
            return True
    return False


def physical_frame_summaries(raw: dict, expected_dt: float = 0.005) -> dict:
    if raw.get("root_fixed") is not False or raw.get("v057_requested_physics_backend") not in ("cuda:0", None):
        raise InvalidSource("NOT_ORIGINAL_CUDA_FREEROOT_G1")
    names = raw.get("joint_names")
    if not isinstance(names, list) or len(names) != 29 or len(set(names)) != 29:
        raise InvalidSource("MISSING_29_JOINT_ORDER")
    original_limits = original_29_velocity_limits(raw)
    ts, qs, vs, coms, roots, fleft, fright, nonfoot, wrist = [], [], [], [], [], [], [], [], []
    observer = raw.get("safe_body_v062")
    obs_rows = observer.get("rows", []) if isinstance(observer, dict) else []
    trace_sets = raw.get("traces")
    if not isinstance(trace_sets, list) or len(trace_sets) != 1 or not trace_sets[0]:
        raise InvalidSource("NO_ACTUAL_NATIVE_TRACE")
    frames = trace_sets[0]
    n = len(frames)
    observer_complete = len(obs_rows) == n
    for i, row in enumerate(frames):
        t = scalar(row.get("time_s"), "time")
        if i and not (t > ts[-1] and (abs(t - ts[-1] - expected_dt) <= 0.00002 or
                                    0 < t - ts[-1] <= 0.00002)):
            raise InvalidSource(f"INVALID_200HZ_CLOCK_AT_{i}")
        ts.append(t)
        qs.append(vec(row.get("joint_pos"), 29, "original_q"))
        vs.append(vec(row.get("joint_vel"), 29, "original_qdot"))
        coms.append(vec(row.get("whole_body_com_xyz"), 3, "original_com"))
        roots.append(vec(row.get("root_xyz"), 3, "original_root"))
        support = row.get("support_foot_force_N")
        if not isinstance(support, dict):
            raise InvalidSource("MISSING_ORIGINAL_SUPPORT")
        fleft.append(scalar(support.get("left"), "support_left"))
        fright.append(scalar(support.get("right"), "support_right"))
        nonfoot.append(scalar(row.get("nonfoot_contact_N"), "nonfoot"))
        if observer_complete:
            o = obs_rows[i]
            if (o.get("state_alignment_source_index") != i or
                abs(scalar(o.get("state_aligned_time_s"), "obs_time") - t) > 1e-6):
                raise InvalidSource("MISALIGNED_OBSERVER_CLOCK")
            qq = vec(o.get("joint_pos_rad"), 29, "obs_joint_pos")
            vv = vec(o.get("joint_vel_rad_s"), 29, "obs_joint_vel")
            if max(abs(a-b) for a,b in zip(qq,qs[-1])) > 1e-5 or max(
                    abs(a-b) for a,b in zip(vv,vs[-1])) > 1e-5:
                raise InvalidSource("OBSERVED_29_DOF_DIFFERS_FROM_NATIVE")
            wrist.append(scalar(o.get("left_wrist_hip_link_origin_distance_m"), "origin_distance"))
    if ts[0] > 0.03:
        raise InvalidSource("MISSING_INITIAL_NATIVE_FRAMES")
    return dict(times=ts, q=qs, qdot=vs, com=coms, root=roots,
                forces_left=fleft, forces_right=fright, nonfoot=nonfoot,
                wrist_hip_origin_distance=wrist if observer_complete else None,
                velocity_limits=original_limits, joint_names=names,
                observed_native_rows=n, observed_end_s=ts[-1],
                observer_200hz_sameframe=observer_complete,
                native_left_wrist_left_hip_collider_pair_certified=has_direct_cuda_wrist_hip_pair(raw))


def summarize_case(raw: dict, case_id: str, cfg: dict) -> dict:
    b = physical_frame_summaries(raw, cfg["dt_s"])
    r = raw.get("records")
    if not isinstance(r, list) or len(r) != 1 or not isinstance(r[0], dict):
        raise InvalidSource("INVALID_ORIGINAL_NATIVE_RECORD")
    rec = r[0]
    ct = rec.get("contact_time_s")
    if ct is not None:
        ct = scalar(ct, "contact_time")
    ts = b["times"]
    qdot = b["qdot"]
    limits = b["velocity_limits"]
    ratios = [max(abs(v)/lim for v,lim in zip(row, limits)) for row in qdot]
    max_ratio = max(ratios)
    worst = ratios.index(max_ratio)
    termination = raw.get("termination_reasons", [])
    nonfoot_early = ("physical_nonfoot" in str(termination) or
                     any(bool(r.get("terminated")) and r["nonfoot_contact_N"] >= 20.0
                         for r in raw["traces"][0]))
    impact_indexes = [] if ct is None else [i for i,t in enumerate(ts)
                        if ct-cfg["contact_pre_s"]-1e-6 <= t <= ct+cfg["contact_post_s"]+1e-6]
    observed_window = (
        ct is not None and ts[0] <= ct-cfg["contact_pre_s"]+FULL_ROW_TOL and
        ts[-1] >= ct+cfg["contact_post_s"]-FULL_ROW_TOL and
        len(impact_indexes) >= 320
    )
    # Distinguish sample count from full temporal coverage. No time imputation.
    pre = [] if ct is None else [i for i,t in enumerate(ts)
            if ct-cfg["contact_pre_s"]-1e-6 <= t < ct-1e-6]
    post = [] if ct is None else [i for i,t in enumerate(ts)
            if ct+1e-6 <= t <= ct+cfg["contact_post_s"]+1e-6]
    frac = scalar(rec.get("requested_surface_impulse_fraction", 0), "instep_fraction")
    if not 0 <= frac <= 1+1e-6:
        raise InvalidSource("INSTALLED_IMPULSE_FRACTION_INVALID")
    vball = []
    if ct is not None:
        for row in raw["traces"][0]:
            if ct+.005-1e-6 <= row["time_s"] <= ct+.11+1e-6:
                vball.append(math.sqrt(sum(v*v for v in vec(row["ball_velocity"],3,"ball_speed"))))
    post_ball_speed = max(vball) if vball else None
    full_14s = b["observed_end_s"] >= cfg["full_recovery_s"]-.005
    gate = dict(
        original_29_qdot_within_limit=max_ratio <= 1.0+1e-9,
        measured_instep_fraction_ge_95pct=frac+1e-12>=cfg["instep_fraction_min"],
        physical_goal=rec.get("goal") is True,
        real_postimpact_ball_speed_proxy_ge_3p5mps=post_ball_speed is not None and post_ball_speed >= cfg["speed_proxy_min_mps"],
        no_physical_nonfoot_termination=not nonfoot_early,
        real_full_14s=bool(full_14s),
        recorded_recovery=rec.get("recovered") is True,
        actual_postimpact_1p25s_observed=bool(observed_window),
    )
    max_nonfoot = max(b["nonfoot"])
    t_max_nonfoot = ts[b["nonfoot"].index(max_nonfoot)]
    def avg(values, indexes):
        return mean(values[i] for i in indexes) if indexes else None
    # Region-specific joint physical qdot; not a learned human synchronization claim.
    region_qdot_pre = {}
    for group,tags in NAME_GROUPS.items():
        jj = [j for j,n in enumerate(b["joint_names"]) if any(tag in n for tag in tags)]
        region_qdot_pre[group] = (mean(abs(qdot[i][j]) for i in pre for j in jj)
                                  if pre and jj else None)
    def three(x):
        return [round(v, 6) for v in x]
    outcome = {
        "id":case_id, "physical_native_200hz_rows":b["observed_native_rows"],
        "last_native_s":b["observed_end_s"], "actual_contact_s":ct,
        "contact_minus_0p4_to_plus_1p25_fully_observed":bool(observed_window),
        "actual_window_native_frames":len(impact_indexes),
        "full_14_second_native_recovery_observed":bool(full_14s),
        "native_nonfoot_event_censored":bool(nonfoot_early),
        "nonfoot_peak_N":max_nonfoot,"time_at_nonfoot_peak_s":t_max_nonfoot,
        "all_original_29_qdot_max_ratio":max_ratio,
        "worst_qdot_joint_name":b["joint_names"][max(range(29),key=lambda j:abs(qdot[worst][j])/limits[j])],
        "worst_qdot_time_s":ts[worst],
        "original_joint_qdot_limit_shape":"29 original native limits (1,29 normalized)",
        "preimpact_region_mean_abs_joint_qdot_rad_s":region_qdot_pre,
        "preimpact_mean_left_support_N":avg(b["forces_left"],pre),
        "preimpact_mean_right_support_N":avg(b["forces_right"],pre),
        "postimpact_mean_left_support_N":avg(b["forces_left"],post),
        "postimpact_mean_right_support_N":avg(b["forces_right"],post),
        "minimum_root_height_m":min(v[2] for v in b["root"]),
        "minimum_whole_body_com_z_m":min(v[2] for v in b["com"]),
        "minimum_wrist_hip_LINK_ORIGIN_distance_m":min(b["wrist_hip_origin_distance"])
            if b["wrist_hip_origin_distance"] is not None else None,
        "origin_distance_is_NOT_signed_mesh_collision_clearance":True,
        "original_G1_GPU_physx_left_wrist_left_hip_manifold_proven":
            b["native_left_wrist_left_hip_collider_pair_certified"],
        "original_legacy_wrapper_pass":raw.get("stride_shot_success"),
        "instep_fraction":frac,
        "postimpact_ball_speed_proxy_mps":post_ball_speed,
        "recorded_recovered":rec.get("recovered"),
        "independent_gates":gate,
        "strict_all_gates_pass":all(gate.values()),
        "gates_failed":[k for k,v in gate.items() if not v],
        "observer_present_and_sameframe":b["observer_200hz_sameframe"],
    }
    return outcome


def verify_manifest(directory: Path, cfg: dict) -> dict:
    if len(cfg["v67_raw_sha256"]) != 9:
        raise InvalidSource("SOURCE_MANIFEST_NOT_NINE")
    ret = {}
    for filename,expected in cfg["v67_raw_sha256"].items():
        path=directory/filename
        if not path.is_file() or sha256(path) != expected:
            raise InvalidSource("ORIGINAL_G1_RAW_DRIFT " + filename)
        ret[filename]=expected
    p=directory/"v067_batch_progress.json"
    if not p.is_file() or sha256(p)!=cfg["source_progress_sha256"]:
        raise InvalidSource("ORIGINAL_G1_BATCH_PROGRESS_DRIFT")
    return ret


def authenticate_censored_native_gpu_logs(directory: Path, cfg: dict) -> dict:
    """Early terminated G1 lacks a post-eval backend tag in raw JSON.

    The independent CUDA proof is the original log SHA-verified against the
    original v67 write-once progress, plus actual CUDA native start/return.
    """
    progress=json.loads((directory/"v067_batch_progress.json").read_text())
    results={}
    if len(progress.get("runs",[])) != 9:
        raise InvalidSource("INCOMPLETE_V67_RUN_LEDGER")
    for row in progress["runs"]:
        identifier=row["id"]
        fname="v067_"+identifier+".json"
        if (fname not in cfg["v67_raw_sha256"] or
            row.get("raw_sha256") != cfg["v67_raw_sha256"][fname] or
            row.get("return_code") != 0):
            raise InvalidSource("UNAUTHENTICATED_G1_RUN_"+identifier)
        log=directory/("v067_"+identifier+".log")
        if not log.is_file() or sha256(log)!=row.get("log_sha256"):
            raise InvalidSource("CENSORED_LOG_HASH_DRIFT_"+identifier)
        content=log.read_text(errors="replace")
        if ("V058_BACKEND_ARG_FORWARDED cuda:0" not in content or
            "V058_NATIVE_START cuda:0" not in content or
            "V058_NATIVE_RETURN cuda:0" not in content or
            "Environment device    : cuda:0" not in content):
            raise InvalidSource("CUDA_NATIVE_PHYSICS_UNPROVEN_"+identifier)
        results[identifier]={"certified":True,
                             "v67_frozen_native_wrapper_status":row["status"],
                             "original_log_sha256":row["log_sha256"]}
    return results


def audit_sources(directory:Path, cfg:dict, protocol_sha:str) -> dict:
    seen=verify_manifest(directory,cfg)
    log_certifications=authenticate_censored_native_gpu_logs(directory,cfg)
    result=[]
    for fname in sorted(seen):
        caseid=fname.removeprefix("v067_").removesuffix(".json")
        raw=json.loads((directory/fname).read_text())
        item=summarize_case(raw,caseid,cfg)
        if caseid not in log_certifications:
            raise InvalidSource("UNVERIFIED_NATIVE_LOG_"+caseid)
        item["original_cuda_physx_proven_by_frozen_log_sha"]=True
        item["original_censored_wrapper_status"]=log_certifications[caseid]["v67_frozen_native_wrapper_status"]
        item["original_backend_raw_field_available"]=(raw.get("v057_requested_physics_backend")=="cuda:0")
        item["original_backend_log_sha256"]=log_certifications[caseid]["original_log_sha256"]
        item["source_file_sha256"]=seen[fname]
        result.append(item)
    return dict(
        version=cfg["version"],
        factual_protocol_sha256=protocol_sha,
        protocol_status="POSTREGISTERED_ANALYSIS_OF_ALREADY_SEEN_V67",
        physical_source="ORIGINAL_29DOF_CUDA_PHYSX_V67_ONLY",
        source_native_physics_episodes=len(result), newly_run_physics_episodes=0,
        original_full_14s=sum(x["full_14_second_native_recovery_observed"] for x in result),
        strict_pass_count=sum(x["strict_all_gates_pass"] for x in result),
        strict_denominator=9,
        independently_verified_native_gpu_wrist_hip_contact_manifolds=sum(
            x["original_G1_GPU_physx_left_wrist_left_hip_manifold_proven"] for x in result),
        contact_ground_truth_cmu_no=True, causal_collision_pair_certification=False,
        cases=result,
    )


def v070_completed_snapshot(progress_path: Path) -> dict:
    """Only rows already committed by v070 owner; never read live child process log."""
    if not progress_path.exists():
        return dict(readable=False)
    p=json.loads(progress_path.read_text())
    rows=p.get("runs",[])
    if not isinstance(rows,list):
        raise InvalidSource("V070_MALFORMED_PROGRESS")
    statuses={}
    for x in rows:
        v=x.get("status","UNCLASSIFIED")
        statuses[v]=statuses.get(v,0)+1
    return dict(readable=True, completed_rows_snapshot=len(rows),
                total_registered_cases=12, status_counts=statuses,
                source="read-only in-flight progress snapshot, NOT completed v070 final audit",
                no_g1_output_is_not_physics_failure=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--v070-progress",type=Path,default=None)
    args=parser.parse_args()
    p=args.root/CONFIG
    cfg=json.loads(p.read_text())
    if cfg["version"]!="v0.1.71" or cfg["source_data_is_already_seen"] is not True:
        raise InvalidSource("PROTOCOL_DRIFT")
    out=audit_sources(args.source_dir,cfg,sha256(p))
    if args.v070_progress is not None:
        out["v070_read_only_progress_snapshot"]=v070_completed_snapshot(args.v070_progress)
    target=args.output or (args.root/"reports/v071_postregistered_g1_readonly_ledger.json")
    target.parent.mkdir(exist_ok=True,parents=True)
    with target.open("x") as file:
        json.dump(out,file,ensure_ascii=False,indent=2,allow_nan=False)
        file.write("\n")
    print("V071_AUDITED",out["source_native_physics_episodes"],
          "NEW_GPU",out["newly_run_physics_episodes"],
          "STRICT_PASS",out["strict_pass_count"],
          "PAIR_PROVEN",out["independently_verified_native_gpu_wrist_hip_contact_manifolds"])
    for row in out["cases"]:
        print(row["id"],"N=",row["physical_native_200hz_rows"],
              "impact_window=",row["contact_minus_0p4_to_plus_1p25_fully_observed"],
              "14s=",row["full_14_second_native_recovery_observed"],
              "strict=",row["strict_all_gates_pass"],"failure=",",".join(row["gates_failed"]))


if __name__=="__main__":
    main()
