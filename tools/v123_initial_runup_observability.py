#!/usr/bin/env python3
"""v123: SHA-locked, CPU-only first-frame/run-up *observability* audit of original G1 PhysX.
This script is NOT a new simulation, learned policy, or athlete biomechanics proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import v121_29dof_power_chain as prior

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v123_frozen_initial_runup.json"
# The master hashes below were reported in the pre-existing, already committed v122 report.
FROZEN_MASTER_SHA = {
    116: "9080ae466ece92603c24b85e00f1424f211693580d2b0ec8fe9a6833ad3a88bc",
    117: "76e5782e62df6cd47c571c100b720a56484024fa02398d6946548bc1838081da",
    118: "800fb091cdf44b61be8bc1674e2930d4138c03be7d1118e2392c7cd7368716d5",
}
DT = 0.005


def first_sustained(t, values, level, frames=3):
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(t) != len(values) or not np.isfinite(values).all():
        raise ValueError("BAD_SUPPORT_TRACE")
    mask = values >= level
    for i in range(len(mask) - frames + 1):
        if bool(np.all(mask[i:i + frames])):
            return {"onset_s": float(t[i]), "left_censored": bool(i == 0)}
    return {"onset_s": None, "left_censored": False}


def rest_like(root_speed, qdot, cfg):
    if not math.isfinite(root_speed) or root_speed < 0:
        raise ValueError("BAD_ROOT_SPEED")
    v = np.asarray(qdot, dtype=float)
    if v.shape != (29,) or not np.isfinite(v).all():
        raise ValueError("BAD_29DOF_VELOCITY")
    return bool(root_speed <= cfg["rest_speed_mps"] and
                np.max(np.abs(v)) <= cfg["rest_joint_speed_rads"])


def first_hard_limit(t, vel, limits, names):
    ratios = np.abs(vel) / limits[None, :]
    where = np.argwhere(ratios > 1.0 + 1e-6)
    first = (int(where[0, 0]), int(where[0, 1])) if len(where) else None
    return {
        "max_ratio": round(float(np.max(ratios)), 6),
        "first_time_s": round(float(t[first[0]]), 6) if first else None,
        "first_joint": names[first[1]] if first else None,
        "first_ratio": round(float(ratios[first]), 6) if first else None,
    }


def first_effect(a, b, threshold):
    if a["names"] != b["names"]:
        raise ValueError("MISMATCHED_29DOF_ORDER")
    if a["q"].shape != b["q"].shape or a["v"].shape != b["v"].shape:
        raise ValueError("CENSORED_MISMATCHED_PAIR")
    if not np.array_equal(a["t"], b["t"]):
        raise ValueError("MISMATCHED_TIME_GRID")
    maxima = np.maximum(np.max(np.abs(a["q"] - b["q"]), axis=1),
                        np.max(np.abs(a["v"] - b["v"]), axis=1))
    ix = np.flatnonzero(maxima > threshold)
    if not len(ix):
        return {"first_s": None, "common_prefix_200hz_frames": len(maxima),
                "max_abs_q_or_qdot_difference": round(float(np.max(maxima)), 9)}
    i = int(ix[0])
    return {"first_s": round(float(a["t"][i]), 6),
            "common_prefix_200hz_frames": i,
            "max_abs_q_or_qdot_difference": round(float(np.max(maxima)), 9)}


def audit_trace(raw, expected_rows, cfg):
    p = prior.validated_physics(raw, expected_rows)
    t, impact = p["t"], p["impact"]
    rows = raw["traces"][0]
    pre = np.flatnonzero(t < impact - 1e-7)
    post = np.flatnonzero((t >= impact - 1e-7) & (t < impact + 1.25 - 1e-7))
    if len(pre) < round(cfg["preimpact_window_s"] / DT):
        raise ValueError("PRECONTACT_CENSORED")
    root = np.asarray([r["root_xyz"] for r in rows], dtype=float)
    root_speed = np.asarray([r["root_speed_xy"] for r in rows], dtype=float)
    tilt = np.asarray([r["root_tilt_deg"] for r in rows], dtype=float)
    if root.shape != (len(t), 3) or np.any(~np.isfinite(root)) or np.any(~np.isfinite(root_speed)) or np.any(~np.isfinite(tilt)):
        raise ValueError("BAD_ROOT_KINEMATICS")
    if not np.all(root_speed >= 0) or np.any((tilt < 0) | (tilt > 180)):
        raise ValueError("BAD_ROOT_SPEED_OR_TILT")
    i0 = pre[0]
    before = root[pre, :2]
    chord = float(np.linalg.norm(before[-1] - before[0]))
    path = float(np.sum(np.linalg.norm(np.diff(before, axis=0), axis=1)))
    sup = p["support"]
    init_len = round(cfg["initial_window_s"] / DT)
    early = np.flatnonzero(t <= cfg["initial_window_s"] + 1e-8)
    if len(early) != init_len or early[0] != 0:
        raise ValueError("BAD_EARLY_200HZ_GRID")
    level = cfg["support_force_active_N"]
    left = first_sustained(t[pre], sup[pre, 0], level, cfg["support_consecutive_frames"])
    right = first_sustained(t[pre], sup[pre, 1], level, cfg["support_consecutive_frames"])
    qdot = p["vel"]
    actual_limits = p["limits"]
    hard = first_hard_limit(t, qdot, actual_limits, raw["joint_names"])
    hard["first_violation_is_preimpact"] = (hard["first_time_s"] is not None and
                                           hard["first_time_s"] < impact - 1e-7)
    hard["first_violation_is_postimpact"] = (hard["first_time_s"] is not None and
                                            hard["first_time_s"] >= impact - 1e-7)
    early_norm = np.max(np.abs(qdot[early]) / actual_limits[None, :], axis=1)
    result = {
        "rows_measured_200Hz": len(t),
        "full_14s_measured": bool(len(t) == 2800 and abs(float(t[-1]) - 14.0) < 1e-5),
        "contact_s": round(float(impact), 6),
        "true_first_frame_s": round(float(t[0]), 6),
        "first_frame_root_xy_speed_mps": round(float(root_speed[0]), 6),
        "first_frame_root_tilt_deg": round(float(tilt[0]), 6),
        "first_frame_max_abs_29qdot_rads": round(float(np.max(np.abs(qdot[0]))), 6),
        "first_frame_max_abs_qdot_over_limit": round(float(np.max(np.abs(qdot[0]) / actual_limits)), 6),
        "first_frame_all29_and_root_rest_like": rest_like(float(root_speed[0]), qdot[0], cfg),
        "first_frame_left_support_force_proxy_N": round(float(sup[0, 0]), 6),
        "first_frame_right_support_force_proxy_N": round(float(sup[0, 1]), 6),
        "initial200_both_support_proxy_below_50N_fraction": round(float(np.mean(np.all(sup[early] < level, axis=1))), 6),
        "initial200_both_support_proxy_at_least_50N_fraction": round(float(np.mean(np.all(sup[early] >= level, axis=1))), 6),
        "initial200_max_original29_qdot_limit_ratio": round(float(np.max(early_norm)), 6),
        "first_measured_sustained_support_left": left,
        "first_measured_sustained_support_right": right,
        "root_xy_precontact_chord_m": round(chord, 6),
        "root_xy_precontact_path_m": round(path, 6),
        "COM_precontact_chord_m": round(float(np.linalg.norm(p["com"][pre[-1]] - p["com"][pre[0]])), 6),
        "postimpact_1250ms_measured_frames": len(post),
        "postimpact_1250ms_censored": bool(len(post) < 250),
        "first_original_29_qdot_limit": hard,
        "coordination_phase_unique_observed_labels": sorted(set(str(r["coordination_phase"]) for r in rows)),
        "initial_start_time_left_censored_if_already_moving": not rest_like(float(root_speed[0]), qdot[0], cfg),
        "foot_force_proxy_zero_does_not_certify_no_contact": True,
        "not_human_athlete_motion_or_natural_runup_certification": True,
    }
    ref = {"names": list(raw["joint_names"]), "t": t, "q": p["q"], "v": qdot}
    return result, ref


def load_and_audit(cfg, source_pattern):
    if cfg["version"] != "v0.1.123" or cfg["dt_s"] != DT or not cfg["no_gpu"]:
        raise ValueError("UNFROZEN_V123_CONFIG")
    if cfg["parent_sha"] != "5b1cddf6d090b52aff4dc0e21f63f7a3341a29f7":
        raise ValueError("PARENT_SHA_DRIFT")
    reference = json.loads((ROOT / cfg["cohort_manifest"]).read_text())
    if reference["cohort"] != cfg["cohort"]:
        raise ValueError("SOURCE_COHORT_DRIFT")
    entries, arrays, hashes = [], {}, {}
    for source in cfg["cohort"]:
        v = source["version"]
        folder = Path(source_pattern.format(version=v))
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("BAD_SOURCE_FOLDER")
        actual_head = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if actual_head != source["commit"]:
            raise ValueError("SOURCE_HEAD_DRIFT")
        master = folder / "reports" / f"v{v}_original_g1_batch.audit.json"
        prior.verify_file(master, FROZEN_MASTER_SHA[v])
        hashes[str(v)] = FROZEN_MASTER_SHA[v]
        manifest = json.loads(master.read_text())
        if manifest["version"] != f"v0.1.{v}":
            raise ValueError("SOURCE_MASTER_VERSION_DRIFT")
        for run in manifest["runs"]:
            if not run.get("certified_original_G1_cuda_200Hz"):
                raise ValueError("NONCERTIFIED_PHYSICAL_RUN")
            path = folder / "reports" / f"v{v}_{run['name']}.json"
            prior.verify_file(path, run["raw_sha256"])
            raw = json.loads(path.read_text())
            if int(raw["seed"]) != int(run["seed"]) or raw["root_fixed"] is not False:
                raise ValueError("RAW_SEED_OR_ROBOT_DRIFT")
            summary, ref = audit_trace(raw, run["actual_traced_200hz_rows"], cfg)
            key = (v, run["scene"], run["control"], bool(run["observer"]))
            if key in arrays:
                raise ValueError("DUPLICATE_SOURCE_TRIAL")
            arrays[key] = ref
            entries.append(dict(version=v, scene=run["scene"], control=run["control"],
                                seed=int(run["seed"]), observer=bool(run["observer"]),
                                name=run["name"], strict_pass=bool(run["hard_gates"]["hard_gate_pass"]),
                                ball_offset=run["ball_offset_xy"], raw_sha256=run["raw_sha256"],
                                measures=summary))
    pairs, observer_pairs = [], []
    by_scene = defaultdict(dict)
    for item in entries:
        if item["observer"]:
            continue
        key = (item["version"], item["scene"])
        if item["control"] in by_scene[key]:
            raise ValueError("DUPLICATE_INDEPENDENT_RUN")
        by_scene[key][item["control"]] = item
    for key, controls in sorted(by_scene.items()):
        if set(controls) != {"v32", "cmu02", "cmu05"}:
            raise ValueError("INCOMPLETE_THREE_CONTROLLER_CELL")
        a = controls["v32"]
        for variant in ("cmu02", "cmu05"):
            b = controls[variant]
            if a["seed"] != b["seed"] or a["ball_offset"] != b["ball_offset"]:
                raise ValueError("UNMATCHED_SEED_OR_SCENE")
            dif = first_effect(arrays[(key[0], key[1], "v32", False)],
                               arrays[(key[0], key[1], variant, False)],
                               cfg["first_control_effect_abs_q_rad"])
            dif["relative_to_baseline_contact_s"] = (round(dif["first_s"]-a["measures"]["contact_s"], 6)
                                                      if dif["first_s"] is not None else None)
            pairs.append({"version": key[0], "scene": key[1], "candidate": variant,
                          "baseline_pass": a["strict_pass"], "candidate_pass": b["strict_pass"],
                          "measured_difference": dif})
    for item in entries:
        if not item["observer"]:
            continue
        akey = (item["version"], item["scene"], item["control"], False)
        bkey = (item["version"], item["scene"], item["control"], True)
        if akey not in arrays or bkey not in arrays:
            raise ValueError("OBSERVER_NO_UNMODIFIED_CONTROL")
        baseline = next(x for x in entries if (x["version"], x["scene"], x["control"], x["observer"]) == akey)
        if baseline["seed"] != item["seed"] or baseline["ball_offset"] != item["ball_offset"]:
            raise ValueError("OBSERVER_UNMATCHED_SCENE")
        observer_pairs.append({"version": item["version"], "scene": item["scene"], "control": item["control"],
                               "measured_joint_delta": first_effect(arrays[akey], arrays[bkey],
                                                                     cfg["first_control_effect_abs_q_rad"])})
    independent = [e for e in entries if not e["observer"]]
    if len(entries) != 29 or len(independent) != 24 or len(pairs) != 16 or len(observer_pairs) != 5:
        raise ValueError("WRONG_FROZEN_COHORT_DENOMINATORS")
    return {
        "version": cfg["version"], "protocol_sha256": prior.sha256(CONFIG),
        "source_master_sha256": hashes, "new_gpu_physx_runs": 0,
        "historical_certified_physx_runs": len(entries),
        "independent_scene_controller_cells": len(independent),
        "independent_strict_pass": sum(bool(x["strict_pass"]) for x in independent),
        "observer_repeats_not_independent": len(observer_pairs),
        "initial_rest_like_independent_cells": sum(e["measures"]["first_frame_all29_and_root_rest_like"] for e in independent),
        "independent_initial_motion_left_censored_count": sum(e["measures"]["initial_start_time_left_censored_if_already_moving"] for e in independent),
        "independent_first_frame_root_speed_mps_range": [min(e["measures"]["first_frame_root_xy_speed_mps"] for e in independent),
                                                           max(e["measures"]["first_frame_root_xy_speed_mps"] for e in independent)],
        "independent_first_frame_tilt_deg_range": [min(e["measures"]["first_frame_root_tilt_deg"] for e in independent),
                                                    max(e["measures"]["first_frame_root_tilt_deg"] for e in independent)],
        "pair_tests": pairs, "observer_pairs": observer_pairs, "runs": entries,
        "no_human_mocap_GRF_CoP_native_contact_pair_or_learned_policy_certified": True,
    }


def write_once_json(path, obj):
    # Complete JSON serialization before opening exclusive destination.
    body = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pattern", default="/home/you/projects/kinetickick-wt-v{version}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = load_and_audit(json.loads(CONFIG.read_text()), args.source_pattern)
    write_once_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "pair_tests", "observer_pairs")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
