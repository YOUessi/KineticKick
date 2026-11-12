#!/usr/bin/env python3
"""CPU-only, write-once, SHA-certified descriptive 29DoF robot power-chain audit.

Robot torque is the existing PhysX estimated actuator torque, never human GRF.
This is deliberately NOT a trained controller, a physical rollout, a human
kinetics data set, or a causal proof of power-transfer sequence.
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

ROOT = Path(__file__).resolve().parents[1]
DT = 0.005
GROUP_PREFIXES = {
    "waist": ("waist_",),
    "right_hip": ("right_hip_",),
    "right_knee": ("right_knee_",),
    "right_ankle": ("right_ankle_",),
    "left_arm": ("left_shoulder_", "left_elbow_", "left_wrist_"),
    "left_support_leg": ("left_hip_", "left_knee_", "left_ankle_"),
    "right_arm": ("right_shoulder_", "right_elbow_", "right_wrist_"),
}
TRANSFER_ORDER = ("waist", "right_hip", "right_knee", "right_ankle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("UNTRUSTED_PHYSICAL_FILE")
    if len(expected) != 64 or sha256(path) != expected:
        raise ValueError("SOURCE_SHA256_DRIFT:" + str(path))


def group_indices(joint_names):
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("NOT_29_UNIQUE_JOINTS")
    groups = {
        group: [i for i, name in enumerate(joint_names)
                if any(name.startswith(prefix) for prefix in prefixes)]
        for group, prefixes in GROUP_PREFIXES.items()
    }
    if any(not value for value in groups.values()):
        raise ValueError("EMPTY_REQUIRED_BODY_GROUP")
    flat = [i for group in groups.values() for i in group]
    if len(flat) != len(set(flat)) or len(flat) != 29:
        raise ValueError("GROUP_OVERLAP_OR_UNASSIGNED_JOINT")
    return groups


def validated_physics(raw, expected_frames: int):
    if raw.get("root_fixed") is not False:
        raise ValueError("NOT_FREE_ROOT_ORIGINAL_G1")
    groups = group_indices(raw["joint_names"])
    traces = raw.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("INVALID_OR_MULTI_TRIAL_TRACE")
    rows = traces[0]
    if not isinstance(rows, list) or len(rows) != expected_frames:
        raise ValueError("REAL_FRAME_COUNT_MISMATCH")
    if len(rows) < 80 or len(rows) > 2800:
        raise ValueError("UNSUPPORTED_REAL_FRAME_COUNT")

    t = np.asarray([x["time_s"] for x in rows], dtype=np.float64)
    if not np.all(np.isfinite(t)) or not np.allclose(
        t, DT * (1 + np.arange(len(rows))), rtol=0, atol=2e-5
    ):
        raise ValueError("NON_200HZ_OR_PADDED_TIME")
    if any(abs(float(row["physics_dt_s"]) - DT) > 1e-8 for row in rows):
        raise ValueError("NON_200HZ_PHYSICS_DT")
    v = np.asarray([x["joint_vel"] for x in rows], dtype=np.float64)
    tau = np.asarray([x["joint_torque_estimate"] for x in rows], dtype=np.float64)
    q = np.asarray([x["joint_pos"] for x in rows], dtype=np.float64)
    support = np.asarray([
        [x["support_foot_force_N"]["left"], x["support_foot_force_N"]["right"]]
        for x in rows
    ], dtype=np.float64)
    com = np.asarray([x["whole_body_com_xyz"] for x in rows], dtype=np.float64)
    nonfoot = np.asarray([x["nonfoot_contact_N"] for x in rows], dtype=np.float64)
    limits = np.asarray(raw["joint_velocity_limits"], dtype=np.float64).reshape(-1)
    if any(x.shape != (len(rows), 29) for x in (v, tau, q)):
        raise ValueError("INVALID_FULL29_Q_QDOT_OR_TAU")
    if support.shape != (len(rows), 2) or com.shape != (len(rows), 3):
        raise ValueError("MISSING_COM_OR_BILATERAL_SUPPORT")
    if limits.shape != (29,) or np.any(limits <= 0):
        raise ValueError("INVALID_ALL29_QDOT_LIMITS")
    if not all(np.isfinite(x).all() for x in (v, tau, q, t, support, com, nonfoot, limits)):
        raise ValueError("NONFINITE_MEASURED_PHYSICS")
    if np.any(support < -1e-6) or np.any(nonfoot < -1e-6):
        raise ValueError("INVALID_CONTACT_FORCE")
    recs = raw.get("records")
    if not isinstance(recs, list) or len(recs) != 1:
        raise ValueError("NO_SINGLE_FROZEN_RECORD")
    impact = float(recs[0]["contact_time_s"])
    if not math.isfinite(impact) or impact < 0.405 or impact > float(t[-1]):
        raise ValueError("UNOBSERVED_OR_TOO_EARLY_IMPACT")
    return dict(t=t, vel=v, tau=tau, q=q, support=support, com=com,
                nonfoot=nonfoot, limits=limits, impact=impact, groups=groups)


def sustained_onset(t, power, indices, threshold_fraction=0.2,
                    floor_W=0.5, frames=3):
    """Pre-registered threshold; never infer unobserved onset."""
    if not len(indices):
        return None
    y = np.asarray(power[indices], dtype=float)
    if len(y) < frames or not np.isfinite(y).all():
        raise ValueError("CENSORED_OR_NONFINITE_ONSET")
    peak = float(np.max(y))
    if peak < floor_W:
        return None
    above = y >= max(floor_W, threshold_fraction * peak)
    for offset in range(len(above) - frames + 1):
        if bool(np.all(above[offset:offset + frames])):
            return float(t[indices[offset]])
    return None


def analyze_physical(raw, expected_frames, config):
    phys = validated_physics(raw, expected_frames)
    t, impact = phys["t"], phys["impact"]
    pre = np.flatnonzero((t >= impact - config["pre_s"] - 1e-7) &
                         (t < impact - 1e-7))
    post = np.flatnonzero((t >= impact - 1e-7) &
                          (t < impact + config["post_s"] - 1e-7))
    expected_pre = int(round(config["pre_s"] / DT))
    expected_post = int(round(config["post_s"] / DT))
    if len(pre) != expected_pre:
        raise ValueError("CENSORED_PRE_IMPACT_400MS")
    if len(post) < 1:
        raise ValueError("NO_POST_IMPACT_PHYSICS")
    v, tau = phys["vel"], phys["tau"]
    power = np.maximum(v * tau, 0)
    measures = {}
    for group, idx in phys["groups"].items():
        y = np.sum(power[:, idx], axis=1)
        amount = float(np.sum(y[pre]) * DT)
        mass = float(np.sum(y[pre]))
        onset = sustained_onset(
            t, y, pre, config["onset_peak_fraction"],
            config["onset_floor_W"], config["onset_frames"])
        centroid = float(np.sum((t[pre] - impact) * y[pre]) / mass) if mass > 1e-9 else None
        # Active at the first measured step means the onset is LEFT-CENSORED.
        onset_left_censored = onset is not None and abs(onset-float(t[pre[0]])) <= 1e-8
        measures[group] = {
            "onset_left_censored_by_pre400_boundary": bool(onset_left_censored),
            "estimated_positive_motor_work_pre400_J": round(amount, 6),
            "estimated_positive_motor_work_observed_post1250_J": round(float(np.sum(y[post]) * DT), 6),
            "peak_positive_motor_power_pre400_W": round(float(np.max(y[pre])), 6),
            "onset_s": round(onset, 6) if onset is not None else None,
            "onset_relative_impact_s": round(onset-impact, 6) if onset is not None else None,
            "positive_power_centroid_relative_impact_s": round(centroid, 6) if centroid is not None else None,
        }

    ratios = np.abs(v) / phys["limits"][None, :]
    worst = np.unravel_index(np.argmax(ratios), ratios.shape)
    violations = np.argwhere(ratios > 1.0 + 1e-6)
    first = tuple(violations[0]) if len(violations) else None
    first_time = float(t[first[0]]) if first is not None else None
    first_idx = int(first[1]) if first is not None else None
    first_phase = ("pre400" if first_time is not None and first_time < impact and first_time >= impact-config["pre_s"]-1e-7 else
                   "post1250" if first_time is not None and impact <= first_time < impact+config["post_s"] else
                   "outside_contact_window" if first_time is not None else None)
    left_support = phys["support"][:, 0]
    support_set = left_support >= config["support_threshold_N"]
    first_hip = measures["right_hip"]["onset_s"]
    support_prior = None
    if first_hip is not None and not measures["right_hip"]["onset_left_censored_by_pre400_boundary"]:
        prior = pre[t[pre] <= first_hip + 1e-8]
        support_prior = bool(any(np.all(support_set[prior[j:j+3]])
                                 for j in range(max(0, len(prior)-2))))
    no_naturalness = True
    onset_order = [measures[g]["onset_s"] for g in TRANSFER_ORDER]
    ordered_observed = (all(x is not None for x in onset_order)
                        and all(x <= y + 1e-8 for x, y in zip(onset_order, onset_order[1:])))
    phase_left_censored = any(measures[g]["onset_left_censored_by_pre400_boundary"] for g in TRANSFER_ORDER)
    # Observed starts at the boundary cannot identify a true transfer order.
    identifiable_order = None if phase_left_censored else bool(ordered_observed)
    first_nonfoot_ix = np.flatnonzero(phys["nonfoot"] > 20.0)
    nonfoot_time = float(t[first_nonfoot_ix[0]]) if len(first_nonfoot_ix) else None
    com_displacement_pre = float(np.linalg.norm(phys["com"][pre[-1]] - phys["com"][pre[0]]))
    return {
        "observed_original_G1_200Hz_rows": len(t),
        "observed_full_14s": bool(len(t) == 2800 and abs(float(t[-1]) - 14) < 1e-5),
        "contact_time_s": impact,
        "pre400_observed_rows": len(pre),
        "post1250_observed_rows": len(post),
        "post1250_censored": bool(len(post) != expected_post),
        "full_29dof_qdot": {
            "maximum_ratio": round(float(ratios[worst]), 6),
            "joint_at_maximum": raw["joint_names"][worst[1]],
            "time_at_maximum_s": round(float(t[worst[0]]), 6),
            "first_hard_limit_violation_time_s": first_time,
            "first_hard_limit_violation_joint": raw["joint_names"][first_idx] if first_idx is not None else None,
            "first_hard_limit_violation_phase": first_phase,
        },
        "left_support": {
            "force_proxy_pre400_integral_Ns": round(float(np.sum(left_support[pre])*DT), 6),
            "fraction_of_pre400_at_least_50N": round(float(np.mean(support_set[pre])), 6),
            "three_consecutive_frames_before_hip_onset": support_prior,
        },
        "right_support_force_proxy_pre400_integral_Ns":
            round(float(np.sum(phys["support"][pre, 1])*DT), 6),
        "COM_pre400_path_chord_m": round(com_displacement_pre, 6),
        "first_nonfoot_force_exceeding_20N_time_s": nonfoot_time,
        "peak_nonfoot_net_force_N": round(float(np.max(phys["nonfoot"])), 6),
        "naive_window_waist_hip_knee_ankle_order_NOT_CERTIFIED": bool(ordered_observed),
        "identifiable_waist_hip_knee_ankle_order": identifiable_order,
        "window_boundary_prevents_transfer_order_identification": bool(phase_left_censored),
        "group_estimated_motor_positive_power": measures,
        "not_human_muscle_power_not_athlete_GRF_not_causal_transfer": no_naturalness,
        "no_qualified_wrist_hip_pair_or_link_distance": True,
    }


def matched_pairs(trials):
    grouped = defaultdict(dict)
    for item in trials:
        if item["observer"]:
            continue
        key = (item["source_version"], item["scene"])
        if item["control"] in grouped[key]:
            raise ValueError("DUPLICATE_INDEPENDENT_CONDITION")
        grouped[key][item["control"]] = item
    pairs = []
    for (version, scene), by_control in sorted(grouped.items()):
        base = by_control.get("v32")
        if base is None:
            continue
        for ctl in ("cmu02", "cmu05"):
            candidate = by_control.get(ctl)
            if candidate is None:
                continue
            if base["seed"] != candidate["seed"] or base["ball_point"] != candidate["ball_point"]:
                raise ValueError("UNMATCHED_FROZEN_PAIR")
            bm = base["observed"]["group_estimated_motor_positive_power"]
            cm = candidate["observed"]["group_estimated_motor_positive_power"]
            pairs.append({
                "version": version, "scene": scene, "baseline": base["name"], "candidate": candidate["name"],
                "baseline_pass": base["hard_gate_pass"], "candidate_pass": candidate["hard_gate_pass"],
                "pre400_estimated_motor_work_delta_J": {
                    g: round(cm[g]["estimated_positive_motor_work_pre400_J"] -
                             bm[g]["estimated_positive_motor_work_pre400_J"], 6)
                    for g in GROUP_PREFIXES
                },
                "group_onset_delta_s": {
                    g: round(cm[g]["onset_s"] - bm[g]["onset_s"], 6)
                    if (cm[g]["onset_s"] is not None and bm[g]["onset_s"] is not None
                        and not cm[g]["onset_left_censored_by_pre400_boundary"]
                        and not bm[g]["onset_left_censored_by_pre400_boundary"]) else None
                    for g in GROUP_PREFIXES
                },
                "qdot_max_ratio_delta": round(
                    candidate["observed"]["full_29dof_qdot"]["maximum_ratio"] -
                    base["observed"]["full_29dof_qdot"]["maximum_ratio"], 6),
                "no_human_power_transfer_or_policy_advantage_claim": True,
            })
    return pairs


def audit(config, source_template, output):
    rows = []
    master_digests = {}
    for c in config["cohort"]:
        version = c["version"]
        source_root = Path(source_template.format(version=version))
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError("SOURCE_WORKTREE_ABSENT")
        head = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if head != c["commit"]:
            raise ValueError("SOURCE_VERSION_HEAD_DRIFT")
        report_dir = source_root / "reports"
        master_path = report_dir / f"v{version}_original_g1_batch.audit.json"
        if not master_path.is_file() or master_path.is_symlink():
            raise ValueError("SOURCE_MASTER_ABSENT")
        master_digests[f"v{version}"] = sha256(master_path)
        master = json.loads(master_path.read_text())
        if master["version"] != f"v0.1.{version}":
            raise ValueError("MASTER_VERSION_MISMATCH")
        for r in master["runs"]:
            if not r.get("certified_original_G1_cuda_200Hz"):
                raise ValueError("UNCERTIFIED_RUN_IN_FROZEN_COHORT")
            name = r["name"]
            raw_path = report_dir / f"v{version}_{name}.json"
            verify_file(raw_path, r["raw_sha256"])
            raw = json.loads(raw_path.read_text())
            analysis = analyze_physical(raw, int(r["actual_traced_200hz_rows"]), config)
            observed_seed = int(raw["seed"])
            if observed_seed != int(r["seed"]):
                raise ValueError("SEED_MISMATCH_MASTER_VS_RAW")
            rows.append({
                "source_version": version, "name": name, "scene": r["scene"],
                "control": r["control"], "observer": bool(r["observer"]),
                "raw_sha256": r["raw_sha256"],
                "seed": observed_seed,
                "ball_point": raw["points"][0],
                "hard_gate_pass": bool(r["hard_gates"]["hard_gate_pass"]),
                "observed": analysis,
            })
    pairs = matched_pairs(rows)
    independent = [r for r in rows if not r["observer"]]
    if len({(r["source_version"],r["scene"],r["control"]) for r in independent}) != len(independent):
        raise ValueError("DUPLICATE_INDEPENDENT_CONDITION")
    out = {
        "version": "v0.1.121",
        "protocol_sha256": sha256(ROOT / "configs/v121_frozen_cpu_cohort.json"),
        "source_master_sha256": master_digests,
        "original_PhysX_rollouts_REUSED_not_NEW": len(rows),
        "reused_200hz_steps": sum(r["observed"]["observed_original_G1_200Hz_rows"] for r in rows),
        "independent_controls_x_scenes": len(independent),
        "independent_strict_pass": sum(r["hard_gate_pass"] for r in independent),
        "observer_parity_repetitions_excluded_from_denominator": len(rows)-len(independent),
        "matched_controller_pairs": len(pairs),
        "independent_unidentifiable_phase_chain_due_to_left_boundary": sum(
            r["observed"]["window_boundary_prevents_transfer_order_identification"]
            for r in independent),
        "first_qdot_hard_violation_counts_by_phase": {
            phase: sum(r["observed"]["full_29dof_qdot"]["first_hard_limit_violation_phase"] == phase
                       for r in independent)
            for phase in ("pre400", "post1250", "outside_contact_window")
        },
        "naturalness_or_biological_power_claim_certified": False,
        "wrist_hip_native_contact_pair_certified": False,
        "human_elite_instep_contact_GRF_CoP_qualified": False,
        "new_learned_policy_actuation": False,
        "new_GPU_PhysX_rollouts": 0,
        "independent_trials": independent,
        "observer_repeat_trials": [r for r in rows if r["observer"]],
        "paired_descriptive_results": pairs,
    }
    with output.open("x") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-template",
                   default="/home/you/projects/kinetickick-wt-v{version}")
    p.add_argument("--output", type=Path,
                   default=ROOT / "reports/v121_leftcensor_corrected_29dof.audit.json")
    a = p.parse_args()
    config = json.loads((ROOT / "configs/v121_frozen_cpu_cohort.json").read_text())
    if config["version"] != "v0.1.121" or config["dt_s"] != DT:
        raise ValueError("UNFROZEN_OR_INVALID_CONFIG")
    result = audit(config, a.source_template, a.output)
    print("V121_REAL_HISTORICAL_PHYSX_REAUDIT", json.dumps({
        k: v for k, v in result.items()
        if k not in ("independent_trials", "observer_repeat_trials", "paired_descriptive_results")
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
