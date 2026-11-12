#!/usr/bin/env python3
"""v122 CPU-only, write-once re-audit of *historical* real G1 PhysX precontact.

This does not generate contact ground truth, human GRF, or new G1 rollouts.
Validate raw source SHA and source HEAD before extracting a single statistic.
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
CFG_PATH = ROOT / "configs/v122_frozen_full_precontact.json"
DT = 0.005
SEQUENCE = ("waist", "right_hip", "right_knee", "right_ankle")


def sha256(path):
    return prior.sha256(Path(path))


def threshold_onset(t, y, indices, fraction, floor, n=3):
    """Return (observed onset, left censored), without extrapolating before t0."""
    idx = np.asarray(indices, dtype=int)
    if len(idx) < n:
        return None, False
    seg = np.asarray(y, dtype=float)[idx]
    if not np.isfinite(seg).all():
        raise ValueError("NONFINITE_POWER")
    peak = float(np.max(seg))
    if peak < floor:
        return None, False
    hits = seg >= max(floor, fraction * peak)
    for k in range(len(idx) - n + 1):
        if bool(np.all(hits[k:k + n])):
            return float(t[idx[k]]), (k == 0)
    return None, False


def sequence_status(measures, fraction_key):
    values = [measures[name][fraction_key] for name in SEQUENCE]
    if any(x["onset_s"] is None or x["left_censored"] for x in values):
        return None
    return all(a["onset_s"] <= b["onset_s"] + 1e-9
               for a, b in zip(values, values[1:]))


def analyze_physics(raw, expected_rows, cfg):
    p = prior.validated_physics(raw, int(expected_rows))
    t = p["t"]
    impact = p["impact"]
    full = np.flatnonzero(t < impact - 1e-7)
    late = np.flatnonzero((t >= impact - cfg["pre400_s"] - 1e-7)
                         & (t < impact - 1e-7))
    if (not len(full) or full[0] != 0
            or not np.allclose(np.diff(t[full]), DT, rtol=0, atol=1e-6)
            or len(late) != int(round(cfg["pre400_s"] / DT))
            or len(full) < len(late)):
        raise ValueError("INVALID_REAL_PRECONTACT_COVERAGE")
    power29 = np.maximum(p["vel"] * p["tau"], 0.0)
    groups = {}
    for name, group_ix in p["groups"].items():
        power = np.sum(power29[:, group_ix], axis=1)
        segment = {}
        for window, ix in (("full", full), ("last400", late)):
            work = float(np.sum(power[ix]) * DT)
            mass = float(np.sum(power[ix]))
            centroid = (float(np.sum((t[ix] - impact) * power[ix]) / mass)
                        if mass > 1e-10 else None)
            for label, fraction in (("primary20", cfg["activation_fraction_primary"]),
                                    ("sensitivity40", cfg["activation_fraction_sensitivity"])):
                onset, left = threshold_onset(
                    t, power, ix, fraction, cfg["activation_floor_W"],
                    cfg["sustained_steps"])
                segment[f"{window}_{label}"] = {
                    "onset_s": round(onset, 6) if onset is not None else None,
                    "relative_to_impact_s": round(onset - impact, 6) if onset is not None else None,
                    "left_censored": bool(left),
                }
            segment[f"{window}_work_J"] = round(work, 6)
            segment[f"{window}_centroid_relative_impact_s"] = (
                round(centroid, 6) if centroid is not None else None)
        segment["early_pre400_work_J"] = round(
            segment["full_work_J"] - segment["last400_work_J"], 6)
        groups[name] = segment

    support = p["support"][:, 0]
    contact_ok = support >= cfg["support_threshold_N"]
    # Support onset uses its own absolute physical threshold (not percentage peak).
    support_ix = None
    if len(full) >= cfg["sustained_steps"]:
        for k in range(len(full) - cfg["sustained_steps"] + 1):
            if bool(np.all(contact_ok[full[k:k + cfg["sustained_steps"]]])):
                support_ix = int(full[k])
                break
    support_at = float(t[support_ix]) if support_ix is not None else None
    hip = groups["right_hip"]["full_primary20"]
    support_prior = None
    if hip["onset_s"] is not None and not hip["left_censored"]:
        support_prior = (support_at is not None and
                         support_at + (cfg["sustained_steps"] - 1) * DT <=
                         hip["onset_s"] + 1e-8)
    # All results are descriptive: neither an observed start nor an observed
    # centroid can identify biomechanical cause or professional-athlete skill.
    full_sequence = sequence_status(groups, "full_primary20")
    full_40 = sequence_status(groups, "full_sensitivity40")
    late_sequence = sequence_status(groups, "last400_primary20")
    max_ratio = float(np.max(np.abs(p["vel"]) / p["limits"][None, :]))
    return {
        "measured_rows": len(t),
        "full14s_measured": len(t) == 2800 and abs(t[-1] - 14) < 1e-5,
        "impact_s": round(impact, 6),
        "full_precontact_rows": len(full),
        "pre400_rows": len(late),
        "full_precontact_first_real_frame_s": round(float(t[full[0]]), 6),
        "qdot_original29_max_ratio_recomputed": round(max_ratio, 6),
        "estimated_robot_motor_power_groups": groups,
        "left_support_full_precontact": {
            "force_proxy_integral_Ns": round(float(np.sum(support[full]) * DT), 6),
            "sustained_50N_onset_s": round(support_at, 6) if support_at is not None else None,
            "onset_left_censored_by_first_sim_frame": support_ix == 0 if support_ix is not None else False,
            "already_established_before_non_censored_hip_onset": support_prior,
        },
        "full_primary20_sequence_order_observable": full_sequence,
        "full_sensitivity40_sequence_order_observable": full_40,
        "last400_primary20_sequence_order_observable": late_sequence,
        "sequence_threshold_robust_if_both_identifiable": (
            full_sequence == full_40 if full_sequence is not None
            and full_40 is not None else None),
        "source_initial_state_may_precede_first_sample": True,
        "no_biological_power_transfer_causality_certified": True,
    }


def matched(trials):
    by = defaultdict(dict)
    for row in trials:
        if row["observer"]:
            continue
        key = (row["version"], row["scene"])
        if row["control"] in by[key]:
            raise ValueError("DUPLICATE_INDEPENDENT_SCENE_CONTROL")
        by[key][row["control"]] = row
    pairs = []
    for (version, scene), ctl in sorted(by.items()):
        if set(ctl) != {"v32", "cmu02", "cmu05"}:
            raise ValueError("INCOMPLETE_FROZEN_SCENE_TRIPLE")
        baseline = ctl["v32"]
        for variant in ("cmu02", "cmu05"):
            cand = ctl[variant]
            if baseline["seed"] != cand["seed"] or baseline["ball_point"] != cand["ball_point"]:
                raise ValueError("UNMATCHED_SEED_OR_BALL_POSITION")
            a = baseline["analysis"]["estimated_robot_motor_power_groups"]
            b = cand["analysis"]["estimated_robot_motor_power_groups"]
            pairs.append({
                "source": version, "scene": scene,
                "candidate": variant,
                "baseline_strict_pass": baseline["strict_pass"],
                "candidate_strict_pass": cand["strict_pass"],
                "full_precontact_motor_work_delta_J": {
                    k: round(b[k]["full_work_J"] - a[k]["full_work_J"], 6)
                    for k in prior.GROUP_PREFIXES},
                "primary20_waist_onset_delta_s": (
                    round(b["waist"]["full_primary20"]["onset_s"]
                          - a["waist"]["full_primary20"]["onset_s"], 6)
                    if (a["waist"]["full_primary20"]["onset_s"] is not None
                        and b["waist"]["full_primary20"]["onset_s"] is not None
                        and not a["waist"]["full_primary20"]["left_censored"]
                        and not b["waist"]["full_primary20"]["left_censored"])
                    else None),
                "no_randomized_or_learned_strategy_claim": True,
            })
    return pairs


def write_once_json(output, result):
    """Serialize before creating the new evidence file: no partial output on type errors."""
    def numpy_scalar(value):
        if isinstance(value, (np.bool_, np.integer, np.floating)):
            return value.item()
        raise TypeError(f"UNSERIALIZABLE_AUDIT_RESULT:{type(value)}")
    contents = json.dumps(result, ensure_ascii=False, indent=2, default=numpy_scalar) + "\n"
    with Path(output).open("x") as h:
        h.write(contents)


def audit(cfg, template, output):
    if cfg["version"] != "v0.1.122" or cfg["dt_s"] != DT or not cfg["no_gpu"]:
        raise ValueError("UNFROZEN_PROTOCOL")
    if sha256(ROOT / "configs/v121_frozen_cpu_cohort.json") == "":
        raise ValueError("MISSING_PRIOR_PROTOCOL")
    records, master_hashes = [], {}
    for source in cfg["cohort"]:
        version = source["version"]
        folder = Path(template.format(version=version))
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("INVALID_SOURCE_WORKTREE")
        source_head = subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if source_head != source["commit"]:
            raise ValueError("SOURCE_HEAD_DRIFT")
        master_path = folder / "reports" / f"v{version}_original_g1_batch.audit.json"
        if not master_path.is_file() or master_path.is_symlink():
            raise ValueError("SOURCE_MASTER_MISSING")
        master_hashes[f"v{version}"] = sha256(master_path)
        master = json.loads(master_path.read_text())
        if master["version"] != f"v0.1.{version}":
            raise ValueError("SOURCE_MASTER_VERSION_MISMATCH")
        for run in master["runs"]:
            if not run.get("certified_original_G1_cuda_200Hz"):
                raise ValueError("UNCERTIFIED_ORIGINAL_G1_ROLLOUT")
            file = folder / "reports" / f'v{version}_{run["name"]}.json'
            prior.verify_file(file, run["raw_sha256"])
            original = json.loads(file.read_text())
            if int(original["seed"]) != int(run["seed"]):
                raise ValueError("SEED_DRIFT")
            analysis = analyze_physics(original, run["actual_traced_200hz_rows"], cfg)
            records.append({
                "version": version, "name": run["name"], "scene": run["scene"],
                "control": run["control"], "observer": bool(run["observer"]),
                "seed": int(run["seed"]), "ball_point": original["points"][0],
                "source_sha256": run["raw_sha256"],
                "strict_pass": bool(run["hard_gates"]["hard_gate_pass"]),
                "analysis": analysis,
            })
    independent = [r for r in records if not r["observer"]]
    pairs = matched(independent)
    if len(records) != 29 or len(independent) != 24 or len(pairs) != 16:
        raise ValueError("UNEXPECTED_FROZEN_COHORT_OR_PAIR_COUNT")
    groups = SEQUENCE
    def count_boundary(window):
        return sum(any(
            r["analysis"]["estimated_robot_motor_power_groups"][g]
            [f"{window}_primary20"]["left_censored"] for g in groups)
            for r in independent)
    summary = {
        "version": cfg["version"],
        "frozen_protocol_sha256": sha256(CFG_PATH),
        "base_parent_SHA": cfg["parent_sha"],
        "source_master_SHA256": master_hashes,
        "historical_real_G1_PhysX_rollouts_reaudited_NOT_NEW": len(records),
        "historical_real_200Hz_rows_reaudited_NOT_NEW": sum(
            r["analysis"]["measured_rows"] for r in records),
        "independent_conditions": len(independent),
        "independent_strict_pass_original_gate": sum(r["strict_pass"] for r in independent),
        "observer_repetitions_not_independent": len(records) - len(independent),
        "matched_v32_vs_CMU_pairs": len(pairs),
        "independent_any_four_group_onset_leftcensored_last400": count_boundary("last400"),
        "independent_any_four_group_onset_leftcensored_full_precontact": count_boundary("full"),
        "independent_full_primary_sequence_identifiable_count": sum(
            r["analysis"]["full_primary20_sequence_order_observable"] is not None for r in independent),
        "independent_full_primary_sequence_order_true_count": sum(
            r["analysis"]["full_primary20_sequence_order_observable"] is True for r in independent),
        "independent_full_both_thresholds_identifiable": sum(
            r["analysis"]["sequence_threshold_robust_if_both_identifiable"] is not None for r in independent),
        "independent_with_non_censored_hip_and_prior_support_identifiable": sum(
            r["analysis"]["left_support_full_precontact"]["already_established_before_non_censored_hip_onset"]
            is not None for r in independent),
        "new_PhysX_runs": 0, "new_29DoF_strategy_actuations": 0,
        "native_wrist_hip_collider_pairs_certified": False,
        "elite_human_instep_GRF_CoP_certified": False,
        "natural_runup_and_explosive_transfer_certified": False,
        "independent_trials": independent,
        "observer_only_repeat_trials": [r for r in records if r["observer"]],
        "frozen_scene_matched_pairs": pairs,
    }
    write_once_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-template",
                        default="/home/you/projects/kinetickick-wt-v{version}")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/v122_complete_precontact_verified.audit.json")
    a = parser.parse_args()
    cfg = json.loads(CFG_PATH.read_text())
    result = audit(cfg, a.source_template, a.output)
    for key, value in result.items():
        if key not in ("independent_trials", "observer_only_repeat_trials", "frozen_scene_matched_pairs"):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
