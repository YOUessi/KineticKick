#!/usr/bin/env python3
"""Independent SHA-pinned 29DoF temporal hazard and coordination falsification for v117.

Only robot-physics observations. Support-force proxies, torque estimates and
cross-correlation are NOT human GRF/CoP, biological power or causal proof.
Never alters the original G1 controller or manufactures recovery after censoring.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v113_phase_biomechanics import load_physical_arrays, phase_metrics, paired_controller_effect
from v116_qdot_causality import physical_arrays, violations, paired_intervention

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_joint_hazard(a: dict) -> dict:
    """Measured original29 violations with event duration, never synthetic future rows."""
    t, v, lim = a["t"], a["vel"], a["lim"]
    if v.shape != (len(t), 29) or lim.shape != (29,):
        raise ValueError("JOINT_LIMIT_MISMATCH")
    ratio = np.abs(v) / lim[None, :]
    if not np.isfinite(ratio).all():
        raise ValueError("NONFINITE_QDOT_RATIO")
    bad = ratio > 1.0 + 1e-7
    joint_events = []
    for j in np.flatnonzero(bad.any(axis=0)):
        first = int(np.flatnonzero(bad[:, j])[0])
        peak = int(np.argmax(ratio[:, j]))
        joint_events.append(dict(
            joint=a["names"][j], first_exceed_s=float(t[first]),
            peak_time_s=float(t[peak]),
            peak_ratio=float(ratio[peak, j]),
            actual_time_above_limit_s=float(np.count_nonzero(bad[:, j]) * 0.005),
            excess_ratio_time_integral_s=float(np.sum(np.maximum(ratio[:, j] - 1.0, 0.0)) * 0.005),
        ))
    joint_events.sort(key=lambda row: row["peak_ratio"], reverse=True)
    impact = float(a["impact"])
    phases = {
        "approach": t < impact - 0.4,
        "pre400": (t >= impact - 0.4) & (t < impact),
        "post1250_observed": (t >= impact) & (t <= impact + 1.25),
        "recovery_observed": t > impact + 1.25,
    }
    stages = {}
    for name, mask in phases.items():
        indices = np.flatnonzero(mask)
        stages[name] = dict(
            real_200hz_rows=int(len(indices)),
            worst_original29_qdot_ratio=float(ratio[indices].max()) if len(indices) else None,
            any_original_joint_over_speed=bool(bad[indices].any()) if len(indices) else None,
        )
    return dict(
        first_original_joint_speed_violation_s=(
            min(x["first_exceed_s"] for x in joint_events)
            if joint_events else None),
        max_original29_qdot_ratio=float(ratio.max()),
        number_exceeding_joints=len(joint_events),
        original29_joint_events=joint_events,
        stage_hazards=stages,
        actual_episode_end_s=float(t[-1]),
        complete_14s_recovery_observation=bool(t[-1] >= 13.995),
        postcontact_1p25s_censored=bool(t[-1] < impact + 1.25 - 1e-7),
    )


def measured_support_and_work(a: dict) -> dict:
    """Physical robot proxies in predefined phase bins; no anthropomorphic overclaim."""
    t = a["t"]
    impact = a["impact"]
    m = (t >= impact - 0.4 - 1e-7) & (t < impact)
    if np.count_nonzero(m) < 75:
        raise ValueError("MISSING_PRECONTACT_400MS")
    support = a["support"][m]
    left, right = (float(support[:, j].sum() * 0.005) for j in (0, 1))
    eps = 1e-12
    signed_power = a["tau"][m] * a["vel"][m]
    return {
        "pre400_left_support_force_proxy_impulse_Ns": left,
        "pre400_right_support_force_proxy_impulse_Ns": right,
        "pre400_support_force_proxy_imbalance_ratio": abs(left-right)/(left+right+eps),
        "pre400_positive_estimated_motor_work_by_group_J": {
            name: float(np.maximum(signed_power[:, idx],0).sum()*0.005)
            for name,idx in a["groups"].items()
        },
        "pre400_negative_estimated_motor_work_by_group_J": {
            name: float(np.minimum(signed_power[:, idx],0).sum()*0.005)
            for name,idx in a["groups"].items()
        },
        "pre400_human_muscle_power_inferred": False,
        "forceplate_grf_cop_calibrated": False,
        "real_athlete_data_included": False,
    }


def _load_certified(root: Path, row: dict) -> dict:
    p = root / "reports" / ("v117_" + row["name"] + ".json")
    if sha(p) != row.get("raw_sha256"):
        raise ValueError("PHYSICS_ORIGINAL_SHA_MISMATCH:" + row["name"])
    obj = json.loads(p.read_text())
    obj["joint_velocity_limits"]=np.asarray(obj["joint_velocity_limits"],dtype=float).reshape(-1).tolist()
    a = load_physical_arrays(obj)
    # Deliberate independent second decoder/validator.
    b = physical_arrays(obj)
    if a["names"] != b["names"] or not np.array_equal(a["t"], b["t"]):
        raise ValueError("INDEPENDENT_200HZ_DECODER_MISMATCH")
    if not np.allclose(a["vel"],b["v"],atol=0.0,rtol=0.0):
        raise ValueError("INDEPENDENT_TRUE_29_QDOT_DECODER_MISMATCH")
    if len(a["t"]) != row["actual_traced_200hz_rows"]:
        raise ValueError("UNCERTIFIED_REAL_PHYSICS_ROW_COUNT")
    return a


def audit(root: Path = ROOT) -> dict:
    master = root/"reports/v117_original_g1_batch.audit.json"
    batch = json.loads(master.read_text())
    if batch.get("version") != "v0.1.117":
        raise ValueError("NOT_FROZEN_V117_BATCH")
    records = []
    refs = {}
    for row in batch["runs"]:
        if not row.get("certified_original_G1_cuda_200Hz"):
            records.append(dict(
                name=row["name"], certified=False,
                startup_unqualified_not_a_failed_shot=True,
                reason=row["certification"]))
            continue
        a = _load_certified(root, row)
        records.append(dict(
            name=row["name"], scene=row["scene"], control=row["control"],
            observer_repeat=row["observer"], original_sha256=row["raw_sha256"],
            certified=True, strict_hard_gate_pass=row["hard_gates"]["hard_gate_pass"],
            all_failed_gates=row["hard_gates"]["failed_gates"],
            hazards=true_joint_hazard(a), support_and_work=measured_support_and_work(a),
            time_phases=phase_metrics(a)))
        if not row["observer"]:
            refs[(row["scene"],row["control"])] = a
    contrasts = []
    for scene in ("AJ","AK","AL"):
        ref = refs.get((scene,"v32"))
        if ref is None:
            continue
        for comparator in ("cmu02","cmu05"):
            other = refs.get((scene,comparator))
            if other is None:
                continue
            pair = paired_controller_effect(ref, other)
            # Use alternative independent raw-parser's all29 onset to reduce hidden coupling.
            direct_diff = np.max(np.abs(ref["vel"][:len(other["t"])] -
                                        other["vel"][:len(ref["t"])]),axis=1)
            ix = np.flatnonzero(direct_diff > 1e-7)
            onset = float(ref["t"][ix[0]]) if len(ix) else None
            first_hazards=[true_joint_hazard(x)["first_original_joint_speed_violation_s"]
                           for x in (ref,other)]
            first_hazards=[x for x in first_hazards if x is not None]
            early = min(first_hazards) if first_hazards else None
            contrasts.append(dict(
                scene=scene, original_controller="v32", comparator=comparator,
                onset_measured_29dof_difference_s=onset,
                earliest_measured_original_qdot_hazard_s=early,
                hazard_precedes_reference_effect=bool(early is not None
                    and (onset is None or early < onset - 1e-6)),
                comparison_time_censored_after_shorter_trace=bool(
                    len(ref["t"]) != len(other["t"])),
                motor_phase_comparison=pair))
    independent = [r for r in records if r.get("certified") and not r["observer_repeat"]]
    return dict(
        version="v0.1.117",
        frozen_protocol_sha256=sha(root/"configs/v117_frozen_matrix.json"),
        batch_master_sha256=sha(master),
        newly_certified_original_G1_executions=sum(r.get("certified",False) for r in records),
        new_true_200hz_physical_frames=sum(
            r["time_phases"]["physics_rows"] for r in records if r.get("certified")),
        independent_frozen_scene_controller_cells=len(independent),
        independent_strict_hard_gate_passes=sum(r["strict_hard_gate_pass"] for r in independent),
        repeat_inclusive_strict_hard_gate_passes=sum(
            r["strict_hard_gate_pass"] for r in records if r.get("certified")),
        real_14s_observed=sum(
            r["time_phases"]["actual_14s_recovery_coverage"] for r in records if r.get("certified")),
        observable_censored_postcontact=sum(
            r["hazards"]["postcontact_1p25s_censored"] for r in records if r.get("certified")),
        matched_controller_contrasts=len(contrasts),
        measured_hazard_before_cmu_effect_count=sum(
            r["hazard_precedes_reference_effect"] for r in contrasts),
        native_actor_collider_contact_manifold_certified=False,
        human_kicking_force_grf_cop_certified=False,
        trained_wholebody_human_policy_physically_actuated=False,
        per_episode=records,
        paired_comparisons=contrasts,
    )


if __name__=="__main__":
    result=audit()
    target=ROOT/"reports/v117_temporal_hazard_and_coordination.audit.json"
    with target.open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print("V117_REAL_HAZARD_AND_COORDINATION",json.dumps({
        k:v for k,v in result.items() if k not in ("per_episode","paired_comparisons")
    }),flush=True)
