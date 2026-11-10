#!/usr/bin/env python3
"""Independent v55 seed-effectiveness and nonindependent-replay audit.

Runs entirely on preserved, SHA-verified original PhysX JSON. Does not change a
simulation, a policy, or a frozen protocol. The two seeds are not independent
robustness evidence if physical traces are bit-for-bit identical.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/physx_pair_dynamics_v055.json"
PROGRESS = ROOT / "reports/v055_pair_batch_progress.json"
OUTPUT = ROOT / "reports/v055_seed_replay_audit.json"
EVIDENCE_KEYS = ("traces", "contacts", "records", "sequence_records",
                 "termination_reasons", "nonfoot_contact_diagnostics")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_seed_pair(a, b):
    """Compare physical events, not metadata-only top-level JSON SHA."""
    if a["seed"] == b["seed"]:
        raise ValueError("Cannot claim distinct seeds")
    if len(a["traces"]) != len(b["traces"]):
        raise ValueError("Incompatible environments")
    per_field = {k: a[k] == b[k] for k in EVIDENCE_KEYS}
    return dict(seeds=[a["seed"], b["seed"]],
                compared_fields=per_field,
                identical_physics_and_contact_evidence=all(per_field.values()),
                identical_200hz_joint_traces=per_field["traces"])


def analyze(config, progress, load):
    if progress["protocol_sha256"] != sha256(CONFIG):
        raise ValueError("Frozen protocol SHA mismatch")
    cases = config["cases"]
    if len(progress["runs"]) != 15 or len(cases) != 15:
        raise ValueError("Incomplete immutable 15-case batch")
    verified = {}
    for case, artifact in zip(cases, progress["runs"]):
        if case["id"] != artifact["id"]:
            raise ValueError("Reordered reports")
        path = ROOT / case["out"]
        if artifact["sha256"] != sha256(path):
            raise ValueError("Primary PhysX evidence was changed")
        episode = load(path)
        if episode["seed"] != case["seed"]:
            raise ValueError("Recorded seed differs from protocol")
        if episode["joint_strike"]["ball_offset_xy"] != case["ball_offset_xy_m"]:
            raise ValueError("Ball placement drift")
        verified[case["id"]] = (case, episode)
    paired = []
    for condition in ("holdoutA", "holdoutB"):
        for policy in ("frozen_v32", "cmu_10_02", "cmu_10_05"):
            name_a = f"{condition}_seed1201_{policy}"
            name_b = f"{condition}_seed1509_{policy}"
            ca, a = verified[name_a]
            cb, b = verified[name_b]
            if ca["ball_offset_xy_m"] != cb["ball_offset_xy_m"] or ca["velocity_y_delta_mps"] != cb["velocity_y_delta_mps"]:
                raise ValueError("Seed-only comparison changes physical condition")
            pair = compare_seed_pair(a, b)
            paired.append(dict(condition=condition, policy=policy, **pair))
    return dict(
        version="0.1.55",
        frozen_protocol_sha256=progress["protocol_sha256"],
        episodes_physically_executed=15,
        center_diagnostics=3,
        holdout_episodes_physically_executed=12,
        holdout_exact_seed_replay_pairs=6,
        pairs_with_identical_all_physical_evidence=sum(p["identical_physics_and_contact_evidence"] for p in paired),
        independent_stochastic_seed_evidence=not any(p["identical_physics_and_contact_evidence"] for p in paired),
        distinct_holdout_conditions_per_policy=2,
        statistical_interpretation="Seed 1201 vs 1509 yields identical 200Hz and contact traces in all compared pairs. These are repeatability checks, NOT 4 independent randomized outcome samples per policy. Do not use n=4 Wilson intervals as inferential generalization.",
        frozen_cmu_policies_not_retrained=True,
        comparisons=paired)


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite preserved independent audit")
    config = json.loads(CONFIG.read_text())
    progress = json.loads(PROGRESS.read_text())
    report = analyze(config, progress, lambda p: json.loads(p.read_text()))
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("V055_SEED_REPLAY_AUDIT", json.dumps(dict(
        exact_replay_pairs=report["pairs_with_identical_all_physical_evidence"],
        total_pairs=report["holdout_exact_seed_replay_pairs"],
        independent=report["independent_stochastic_seed_evidence"])))
    return report


if __name__ == "__main__":
    main()
