#!/usr/bin/env python3
"""Read-only fail-closed v64 scientific claim gate; independent of task PASS.

Consumes the SHA-linked 12 original G1 PhysX audit, not the raw large files.
It can license only *observed causal robot motion effect*, never automatically
human naturalness, elite instep power, certified contact safety or hardware use.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/v064_independent_physics_audit.json"


def classify_pair(pair: dict) -> str:
    if not pair.get("valid_real_PhysX_pair", False):
        return "UNSCORABLE_NO_PREACTION_PARITY"
    parity = pair.get("full_25_field_200hz_physics_parity_vs_original")
    if not isinstance(parity, dict):
        return "UNSCORABLE_MISSING_WHOLE_EPISODE_PARITY"
    steps = pair.get("candidate_20ms_actual_steps")
    amplitude = pair.get("candidate_actual_max_command_delta_rad")
    actual = pair.get("actual_measured_g1_29dof_joint_position_difference_max_rad")
    if (type(steps) is not int or steps < 0 or
        not isinstance(amplitude, (int, float)) or not isinstance(actual, (int, float)) or
        not (0 <= amplitude < float("inf") and 0 <= actual < float("inf"))):
        return "UNSCORABLE_MISSING_ACTUATION_EVIDENCE"
    identical = parity.get("matches_at_abs_tol_1e_6") is True
    maxdifference = parity.get("max_abs_numeric_difference")
    if steps == 0:
        if amplitude != 0 or actual != 0 or not identical or maxdifference != 0:
            return "INCONSISTENT_INERT_CANDIDATE_STATE_CHANGED"
        return "INERT_POLICY_BASELINE_REPLAY_ONLY"
    if amplitude <= 1e-8 or actual <= 1e-8 or identical:
        return "UNVERIFIED_ACTUATION_NO_OBSERVABLE_CAUSAL_MOTION"
    if pair.get("preaction_parity", {}).get("matches_at_abs_tol_1e_6") is not True:
        return "UNSCORABLE_PREACTION_STATE_DRIFT"
    return "ROBOT_MOTION_EFFECT_OBSERVED_NOT_NATURALNESS"


def assess(audit: dict) -> dict:
    if audit.get("version") != "v0.1.64":
        raise ValueError("Wrong immutable study version")
    rows = audit.get("all_episodes", [])
    pairs = audit.get("preregistered_pair_results", [])
    if (len(rows) != 12 or len(pairs) != 8 or
        len({x.get("id") for x in rows}) != 12 or
        len({(x.get("context"), x.get("policy")) for x in pairs}) != 8 or
        audit.get("actual_original_G1_PhysX_attempts") != 12):
        raise ValueError("Not the complete registered 4 x 3 physics experiment")
    labels = [classify_pair(pair) for pair in pairs]
    effects = sum(label == "ROBOT_MOTION_EFFECT_OBSERVED_NOT_NATURALNESS" for label in labels)
    inert = sum(label == "INERT_POLICY_BASELINE_REPLAY_ONLY" for label in labels)
    if effects + inert != len(labels):
        return {"scientific_activation_gate": "REJECT_INCONSISTENT_OR_UNSCORABLE",
                "candidate_effect_pairs": effects, "inert_pairs": inert,
                "pair_labels": dict((p["context"] + ":" + p["policy"], v) for p, v in zip(pairs, labels)),
                "original_task_passes": audit.get("actual_original_PhysX_all_gate_pass"),
                "professional_kinetic_claim_allowed": False,
                "naturalness_claim_allowed": False}
    human_contact = audit.get("professional_high_power_true_foot_ball_contact") is True
    human_grf = audit.get("measured_human_ground_reaction_force") is True
    heldout_people = audit.get("independent_human_subject_test") is True
    return {"scientific_activation_gate": "OBSERVED_ROBOT_EFFECT_ONLY" if effects else "REJECT_ALL_CANDIDATES_INERT",
            "candidate_effect_pairs": effects, "inert_pairs": inert,
            "pair_labels": dict((p["context"] + ":" + p["policy"], v) for p, v in zip(pairs, labels)),
            "original_task_passes": audit.get("actual_original_PhysX_all_gate_pass"),
            "original_PhysX_attempts": audit.get("actual_original_G1_PhysX_attempts"),
            "independent_people_test_available": heldout_people,
            "professional_kinetic_claim_allowed": False,  # No verified elite contact-force/power benchmark or human blind rating
            "naturalness_claim_allowed": False,
            "naturalness_reason": "Human observer/GRF/contact truth and independent blind motion rating were not certified"}


def main() -> None:
    data = json.loads(AUDIT.read_text())
    result = assess(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["scientific_activation_gate"] != "REJECT_ALL_CANDIDATES_INERT":
        raise RuntimeError("v64 published conclusion changed; recheck frozen evidence, do not silently change claims")


if __name__ == "__main__":
    main()
