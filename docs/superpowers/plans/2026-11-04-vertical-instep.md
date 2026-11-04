# Near-Vertical Instep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the instep toes point nearly vertically down at real ball contact while preserving the free-root physical chain.

**Architecture:** Reuse the unchanged physical evaluator and posture audit, add stricter orientation/clearance measurements, and tune only a new versioned joint-control profile.

**Tech Stack:** Python/NumPy, original ONNX policy, IsaacLab/PhysX, existing URDF kinematics.

**Spec:** `docs/superpowers/specs/2026-11-04-vertical-instep-design.md`

## Global Constraints

- version/v0.1.4; original physical asset and joint limits; no root/ball injection or training.
- At first contact and by impulse weighting: toe within15 degrees of down, dorsal-forward≥0.90. Conservative near-contact foot clearance≥1cm.
- At most12 trials per diagnostic batch, inspect before the second. No repeated blind sweeps.
- Freeze before6 perturbation trials/style; at least5/6 combined successes, no falls.

## Review Focus

- A single upright frame must not hide flat orientation at most of the impulse.
- Missing foot-pose coverage cannot certify toe clearance.
- World/environment translations must not corrupt the clearance estimate.
- Strict vertical criteria must supplement contact/goal/recovery, not replace them.
- New default preview must keep old target/power calibration and explicit configs intact.

### Task 1: Audit and bounded physical diagnosis

**Files:** `tools/vertical_instep.py`, `tests/test_vertical_instep.py`; ignored probe manifests.
**Interface:** `audit_vertical_instep(run,index)` consumes the existing physics report, returns measured toe-down angle, weighted alignment, conservative near-contact clearance and combined success.

- [x] Test that55° fails,80° passes, later flat impulse fails, insufficient/missing clearance fails; run RED then implement and run GREEN.
- [x] Derive feasible joint/timing hypotheses, run two12-case batches with diagnosis between them. Additional4-case telemetry/reference and2-case position/timing checks are documented rulings; all failed the combined goal.

### Task 2: Frozen profile, evidence and publication

**Files:** New `configs/free_shooting_vertical_v1.json` if supported; preview default, tests, README and version report.
**Interface:** Ordinary evaluator trial descriptors; strict matrix audit composes the existing independent replay and vertical checks.

- [ ] Not achieved: no supported candidate to freeze; no independent acceptance or stable profile claimed.
- [x] Render a complete failed feasibility replay alongside the stable v0.1.3 sequence, with explicit failure labeling. Near-vertical pose occurs in air, not at accepted contact.
- [x] Run112 tests and one fresh final review; fix finite-timestamp validation with RED→GREEN coverage. Document and publish the diagnostic branch; physical outcome remains not achieved.
