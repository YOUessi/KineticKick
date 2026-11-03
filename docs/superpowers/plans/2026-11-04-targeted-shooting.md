# Targeted Free-Root Shooting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Control and verify horizontal aim and outgoing ball speed through free-root physical kicks.

**Architecture:** Separate fixed-goal measurement from the policy's virtual aim. Calibrate bounded actuator commands using actual impacts, freeze their mapping, and validate independent target/speed commands with complete traces.

**Tech Stack:** Python, NumPy, existing ONNX policy, Tang-local Isaac Lab/PhysX, unittest.

**Spec:** `docs/superpowers/specs/2026-11-04-targeted-shooting-design.md`

## Global Constraints

- Branch `version/v0.1.2`; previous version configuration stays available.
- Fixed goal center (0, -3.7), normal (0, -1), opening 1.6 × 1.1 m.
- Outgoing speed at first contact +0.10 s; aim error ≤0.15 m; speed error ≤max(0.20 m/s, 12% of request).
- Preserve free root, real contact surfaces, no post-initialization state writes and contiguous standing recovery.
- Keep generated outputs, models and upstream assets local. One simulator at a time.

## Review Focus

- A target outside the calibrated envelope must produce an actionable error before simulation.
- Goal measurement must remain fixed when virtual aim or requested target changes.
- Missing, reverse or out-of-aperture crossings cannot masquerade as a hit.
- An incomplete trace or changed controller cannot certify acceptance.
- Direction must remain controlled while strength changes, and vice versa.

### Task 1: Fixed-goal and outgoing-speed measurement

**Files:** Create `tools/targeted_shooting.py`, `tests/test_targeted_shooting.py`; modify `tools/eval_free_shooting.py`.

**Interfaces:** `TargetShotTracker(target_x, speed_mps, goal_y=-3.7)` consumes actual timestamped ball positions/velocities/contact forces and produces crossing, target error and speed error. Trial manifest `policy_goal` is distinct from fixed `goal` and `goal_forward`; optional `target` activates the additional metric.

- [x] Write failing tests for fixed-plane interpolation, first crossing outside the opening, reverse crossing, speed interpolation at contact+0.10 s and incomplete evidence.
- [x] Run the targeted unit file and observe the expected missing implementation failure.
- [x] Implement the pure metric and independent goal-conditioning input in the shared live/batch harness.
- [x] Pass unit tests, then run bounded sensitivity probes of virtual aim and strike controls; preserve raw traces and diagnose results.
- [x] Commit the independently usable measurement/probe interface.

### Task 2: Calibrated aim and strength controller

**Files:** Extend `tools/targeted_shooting.py`, tests and evaluator CLI; create `configs/targeted_shooting_v1.json`.

**Interfaces:** `resolve_targeted_shot(config, style, target_x, speed_mps)` validates the supported envelope and returns an ordinary trial descriptor with pinned physical goal, virtual aim and actuator settings. Low/medium/high presets resolve to physical speed requests per style. Exact coefficients and supported nodes are frozen from Task 1 evidence.

- [x] Add failing tests for finite inputs, unsupported aim/speed, exact/intermediate command mapping, fixed geometry and incompatible calibration metadata.
- [x] Implement the smallest controller justified by probes and a CLI for target/power inputs.
- [x] Calibrate actuator controls in bounded batches, test intermediate commands if interpolation is exposed, and freeze feasible command ranges before held-out evaluation.
- [x] Pass tests and verify that each axis changes the corresponding physical outcome without invalidating contact/recovery.
- [x] Commit the frozen control configuration and interface.

### Task 3: Held-out acceptance and usable demonstration

**Files:** Create `tools/check_targeted_shooting_acceptance.py`, its tests and `reports/2026-11-04-targeted-shooting.md`; update `README.md`, live visuals as needed.

**Interfaces:** Replay all records through `ShotTracker` and `TargetShotTracker`; validate unique complete matrix/seeds, pinned configuration/model/assets and trial duration. Report per-style, per-speed and total success/error rather than trusting saved success flags.

- [ ] Write failing tests for duplicate/missing commands, incomplete traces and changed controller; implement the checker and pass tests.
- [ ] Run the frozen 27 commands at two new seeds (split deterministic batches ≤24 environments), replay the traces, and inspect every failure against the spec.
- [ ] Capture representative actual simulation with requested and measured aim/speed; inspect complete frames and physics record.
- [ ] Run the relevant regression suite, one fresh whole-branch review and one fix pass if necessary; document exact achieved limits and reproducible commands.
- [ ] Commit, push with upstream and verify the remote head. Delete this plan's scratch workspace only after completed review.
