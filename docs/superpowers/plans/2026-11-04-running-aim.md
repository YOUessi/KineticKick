# v0.1.8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Extend free-body approach/shot/recovery with measured running, outside contact and calibrated aim/power.
**Architecture:** Preserve the frozen recurrent policy and physical evaluator. Add isolated reference conversion and acceptance helpers; use trial manifests for bounded physical investigations. Freeze only successful combinations after holdouts.
**Tech Stack:** Python, NumPy, ONNX Runtime, Isaac Lab/PhysX, unittest, ffmpeg.
**Spec:** docs/superpowers/specs/2026-11-04-running-aim-design.md

## Global Constraints
- version/v0.1.8; preserve previous versions and ignored Tang assets.
- Original 29DoF actuator limits, free root, physical ball; no runtime state writes.
- Fixed goal [0,-3.7], aim <=0.15 m; adjacent measured strength levels >=0.30 m/s.
- Running excludes reset flight; full sequence includes legal contact and stable recovery.

## Review Focus
- Initial drop or missing samples must not certify running (task 1).
- A target hit after a body collision or fall must not pass (task 3).
- A shifted measurement goal must not masquerade as aiming (task 3).
- Old calibration hashes cannot certify new controllers (task 3).
- Unsupported mode/style/target/power must fail explicitly (task 3).

### Task 1: Running reference and physical flight gate
Files: tools/running_metrics.py; tests/test_running_metrics.py; tools/build_run_reference.py if conversion is viable.
Interface: summarize_running(trace, initial_root, forward) -> dict with flight intervals, alternating supports, progress, failure reasons and running_success.
- [x] Test initial drop rejection, alternating 3 flights acceptance, missing trace/fall rejection.
- [x] Implement metrics and convert same-joint RUN reference with explicit order, fps and provenance; inspect velocities/seams.
- [x] Run <=3 short physical probes on the original actor, inspect tracking and contact telemetry. Diagnose failure before any second approach.

### Task 2: Outside full sequence
Files: configs/runup_shooting_v2.json only after success; generated reports remain local.
Interface: standard evaluator per-trial motion_file/coordination/strike manifest; sequence_records must pass intended outside contact.
- [x] Derive <=4 geometry-supported candidates from observed root/foot pose, run one batch.
- [x] Diagnose contact/root tracking before one correction batch; promote only legal full sequences and holdout perturbations.

### Task 3: Aim/power joint acceptance
Files: tools/runup_targeting.py; tests/test_runup_targeting.py; configs/runup_targeting_v2.json if calibrated; preview wrapper integration.
Interface: validate_runup_targeting(config, root); resolve_runup_target(config, mode, style, target_x, power); summarize_target_matrix(results, config).
- [x] Test missing/duplicate commands, source mismatch, unsupported command, false sequence success and per-target speed separation.
- [x] Quantify bounded local sensitivity on three accepted motions with fixed physical goal.
- [ ] Derive calibration only inside demonstrated response region; freeze before unseen-seed/ball-offset holdouts.
- [x] Preview actual accepted trials; keep experimental/failed modes distinct.

### Task 4: Release evidence
- [x] Run focused tests and full asset-dependent suite; independently review whole diff once and fix material issues.
- [x] Record exact physical passes/failures and remaining limitations in reports/2026-11-04-running-aim.md and VERSION_CONTROL.md.
- [x] Commit and push version/v0.1.8 with upstream; verify remote SHA; provide actual physical preview and concise outcome.

## Execution record

The feature objective remains partially complete: outside full sequence6/6 and native-strength aim18/18 passed; sustained running and jointly calibrated three-level power did not. No three-level calibration was frozen. Two distinct running probes isolated reference-error termination from actual physical falling. Direction probes converged; velocity feedforward probes and two joint matrices did not yield robust three-level control, so that table search stopped.

Verification: full Tang-local suite197/197; one independent whole-branch review independently replayed acceptance and checked fingerprints, with no Critical/Important findings. Final: minor (deferred): ordinary preview console's final line prints sequence status without the target error; target results remain in the complete output/evidence. No controller/code fix was required by review.

Ruling: promote only fixed native-strength aiming and the six-case outside envelope, not the requested full running/three-level capability — raw evidence does not meet those gates — broader use remains unsupported. Existing user authorization selects publishing the version branch, with main and earlier branches preserved; no extra integration menu is needed.

Publication verified: initial implementation commit084b6c5578b249dae229c39091101792859c0e60 was pushed with upstream tracking. Remote main and v0.1.7 hashes were unchanged. Both final H.264 videos decoded fully; all four rendered trials passed their applicable sequence/target gates.
