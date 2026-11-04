# Kick Posture Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct dorsal-foot orientation and excessive outside-foot twisting while preserving free-root approach, strike and recovery.

**Architecture:** Measure posture from actual contact quaternions, kinematics and impulses; derive feasible leg poses from the local URDF; test support/strike coordination under the existing policy. Keep observations and physical outcomes distinct from requested labels.

**Tech Stack:** Tang Python/NumPy/SciPy, existing ONNX controller, Isaac Lab/PhysX.

**Spec:** `docs/superpowers/specs/2026-11-04-kick-posture-design.md`

## Global Constraints

- version/v0.1.3; no root fixation, added foot geometry, ball impulse/state injection or hidden resets.
- Preserve older profiles and raw evidence. Original directed-power objective remains open.
- Instep top forward≥0.75, toe Z≤−0.60; outside hip yaw≤1.90rad and pelvis-heading error≤45°; preserve real requested surfaces.
- After two ineffective variants of a mechanism, diagnose and change the mechanism. One GPU simulator at a time.

## Review Focus

- Quaternion conventions, env origins and shot direction must not corrupt posture metrics.
- First contact versus later large impulses must both remain visible.
- A correct-looking pose with a wrong contact surface or fall cannot pass.
- New posture controls must not silently certify the old frozen power table.
- Nominal visual success must not replace perturbation trials.

### Task 1: Posture audit and geometric targets

**Files:** `tools/shooting_posture.py`, `tests/test_shooting_posture.py`; ignored calibration manifests.
**Interface:** `audit_posture(run, index)` returns actual foot axes, shot alignment, hip yaw, heading error and contact-patch position, plus separate posture and full-chain flags.

- [x] Write failing tests for wxyz rotation, arbitrary shot heading, misleading top-edge contact and excessive hip twist.
- [x] Implement and pass the audit tests; quantify the existing instep/outside evidence.
- [x] Derive bounded joint targets using measured roots and existing LegKinematics; run one≤24-trial probe, inspect every failure.

### Task 2: Corrected control and physical comparison

**Files:** New versioned posture config/runner only where evidence supports it; existing generic evaluator remains the physical harness.
**Interface:** Ordinary trial descriptors with explicit goal/strike/reference controls, measured by Task1 and existing ShotTracker.

- [x] Diagnose first probe before a second controlled adjustment; avoid repeating previous failed full-pose takeover.
- [x] Freeze viable corrected poses, run the six perturbation trials per changed style and audit both appearance geometry and complete physical success.
- [x] Capture matched before/after full-body and foot views; report unresolved criteria instead of relabeling the old motions.
- [ ] Run relevant tests and one fresh final review; fix material findings once; document and push the version branch with verified status.
