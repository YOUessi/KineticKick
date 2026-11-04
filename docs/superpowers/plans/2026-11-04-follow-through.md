# Follow-Through Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve a forward leg swing after actual ball departure, then decelerate and recover without losing the existing kick/posture success.

**Architecture:** Audit recorded foot/root motion and apply post-impact timing/reference changes using the existing controller. Freeze only physically supported profiles. Final scope: diagnostics and explicitly unsuccessful experiments only. No new default/stable profile; full follow-through goal remains unresolved.

**Tech Stack:** Python/NumPy, unchanged ONNX controller and IsaacLab/PhysX.

**Spec:** `docs/superpowers/specs/2026-11-04-follow-through-design.md`

## Global Constraints

- version/v0.1.5; preserve old profiles; a new optional continuation action phase changes the source fingerprint, so old target/power provenance must reject this branch.
- Forward travel≥0.22m, pelvis-relative advance≥0.15m by+0.25s; velocity at+0.12s≥0.60m/s; sustained reversal≥+0.20s; return speed≤2m/s through+0.45s and≤2.5m/s through+0.75s.
- Additional contact above2N from+0.08 through+0.75s fails. Original real contact/goal/posture/recovery still required.
- One GPU process,12-case first probe, diagnose before refinement; no repeated blind grids.

## Review Focus

- Body translation alone cannot be mistaken for leg follow-through.
- NaN/gapped/short kinematics cannot certify continuity.
- Late ball pushing cannot replace a released shot.
- Changed reference conditions must remain part of the frozen control manifest.
- Unsuccessful experimental motions must not become the default preview.

### Task 1: Measured follow-through and bounded probes

**Files:** `tools/follow_through.py`, `tests/test_follow_through.py`, `tools/shooting_control.py`, `tests/test_shooting_control.py`; ignored manifests.
**Interface:** `audit_follow_through(run,index)` returns real forward/relative travel, measured velocity/reversal/return speed, extra contact and combined success. Matrix check composes original posture acceptance.

- [x] Write RED tests for abrupt stop/fast return, body-only movement, wrong/extra contact and invalid timestamps; implement and pass.
- [x] Run12 isolated post-impact cases; inspect physical pose, contact, forward swing and recovery before refinement.

### Task 2: Freeze, visual comparison and publish

**Files:** New follow-through configuration if supported, preview routing/test, README and report.
**Interface:** Ordinary frozen evaluator config, inherited physics and actual recorded traces.

- [ ] Freeze supported controls and verify three ball positions × two new seeds for all styles.
- [x] Render and inspect before/after motion; report any unmet criteria.
- [ ] Run appropriate tests, one independent final review, document and push version branch.

Final outcome: Task2 stable-control acceptance not achieved. Publish an explicitly failed/experimental branch, preserve v0.1.3 default and keep this task ledger for unresolved work. No task-success claim.
