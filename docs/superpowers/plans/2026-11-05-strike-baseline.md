# Frozen Strike Baseline Implementation Plan

> Execute inline under the user's approved baseline-evaluation scope. Use the existing checkout, preserve earlier versions and use targeted tests; no repetitive full-suite runs.

**Goal:** Produce an honest, reproducible v0.1.15 force/coordination baseline and seven-case sensitivity evaluation.
**Architecture:** Unchanged original controller behind a reset-configuration-only wrapper; pure offline measurement module; frozen manifest; generated local evidence and tracked research report.
**Tech stack:** Existing Python, NumPy/SciPy, Matplotlib, Isaac Lab/PhysX; one GPU simulator at a time.
**Spec:** ../specs/2026-11-05-strike-baseline-design.md

## Constraints and review focus

Do not tune controls or goal while evaluating. Freeze inputs before results. Missing contact, truncated force episodes, missing samples and nonfinite data must not become favorable zeros. Keep signed power and correct per-sample duration. Do not call seven deterministic cases a success-probability estimate. Validate actual initialization perturbations and exact unchanged-center replay.

- [x] Add tests for real force episodes, signed power/work, collision-bin exclusion, yaw wrap, missing evidence and frozen-plan changes.
- [x] Implement `tools/strike_baseline_protocol.py` and `tools/run_strike_baseline_case.py`: only allowed case offsets, hash checks, original runner import, deterministic reset velocity and initial-state provenance.
- [x] Implement `tools/strike_baseline_metrics.py` and offline report/plot entry point; analyze historical normal and strong records.
- [x] Replay the center once, compare historical physics; run the six predeclared perturbations sequentially and retain every physical failure.
- [x] Produce measured failure breakdown, scientific figures and research report; update README, VERSION_CONTROL and research plan; commit/push v0.1.23.

## Execution evidence and review

- Frozen matrix: seven conditions, no retuning; eight physical episodes total including original-entry headless diagnostic.
- Center wrapped vs original headless: all2800 trace rows identical. Historical rendered trajectory differs after1.785s; measured ball trajectory identical. This difference is reported, not hidden by tolerances.
- Terminal marker parsing: confirmed producer adds a copied t+1us status; covered by a failing regression then corrected without GPU reruns.
- One independent read-only review of new evaluation code found incomplete-log success and missing standalone provenance checks. Both received reproducing tests, then were fixed; 16 targeted checks passed. No second review or broad suite.
- Review exclusions (accepted scope): no new GPU runs by reviewer; parent verified replay equality; human naturalness and hardware measurements are outside this baseline task. Final scientific interpretation and limits are documented in the report.
- Scientific figures inspected; full raw failures retained. No model training or control tuning performed.
