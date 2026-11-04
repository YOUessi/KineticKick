# Reactive receiving implementation plan

> Execute inline in this session; preserve the existing v0.1.20 implementation.

**Goal:** Reduce residual ball motion and receive ground/aerial balls from a common feet-down stance, reacting to observed height, direction and velocity.

**Architecture:** A new motion planner consumes live ball position/velocity, predicts a reachable interception and schedules a continuous foot translation/rotation from the standing pose. An isolated runtime reuses the original free-root simulation, support feedback and measured-foot-placement motor stance. Physical telemetry decides whether the ball is actually retained and the robot recovers.

**Tech stack:** Python, NumPy/SciPy, Isaac Lab/PhysX, original PAiD G1 and ONNX.

**Spec:** Latest user request in this chat; previous result and limits in `reports/2026-11-05-balanced-inside-receiving.md`.

## Constraints and review focus

- New branch version/v0.1.21, no main or previous-version history changes.
- No physics-state writes after initialization, extra ball impulses, changed colliders or expanded motor limits.
- The controller must not read scenario launch metadata to choose interception height or timing.
- Same initial grounded feet for both arrival types; changed measured ball height/speed must change the foot target and action time.
- A stopped/diverging/unreachable ball must not trigger a blind scheduled kick.
- Acceptance requires real inside-foot contact, active yielding, no illegal contacts or fall, safe release, final ball speed <0.02m/s and spin <0.3rad/s, and recovered stance. Report partial results without widening thresholds.
- Avoid broad scans. First validate dynamic preparation in one ground/aerial case; diagnose failures before adapting the approach. Then run a small direction/speed set and render representative episodes with recorded source/config provenance; reuse earlier successful recordings only when the changed paths are inactive/equivalent for their measured inputs.

## Work

- [x] Add `tests/test_reactive_receive_motion.py` for live-height, live-speed and stationary/diverging-ball responses; observe the absent planner failure.
- [x] Add `tools/reactive_receive_motion.py` with `update_prediction(now, ball, velocity, touched)` and `targets(now)` compatible with the existing runner, plus measured event telemetry.
- [x] Add `tools/run_reactive_receive.py` and common initial stance/scenarios; generate local references with `tools/prepare_aerial_reference.py`.
- [x] Verify dynamic physical contact and recovery, then reduce residual motion based on actual contact/ball traces.
- [x] Add stronger acceptance, run the bounded direction/speed cases, and record actual failures as well as successes.
- [x] Render actual simulation previews; document inputs, provenance, scope and results; commit/push this version with upstream.
