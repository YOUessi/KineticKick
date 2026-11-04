# Receive, reposition and directional pass implementation plan

> Execute inline in this session using the existing checkout. User authorized continued implementation; preserve v0.1.21.

**Goal:** Continue the physical receiving episode with a step based on the measured stopped-ball location, a commanded inside-foot ground pass, and balanced recovery.

**Architecture:** Reuse the verified receiving runtime through an isolated new runner. A measured-state gate begins a short right-foot repositioning step; actual unloading and landing must occur before preparing and executing the pass. The left foot remains the support foot in this first bounded implementation. Ball position and requested world direction determine the step and strike geometry; no launch-time ball coordinates drive the pass. Recovery returns toward the measured original ready stance, then holds the observed landing pose with motor support feedback.

**Tech stack:** Existing Python/NumPy/SciPy, Isaac Lab/PhysX, original free-root 29DoF G1. No training or new dependencies.

**Spec:** Latest request in this chat: stop the ball, adjust steps according to its obtained position, then pass in a specified direction. Default choice: inside-foot ground pass, world −Y with selectable angle/target; verify center and two nearby directions. This is local footwork, not arbitrary walking around the ball.

## Constraints and acceptance

- Branch version/v0.1.22 was created from verified, pushed v0.1.21 before edits.
- Retain physics, joint limits, original assets, all previous versions and raw evidence. No state writes after initialization.
- Before planning: actual prior reception, near-rest ball speed <0.02m/s and spin <0.3rad/s for >=0.85s, stable double support >=0.6s.
- Footwork: actual foot liftoff, >=3cm horizontal relocation, and measured ground contact before striking; reject merely moving the target or airborne swing as a step.
- Pass: distinct later inside-foot contact, ball crosses a 1m downrange plane with direction error <=8deg, no unintended body-ball contacts or fall, and final balanced double support >=0.5s. Log actual speed and height; do not relabel a deflection during footwork as a pass.
- Ground/aerial reception evidence is evaluated only before the step trigger; the intentional pass must not make the receiving evaluator reject the whole episode merely because the ball later travels away.
- Bound diagnostic runs; inspect failure traces before changing one causal element. No wide parameter scans or full-suite reruns.

## Work

- [x] Add planner tests for stopped-ball/support gates, real measured landing, changed ball position, direction and safe foot geometry.
- [x] Add `tools/receive_pass_motion.py`: stateful measured-state step/strike/recovery planner, desired foot pose/velocity/load, and event telemetry.
- [x] Add isolated `tools/run_receive_pass.py`, shared receiver plan plus passing controls, expanded bounded duration, source provenance and phase telemetry.
- [x] Add `tools/receive_pass_metrics.py` and targeted tests for phase separation, actual footwork, heading and final stance.
- [x] Run one continuous ground receiving/step/pass pilot; diagnose contact/support failures. Then test the small direction set and an aerial receiving handoff.
- [x] Record actual previews, describe exact scope/failures, update README/VERSION_CONTROL, commit and push this version.
