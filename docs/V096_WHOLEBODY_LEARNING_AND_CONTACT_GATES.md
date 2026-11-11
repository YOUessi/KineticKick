# KineticKick v0.1.96 — Whole-body coordination research and safety boundary

## Falsifiable research direction

The original robot is free-root 29DoF Unitree G1. The scientific contribution must be learned **human-like whole-body timing and coordinated power**, not a small angular residual, single-frame cosine regression, scripted ball touch, or robot-trajectory prediction alone. The proposed contact-conditioned phase graph has temporal nodes: run-up, plant/support acquisition, pelvic lead, trunk rotation, proximal-to-distal right-leg acceleration, counter-arm swing, dorsal instep contact, follow-through, weight transfer and stable recovery. A candidate outputs full 29DoF time-indexed desired motion and explicit support/impact intention. It is not considered physically safe until paired contact geometry, friction, joint torque/speed/angle constraints and 14-second recovery are independently validated.

## Human data qualification

Frozen human train/test splits must be by subject and recording session, with validated licensing, motion rate, calibration and annotations. Measured human left/right GRF and CoP, if available, are separate from *observed soccer ball-foot impact impulse*. Neither a force-free 120Hz CMU motion nor a generic human motion/force dataset is professional football contact truth. Existing six CMU snippets are inadequate to supervise an expert explosive instep collision; do not fabricate missing force or expert provenance. GroundLink can be considered for support-phase supervision only after verifying its rights and measurements. Previously studied original G1 E/F positions are not independent publication-grade blind tests.

## Physical hard gates

Every true new robot trial must pass all of: measured 29 original qdot and torque/angle constraints, >=95% instep collision impulse share, legal goal, ball speed >=3.5 m/s, no nonfoot collision > original 20N termination threshold, full 14s trajectory and stable recovery. A 3.5m/s floor is NOT high-velocity expert shooting. Human-like coordination must be separately scored with observed phase lags, planted support, arm-trunk-leg synergy, smooth run-up and follow-through.

## Current v96 negative results

Nine physical 200Hz isolated sphere/box calibration cases (1620 actual steps), five exact observed-state OFF/ON pairs, one CUDA tensor contact positive and one separated negative. Native contact callbacks, headers and shape IDs remain zero under body schema creation during spawn, manually, root+collider, or with a dynamic partner. Zero callbacks cannot disprove collision; a filtered tensor pair is not a shape-level contact manifold. No original G1 physical action is newly authorized. The historical G1 E/F cases are still 0/4 strict and historical CMU10_05 wrist/hip equal/opposite net forces remain only indirect evidence.

## Next preregistered dependencies

First verify the installed PhysX runtime scene and stage binding and whether native CPU contact reporting is supported with the actual step execution path. Then establish native positive AND separated negative controls with real actor/collider IDs, normal, point and impulse; check observer nonperturbation. Only after true isolated positive calibration can unchanged original G1 3-way controller OFF/ON contact-instrumentation pairs be run with complete per-frame parity. Train human-conditioned full-body trajectories only on qualified human sequences and compare full vs foot-only, frozen-arm, reverse-timing, state-only baselines with frozen new ball positions/seeds and strict physics.

Useful API references:
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physx/docs/api/python.html
https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/read_write/readable.html
