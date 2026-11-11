# KineticKick v0.1.94 — Executable OFFLINE support/collision QP interface and research gate

## Implemented, not physically authorized
File: tools/v094_safe_wholebody_projection.py. It accepts all-29 desired and measured joint velocities, original per-joint speed and acceleration limits, measured two-foot support loads, current 3×29 stance-foot Jacobians, and a geometry-SHA-verified signed collider-pair distance with 29-variable gradient. It performs a constrained CPU quadratic projection with hard velocity/acceleration box constraints, support-foot slip limits and linearized next-step collision clearance. After optimization it independently rechecks constraints before returning a candidate.

Missing/stale native shape geometry, uncalibrated native stream, no planted support or incorrect data produce explicit FAIL_CLOSED. It never dispatches a G1 motor command: physical_actuation_authorized is always False, including when offline feasible, because inverse dynamics torque, full collision topology, G1 native manifold and source-human qualification are still unavailable. A sphere-box tensor witness cannot substitute a G1 wrist-hip native manifold.

The offline filter is a concrete control interface and safety scaffold, NOT a trained human action generator, not a certified inverse-dynamics controller, and not a naturally learned high-power soccer kick.

## Falsifiable paper hypothesis, not yet proved
A learned expert-human wholebody action model conditioned on run-up, actual plant-foot support, pelvis drive, trunk rotation, swing-leg kinetics, counterarm phase and follow-through/landing may improve blind-rated naturalness and unseen-geometry G1 physical legality against v32, leg-only, frozen-arm, reversed-phase and no-support ablations. It must generate a complete 29DoF action time series, not a weak 1mrad perturbation or an observer output, and be constrained by motor torques and support/collision QP in the original G1 physics.

## Human dataset qualification and gate
Current six authentic 120Hz CMU C3D clips lack verified elite-athlete identity, synchronized force-plate GRF/CoP, measured right-instep impulse and ground-truth foot-ball contact. Subject folders are split 5:1; post-hoc stricter 90% visibility yielded 0/6 qualified geometric phase clips. These optical markers cannot be force or collision truth. Obtain licensed multi-subject expert instep shots with full-body 3D, verified planted-foot contacts, high-fps ball velocity, synchronized GRF and genuine original impact labels. Split by subject and session prior to model training. Keep human motion naturalness and original G1 impact reward independent.

## Subsequent pre-registered physical matrix (not executed in v94)
Freeze full original 29DoF USD/geometry SHA, train/val/test subjects, actual perturbed initial ball positions plus seeds, 200Hz scenario matrix, action magnitudes and baselines. First obtain native G1 actor/collider paths with positive point/normal and impulse and test OFF/ON exact unchanged physics; then verify inverse dynamics, support/geometry certificates and a qualified human generator before motor actuation. Every G1 episode uses independent gates: >=95% instep impulse, actual goal, >=3.5m/s ball speed, all original 29 measured qdot, nonfoot <20N and true 14s recovery. 3.5m/s is only a minimum eligibility threshold, not proof of high-speed explosive soccer shooting.

## v94 outcome limitation
The six new episodes are isolated sphere-box real PhysX probes, not original G1 kicking. Positive GPU contact tensor and negative separated-box controls have not repaired zero native contact headers. There are zero new G1 kicking trials and no new learned fullbody controller physical trials.