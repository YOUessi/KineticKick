# KineticKick v0.1.96 preregistered calibration protocol
Parent SHA: ee7095fe4fc0ec2d121a5f38c485236c8e6e604f (v95). Date: 2026-11-12.

Purpose: Resolve why physically calibrated tensor positive observations coexist with zero native contact callbacks. Test three PhysxContactReportAPI authoring locations (manual owning body, spawn-time creation, body+child collider); vary CPU vs CUDA and dynamic vs kinematic partner. These changes are **isolated calibration model variants**, not new original G1 physics and not an action policy.

Nine one-shot 200Hz/180-frame fixed cases are specified in configs/v096_frozen_matrix.json and must be immutable at first physics launch. A single shared flock serializes ALL Isaac/PhysX GPU or CPU episodes. Verify no foreign GPU jobs, >=20GiB free, clean source branch, exact frozen SHA, and unique output names before each trial. Stop after any child/physics infrastructure error. Store raw JSON/log, parsed audit, and SHA locally only; do not overwrite preexisting evidence. No previous controller, URDF, human data, or weights may be modified.

Parities compare only observed saved coordinates, NOT claim equality of all internal physical variables. Off/on schemes are not a proof of unperturbed original G1 physics; if native positive calibration is obtained here, separate immutable original-G1 OFF/ON experiments with full-state parity remain mandatory before deployment. Zero callbacks does not negate observed real contacts. Do not interpolate censored episodes. Do not count any calibration run as original G1 shooting.

Secondary analysis must independently validate every raw SHA, native positive classification, observed tensor/pose collision, CPU vs GPU, all 5 frozen state parity pairs, and present failure/recovery limitations. Future human motion training requires license, subject holdout and verified expert impact/GRF, no claims from robot-only MAE.
