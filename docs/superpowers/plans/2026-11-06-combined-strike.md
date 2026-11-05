# Combined Strike Adaptation Implementation Plan

> Execute inline with superpowers:executing-plans. User has authorized continued diagnosis, implementation, bounded local GPU experiments and version branch publication; no new approval menu.

**Goal:** Fix the known combined-condition physical failure, preserve working cases and measure new combinations honestly.
**Architecture:** Frozen feedback experts plus measured-state adaptation; choose collision-clearance versus early gait adaptation from body-level evidence.
**Tech Stack:** Existing IsaacLab/PhysX, Torch, ONNX, original G1 assets.
**Spec:** docs/superpowers/specs/2026-11-06-combined-strike-design.md

## Constraints and review focus

- version/v0.1.27 created from clean92141f6 after fetching remote branches; preserve old branches and19 original frozen files.
- Never replace physical success with reward; keep original contact/body/goal/running/recovery gates.
- Measured geometry, velocity and support drive commands; no case-id switches or post-reset state writes.
- Do not regress original or successful endpoint cases. Reuse unaffected frozen evidence only with explicit source/equivalence limits.
- New held-out combinations are specified in the design before tuning; known failed combination is development data.

## Tasks

- [ ] Diagnose: tools/run_placement_strike.py optional --contact-diagnostics records nonfoot body names, forces and positions without affecting controls or termination. Replay known failure once; compare physical outcome with v0.1.26 and explain collision cause.
- [ ] Fix: implement the smallest measured-state control correction supported by diagnosis in an isolated helper/config; use actual failed geometry as a CPU regression fixture if geometry logic is needed. Keep original configuration behavior available.
- [ ] Validate: independent known failure, affected original conditions and prespecified new combinations; one actual preview and a minimal ablation. No broad redundant suite.
- [ ] Publish: exact results and limitations in reports/2026-11-06-combined-strike-adaptation.md, README and VERSION_CONTROL; targeted checks, version commit and upstream push.

Diagnosis update: old physical samples show right hip pitch−1.4490rad/velocity−3.710rad/s at1.640s with+0.2237mm pelvis/thigh gap; at1.655s pitch−1.49774rad/gap−0.0413mm coincides with1031.7N nonfoot force. Other nonfoot geometry remains>9cm above ground; no illegal body-ball contact. This strongly supports pelvis/right-thigh self-collision from post-contact overshoot. Optional body-force replay is blocked before scene startup by CUDA cuInit999, also reproduced outside Isaac in a clean library environment. nvidia-smi and loaded driver/library580.178.04 agree; UVM use count0, sudo requires password. User has been asked asynchronously to reload only nvidia_uvm; cause of CUDA failure not assumed. No simulation or training remains running.

Selected fix: exact sphere/capsule geometry with roll-dependent nearest segment point, a measured stopping-angle estimate and bounded positive hip target correction only after real foot-ball separation and strike-pulse completion. Config state_conditioned_strike_v2.json; all v0.1.26 parameters remain unchanged. Two geometry/braking CPU contracts went RED(missing helper)→GREEN, including the recorded penetration and an internal-segment counterexample to endpoint-only geometry. Physical effect remains unverified. Nominal gate and original collision/termination criteria remain enforced.

Fresh focused review completed by hip_clearance_review. TwoP2 findings were fixed in one pass: (1) explicit strike-pulse time-window completion, beyond support-gated zero velocity; regression test_support_gated_zero_velocity_is_not_pulse_completion RED→GREEN. (2) shared case registry and explicit no-historical-baseline audit path, with v26 provenance read from92141f6; test_declared_combination_uses_original_initial_state_and_requested_delta RED→GREEN and altered actual initial velocity rejected. Final targeted CPU command: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_hip_clearance.py tests/test_combined_strike_audit.py tests/test_placement_strike.py tests/test_closed_loop_strike.py tests/test_closed_loop_acceptance.py →10passed in1.20s. No extra review loop.

CUDA remained999 on final direct probe; all19original frozen files unchanged. No owned simulator process active. Physical replay, final model selection, affected-condition regression, new combination evaluation and actual preview remain pending on CUDA recovery. The candidate remains local/unpublished; retain the successful v26 publication as the last verified version. See reports/2026-11-06-combined-strike-adaptation.md for the concrete prepared patch and required administrator action.
