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

- [x] Diagnose: tools/run_placement_strike.py optional --contact-diagnostics records nonfoot body names, forces and positions without affecting controls or termination. Replay known failure once; compare physical outcome with v0.1.26 and explain collision cause.
- [x] Fix: implement the smallest measured-state control correction supported by diagnosis in an isolated helper/config; use actual failed geometry as a CPU regression fixture if geometry logic is needed. Keep original configuration behavior available.
- [x] Validate: independent known failure, affected original conditions and prespecified new combinations; one actual preview and a minimal ablation. No broad redundant suite.
- [x] Publish: exact results and limitations in reports/2026-11-06-combined-strike-adaptation.md, README and VERSION_CONTROL; targeted checks, version commit and upstream push.

Diagnosis confirmed after CUDA recovery: paired, equal-and-opposite PhysX forces on pelvis and right_hip_roll_link reproduce the old failed combination exactly. The diagnostic simulator clock includes 0.010s initialization; the original episode recorder remains authoritative. No ground or ball body contact caused this termination. Existing physical rows of the v26 failure and the new guard-disabled replay are exactly equal.

Selected fix: exact sphere/capsule geometry with roll-dependent nearest segment point, a measured stopping-angle estimate and bounded positive hip target correction only after real foot-ball separation and strike-pulse completion. Config state_conditioned_strike_v2.json retains all v0.1.26 adaptation parameters. Its preparation metadata and hash are frozen before physical testing; current acceptance status is maintained in the report, not by rewriting evaluated configuration bytes.

Fresh focused review completed by hip_clearance_review. Two P2 findings were fixed in one pass: explicit strike-pulse time-window completion beyond support-gated zero velocity, and shared case registry/no-historical-baseline audit path with v26 provenance read from92141f6. Targeted CPU command: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_hip_clearance.py tests/test_combined_strike_audit.py tests/test_placement_strike.py tests/test_closed_loop_strike.py tests/test_closed_loop_acceptance.py → 10 passed in1.20s. No extra review loop.

CUDA recovered on 2026-11-06 after a local GUI terminal ran the previously requested non-forced nvidia_uvm unload/reload; an independent cuInit probe returned0. No driver reinstall, reboot or physical-asset change. The CUDA failure cause itself was not proven.

The first physical candidate passes the known failed combination: contact1.585s, outgoing horizontal speed3.4549m/s, actual right instep, goal, full14s, final standing9.77s. Four post-contact hip corrections at1.60–1.66s prevent the self-collision; minimum actual geometry clearance+0.03894mm versus−0.06156mm without guard. The small clearance is a limitation, not a guaranteed safety margin. Candidate and ablation physical states match exactly through1.60s (320rows). Independent original-criterion/provenance audit passes. The remaining old cases and four prespecified new combinations were subsequently run; outcomes and refinements are recorded below.

First complete evaluation: guard-only v2 passes all9 known conditions, and3/4 new combinations. heldout_right_far_slower has an ambiguous toe-biased normal at1.61s, and penetrates the same hip geometry at1.675s despite saturated braking. It is now development data. The swing IK is rejected after1.46s because orientation error exceeds its original limit; no surface criterion is changed. Next bounded configuration trial lowers the state-scaled pre-contact foot target from−0.025 to−0.040m to increase instep exposure, and increases geometric target clearance0.5→1mm / bounded hip correction0.25→0.35rad because the recorded stopping prediction exceeds the original correction limit. Test this condition first; do not run a broad matrix until it physically passes. At most two informed configuration trials; if both fail, diagnose IK feasibility rather than more parameter guessing. Original v2 and all first-round outputs remain frozen.

Before inspecting refinement results, reserve two additional confirmation combinations excluded from this refinement: heldout_confirm_left_far_slower = ball(−0.020,−0.020)m / worldvy+0.100m/s; heldout_confirm_right_near_slower = ball(+0.015,+0.015)m / worldvy+0.150m/s. Their outcomes must be reported separately from the original four combinations now already inspected. No tuning on these while labeling them held out.

Refinement A (v3): larger post-contact clearance prevents termination and recovers7.465s, but the contact normal remains ambiguous. Lowering the requested foot height did not fix contact because IK still rejects the command. Stop height guessing. A CPU replay of the actual1.50s pose reproduces full-target orientation error0.3185rad; half the requested translation reaches0.1063rad with the same original joint bounds, while zero correction wastes available reach. Refinement B (v4) restores the original height and adds a six-iteration local feasibility search over target fraction. The original full foot-orientation requirement0.12rad and position0.025m stay unchanged. Full feasible commands pass through unchanged; only failed requests search toward zero. A real-URDF regression goes missing-method RED→3 targeted tests GREEN, with independent original FK verifying useful world displacement, orientation and limits. This is a local kinematic correction, not a learned human coordination model.

Refinement B physical trial passes: real instep at1.605s, goal, final standing9.59s;22 projected control frames. Fresh focused review of this new feasibility change reports no actionable correctness findings. Validate affected old conditions first, include actual rendering of the newly fixed combination, then run the remaining declared combinations with the same source/config. Original first-round holdout3/4 remains reported; after refinement those four are development data.

Final frozen v4/source audit: 14/15 total, known9/9, reused initial evaluation4/4, separately reserved confirmation1/2. Original frozen comparison1/7 versus current7/7. Actual rendered episode independently passes, video21.04s and contact/recovery frames inspected.11 targeted CPU tests pass;19 original file hashes unchanged. Version publication is on version/v0.1.27 with upstream tracking; previous version branches and main are retained. The unresolved confirmation failure is explicitly preserved, not counted as task success.

The final diagnostic reproduces the independent confirmation failure and pairs left_knee_link/right_knee_link forces425.20N at episode1.55s. Right hip/pelvis clearance is positive, and torso tilt is under5deg: this is a distinct pre-contact two-shank collision. No further tuning used that confirmation condition. General adaptation remains incomplete; next research requires collision-aware approach/swing paths rather than another endpoint-only adjustment. All owned simulator runs have completed.
