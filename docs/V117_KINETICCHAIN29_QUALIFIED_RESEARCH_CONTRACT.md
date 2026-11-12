# KineticChain-29 — v117 research contract after preregistered AJ/AK/AL falsification

**Status (2026-11-12): research design + executable offline action-admission gate; not a trained human skill and not physically actuated.** Physical source evidence: `reports/2026-11-12-v117-AJ-AK-AL-original-G1-physics-wholebody-evidence.md`. Frozen model/physics code before experiments: `0fcb4af3a4152e0ce592c61781d5f0064d4d65fa`.

## Testable primary hypothesis

H1: A human-kinematic-conditioned, support-aware, **phase-continuous 29DoF full-body** generator, constrained by native contact-pair geometry and real robot dynamics, can produce more human-like approach, pelvis/torso–leg–opposite-arm timing and follow-through than original v32 and shallow CMU phase-references **without reducing** qualified goal, instep share, ball speed, 29-joint q/qdot/qddot/torque, collision, stability and 14s recovery performance. It must beat stronger robot-motion-only generation as well as a frozen v32 policy.

H0: Any apparent improvement is a ball-position or low-dimensional precontact collision artifact; phase-conditioned robot-only motion or better ball contact explains it without learning any authentic human spatiotemporal coordination.

## What the new data can and cannot supervise

- Human: six SHA-qualified marker C3D sequences, two subjects, 120Hz, 12 eligible wrist/shoulder/waist/thigh/ankle/toe marker streams, plus derived phase candidates. Their labels **do not include** synchronized ball impacts, force-plate GRF/CoP, true professional soccer identity or normative maximal instep shooting; retain missing-label masks.
- Robot: actual 200Hz simulator state, measured original29 qdot, estimated torques, foot support force proxies, COM, target ball/goal and actual contact events. Do **not** treat robot simulated force as human GRF or learn human ground-force truth from it. Raw contact-pair identity is not yet certified.
- If professional/elite labeled data remain unavailable, label the task explicitly *human motion-kinematics-informed robotics*, rather than claiming learned professional explosive mechanics. Do not use manually selected post hoc success trials as a training set or calibrate on frozen holdout.

## Concrete learnable whole-body action model (not yet trained)

Use a single temporally continuous phase latent `z_t` spanning **approach → support landing → backswing → hip/torso acceleration → instep impact → follow-through → landing/recovery**. The generator takes time-ordered human relative 3D markers and robot proprioception/foot support/COM state, action history, target-ball geometry and contact safety margin, and outputs all 29 joints with shared temporal latent, rather than a foot-only offset or a late 1mrad reference.

A candidate end-to-end target (after honest qualification) is the finite horizon `(q*_1:T, dq*_1:T, tau*_1:T, z_1:T)`, sampled at the robot controller cadence. Enforce monotone phase progress and couple feet and arms through physically valid centroidal angular momentum and base-support dynamics. Retarget human relative pelvis/shoulder, swing leg and opposite-arm *temporal features*, not absolute humanoid joint angles; include marker confidence and trial qualification masks. Human contact timestamp and athlete force loss terms must be **disabled when unobserved**. A learned real-human skill cannot be inferred from absent targets.

Safety-aware control should solve per-control-cycle constrained tracking in torque/acceleration space subject to original 29 q/qdot/qddot/torque and active support/friction/COP constraints, signed robot self-collision distance, certified PhysX native witness identity and chronological recovery observation. Collision distance and native contact path availability are prerequisites, **not rewards traded against goal success**. Even if all mathematical inequalities pass, the actuator interface must refuse without qualified measurement provenance.

## Implemented executable safety API now

`tools/v117_witnessed_horizon_admission.py` validates finite Nx29 arrays and ordered phase IDs, then uses `tools/v107_motor_admission.py` for every observed 0.005s frame. Missing native actor/collider pair, source witness or stance certification makes the whole trajectory **REFUSED**. The API always emits `physical_motor_command_issued=False`. Tests include missing witness, fake qualified labels, phase disorder, dimension/nan and short horizons. This is a **screen**, not a rollout controller; it does not prove controller feasibility under coupled whole-body dynamics.

## Pre-registration and non-leakage for the next real policy

1. Before optimization, freeze a fresh code SHA, original G1 physical assets, independent (train/validation/holdout) human subject/trial manifest, and robot ball pose/velocity/seed holdout. AJ/AK/AL/AG/AH/AI and all previously viewed failures belong to **development evidence** and may never become an unobserved test set.
2. Establish instrument calibration in an isolated real PhysX *positive* pair with collision collider paths, contact point/normal/impulse; a separated negative; then actual G1 wrist–hip case with repeat OFF/ON bitwise parity. Until that is certified, no new learned motor policy actuation.
3. Build a simulator-side whole-body retargeting feasibility baseline with correct original joint limits and stance + collision protection. **Never train purely to reduce joint MSE or cosine** and call that natural motion.
4. Compare preregistered original v32, original CMU, phase-shuffled human features, robot-only phase encoder, arm-locked, pelvis-locked, support-blind and the complete full-body model. Equalize speed targets/ball distributions/seed counts/compute and report all negative failures. Keep training dynamics and trial data disjoint from test.
5. Natural coordination metrics require continuous measured approach/support/contact/follow-through event timing, pelvis–torso phase lag, bilateral support transfer, counter-arm angular momentum and qualified expert judgment where feasible. Report time-aligned trajectory plots, not only offline global MSE.
6. Primary *hard* success remains conjunctive: legal instep impulse share ≥0.95, goal entry, ball speed proxy ≥3.5m/s, all original joint limits and safety constraints, verified complete 14s standing recovery and no non-foot hazard. Report speed as a measured robot proxy, **not professional-level** impact speed.
7. If any episode stops early, mark subsequent post-contact recovery as *censored*, not failure-at-rest or zero. Report per independent seed/scene, paired OFF/ON separately, uncertainty intervals and worst-case safety violations.

## Falsification from actual v117

AJ shows CMU reference can introduce qdot violation (peak 1.332× and 1.272×) despite goal, full recovery and legal instep; AK improves instep legitimacy but with *lower* ball-speed proxy; AL passes all frozen gates in every original controller. The first measured control effect remains at 1.405s, shortly before contact. None is adequate evidence for the primary hypothesis. Further high-quality human qualification and native manifold proof are non-negotiable prerequisites.
