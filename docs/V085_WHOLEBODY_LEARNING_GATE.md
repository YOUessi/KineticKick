# KineticKick v0.1.85 — Whole-body coordination learning contract and safety-driven training decision

**Status:** implementation-grade research contract, **NOT** a trained human-football controller and not approved for G1 physical rollout. The contact-truth readout remained uncalibrated in v85. The synthetic v83 phase-chain reference is not a training target, professional style label or demonstrated naturalness.

## Contribution to test, not yet claim

The scientific hypothesis is that a *human-trained, support-conditioned whole-body temporal policy* should produce more natural and effective G1 football locomotion and instep impact than (a) original G1 v32, (b) foot-only residual control, (c) geometry-only CMU phase priors, while meeting *every* original hard safety gate. A pass at 3.5m/s is **only minimum acceptable ball motion**, not a high-speed power shot. Compare naturalness and physical efficacy as independent primary outcomes and keep all failures.

## Trainable state/action contract

At each actual 200Hz simulation step, the observation needs at least:
`q[29], qdot[29], base SE(3), base twist, wholebody COM, foot poses/velocities, per-foot support + confidence, ball-relative pose/velocity, target line, previous phase, collision clearance by actor-pair/collider, contact source reliability flags`. Missing trusted contact-pair evidence is **not** replaced with equal-opposite net force. If GRF/CoP is not measured, supply `null` with an explicit missingness mask, not a guessed label.

The learned model predicts a distribution over event phase `z`, duration/transition uncertainty, whole-body `q_ref[29]`, `dq_ref[29]`, desired pelvis/root momentum and support-foot sequence. Proposed eight phases: approach, support plant, hip drive, torso transfer, swing drive, instep impact, follow-through and landing. The phase is **state/event-conditioned**, not an imposed animation clock. Right-leg dominant kicking should exhibit learned counter-rotation of trunk/arms, not simply a coordinated-looking sinusoid.

## Contact-safe controller design

Prototype an inverse-dynamics QP at each accepted controller step. Minimize joint acceleration and torque deviation from the learned intention while enforcing articulated dynamics `M(q)qdd+h(q,qdot)=S^T tau+J_c^T lambda`, support-friction pyramid, actuator torque/velocity/joint limits and the support-foot no-slip velocity constraint. Include a verified signed-distance conservative collision barrier for wrist/hip and other non-foot actors `d(q)+grad(d)·qdot Δt >= d_safe` only when the geometry and Jacobian can actually be established from the same G1 assets. **A barrier on unverified geometry is not a collision guarantee.**

Fail-closed on missing pair geometry, support confidence, invalid qdot vector, nonfinite observation, QP infeasibility or near-limit hardware conditions. Do not mask imminent collisions with a positive ball score. During R&D, maintain unchanged baseline as fallback; do not advertise physical safety for unverified execution.

## Human data qualification before training

Six existing CMU 120Hz football C3D clips are observational, provide only geometry/sequence phase cues, and lack verified professional player identity, synchronized GRF/CoP and measured instep/ball impulse. No human high-power force labels may be fabricated from these optical samples. Any human pre-training material must include license/consent, unique person and recording session IDs, timestamp and calibration uncertainty, plant/swing/ball visibility, 3D joint coordinate system, missing labels and provenance hashes. Train/validation/test split **by person and capture session**; do not tune on the frozen physical holdout. Label optical planting/contact cues with confidence, distinguish them from actual contact manifold ground truth.

The first model trained without GRF must be called **kinematic imitation only**, with force/power acquired solely from independently validated physical optimization and separate instrumented evidence. v85 does not trigger training because the required professional high-quality contact-qualified human corpus is absent.

## Locked evaluation matrix after qualified data arrives

Pre-register initial-condition distribution (actually perturb reset), held-out targets/ball offsets, unique seeds, physics rate, source and asset SHA, all original strict-gate thresholds, and blinded human evaluators before optimization. Equal budgets and independent identical initial states across: original v32; CMU proxy; full-body learned; leg-only learned; arms-frozen learned; phase reversed; no-support-risk constraint; full model with collision barrier. Score every physical run via 95%+ instep impact fraction, actual goal, 29 real qdot limits, nonfoot, recovered 14s and measured ball speed. Independently report motion naturalness via blinded videos, jerk/torque-rate and phase causality; **do not** call smoothness or offline MSE alone naturalness. Include confidence intervals clustered by held-out scene and human identity and all negative runs.

## v85 decisions and next external qualification

The 3x2 original-center native-reader paired controls give **3/3 exact parity**, but at 1.57s each ON run has a recorded foot-ball impact while the native contact reader returns no headers and no callbacks. Thus it is **uncalibrated**, and neither direct wrist-hip collision nor its absence has been proven. The CMU10_05 nonfoot failure remains real and repeated exactly. Next verify a native foot-ball positive control in a separate bounded isolated test scene with report API authored **before** physics creation, then test original USD reporting lifecycle/DirectGPU support without modifying the baseline physics. No contact geometry, no learned policy rollout. Keep the 200Hz true physical trace separate from a terminal annotation row.
