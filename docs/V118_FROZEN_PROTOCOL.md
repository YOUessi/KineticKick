# KineticKick v0.1.118 — committed preregistration for actual PhysX and whole-body falsification

**Parent Git SHA:** `24857ba8e5544dcf72ce904d38b2d5699e6426fb`.
Prepare a clean independent version/v0.1.118 worktree. Commit this immutable matrix and runner before executing any physics. No historical raw JSON, original controller, CMU/C3D, checkpoint, or existing branch may be changed.

## Frozen independent experiment matrix

Two held-out geometry/seed choices fixed without observing their outcomes: AM offset (+0.022,+0.056)m, ball velocity delta +0.006 and seed 3001; AN offset (-0.067,+0.012)m, velocity delta -0.009 and seed 3049. For each: original v32, original CMU10_02 and original CMU10_05. For v32 only, repeat OFF then ON the existing read-only G1 Tensor observer. **8 true physics executions; 6 independent scene×controller cells; 2 parity repeats not counted as independent data.** No post hoc re-selection, threshold changes or controller tuning.

## Certification and safety

Original 29DoF free-root Unitree G1, unchanged committed original v080 and v098 runners, SHA-qualified RL and recovery checkpoints, 200Hz CUDA PhysX and 14s recovery. Per-shot source SHA, controller SHA and human-proxy source SHA checks; strict original hard-gate evaluation for goal, observed instep impulse share >=95%, ball speed proxy >=3.5m/s, all actual individual 29 qdot within original limits, nonfoot >=20N rejection and observed standing recovery 14s. Identical all-field 200Hz OFF/ON parity required or the rest of the matrix stops. Non-foot wrist–hip native collider-pair uncertainty is a *separate* unresolved certification blocker.

Global GPU lock `/home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock` is mandatory, other GPU compute jobs forbidden; disk free must be >=20GiB before each child; each child watchdog=140s, only child group terminated on timeout; never kill external processes. New evidence write-once in ignored local v118 JSON/log/audit files. A failed or incomplete launch is not a scored shot.

## Independent analysis frozen before experiments

Use independent measured 200Hz 29-joint qdot, pelvis/root COM and bilateral support force proxies, opposite-arm/hip/leg mechanical work, joint hazard first-event time/duration, true 0.4s pre and >=1.25s post contact windows and actual 14s recovery. Compare all 4 matched CMU-versus-v32 conditions, and apply a deliberately falsifiable **lead-before-impact and after-impact safety** criterion. Do not substitute raw joint MAE or cosine MSE for learned naturalness. All failed and censored observations retained.

## Native collision and human evidence boundaries

Known isolated Tensor positive is NOT a native actor/collider contact manifold. Native positive/negative and left-wrist–left-hip paths are required; absent path/point/normal/impulse is **NOT CERTIFIED**. Six existing real C3D trials (two subjects) contain no verified elite-kicking contact timestamps, force-platform GRF/CoP or foot-ball impulse; do not train or actuate an unqualified generated policy. New CPU-only research code may investigate interpretable phase/support constraints but never command motors without native and torque/support witness.

Document source and raw SHA, actual outcome, limitations, all tests and remote Git HEAD; never rewrite this committed protocol after seeing outcomes.
