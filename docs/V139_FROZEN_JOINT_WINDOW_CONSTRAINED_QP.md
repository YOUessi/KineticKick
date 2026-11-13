# v139 preregistration — window-wide constrained 29DoF coordination (CPU only)

**Parent**: `version/v0.1.138` SHA `a3a119932978be7e18b3483e06add87bb89dbd9c`. **Date**: 2026-11-13. Original G1 URDF SHA-256 is fixed in `configs/v139_frozen_joint_window_qp.json`. This protocol must be committed independently before implementation or experiments.

## Why this experiment
v138 used an exact one-instant 29DoF `qdot` witness and a fixed `C2` bump. The 1 and 3m/s targets failed a descriptive 40rad/s² screen in every one of 30 attempts. That failure might be due to the **restricted bump family**, not genuine time-domain infeasibility. This new falsifiable control-oriented candidate solves the *entire compact 200Hz window* in one optimization, not instant LP plus independent smoothing. It remains engineering synthetic kinematics, not skilled human motion learned or real dynamic execution.

## Preregistered grid
The exact 45 cells are `5 amplitudes (0,1,2,4,-1) × 3 relative ankle-link-origin x velocity targets (0,1,3m/s; y=z=0) × 3 compact halfwidths (0.4,0.8,1.2s)`. Use only the v135 fixed seed reference, original 29 G1 joints and URDF, v137 mass COM / analytic Jacobian. No optimizer selection, constraint/gate change or tuning using original v54 failed replay or 45-cell outcomes. Keep all attempts and reject unresolved solver states. Contact time 2.95s is synthetic and must never be claimed real.

## Decision and true scope
Per joint optimize four free coefficients of `W*s^(odd_power)*(1-s²)^4` (`odd_power∈{1,3,5,7}`, `s=(t-2.95)/W`, zero outside `|s|<1`). Value, first and second derivatives are zero at edges; optimize all coefficients *simultaneously*. At the center, `q` is unchanged and first basis derivative equals 1; other basis derivatives vanish. Use **one sparse convex QP** with sampled hard original29 URDF position and velocity limits, frozen 40rad/s² **research-only** acceleration screening as a *QP constraint*, exact 3D linearized relative ankle-link-origin velocity at 2.95s, center linearized virtual-ankle-relative COM cap 0.1m/s and ±0.25rad/s center arm velocity-departure cap. Quadratic objective integrals on reference perturbation q, qdot, qddot plus ridge as in JSON. Solve with OSQP, no slack or hidden relaxations. If infeasible, report as negative, not tuning a different threshold.

## Independent evaluation / interpretation limits
At **all 200Hz samples of the complete 14s**, recompute original29 q, dq, ddq analytical derivatives and position/velocity/accel max violations independently of the sparse matrices; evaluate geometric wrist/hip clearance and virtual left ankle / COM coordinates in [-0.4s,+1.25s] for candidates with postsolve witness. Report every one of 45 cells, including rejected and solver errors. Cross-check analytic temporal derivatives versus centered finite differences, immutable config byte SHA, and physics fail-closed. Investigate virtual stance root, C2 boundaries, and negative controls. A mathematical QP solution is NOT a motor command, no measured stance patch, GRF, CoP, base rotation, friction, torque feasibility, actual instep contact impulse, ball speed, 14s physical recovery, or professional human skill.

**Physical work blocked** while prior v120 test occupies GPU. Do not kill it. If GPU frees, separately freeze and execute v133/v134 native contact positive/negative fixtures and unchanged original v54 OFF/ON. Do not claim original left-wrist/hip collider path from URDF geometry.

## Safety, reproducibility, release
Only new isolated `version/v0.1.139` branch from clean v138, immutable protocol SHA committed **before code**; raw per-run JSON, figures and large assets remain local via `.gitignore`. CPU-only experiment can be run while v120 GPU pytest lives. Do not modify old CMU/C3D, PhysX traces or weights, and no forced push. A research pass is only a bounded *synthetic kinematic* screen, NEVER strict physical success.
