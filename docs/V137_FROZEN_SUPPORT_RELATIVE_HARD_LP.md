# v137 — Pre-registered support-relative whole-body feasibility (CPU, no actuation)

Parent is verified v136 commit `4cc754c1230e488b8ec81526f14b96125596fd22`. This document and `configs/v137_frozen_support_relative_velocity_lp.json` must be **committed before** implementing the optimizer or observing new v137 results. A separate Git commit SHA permanently locks the matrix.

## Research question

v136's *weighted* task residuals cannot establish hard kinematic infeasibility. Can the original 29DoF G1 URDF joint-velocity box reach a specified **right-ankle-link origin relative to left-ankle-link origin** velocity, with a purely kinematic virtual left ankle translation anchor, an inertial-mass COM-relative velocity cap, and constraints preserving arm motion? This is an **engineering reachability question**, not a physical kick or demonstration of human-like power.

## Prior-fixed comparisons

Five synthetic full-body v135 sources (multipliers 0,1,2,4,-1), at 2.25, 2.95, 3.60 s. Four exact relative ankle-origin velocity targets in root +x: 0, 0.5, 1, 3 m/s; y,z must equal zero. Compare free versus bounded COM relative velocity (all 3 components ≤0.1m/s), and free versus arm angular velocity deviation from the corresponding v135 reference (all 14 arm joints ±0.25rad/s). Do not retune the grid after seeing results.

For pose q, let J_L, J_R be 3×29 ankle-origin position Jacobians, J_C be mass-weighted 37-inertial-link COM Jacobian. A fictitious root *translation* rate −J_L qdot locks the left ankle origin, hence relative right ankle rate is (J_R−J_L)qdot and COM rate in that moving frame is (J_C−J_L)qdot. Use 29 original joint-velocity hard bounds. Solve exact convex linear feasibility with SciPy HiGHS for each target and maximize +x for each frozen condition (60 maximum envelopes and 240 feasibility calls). Numerical solutions must independently satisfy hard inequalities and equalities at prespecified residual tolerances, or the run is INVALID, not PASS. Derivatives must agree with independent symmetric differences at 1e−6rad perturbation (tolerance 2e−5).

A true fixed sole requires orientation, a measured contact patch, CoP and friction forces, not a virtual ankle origin. Actual root orientation, torque, dynamics, actual instep position, ball speed, contact impulse, non-foot collision and full recovery are not demonstrated. A large LP envelope can **never** be called executable. No motor API is permitted and all rows must explicitly report physical certification false. Failed constraints and absent evidence must remain visible.

## Negative controls and acceptance

Tampered protocol and URDF must fail closed; target not in frozen grid must be rejected; size, NaNs and insufficient frames rejected; zero synthetic motion is not a true stance calibration; exact 29DoF velocity box and exact y,z are mandatory; missing GRF, native pair, motor torque, real floating base rotation, and human pro-kick GRF are hard NOT-ATTESTED. Preserve v54 negative evidence without writing to any prior data. Run v137 tests and broad v127–v137 CPU regression; do not block or cancel the existing GPU-bound v120 test.

Local results and plots use exclusive new paths, are not committed. Commit only protocol, reproducible tools, tests, and written analysis on branch `version/v0.1.137`; verify remote exact SHA after push. No inference from v54 failed trajectory is used to alter parameters.
