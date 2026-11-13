# V138 pre-registered temporal 29DoF contact-proxy feasibility protocol

Protocol parent: `aabc122da8497583a8d1322185d7c8d8c96f7d70` (version/v0.1.137). The **entire protocol must be committed before implementation**, and later runs must prove byte equality to this commit using `git show`.

## Research question and explicit limits

v137 found instantly feasible right-ankle-*origin* relative velocities up to 3 m/s using a virtual translating left-ankle anchor, but its LP witnesses saturated actuators and had hypothetical 5 ms acceleration discontinuities. This batch tests whether **minimum weighted joint-velocity change** and an analytically C2-joined finite-duration perturbation can close that discontinuity without joint position, velocity, acceleration-proxy, stance-relative COM or limb-collider warnings. This is a falsifiable CPU engineering study, **not** motion learning, real instep/ball collision, floating-base stabilization, motor admissibility, genuine torque or professional kick evidence.

## Fixed matrix and baselines

Exactly 5 synthetic v135 multipliers `[0,1,2,4,-1]` × 3 instantaneous support-relative right-ankle **link-origin x velocity** targets `[0,1,3] m/s` under COM-relative 3-axis velocity cap 0.1m/s and arm-deviation cap 0.25rad/s: 15 qdot quadratic objective SLSQP seeds. For each seed attempt halfwidth `[0.10,0.20,0.40]` seconds: exactly **45 predeclared attempts**, no replacing failures. Baseline synthetic v135 is reconstructed from SHA-pinned v135 file and has 2800 steps @ 200Hz over 14s; nominal proxy at 2.95s.

Required equality at center: `(J_right - J_left) dq = [target,0,0]` at that q, and velocity bounds from original 29DoF G1 URDF. For each solution minimize weighted squared distance to preexisting synthetic dq; solver failure must be reported honestly as unqualified. Base translation cancels only the left ankle **origin**, not left-foot contact patch orientation or friction.

Perturbation per joint: `q_new(t)=q_v135(t) + width * (vstar - dq_v135(tc)) * h((t-tc)/width)`, compact `h(s)=s(1-s²)^4` inside |s|<1, zero outside. Use analytic `h'`, `h''`; center q is unchanged, center dq becomes vstar, endpoints C2. Compare q, dq and ddq against original hard position/qdot and a **descriptive** 40rad/s² qddot screen, not real motor dynamics. Check 200Hz sampled trajectories and mathematically verify continuity/derivative. Report screen violations, not just success rates.

## Source and window evidence

Use SHA-pinned original paid URDF (not user-created model), 37 inertial masses and original joint order. Compute full 14s q/dq/ddq, plus left-ankle virtual-frame COM and swing-ankle relative trajectories in frozen [-0.4s,+1.25s] window. Assess the original four URDF collider capsules, respecting that positive conservative URDF separation is **not** runtime PhysX manifold truth. 20mm proximity is a warning, never a certified collision. Separate unknown evidence from negative physics outcomes.

## Audit, reproducibility and gates

No CPU candidate authorizes motor execution. Do not change v54, v52, v120, v131 uncommitted state, original C3D/motion/NPZ/30MB JSON/media, any PhysX evidence, or unrelated worktrees. Save raw numeric audits and PNG under ignored Tang `reports/v138_*.*`; only commit script/config/docs/tests/concise summary. No new PhysX while GPU owner PID 242644 continues or disk free <20GiB; no kill. Treat right-ankle origin velocity as **not** instep speed or ball speed. Strict true PhysX success requires ≥95% legal instep impulse, legal goal, ball speed ≥3.5m/s, original29 qdot, no nonfoot contacts and verified 14s recovery. HUMAN professional force learning and genuine GRF unqualified.

## Required tests and report

Test frozen SHA immutable, wrong-URDF fail, input shape and nonfinite rejection, bump endpoints and centered derivatives, target equality and limits, zero-target negative, arm COM constraints, all 45 attempts retained, 200Hz trajectory stitching/accel, no false physical gate; run CPU regression; preserve failures and disclose skipped GPU tests. Output provenance, counts, key worst cases, and reproduction instructions, then push **only** `version/v0.1.138` and independently verify remote SHA.
