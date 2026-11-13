# KineticKick v0.1.154 frozen protocol — actuation-aware floating-base whole-body coordination gate

Parent SHA `c4777d50265f90bd052b449f5fe6497c8b0f35c5`. New version worktree is isolated from the dirty original root and all other versions. This protocol is committed **before** optimization experiments or implementation. A module must verify the frozen protocol against a named Git SHA (not mutable working-copy content).

## Hypothesis
A full-body football coordinator should be **feasibility-gated against coupled floating-base rigid-body dynamics**, ground contact, friction, all original 29 joint actuation/position/speed limits and non-foot collision clearance, instead of accepting weak ~1mrad posture residuals or a kinematic QP. The first executable baseline is a **35-dimensional constrained acceleration/torque/contact-force quadratic program**, with external mass matrix, bias, foot Jacobian and derivative term, measured contact attribution, and link separation gradients. It is a *CPU numerical infrastructure test*, not a natural kick or physical proof.

## Hard invariants
1. State `[floating base 6D, original 29 joint velocities]`; solve `M a + h = S^T tau + J_L^T f_L + J_R^T f_R`. Foot stance acceleration constraints `J a + Jdot_v = 0`; single or double stance explicit.
2. Unilateral **signed normal** `f_z >= 0`, world-Z contact frame and Coulomb friction pyramid `abs(f_x), abs(f_y) <= mu f_z`. Do not infer signed GRF or CoP from history's foot net-force magnitudes.
3. All 29 joint torque bounds, one-step joint q and qdot, acceleration ceiling, and collision-separation second-order linearized inequalities. Reject missing, uncalibrated or wrong-tick external Jacobians; missing collision guards can never certify real motion.
4. Track full 29DoF phase-target acceleration, including legs, waist, arms, but **all target patterns in this batch are deterministic synthetic fixtures**. No human professional style, GRF, contact instant, ball speed or instep impulse ground truth exists. Real use is blocked until exact-source M/h/J/geometry and contact reporting are available.
5. Solver outcome must be independently audited by an implementation not calling the optimizer: exact dynamics residual, stance residual, 29-actuator constraints, friction, collision guard, floating-base dimension, and provenance. No apparent goal / recovered root attitude can waive any physical gate.

## Frozen CPU experiment
- Six named action phases × two stance modes × four seeds = **48 nominal synthetic cases**. Each generated from *unchanged* synthetic mass/contact fixtures; every solution either passes independent constraints or remains a failure.
- Additional adversarial negative cases cover missing M, missing ground-only provenance, omitted collision jacobian, wrong dimensionality, NaN, zero support, infeasible clearance, and overdriven horizontal base balance. Include missing-human-qualification rejection.
- Compare with deliberately unconstrained phase accelerations in the very same fixture; quantify unsafe cases and effort tradeoffs. Treat synthetic successes as computational feasibility *only*, not KineticKick physical trial success.
- Historical 200Hz source checksum and original row counts 2800+2800+1028 are re-checked as **read-only**; historical support sensor magnitude cannot supply ground-only signed normal or moments, the required spatial matrices or observed collider separation. No interpolation of post-terminal state.
- Independent validation: source freeze SHA, deterministic matrix, negative checks, tests, full CPU regression, Markdown failure report, remote SHA verification. Generated JSON/log/PNG remain gitignored locally.

## Safety and stopping
No new IsaacLab/GPU launches while v120 PID or any other foreign CUDA context exists; do not kill or preempt. If disk under 20GiB, abort any physics. Original CMU/C3D/NPZ/weights/PhysX JSON immutable. The 6 paired real OFF/ON trials, native PhysX collider manifold, high-speed instep legality, 14s recovery and real human football dynamics remain **not tested here**. Freeze algorithm/weights before any unseen real evaluation.
