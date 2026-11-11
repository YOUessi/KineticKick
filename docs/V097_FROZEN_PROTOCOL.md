# KineticKick v0.1.97 immutable preregistration — six original G1 CUDA PhysX trials

Parent **f143a1cc70f5e5c92c6ed2c803c16b1a1934faea** (v96), frozen before physics on 2026-11-12.

## Hypothesis and scope
Does a pre-existing 120Hz CMU optical phase proxy (10_02,10_05) cause robust improvements over the unchanged v32 controller under two independently preregistered ball positions? This is a *falsification and safety* study; it cannot certify professional technique, direct collider manifolds, or a new learned control policy. The historically riskier 10_05 is explicitly a **negative control**. Comparison is within ball placement at the same seed; the seed does **not** implement independent perturbations to root/joints, so it cannot establish stochastic generalization.

**Before viewing any new results:** G=(+0.035,-0.095)m with velocity_y=+0.015, seed=1409; H=(-0.045,+0.095)m, velocity_y=-0.015, seed=1423. Each runs in order v32, CMU10_02, CMU10_05. Six fixed jobs, at most 14s original 29DoF free-root 200Hz G1 GPU PhysX each, max180s wall time. Source configs are unchanged and verified by SHA; no actuator gain/physics, ball/foot dynamics, or collision threshold changes.

## Safety and reproducibility
GPU is exclusively serialized by existing shared flock /home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock. Verify disk >=20GiB and no external CUDA computation before *every* episode. Require clean tracked tree, exact frozen protocol and executor bytes from Git HEAD, write-once JSON/log/audit in .gitignore reports. Stop on infrastructure/physics certification failure, preserving partial outputs. Do not kill any processes except one's own child under timeout. Do not delete or overwrite original CMU, C3D, NPZ, weights, models, experiments or PhysX.

## Hard scoring (all conjunctive)
Foot/ball witnessed, instep impulse >=95%, through goal, actual 5-110ms ball-speed proxy >=3.5m/s, each of actual 29 measured joint velocities <=original limits, peak nonfoot net force strictly <20N without nonfoot termination, complete 14s physics and stable recovered. All partial/censored/early-stop runs are failures of full task, **not** evidence of unobserved post-contact dynamics. Original ball speed is a proxy, not a power metric. Subset signs, antagonistic link forces and empty native report must NOT certify actor0/1 collider0/1 or rule out self-collision.

## Independent analysis
Separate post-evidence script rechecks SHA of each original JSON/log, contiguous 200Hz real state incl terminal-annotation excision, true 29DoF speeds/COM/ground-force/support vs trunk/arms/leg lag, event coupling and all hard gates with failure figures. Evaluate paired three-way failure types and report signs even if all fail. Avoid fitting thresholds/policy to G/H.
