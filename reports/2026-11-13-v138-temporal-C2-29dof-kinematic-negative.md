# KineticKick v0.1.138 | Frozen 29DoF temporal C2 coordination feasibility and hard-negative audit

**Date:** 2026-11-13 (Tang, Asia/Singapore). **Parent:** `version/v0.1.137` / `aabc122da8497583a8d1322185d7c8d8c96f7d70`. **Pre-frozen protocol SHA:** `c032c6f2de72ef0ace3d4e00834dd265164b98f0`. **Target:** original G1 with 29DoF, original URDF SHA-256 `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`. This is a **CPU-only synthetic kinematic experiment, not a new G1 PhysX experiment, learned football skill or authorized motor command.**

## 1. Actual safety/source checks BEFORE implementation

- Tang online, `/home/you/projects/kinetickick` available, ~30GiB free (system root ~97% full), above minimum 20GiB.
- Existing `/home/you/projects/kinetickick-wt-v120` pytest PID 242644 was consuming ~260MiB RTX4090 memory, >15h in `R` process state; full test log empty at preflight; no kill, restart or new GPU/IsaacLab job.
- Root worktree `version/v0.1.64` had unrelated untracked content. v131 worktree also had `?? tools/v131_shared_latent_29dof.py`. **Neither worktree was modified.**
- Source `version/v0.1.137` worktree clean; verified `version/v0.1.138` not used remotely or locally, created new isolated worktree `/home/you/projects/kinetickick-wt-v138`.
- Protocol JSON + Markdown committed **before** solver or tests. All pre-existing CMU/C3D/models and original JSON/NPZ/PhysX traces untouched. The generated audit JSON and two plots are ignored under `reports/` on Tang.

## 2. Falsifiable method, not the prior 1mrad reference

v137 proved **instantaneous** 29DoF relative-ankle link-origin x velocity feasible, but its LP vertices saturated velocity constraints and were discontinuous. This batch used **the prior frozen five full-body synthetic references**, not v54 failure trajectory labels. There are five full-body seed amplitudes (0,+1,+2,+4,-1); for each, solve 29DoF minimum weighted velocity departure for right-ankle-origin relative targets 0/1/3m/s, hard original URDF velocity limits, COM-relative velocity cap 0.1m/s and arm velocity deviation cap 0.25rad/s at the artificial `t=2.95s` proxy. CPU SLSQP, initialized with an exact hard-feasible LP witness, must pass post-optimization equality/bounds/COM checks.

To test **time continuity**, do not teleport a one-step qdot. For each of those 15 fixed qdot targets and three frozen halfwidths (0.10, 0.20, 0.40s), construct

`q_new(t) = q_v135(t) + width * Δqdot * h((t-2.95)/width)`,

where `h(s)=s(1-s²)^4` for `|s|<1` and is zero elsewhere. Analytic first/second derivatives are used, so the joint trajectory is **C2** at the compact support boundaries; its exact center qdot equals the optimized target, with q position unchanged at center. Every candidate spans the 14s 200Hz joint timeline (2800×29), though the 14s evolution is only synthetic. The 331-frame [-0.4,+1.25]s window undergoes original-URDF FK/37-inertial-mass COM, virtual left ankle anchor, right ankle relative motion, and four original URDF wrist/hip capsule geometry screening.

**Limitations:** The left-ankle origin is not a real planted sole. Virtual translation cannot provide base rotation, support polygon, GRF, CoP, slipping, floating-base dynamics or torque. Ankle link-origin speed is not instep/ball speed. The descriptive **40rad/s² acceleration screen is not a certified G1 actuator specification**. Positive URDF cylinder/capsule separation is not native USD/PhysX manifold. These are deliberately preserved distinctions.

## 3. Complete frozen grid and the actual negative results

All **15/15** minimum-change optimizer solutions converged with post-solve checks; exactly **45/45** registered 14s synthetic candidates were evaluated. No failed cell was dropped or replaced.

| 200Hz 14s candidate outcome (non-mutually-exclusive warnings) | Out of 45 |
|---|---:|
| Passed all **descriptive CPU** position, velocity, 40rad/s² acceleration and 20mm four-pair warnings | **12** |
| Any original 29DoF joint position limit violation | **11** |
| Any original 29DoF joint velocity limit violation | **0** |
| Any descriptive 40rad/s² acceleration screen violation | **33** |
| Any four-pair URDF conservative proximity warning below 20mm | **1** |
| New true PhysX rollout / strict goal+impulse+speed+recovery certified | **0 / N/A** |

Per-target: 0m/s center-ankle target produces **12/15** CPU-screen passes; 1m/s and 3m/s produce **0/15** each. Per halfwidth (each has five seeds × three velocity targets): **4/15 CPU-screen passes at each 0.10, 0.20 and 0.40s**. Peak *analytic* joint acceleration across all 45 candidates ranges 0.0415–1154.25rad/s²; the worst peaks by width are 1154.25 at 0.10s, 569.76 at 0.20s and 285.39rad/s² at 0.40s. Longer smoothing reduces peak acceleration but increases joint position excursions: joint position violations by width are 0/15, 5/15 and 6/15 respectively. This is a concrete, frozen **smoothness vs position feasibility trade-off**.

The maximum original-URDF joint-speed utilization over all candidates was **0.8428**, despite the large acceleration and position failures. The frozen 3m/s center velocity targets required maximum individual joint qdot changes of **15.31–26.14rad/s** relative to the untrained v135 full-body reference depending on amplitude. Peak *extra* window right-minus-left-ankle relative path excursion reached **0.301m**; peak extra virtual-stance-relative COM path difference reached **0.120m**, neither indicating real support stability. One `+2` amplitude, 3m/s, 0.40s attempt had four-pair URDF capsule margin **6.604mm**. That is a **proximity warning, not a proven PhysX self-collision**.

Two SciPy SLSQP warnings were emitted during intermediate clipped optimization steps; final 15 solutions were **not** blindly accepted on solver status alone: independent equality, joint velocity bounds and COM inequalities were checked against frozen `2e-5` tolerance. Source SHA, retained attempt counts and CPU-only provenance checked by regression.

## 4. Research interpretation and hypotheses

1. v137's exact qdot reachability does not imply a **time-continuous dynamically admissible 200Hz** whole-body soccer action. At every nonzero ankle-origin target the imposed 40rad/s² screening fails at least once, while changing the bump duration produces a separate joint-position issue. This is a negative result **for this frozen bump family**, not a theorem that all G1 controllers are incapable.
2. A true whole-body method must jointly parameterize support transfer, coordinated hip/torso/arm sequence, swing-foot path and timing, center-of-pressure/friction and recovery with *jerk/torque-aware* temporal constraints. Instantaneous task-space LP and separate joint-level smoothing do not replace physics.
3. A defensible next policy would learn a **shared whole-body phase/synergy latent** from qualified multi-frame human skill data while using original29 constrained whole-body inverse dynamics or an optimization-based controller with floating base. Foot collisions/ball impulse and 14s balance must remain independent **strictly verified** gates, not objectives that can be traded away for visually plausible trajectories.
4. v54's left-wrist/left-hip native PhysX contact actor/collider/manifold remains **UNKNOWN**. v138 performed no new native callback replay, due GPU mutual exclusion. Neither URDF 6.604mm proximity nor original equal/opposite net force is a native contact pair certificate.

## 5. Verification, reproducibility and physical test plan

- Test `tests/test_v138_temporal_full29.py`: 39 passed (byte-locked protocol, independent parent SHA, original URDF tamper, analytical derivatives/FD, negative controls, arm/COM/qdot hard constraints, genuine 14s 200Hz C2, 331-frame window and all 45 attempts, physical fail-closed).
- Extended targeted **CPU-only v124–v138 regression**: **524 passed, 2 skipped, 2 SLSQP clipping warnings, 0 failed**, 16.76s. Two skipped tests must **not** be represented as full all-project or GPU test completion; the long-running old v120 GPU pytest remains separate.
- Code: `tools/v138_temporal_full29.py`, `tools/v138_plot_temporal_negatives.py`, `tests/test_v138_temporal_full29.py`; protocol `configs/v138_frozen_temporal_coordinated_29dof.json` and `docs/V138_FROZEN_TEMPORAL_FULL29_C2_AND_STANCE.md`.
- Raw ignored outputs on Tang: `reports/v138_temporal_audit.json`, `reports/v138_temporal_analytic_accel.png`, `reports/v138_temporal_joint_and_collision_negatives.png`, `reports/v138_cpu_regression.log`.
- Reproduce on Tang in isolated v138 worktree:

```bash
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v138_temporal_full29.py --output reports/v138_temporal_reproduction_NEW.json
$PY tools/v138_plot_temporal_negatives.py --audit reports/v138_temporal_reproduction_NEW.json --stem reports/v138_reproduction_NEW
$PY -m pytest -q tests/test_v12[4-9]*.py tests/test_v13[0-8]*.py
```

**GPU-dependent next gate:** only after verifying other GPU user is finished and disk free >20GiB, run v133/v134 revised *native vector* PhysX contact callback in isolated positive and negative collider fixtures; validate runtime USD actor paths, phase time, vector impulse, same-step q and ON/OFF physics state equality; reproduce original v54 three controllers unchanged. Subsequently freeze genuine floating-base and real stance constraints, then execute a multiseed unseen-position real PhysX matrix with strict instep impulse ≥95%, legal goal, ball speed ≥3.5m/s, every original29 qdot, zero nonfoot collision, full 14s recovery. **Until then no new policy can claim strict shooting success.**
