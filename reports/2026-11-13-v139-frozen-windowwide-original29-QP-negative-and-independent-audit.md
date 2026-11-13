# KineticKick v0.1.139 | Preregistered window-wide whole-body constrained QP and independent negative audit

**Date:** 2026-11-13. **Parent:** `version/v0.1.138` / `a3a119932978be7e18b3483e06add87bb89dbd9c`. **Protocol committed BEFORE implementation:** `eae275db0695ad6df6ca89a98192817a58e95bcd`. Original 29DoF G1 URDF SHA-256: `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`.

**Evidence boundary:** All **45 new trials are CPU-only synthetic kinematic QPs**. **New real IsaacLab/PhysX G1 shots: 0.** No human elite-instep learning, no motor operation, no measured ground reaction forces, no valid native contact manifold, no claim of physical speed/impulse or successful 14s recovery.

## 1. State preflight and safety

- Tang online at `/home/you/projects/kinetickick`; root worktree `version/v0.1.64` had untracked material `?? =100===`. Numerous older version worktrees exist. Existing `v131` worktree had unrelated untracked code. None touched.
- Original `v138` worktree is clean; new independent `/home/you/projects/kinetickick-wt-v139` created from `a3a1199` after confirming v139 was absent via GitHub branch search.
- `df -h`: only about **30GiB free** on root (97% used), above the user's 20GB protection cutoff. Original CMU/C3D, historical PhysX trajectories, weights, and large media never modified.
- **GPU is unavailable for new Isaac/PhysX:** PID `242644` remained active in `/home/you/projects/kinetickick-wt-v120`, consuming about 260MiB RTX4090 memory; during this batch it had run roughly **16h55m** and process state `R`. No GPU experiment, kill, restart or competing IsaacLab process.
- `git fetch` was attempted for remote review; GitHub branch existence was separately checked through the GitHub integration before branch creation. No other branch was force-pushed or reset.

## 2. Research question, frozen data and different algorithm

v138 proved that 15/15 instantaneous 29DoF qdot solutions exist but separate local C2 bump stitching gave 0/15 descriptive passes at either **1m/s or 3m/s** ankle-link-origin center targets. This result was limited to the **fixed single-bump shape**. v139 asks: **Can simultaneous optimization of 4 compact C2 basis coefficients per joint over a whole time window produce genuinely 200Hz time-consistent, bounded 29DoF synthetic motion without relaxing the same original joint boxes?**

For each of the original G1's 29 joints, four free coefficients multiply the analytical compact basis `W*s^n*(1-s²)^4`, `n=1,3,5,7`, `|s|<1`; outside that interval they are zero. Thus there are **116 coupled optimization variables** (not 29 independent posthoc smoothers), a common window center at **2.95s**, and exactly the registered **5 baseline synergy amplitudes × 3 virtual right-minus-left ankle-origin x targets × 3 halfwidths = 45 cells**:

- amplitudes `[0,+1,+2,+4,-1]`
- ankle-link-origin x speed target `[0,1,3] m/s`, desired y/z zero
- compact halfwidth `[0.4,0.8,1.2]s`
- all 29 original URDF **position and velocity bounds** are sampled hard constraints across the window, plus a **40rad/s² research-only acceleration bound**. This cap is NOT a verified motor acceleration limit.
- At center, use the original G1 mass-weighted COM Jacobian, synthetic virtual stance-ankle-relative COM speed cap **0.1m/s** and both-arm qdot change cap **0.25rad/s**.
- Objective jointly minimizes time-integrated perturbation q, qdot and qddot with preregistered weights `1, 0.1, 0.002` and ridge `1e-6` using **OSQP 0.6.7.post3**. An unsuccessful or unresolved QP must be retained as a failure; no slack variables, manual target replacement or after-the-fact tuning.
- The center constraint uses a **linearized Jacobian at synthetic reference pose**. It does not solve floating-base dynamics, frictional stance, torque, impulse or true instep speed; only independently post-checks the entire **2,800×29** 14s 200Hz synthetic trajectory.

All 45 outputs, including solver nonconvergence and postsolve violations, appear in the **ignored local JSON**. Independent code reconstructs the denominator and refuses duplicated/missing cells, forged physics status and inconsistent counts.

## 3. Actual full 45-cell results (every attempt retained)

| Synthetic right-ankle relative velocity | W=0.4s | W=0.8s | W=1.2s | Admitted / 15 |
|---|---:|---:|---:|---:|
| **0m/s negative control** | 5/5 | 4/5 | 4/5 | **13/15** |
| **1m/s target** | 0/5 | 4/5 | 3/5 | **7/15** |
| **3m/s target** | 0/5 | 0/5 | 0/5 | **0/15** |
| **Total** | **5/15** | **8/15** | **7/15** | **20/45** |

**20/45** met the frozen postsolve *sampled kinematic* admission checks; **25/45 were rejected**, broken down into:
- **18** OSQP `primal infeasible` decisions.
- **1** OSQP `primal infeasible inaccurate` (NOT a formally certified infeasibility witness).
- **5** OSQP `maximum iterations reached` (UNRESOLVED; not reclassified as infeasible).
- **1** solver reported `solved` but **independent postsolve position violation** was **0.0003236896rad**, exceeding preregistered **0.0003rad**, hence REJECTED instead of counted as a pass.

The `summary.qp_infeasible=19` combines the 18 exact and 1 inaccurate status; use the separated counts above when interpreting mathematical feasibility.

Among admitted candidates: largest analytic joint acceleration **39.9998391rad/s²** (very close to the imposed bound), highest original29 joint speed-limit utilization **0.2084922**, largest perturbation in q **0.9048661rad**, and minimum among *four screened* original URDF wrist/hip capsule lower bounds **0.127749m**. The geometry screen reported **0 warnings among admitted**; **rejected/unproduced trajectories were not cleared of collision risk**, and other links, runtime USD colliders and native PhysX contacts were not certified.

A particularly important negative control is `+4`: its **unmodified synthetic reference** itself peaks at about **58.5216rad/s²**, above the new 40rad/s² screen; some cells are not merely limited by the desired ankle speed. The 0m/s baseline is **not** zero muscle effort or an accurate human behavior; it is a mathematical probe only.

These results are consistent with **the hypothesis that joint temporal optimization can admit certain moderate 1m/s ankle-origin goals that v138's restricted bump could not**, but they are **not an apples-to-apples statistical performance comparison** because the frozen width grids differ (`v138: 0.1/0.2/0.4s`; `v139: 0.4/0.8/1.2s`) and v139 imposes hard sampled accelerations rather than posthoc screening. **3m/s remained 0/15 admitted under this frozen QP family, which is not a theorem on G1 achievable kick speeds.**

## 4. Quality controls, transparency and limitations

- Implemented source SHA-locked original G1 model loading and parent protocol checks, full sparse QP build, 29DoF original-box limits, multi-basis analytic q/dq/ddq, center Jacobian constraints, and independently re-evaluated postsolve gate. **Full trace shape is 14s×200Hz×29 joints**, but outside compact window it is the synthetic v135 reference, NOT simulated 14s stability.
- The initially deployed code encountered a Python identifier syntax error (`200hz` prefix) and **OSQP 0.6 legacy ABI** (`polish`, `solve()`, not new `polishing` / `raise_error`). Fixed before the successful complete run; earlier errors were not represented as successful physical evidence.
- The initial unit tests caught submachine-precision nonzero basis values at mathematically exact compact endpoints. Explicit numerical snapping within `1e-12` of |s|=1 was added, tests rerun and **a distinct `v2` audit file** written rather than overwriting the original. The original first-run audit and failing test log remain locally.
- Independent 45-cell evaluator re-checked source labels, unique grid keys, reported solver states, 200Hz sample counts, all success tolerances, summary denominator and zero new physics. Two locally generated plots represent **the admitted grid** and **rejection categories** without pretending any plot is true G1 PhysX.
- **Technical non-certifications:** Continuous-time between-frame extrema are not bounded solely by 200Hz samples. COM and ankle task dynamics use center linearization and virtual support at a link origin; floating base/foot sole stance friction/CoP/GRF/torques and native actor/collider/path manifold are UNKNOWN. The frozen 2.95s contact time is synthetic, and right-ankle **origin velocity is not actual instep velocity or ball speed**.
- The original historical v54 `CMU10_05` 25.89N equal-opposite nonfoot net forces do **not** uniquely identify the native left-wrist/left-hip PhysX contact pair. GPU occupancy prevented new v133/v134 callback fixtures and same-seed OFF/ON physical replays.

## 5. Reproduction (CPU-only, never launch Isaac)

```bash
cd /home/you/projects/kinetickick-wt-v139
PY=/home/you/conda_env/env_isaaclab/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
$PY tools/v139_joint_window_qp.py --output reports/v139_repro_new_01.json
$PY tools/v139_independent_grid_audit.py --input reports/v139_repro_new_01.json --output reports/v139_independent_new_01.json
$PY tools/v139_plot_qp_failures.py --independent reports/v139_independent_new_01.json --stem reports/v139_new_01
$PY -m pytest -q tests/test_v139_joint_window_qp.py
$PY -m pytest -q tests/test_v12[4-9]*.py tests/test_v13[0-9]*.py
```

Expected latest actual logs: **60 passed** for v139 focused tests, **584 passed, 2 skipped, 2 SLSQP warnings** for extended CPU v124-v139 suite. **No full-project GPU suite was run as part of v139**. Logs under `reports/` are deliberately ignored, not silently uploaded or substituted for old raw traces.

Repository deliverables: `tools/v139_joint_window_qp.py`, `tools/v139_independent_grid_audit.py`, `tools/v139_plot_qp_failures.py`, `tests/test_v139_joint_window_qp.py`, preregistered `configs/v139_frozen_joint_window_qp.json`, `docs/V139_FROZEN_JOINT_WINDOW_CONSTRAINED_QP.md`, this report.

## 6. Next actual physics gate / paper hypothesis

**Next meaningful CPU step:** A **floating-base-aware** nonlinear sparse contact-geometry trajectory optimization with real sole polygons and collision paths/actuator torque, not just phase hallucination, after obtaining qualified source or physically valid support labels. **Next GPU gate:** on confirmed exclusive 4090 (and >20GiB disk), use v133/v134's corrected vector-impulse native PhysX callback in isolated positive and negative contact fixtures, capture both collider paths with a *same-step* original29 state; then rerun the original v54 three controllers OFF/ON with full common-state equality. Only then test new whole-body coordinated actuation on preregistered unseen ball positions/multiple seeds with strict instep impulse ≥95%, legal goal, ball speed ≥3.5m/s, all original qdot bounds, zero nonfoot events, and real 14s recovery.

**Paper hypothesis remains unproven:** High-quality authentic human soccer motion can teach phase-coupled whole-body joint coordination and acceleration patterns for a safe, natural, high-speed G1 instep shot. v139 provides an engineering feasibility **negative/positive CPU screen**, neither a learned model nor a final physical result.
