# KineticKick v0.1.140 — 1ms 29DoF densification, exact-pose FK and v139 false-admission falsification

**Date:** 2026-11-13. **Parent immutable source:** v139 / `a9dac3c32532745f2782e4213a51a390017f8e78`. **Pre-committed protocol BEFORE code and experiments:** `d1334ec692a81ebb1d146e121940ff9c6a72542b`. **Original G1 URDF SHA256:** `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`.

**Evidence boundary:** All new trials are **CPU-only mathematical synthetic-kinematic re-audits**, **not** new PhysX robot shooting trials. New true native pair/manifold contact events = **0**. Real PhysX rollout attempts = **0**. Actual new human elite-instep motion/GRF labels = **0**. Motor permission = **NO**. No inference of real full-body stable recovery or ball exit speed.

## 1. Safety and repository preflight

- Tang responded online through the authorized Remote Desktop Commander device. GitHub remote branch tips were verified by `git ls-remote`, numerically sorted: latest available before work was `version/v0.1.139`. `version/v0.1.140` absent both local and remote before creation.
- Original working directory `/home/you/projects/kinetickick` remained at **dirty** `version/v0.1.64` with `?? =100===`; existing v131 and other historical worktrees were not modified. The `v139` source worktree was **clean**. Created separate `/home/you/projects/kinetickick-wt-v140` from frozen parent source.
- Root disk free **30 GiB** (97% utilized), above user-defined minimum of 20GiB. GPU still held by *pre-existing* v120 whole-repository pytest process **PID 242644**, approximately 260MiB VRAM, ~17h45m elapsed at initial check; existing CPU-intensive run also left untouched. No new IsaacLab, GPU work or process termination.
- Original CMU/C3D, pretrained weights, original v54 PhysX evidence and all prior JSON/PNG untouched. New per-attempt JSON, independent audit JSON, plots, and test logs remain local under ignored `reports/`, never committed to GitHub.

## 2. Frozen question, methodology and independence

v139 simultaneously optimized 116 independent C2 coefficients (4 per original joint), through one shared sparse QP. It reported **20/45** admitted on a 200Hz sampled grid but had two unexamined issues: (i) sampled hard bounds do **not** imply a bound between 5ms samples, and (ii) its ankle/COM speed tests used frozen reference-pose Jacobians. v140 independently addresses both, WITHOUT changing old QP hyperparameters, source, targets, width, objective, or original solver outcomes:

1. Re-execute **the complete 45-cell frozen parent grid**, `5 amplitudes [0,+1,+2,+4,-1] × 3 synthetic relative ankle-origin target speeds [0,1,3]m/s × 3 widths [0.4,0.8,1.2]s`. Use the unchanged original 29DoF URDF joint position/velocity bounds and a **research-only** 40rad/s² acceleration screen.
2. Independently recreate the full **14-second** seven-phase synthetic shared-latent trajectory at **1ms**: **14,000 frames × 29DoF**, with analytic phase derivatives. The implementation was cross-checked against all 2,800 parent reference samples for each amplitude and finite-difference derivative tests; no temporal interpolation of frame values pretending to be analytic motion.
3. Independently reconstruct the 116 C2 coefficients from each reported parent 200Hz witness using a least-squares design matrix. Check reconstructed q, qdot and qddot against parent witness to respectively 1e-7rad, 1e-6rad/s and 1e-5rad/s² BEFORE accepting a dense reconstruction. Do not fabricate coefficients for unsuccessful QPs.
4. Recompute full-horizon dense limits with same frozen postsolve tolerances: position 0.0003rad, velocity 0.0003rad/s, acceleration 0.005rad/s². At the exact 2.95s synthetic phase center, recompute original-URDF nonlinear full 29DoF Jacobians for left/right ankle-link origins and mass-weighted COM, re-evaluate relative target (tolerance 0.01m/s) and virtual-stance-relative COM speed (0.1+0.002m/s).
5. An original QP failure stays a failure; any dense/nonlinear violation downgrades a parent admission. Independent evaluator audits 45 unique cells, failure counts, physical provenance and all numerical gate definitions. It rejects forged PhysX/GRF/human evidence or attempts to upgrade a parent failure.

**Not a continuous-time interval certificate:** 1ms still samples rather than mathematically bounding every intermediate instant. Analytical qddot is real *for the synthetic reference function*, but URDF effort limits, real base motion, foot-ground contact, real motor torque and ball collision are **not** simulated.

## 3. Actual 45-cell numerical results — a nontrivial negative finding

| Virtual right-minus-left ankle LINK ORIGIN x-speed | v139 admitted at 200Hz | v140 admitted after 1ms + nonlinear FK |
|---|---:|---:|
| 0m/s (negative control) | 13 / 15 | **12 / 15** |
| 1m/s | 7 / 15 | **7 / 15** |
| 3m/s | 0 / 15 | **0 / 15** |
| **All 45 cells** | **20 / 45** | **19 / 45** |

- **All 45/45 original solver statuses, admission flags and reason strings exactly matched** v139's original ignored `v139_joint_window_audit_v2.json`; none silently reclassified by rerunning QP. Original ledger SHA256: `6ae9e3a4a53e4b319ca1fa587354e54fc846beefc93b3f37b59ce4fafb0c4415`; v140 new ledger SHA256: `03cecd98e74e6397ea9eb059566b2d324457e4095fa854bcda8eeee7d93eb031`.
- Among the 20 original parent admissions, **one newly falsified**: multiplier **+4**, target **0m/s**, compact halfwidth **0.4s**. Its true analytical qddot evaluated at 1ms reaches **40.12341347rad/s²**, exceeding the frozen **40rad/s² research screen** by **0.12341347rad/s²**, despite passing the 200Hz sampled check.
- The other 19 parent admissions passed this *stronger numerical screen*. There were **no additional q/qd bound, exact-pose center ankle target, or exact-pose COM rejections**. This does not mean any robot executed them; all 19 remain *not PhysX certified*.
- **Critical correction to the research assumption:** at the *exact center* of the v139 basis, each odd-power compact position basis has zero displacement, so the fitted center posture is **identical to the reference center posture** (observed max 6.66e-16rad). Consequently, the Jacobian evaluated by v139 at that same center **was already at the exact fitted posture**, rather than an approximation arising from a changed center q. Measured max exact-pose velocity residual among all 20 parent admissions was **1.44035e-7m/s** and max virtual-center-COM component speed **0.100207525m/s** (within the 0.102m/s frozen cap). Thus v140's proposed 'center linearization defect' is **falsified**; the real additional problem here is **between-sample acceleration**. Away from the center there can still be nonlinear Jacobian drift; not certified by this test.
- The **three-m/s target remains 0/15**, but this is a failure of the *specific fixed finite C2 basis + current QP + synthetic reference*, not a physical theorem on achievable Unitree G1 foot or ball speeds.

Initial debug run caught `abs(list)` in the COM metric, fixed before the first completed experiment; the failed log `v140_nonlinear_dense_run.log` was not overwritten. Initial focused unit tests caught 4 tiny floating-roundoff differences in dense analytic qddot at phase boundaries (~3.67e-13rad/s²); independent analytic comparisons were correctly relaxed to a 1e-12 numerical tolerance, logged as `v140_focus_test_v1.log`, with the passing rerun in `v140_focus_test_v2.log`. Frozen scientific thresholds remained **unchanged**.

Independent final audit: **45 cells, parent 20, new 19, additional veto 1, original rejected 25, new genuine physical shots 0**.

## 4. Engineering / tests / evidence products

- `tools/v140_nonlinear_dense_reaudit.py`: original SHA-locked protocol, full synthetic 14s 1ms analytic latent reconstruction, old-parent coefficient cross-examination, real original-URDF exact-pose FK/Jacobian and COM, 45 QP replays, source/physics labeling.
- `tools/v140_independent_nonlinear_audit.py`: independent JSON-only numerical threshold and completeness checker, ensures no solver rejection can be upgraded and no GPU/human/contact evidence can be forged.
- `tools/v140_plot_nonlinear_dense.py`: two figure outputs from the independently audited JSON: 200Hz vs 1ms+exact-pose admissions and the one additional acceleration rejection. PNGs local only.
- `tests/test_v140_nonlinear_dense_reaudit.py`: **37 new CPU tests** including five all-frame original-reference comparisons, analytic derivatives, source/config tampering, reconstructed coefficient validation, no-contact/nonmotor proofs and forged grid/physics negative cases.
- Full selected v124-v140 CPU regression: **621 passed, 2 skipped, 2 inherited SciPy SLSQP clipping warnings, 0 failed**. Focused: **37 passed**. No full GPU-project suite claimed.

## 5. Reproduction

```bash
cd /home/you/projects/kinetickick-wt-v140
PY=/home/you/conda_env/env_isaaclab/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# New unique filenames: script refuses to overwrite any prior research evidence
$PY tools/v140_nonlinear_dense_reaudit.py --output reports/v140_reproduction_01.json
$PY tools/v140_independent_nonlinear_audit.py --input reports/v140_reproduction_01.json --output reports/v140_independent_repro_01.json
$PY tools/v140_plot_nonlinear_dense.py --independent reports/v140_independent_repro_01.json --stem reports/v140_repro_01
$PY -m pytest -q tests/test_v140_nonlinear_dense_reaudit.py
$PY -m pytest -q tests/test_v12[4-9]*.py tests/test_v13[0-9]*.py tests/test_v140_nonlinear_dense_reaudit.py
```

Log and main figures from this actual run: `reports/v140_nonlinear_dense_v2.json`, `reports/v140_independent_v2.json`, `reports/v140_nonlinear_dense_01_passes.png`, `reports/v140_nonlinear_dense_01_veto.png`, and `reports/v140_extended_cpu_regression.log`.

## 6. Next dependent research / falsifiable paper hypothesis

**A, first real GPU gate:** When the v120 GPU test exits and disk remains >=20GiB, isolate v133/v134 native PhysX actor/collider pair + vector impulse callback in known-contact and known-noncontact fixtures, record same-step G1 original29 state, and rerun **all original three v54 controllers with native observer OFF/ON** without changing the controller, verifying every shared physical state numerically. Wrist-hip is **UNKNOWN** until exact actor/collider pair/manifold and same-step witness are captured; equal-opposite 25.89N and URDF proximity are insufficient.

**B/C, upgraded dynamic candidate:** Use real floating-base states, actual foot sole collision geometry/polygons, double-/single-support CoP and GRF, physical contact impulse and original G1 dynamic actuator torque limits; only then attempt full-window stance-constrained temporal coordination, not merely ankle origin kinematics. Move toward **interval-certified** analytic joint acceleration and collision separation bounds to avoid the v140 200Hz sampling leak. Freeze unseen ball positions/seeds and strict collision/motor/recovery gates before any training or retuning. Do not train using failures already observed as a hidden test-set oracle.

**D/E, publication truthfulness:** Genuine high-quality human *instep soccer kicking*, ideally synchronous motion capture + GRF + foot-ball contact, remains an unqualified data prerequisite. Until obtained, the full-body latent generator is an engineered coordination *hypothesis*, not elite motion imitation. Strict live shot must satisfy instep impulse share >=95%, legal goal, ball speed >=3.5m/s, all original29 velocity and safety limits, no nonfoot contacts, and 14s physical recovery — **no such result exists in this new batch**.

**Scientific conclusion:** A nominal v139 200Hz admitted trajectory can violate even a synthetic acceleration bound **between** samples. Combined with the exact center odd-basis invariance, this identifies a genuine validation flaw while eliminating a proposed nonlinear-center explanation. None of this is evidence of natural or professionally learned kicking.
