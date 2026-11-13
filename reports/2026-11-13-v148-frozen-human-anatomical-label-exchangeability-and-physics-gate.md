# KineticKick v0.1.148 — Independent real-C3D anatomical-exchangeability falsification and football data admission

**Executed:** 2026-11-13 on authorized Tang Ubuntu CPU. **Parent:** clean v147 `f4761baefd1cdbcf2ae951ecc97b086139818c44`; protocol/config separately frozen before code/analysis in `8da717747c86c77aca0c146c9f6c40c875adec93`. Another long-running v120 pytest PID 242644 retains RTX4090 context, so **no new PhysX, no training, no model or old state mutation**.

## 1. Why this is a scientific falsification, not another synthetic speed optimization

The prior 720-source-cell v145 study observed pelvis→right thigh→right shank peak timing on two original CMU120Hz C3D kick clips; **only one human subject, two uncalibrated ball-launch event anchors, no synchronized GRF/CoP, no true instep impact timestamp, no elite competence certification**. Existing 360 right cells are overlapping algorithm choices (span, median filter, event shift, window), not 360 independent observations. v148 registered an exact six-way anatomical relabelling per clip, 36 combinations overall, retaining the whole correlated grid and invalid-marker masks. This is an exploratory anatomical-label exchangeability diagnostic: NOT a randomized population experiment, NOT a p-value, NOT contact force or motor dynamics.

## 2. What was genuinely executed

- `tools/v148_human_anatomical_exchangeability.py`: checks independently SHA-verified original 6×CMU C3D/AMC, recomputes 720 real marker-rate grid cells, validates source-integer peak indices and 2-frame minimum margins; rejects contradictory statuses, missing/duplicated grid keys, incomplete coverage, unregistered subject and forged contact claims; computes 6 actual per-clip relabellings and all 36 jointly paired scores. Executes original v54 three SHA-qualified PhysX file checks for **6,628 real 200Hz states** and distinguishes historic runs from new runs.
- `tools/v148_independent_peak_recheck.py`: independent pure-Python recomputation from recorded source-derived 720 peaks; verifies every shift count, exact 36-case scores and source case SHA values, rejects tampered physics/human claims. It does not call the primary v148 summarizer.
- `tools/v148_plot_falsification.py`: plots actual anchor-shift sensitivity of all four clip-side combinations, write-once PNG saved locally.
- `tests/test_v148_source_exchangeability_and_safety.py`: 42 positive and adversarial tests; includes boolean/noninteger peaks, wrong 120Hz shift, missing markers, forged exact human contact/GRF/professional status, fake same-tick native manifold, false new PhysX run, changed source SHA, duplicate grid, model false admission, plot overwrite.
- `docs/V148_QUALIFIED_FOOTBALL_HUMAN_DATA_SHORTLIST.md`: bounded dated real-data lead review with URLs and explicit per-source true/false/UNKNOWN physical labels; no 987MB downloads, GRF hallucination or false 3D labels.

**The first v148 attempt failed** only because the legacy original PhysX case names were lowercase `cmu_10_02` / `cmu_10_05` rather than the display names `CMU10_02` / `CMU10_05`; fixed the parsing map before accepting any output. Initial failure remains in local log `reports/v148_human_run.log`.

## 3. Actual frozen results: raw human marker timing, two weak anchors

| Real raw clip | Right chain order | Left-side control | Right shift −100ms | −50ms | 0ms | +50ms | +100ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMU10_02 | **129/180=71.7%** | 0/180 | 0/36 | 36/36 | 36/36 | 36/36 | 21/36 |
| CMU10_05 | **108/180=60.0%** | 4/180 | 0/36 | 27/36 | 27/36 | 27/36 | 27/36 |
| Total (same subject) | **237/360=65.8%** | 4/360=1.1% | 0/72 | 63/72 | 63/72 | 63/72 | 48/72 |

Original 120Hz rate and ±12 frame = ±100ms shift. These are frequencies over *correlated processing choices* only; no confidence interval or effective independent N inferred. All 180 triplets for each side/clip were marker-complete under registered v145 criteria. Right-chain gap medians over the 180 choices are 5 and 6 frames (CMU10_02 pelvis→thigh and thigh→shank); 4 and 7 frames (CMU10_05). The negative tail at some shifts shows that even phase ordering can reverse in other plausible analysis windows.

**Exact anatomical relabelling diagnostic:** six within-clip scores were `[129,0,0,15,27,0]` (CMU10_02) and `[108,0,0,0,0,0]` (CMU10_05). The identity map has aggregate score **237**. Only **1/36** of paired anatomical reassignments scored ≥237 (observed itself); paired score range **0–237**. This demonstrates that the observed peak ordering is strongly linked to the particular right-side anatomical labels under *this restricted post-hoc source grid*. It does **NOT** establish statistical significance across footballers, professional shot style, actual kinetic transfer, impact impulse or causal force.

**Pre-frozen data intake gates:** two clips individually fail 80% robust-order requirement, both fail the ≥60% every-anchor-shift requirement because −100ms gives zero, and only one source subject is available rather than three impact-annotated independent subjects. Exact human instep time, force-plate GRF/CoP, elite skill certification and subject-disjoint test are all **false/unknown**. Thus `qualified_for_human_to_original29_training=false`. The test is an *honest negative qualification*, not an algorithm weakness hidden by averaging.

## 4. True original G1 physics versus CPU diagnostic

All three prior native 200Hz G1 v54 traces were read again with SHA validation: 2,800 frozen_v32 + 2,800 cmu_10_02 + 1,028 cmu_10_05 = **6,628** historical real 5ms states. Original nonfoot CMU10_05 event at 5.135s remains a 25.890153...N aggregated net-force maximum; its terminal annotation is **not an extra physics frame**. Original v53 four-ball strict physical outcomes remain 0/4 for each of three policies; our examination of historical data did not create new shots.

This run performed **0 actual new PhysX episodes**, so new strict goal+≥95% legal instep impulse+≥3.5m/s ball speed+original 29qdot+actuator+nonfoot+14s safe-recovery success is **not measurable**. The left-wrist versus left-hip PhysX collider paths/manifold remain **UNKNOWN**. Per-body net sensor forces and root tilt cannot certify self-collision, isolated GRF or a fall. The sample memory of 25.89N opposite vectors is not a verified native pair.

## 5. Training hypothesis and physics controls for next cycle

v148 provides a stronger reason to **withhold human-to-robot motor training** on the two available weakly labeled clips. A prospective trainable method should learn all-body correlated phase latents and contact stance from *independent professional-kick human subjects* with true ground supports, with counter-swing arms, moving pelvis/torso, timed swing-leg energy and free 6-DoF base. The decoder must be projected into actual original29-G1 dynamics, friction/CoP margins, collision barrier, torque/velocity actuator bounds, and original ball/contact/recovery hard gates. Register held-out ball positions and random seeds **before training**; compare full latent against no-hip/trunk, no-arm, shuffled-phase, kinematic-only, and v32/CMU weak-reference baselines without using test failures to retune. These are proposed protocols, not claimed v148 robot policies.

Physical execution is still blocked at start by **other v120 pytest PID 242644 (~260MiB device CUDA context)** with only ~30GiB free disk. Did not kill or signal that process, did not take the GPU mutex, and did not touch original CMU/C3D, pretrained models, v53/v54 JSON/NPZ, renderer media or any preexisting dirty worktree. The original `version/v0.1.147` prior completed SHA was also successfully pushed and verified before branching v148; it had not been on the remote.

## 6. Reproduction without overwriting evidence

```bash
cd /home/you/projects/kinetickick-wt-v148
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v148_human_anatomical_exchangeability.py --output reports/v148_repeat_UNIQUE.json
$PY tools/v148_independent_peak_recheck.py --source reports/v148_repeat_UNIQUE.json --output reports/v148_independent_UNIQUE.json
$PY tools/v148_plot_falsification.py --source reports/v148_repeat_UNIQUE.json --output reports/v148_plot_UNIQUE.png
$PY -m pytest -q tests/test_v148_source_exchangeability_and_safety.py
$PY -m pytest -q tests/test_v12*.py tests/test_v13*.py tests/test_v14*.py
```

The actual first accepted write-once local output SHA256 is `cf081b225bf8d247321fb76a1bb21e1424d7e433611525ea0611dd2706f8d152` at `reports/v148_human_source_anatomical_exchangeability_locked.json`; follow-up independent JSON, plot and terminal logs are kept ignored locally. Only code, frozen protocol/config, tests and text are committed to GitHub. No false positive safety or elite professional claim.

## 7. Final test / safety / remote pre-push audit

- New adversarial specialist suite: **42 passed / 0 failed**.
- Expanded *CPU-only v12x–v14x* suite (all `tests/test_v12*.py tests/test_v13*.py tests/test_v14*.py`): **998 passed, 2 skipped, 0 failed, 2 prior SLSQP bound-clipping warnings**, logged at `reports/v148_expanded_cpu_verified.log` with **exit code 0**. This does **not** mean the full IsaacLab GPU suite has run.
- Independent source-derived triple analysis: **720/720 cells accepted** and right-chain identities 129+108=237; exact 36 label combinations verified by an implementation not calling the primary summarizer.
- Original PhysX source SHAs reverified at 3/3, historical 5ms state denominator **6,628**; there were **0 new simulated physics runs**, so no new physical success estimate.
- Current NVIDIA compute process remained **PID 242644 / 260 MiB**, disk available **~30 GiB**, so GPU-exclusive native calibration and six OFF/ON physics runs remain blocked.
- False-positive opportunity: the exact-label-only maximum **1/36** can look statistically convincing but cannot be interpreted as 36 independent training examples, a population `p`-value or elite biomechanical causality. The frozen timing-shift robustness controls fail; preserve this negative result.

The only eligible next physical batch requires the existing foreign GPU job to finish without our intervention, >20GiB free, native callback positive/negative collider-pair calibration and exact OFF/ON 29q/state parity, followed by strict multi-seed held-out original G1 14s football shooting.
