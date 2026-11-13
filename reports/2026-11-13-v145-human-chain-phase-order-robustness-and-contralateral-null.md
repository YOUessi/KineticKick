# KineticKick v0.1.145 — Human phase-order falsification and contralateral negative control

Date: 2026-11-13. Independent version/v0.1.145 branch from clean parent v144 SHA 51c99a9a72816a981c47cc046f38ffebaf4c0809. Preimplementation protocol SHA **6a277845fd858a67361721b94a87f119756d68a0** committed before this batch's source-grid computation. This is transparently **post-hoc sensitivity/falsification**, not a new-subject confirmation experiment.

## Safety, source, and live GPU state

Tang online. Main v64 worktree dirty; v143 has an untracked native poststep bridge; neither touched. The root filesystem has ~30GiB free (97% utilized), above the 20GiB hard stop. Original v120 pytest PID 242644 has held ~260MiB RTX4090 context for >21 hours; its log remained 0 bytes. Do not kill this protected process or launch competing Isaac Sim. New GPU PhysX executions **0**, new trained policies **0**, new verified native wrist-hip contact manifolds **0**.

SHA-verification reread all six original 120Hz CMU C3D and matching AMC sources: 3,358 real human marker frames across subjects 10 and 11. Only 10_02 and 10_05 have previously qualified **weak uncalibrated ball-marker launch** anchors at frames 353 and 303; both are subject 10. Four other raw clips are preserved in the source eligibility denominator, not falsely assigned true human kicking impact. The two selected clips have no verified professional identity, true right-instep ball impact, synchronized GRF/CoP, ball surface collision or anthropometric energy-transfer truth.

## Frozen 720-decision source-data sensitivity grid

The original marker vector direction-rate estimator has no imputation. We freeze derivative half-span [1,2,3,4] frames; weak-launch anchor shifts [-12,-6,0,6,12] (±100ms); prelaunch windows [0.4,0.6,0.8] seconds; local median width [1,5,9] (invalid data never filled). For each of two CMU event clips and both right/left pelvis→thigh→shank chains: 4 × 5 × 3 × 3 = 180 choices, total **720**. Passage requires strictly ordered peak frame indices with >=2 frames between successive segment peaks. There were no missing parameter cells or low-marker-coverage cells.

| Original clip; same subject 10 | Right-chain pass | Left-chain pass | Original v144 baseline |
|---|---:|---:|---|
| 10_02 | **129/180 (71.67%)** | **0/180** | Right passes; left rejects |
| 10_05 | **108/180 (60.00%)** | **4/180 (2.22%)** | Right passes; left rejects |
| All frozen sensitivity choices | **237/360 (65.83%)** | **4/360 (1.11%)** | 2/2 right, 0/2 left |

For the exact baseline (span=2, prewindow=.6s, anchor shift=0, no median smoothing), the observed real-data frame peak indices are 337→343→350 for CMU10_02 (right adjacent leads 6,7 frames); 286→292→300 for CMU10_05 (6,8 frames). These values are **relative to weak ball marker launch**, not certified foot-ball contact.

**Important negative sensitivity**: pooled right-chain passes with anchor shifts −12, −6, 0, +6, +12 frames are respectively **0/72, 63/72, 63/72, 63/72, 48/72**. The originally reported chain is therefore not robust to a 100ms *earlier* weak anchor; the observation window can lose its near-launch segment peaks. For 1-, 5-, 9-frame median smoothing, pooled passes are **93/120, 93/120, 51/120**. No p-values or subject-level independence may be inferred from grid frequency.

Paired anatomical specificity table over 360 matched parameter settings: **right-only 234**, **left-only 1**, **both 3**, **neither 122**. Descriptive right-side specificity does not imply right instep legality, causal hip→leg power transfer, or athlete expertise.

## Independent numerical falsification

A separately coded atan2(norm(cross),dot) 3D angular rate estimator independently loaded the exact original SHA-verified C3D sources and rebuilt all registered choices. **720/720** decision statuses and **720/720** triple peak-frame index sets matched the primary arccos estimator; maximum numerical discrepancy across chosen peak rates was **2.2248869e-13 rad/s**. This tests computational reproducibility, not human physical truth.

One initial new test expected ValueError rather than the underlying existing safe_write_once FileExistsError when overwrite was attempted. This was a test expectation mismatch, not experiment tuning; first failure log retained in ignored Tang reports, corrected expectation and reran.

## Original G1 physically real negative results, SHA checked again

Original v53 rigid-body PhysX: 12 historical trials, four frozen ball placements × frozen_v32 / CMU10_02 / CMU10_05; each policy retains **0/4 strict passes**. Original v54: three historical real rollouts and **6,628 actual original29 200Hz state rows** reread, not reexecuted. The failed CMU10_05 episode has authentic sampled state through 5.140s, named net body sensor symptom around 5.135–5.145s. There is no original same-step 5.145s full29 state or postevent recovery horizon. **Left wrist—left hip native PhysX actor/collider manifold remains UNKNOWN**.

Original strict instep impulse >=95%, valid goal, ball speed >=3.5m/s, complete original 29 qdot speed, no nonfoot collision, and recovery to 14s all remain compulsory. No goal or one peak rate overrides a failed gate. This v145 experimental batch had **0 new PhysX shots, 0 new learned motor policies and no validated high-speed instep success**.

## Bounded external data qualification (metadata only; no large download)

GroundLink provides genuine synchronized fullbody motion, measured ground reaction forces and CoP, including marker C3D and foot-specific force annotations. It is a candidate for generic **support/contact** pretraining; public described tasks are not verified elite instep soccer striking with flight/impact truth. Source: https://csr.bu.edu/groundlink/ and https://arxiv.org/abs/2310.03930.

Inertial Sensor-Based Motion Tracking in Football provides processed football IMU plus Vicon data. Public metadata do not establish a qualified synchronized force plate, precise right-instep collision or professional explosive striking. Candidate for **source/license and task metadata** validation only, not approved high-power supervision. Source: https://zenodo.org/records/3732988.

BadmintonGRF 2026 presents substantial measured force data for another sport, not professional soccer instep contact truth. https://arxiv.org/abs/2605.01876. No large new datasets downloaded to a 97%-full disk.

## Implementation, reproducibility, independent tests

- tools/v145_phase_order_robustness.py — strict SHA frozen source/contract validation, no-imputation 720-cell genuine C3D sensitivity grid, paired right-left control, original physical G1 SHA negative audit, write-once local JSON.
- tools/v145_independent_phase_audit.py — different atan2 geometry, 720 independent source-grounded comparisons and fail-closed provenance guard.
- tools/v145_plot_phase_robustness.py — two factual local PNG diagnostics; original files never overwritten.
- tests/test_v145_human_phase_robustness.py — 36 passed targeted tests: six genuine C3D/AMC SHA checks, data missingness and known synthetic-rate cases, hard subject split, frozen baseline, negative controls, tamper and write-once rejection, full independent 720-cell audit and historical PhysX count.

**Expanded CPU regression:** 32 v124–v145 test modules, **824 passed / 2 skipped / 0 failed**, 2 existing SciPy SLSQP warnings; NOT a full original PhysX GPU integration suite. Append-only ignored Tang artifacts: reports/v145_posthoc_phase_grid_first_pass.json, reports/v145_independent_atan2_full_grid.json, reports/v145_real_cmu_bilateral_specificity.png, reports/v145_real_cmu_anchor_sensitivity.png, two specialized test logs and expanded CPU regression log. No raw 30MB JSON/NPZ/C3D/AMC or media or model weights pushed.

To reproduce with uniquely named output files on Tang:

~~~bash
cd /home/you/projects/kinetickick-wt-v145
export PYTHONPATH=tools
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v145_phase_order_robustness.py --output reports/NEW_PHASE_GRID.json
$PY tools/v145_independent_phase_audit.py --input reports/NEW_PHASE_GRID.json --output reports/NEW_INDEPENDENT.json
$PY -m pytest -q tests/test_v145_human_phase_robustness.py
~~~

## Research conclusion and next dependency

**Supported**: original-source same-subject right pelvis→thigh→shank unsigned direction-rate ordering is observed in a majority of registered sensitivity choices and almost absent in the left contralateral control. **Unsupported/falsified**: the phase pattern is invariant to weak-anchor timing, has been shown to cause proximal-to-distal mechanical power transfer, implies elite kicking or trains a legal high-velocity instep G1 strike. There is still no paired GRF, verified true contact, or subject-heldout football training benchmark.

**Next real dependency:** after protected GPU process release and disk recheck, isolate genuine PhysX positive/negative collider calibrations; then execute original three-controller OFF/ON six physical rollouts with matched seeds, verified 200Hz state parity and real collider ownership/manifold time-of-contact. Until then, continue bounded high-quality force-contact source audit and design floating-base, full29 torque/stance/friction-aware synergy with no test-set tuning. Do not promote synthetic QP or human pose rates as the paper endpoint.
