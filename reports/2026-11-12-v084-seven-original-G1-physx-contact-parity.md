# KineticKick v0.1.84 — frozen independent-positions original G1 PhysX batch, strict hard gates, contact-reader parity and falsifiable whole-body next method

**Date:** 2026-11-12 (Tang local +08:00). **Branch:** `version/v0.1.84`. **Freeze:** `0c85f4b1384e68d123122df2ca8b1da30ef8f95a` (committed *before* any v84 physics). **Frozen protocol SHA256:** `528e3bd8ff5359fdf83adbd8e8f177d21057038e8ed30622d32b6b0b4a67aafb`. New branch based on committed v83 `9f51c09`; **v83 untracked/unfinished files were not edited or imported**. Local v83 unfinished work is *not* a remote-pushed v83 research report.

## Key outcomes and scientific interpretation

- Tang RTX4090 CUDA PhysX, *original free-root 29DoF G1*, genuine 200Hz (0.005s). **7 new physical executions, all 7 completed 14s (2800 frames each), all 7/7 passed the existing strict hard gates.** These are **6 unique new ball/control conditions plus 1 identical-condition readonly on/off instrumentation replay**, not seven independent initial-condition samples.
- 2 new frozen scenes: C ball XY offset `(+0.085,-0.040)m`, lateral command delta `+0.035`, seed 1219; D `(-0.075,+0.055)m`, delta `-0.035`, seed 1223. Each crossed untouched original v32, original CMU10_02 proxy, original CMU10_05 proxy. Actual root/joint reset perturbation is **disabled**, so seed variation is *not* independently perturbed reset physics.
- All 7 have measured original 29-joint qdot at 200Hz and genuine contact-0.4s to contact+1.25s evidence; all 7 have full 14s observed and recovered. No early terminations, nonfoot net peaks 0N in this batch. Original maximal qdot/limit ratio **0.99703361** (near hard limit).
- All 7 have measured instep impulse share 100% and goal=true. But the observed **post-contact 5–110ms ball-speed proxy is only 3.509–3.513m/s in scene C and 5.495–5.500m/s in scene D.** Passing the project **3.5m/s minimum gate does not imply a high-speed natural human power shot.** No naturalness or professional style gain is supported by this result.
- Original v32 and CMU proxy policies already pass all six new unique positions. This batch therefore **does not show improvement from human prior**. Separately retain earlier v83 4/6 (two goal failures on B, not hidden), v67 4/9, and the v83 staged tensor observer's 95s watchdog timeout. Denominators are not pooled.
- v32 scene C readonly OFF/ON run was physically equal **across all shared 2800-frame trace fields, original contacts, records and terminations**; **1/1 exact parity PASS**. Readonly poll executed **160 calls** during the 4.70–5.50s window and returned **0 native headers / 0 native points / 0 direct wrist-hip witnesses**, with no poll errors. This is **UNKNOWN CONTACT TRUTH**, not evidence disproving self-collision; wrist/hip opposite net forces from earlier experiments still cannot establish same contact manifold.
- 7 original JSON PhysX recordings total approximately **209.12MiB remain local**; their individual SHA256 identities are verified by postregistered independent audit and not committed. 7 corresponding SHA-bound physical evidence plots remain in local `media/v084_*`. No original CMU, C3D, weights, PhysX records, or other worktrees were overwritten.

## Raw physical per-condition independent evaluation

| Scene/control | Independent physical rows | Strict gates | Instep | 5–110ms ball speed proxy (m/s) | Max measured original qdot/limit | Nonfoot peak |
|---|---:|---|---:|---:|---:|---:|
| C / original v32 readonly OFF | 2800 | PASS | 100% | 3.510465 | 0.909141 | 0N |
| C / original v32 readonly ON | 2800 | PASS | 100% | 3.510465 | 0.909141 | 0N |
| C / CMU10_02 | 2800 | PASS | 100% | 3.508981 | 0.961548 | 0N |
| C / CMU10_05 | 2800 | PASS | 100% | 3.512750 | 0.909141 | 0N |
| D / original v32 | 2800 | PASS | 100% | 5.494713 | 0.997034 | 0N |
| D / CMU10_02 | 2800 | PASS | 100% | 5.499724 | 0.907497 | 0N |
| D / CMU10_05 | 2800 | PASS | 100% | 5.500031 | 0.981042 | 0N |

All seven: physical full 14 seconds, complete true-time 0.4s pre/1.25s post contact window, positive goal, standing recovery, original all-29 joint velocity within limits, no named nonfoot contacts. The original G1 free-root foot touchdown, support, center-of-mass, root height and root tilt time series were saved and audited; e.g. C/v32 support classification **68.179% bilateral**, **14.393% left-only**, **12.750% right-only**, **4.679% flight** and wholebody COM height approximately **0.640–0.734m**. Do **not** treat these descriptive ratios as evidence of learning anthropomorphic coordination. The right-ankle contact-event geometry is the existing recorder's modeled event, not direct native shape-pair proof.

## Independent software and evidence checks

New frozen files:
- `configs/v084_frozen_generalization.json`: seven preregistered, unchanged-controller trials, disk/GPU/strict physics gates and truth limits;
- `docs/V084_FROZEN_PROTOCOL.md`: before-experiment hypotheses, matrix and interpretation;
- `tools/v084_batch.py`: source + SHA + clean working-tree preflight, global GPU mutex, safe write-once subprocess timeouts, physical output certification and independent native-poll on/off bytewise-state comparison;
- `tools/v084_evidence.py`: independent full 200Hz 29-joint, COM, foot supports, named nonfoot, velocity-limit and contact-window audit;
- `tests/test_v084_independent_evidence.py`: invalid frame gap, missing qdot, seed mismatches, false contact-manifold witnesses, zero native header UNKNOWN and paired-state divergence regressions.

Postregistered CPU-only additions: `tools/v084_evidence_py310.py` is a narrowly scoped Python 3.10 streaming-SHA compatibility shim. The initially frozen independent audit attempted Python 3.11's `hashlib.file_digest` and **failed**, faithfully retained in `reports/v084_independent_console.log`. The shim changed **no frozen script, controller, threshold, trial or physics bytes**; it only replaces the digest function in memory. The retry succeeded: **7 genuine episodes, 7/7 strict pass, 7/7 complete 14s and 7/7 genuine contact windows**. `tools/v084_plot.py` generated 7 immutable local physical-data figures, including axes for double/single support, true COM height, all-29 maximum qdot/limit and force/ball-speed with vertical contact timestamp. The plots are descriptive, not motion-quality labels.

At preregistered test stage, the isolated worktree lacked original local report inputs, causing **25 missing-file failures**, preserved in `reports/v084_cpu_full_suite.log`. We restored **only readonly symlinks to 1895 original local report files** (no bytes copied/overwritten); complete original CPU regression then passed **949 tests, 2 skipped, 69 subtests passed, 0 failures**. After adding the postregistered shim and visualization tests, rerun is recorded separately.

## Remaining A–E scientific blockers

**A (unsolved):** Native actor0/actor1/collider0/collider1/contact points/normals/positive impulses for left wrist–left hip still missing. An empty contact polling result cannot exclude collision. The v83 filtered tensor initialization succeeded through three view creations but subsequent physics did not exit before 95s watchdog; do not retry an unbounded view. NVIDIA's Omni Physics contact-report documentation describes subscribing to the physics-step callback and requires PhysxContactReportAPI attachment to an eligible rigid body. Audit existing API **thresholds/step registration** and existing sensor raw data in a new frozen readonly probe; distinguish reported contact absence from reporter dispatch failure using a known physical foot–ball positive-control event. Do not author or mutate USD schema in the original equivalence experiment.

**B (partially met):** 7 complete real 200Hz whole-body/support/COM traces plus measured nonfoot and real qdot. Foot–hip–wrist **native clearance/contact pair geometry** is unavailable; that feature cannot be certified from net forces.

**C (research hypothesis only):** v83 phase-conditioned support→hip→torso→swing→opposite-arm→follow-through is a coherent, approximately 0.154rad peak intended 29DoF synthetic control *interface*, **not a learned excellent-human football controller**, and is blocked from new PhysX execution without verified collision/support safety witness. No training or post-hoc tuning on these newly seen scenes. See `docs/V084_FALSIFIABLE_WHOLEBODY_METHOD.md` for the pre-learning data/retarget/physical PPO/QP plan.

**D:** Original unchanged-control/new-scene genuine GPU control matrix completed as frozen, but **no new learned wholebody arm/hip coupling physics ablation** yet. Passing 3.5m/s by 0.009–0.013m/s in C reinforces the need for independent strong-shot quality criteria; do not label this as high speed.

**E:** All setup failures, prior negatives and analytical SHA compatibility failure retained locally. GitHub holds reproducible code/protocol/tests and this textual report, while the SHA-certified raw JSON/plots remain on Tang. No professional human identity, synchronized GRF, force-plate or human instep impulse ground truth exists in the six audited CMU C3D files.

## Reproduce (on the verified Tang asset tree, new output namespace needed)

```bash
cd /home/you/projects/kinetickick-wt-v084
git show 0c85f4b1384e68d123122df2ca8b1da30ef8f95a:configs/v084_frozen_generalization.json | sha256sum
# v084_batch is write-once and refuses existing outputs, preserving original evidence.
PYTHONPATH="$PWD/tools" /home/you/conda_env/env_isaaclab/bin/python tools/v084_batch.py
/home/you/conda_env/env_isaaclab/bin/python tools/v084_evidence_py310.py
/home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
```

## Final all-CPU regression and hardware safety

The final **postregistered** CPU suite passed **951 tests, 2 skipped, 69 subtests passed, 0 failed** (\`reports/v084_full_suite_final.log\`). Previous Python 3.10 audit SHA error and unlinked-test-input failure logs are preserved, not rewritten. At closeout Tang had about 38GiB free disk, no active CUDA compute jobs, and no other worktree source changed. Raw simulation and all seven plotted images remain Tang-local.
