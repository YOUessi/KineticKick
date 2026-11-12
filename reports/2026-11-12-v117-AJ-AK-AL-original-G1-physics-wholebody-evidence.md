# KineticKick v0.1.117 — frozen AJ/AK/AL original G1 physics, temporal safety falsification and human evidence qualification

Date: 2026-11-12 (Asia/Shanghai). This is a completion report of the **already preregistered** v117 experiment, not a post-hoc retuned controller. Frozen experiment source SHA `0fcb4af3a4152e0ce592c61781d5f0064d4d65fa` and frozen matrix SHA256 `5e61561635bdb6cabf399d296f2f517b6e6e2af45672e356980c7682cd0b21bd`. Parent source SHA `402b87f019dd7cd31f9d5af980ad38a2b63a7e3f`.

## Provenance and safety

- Tang / original free-root 29DoF Unitree G1 / original SHA-qualified v32, CMU10_02, CMU10_05 controller and recovery checkpoint / 200Hz CUDA PhysX / 14s per trial.
- The preregistered matrix has AJ [-0.018,+0.041] m, delta velocity -0.013, seed 2843; AK [+0.035,-0.029] m, +0.012, seed 2897; AL [-0.059,-0.022] m, -0.017, seed 2939. Three original controllers per scene (9 independent scene×controller combinations), with two additional repeated ON/OFF read-only observers (AJ/v32 and AL/v32): **11 actually executed, 11 certified, 30,800 recorded genuine 200Hz steps**.
- The batch driver checks asset SHA, original frozen committed code, GPU compute exclusivity, disk >=20GiB, shared GPU flock and per-child watchdog. Original data, C3D, checkpoints and prior PhysX evidence were not overwritten. The new large physical JSON, log, audit JSON and figures are write-once and excluded from Git.
- Raw SHA and independent 29DoF source joint velocity decoder checked. The two ON/OFF comparisons matched **every common recorded physical 200Hz trace field and recorded event/shot/torque field** exactly (max difference 0 across 2 × 2,800 step pairs). This means no observed perturbation to those fields; it cannot certify every internal PhysX state or missing native manifold.

## Strict scoring: don't confuse a goal with a safe kick

Pass requires all original gates: entered goal, observed legitimate instep impulse share >=0.95, ball speed proxy >=3.5m/s, *all original 29 individual measured qdot ratios <=1*, non-foot net contact <20N, and observed 14s recovery. These are **robot simulation criteria**, not elite speed or biomechanics proof.

| Scene and original control | Observer | Goal | Instep share | Ball-speed proxy (m/s) | Worst measured qdot/limit | Strict result |
|---|---|---|---:|---:|---:|---|
| AJ / v32 | OFF | yes | 1.000 | 4.8720 | 0.908 | PASS |
| AJ / v32 | ON repeat | yes | 1.000 | 4.8720 | 0.908 | PASS (not independent) |
| AJ / CMU10_02 | OFF | yes | 1.000 | 4.8068 | **1.332** | **FAIL: original qdot** |
| AJ / CMU10_05 | OFF | yes | 1.000 | 4.8230 | **1.272** | **FAIL: original qdot** |
| AK / v32 | OFF | yes | **0.208** | 4.7162 | 0.908 | **FAIL: instep impulse share** |
| AK / CMU10_02 | OFF | yes | 1.000 | 4.6199 | 0.908 | PASS |
| AK / CMU10_05 | OFF | yes | 1.000 | 4.6199 | 0.908 | PASS |
| AL / v32 | OFF | yes | 1.000 | 5.1506 | 0.908 | PASS |
| AL / v32 | ON repeat | yes | 1.000 | 5.1506 | 0.908 | PASS (not independent) |
| AL / CMU10_02 | OFF | yes | 1.000 | 5.1588 | 0.908 | PASS |
| AL / CMU10_05 | OFF | yes | 1.000 | 5.2203 | 0.908 | PASS |

**Unrepeated independent conditions: 6/9 (66.7%); all physical executions including repeated observer conditions: 8/11 (72.7%).** These percentages represent this small, preregistered matrix only; the 2 ON repeats must not be treated as independent samples. All 11 recorded 14s; none had a censored post-contact 1.25s window. All had a recorded non-foot net force peak of 0N under the available sensor; that is **not** a collider-pair-level self-collision exclusion.

**Important mixed causal result:** The CMU references convert AK's *measured* instep share from ~20.8% to 100% and satisfy the frozen gates in AK. In AJ, both CMU controllers introduce an original joint speed violation at 2.245s (CMU02 max 1.332 and CMU05 max 1.272) even though v32 stays below limit. In AL all controllers satisfy the existing gates. There is neither uniform CMU benefit nor uniform deterioration and no basis to ascribe the favorable AK change to learned natural human full-body coordination.

## Full-body 200Hz temporal audit

The independent `tools/v117_causal_safety_statistics.py` validates every raw SHA and two joint decoders, checks every one of 29 measured velocities, computes observed hazard onset/duration, separates actual approach / 400ms pre-contact / 1.25s post-contact / observed recovery, estimates torso/legs/arms joint mechanical work from simulated actuator torque × qdot, and measures left/right foot support-force **proxies**. Neither the simulated support proxy nor estimated motor work is human GRF/CoP/muscle power.

Six same-scene controller contrasts (3 scenes × 2 CMU conditions) first differ in *measured* all-29-joint velocities at **1.405s**. Actual contact occurs at 1.565–1.580s, so this reference has a short measurable precontact lead of **0.160–0.175s**. No pair demonstrates an early, continuous, learned full approach–stance–pelvis/torso–swing–arms–follow-through coordination sequence. In AJ, first original qdot hazard occurs later at 2.245s; consequently the pre-effect hazard claim from some earlier scenes is **not** reproduced here. Do not cherry-pick earlier cases. The actual total RMS 29DoF qdot differences are merely controller-response measures, not a naturalness metric or biomechanical identification.

Reviewable local figures (not in Git): `reports/v117_AJ_actual_qdot_support_com.png`, `v117_AK_actual_qdot_support_com.png`, `v117_AL_actual_qdot_support_com.png`; a separate real marker-kinematics diagnostic `v117_real_human_c3d_no_contact_truth.png`.

## Contact witness: negative / inconclusive is still important evidence

`tools/v117_contact_evidence_reconciliation.py` separately SHA-verifies the previous isolated GPU Tensor positive/negative controls and current G1 read-only filtered sensor pairs. The isolated Tensor ball/box witness validates **only** that isolated setup; the native channel did not provide a complete actor0/actor1, collider0/collider1, point/normal/impulse manifold. New G1 observers also failed to certify a direct left-wrist–left-hip native pair. A zero reported filtered Tensor contact force does not prove absence of self-collision. **Native contact-pair certification remains an OPEN safety blocker.**

Do not reclassify equal-and-opposite wrist/hip net forces as identified collider-pair ground truth. No learned policy was authorized to actuate the G1.

## Real human data eligibility: source measured versus inferred

`tools/v117_human_marker_temporal_eligibility.py` SHA-checks six genuine 120Hz C3D trials from only two subjects (3,358 measured frames), inspects 12 marker labels (toe/ankle, thigh, waist, shoulder, wrist), and derives temporal *kinematic* observables. It finds **zero** force-plate GRF/CoP trials, **zero** annotated true instep contact timestamps, **zero** professionally certified kick-force impulses and no verified athlete skill labels. Raw finite-difference marker speed spikes can be artifacts; do not equate peaks with calibrated ball speed, torque, athlete performance, or true contact. Cross-subject generalization is not established.

## Multi-phase full-body safety interface (not a validated new motor policy)

`tools/v117_witnessed_horizon_admission.py` adds a whole-trajectory 200Hz 29DoF admission screen across ordered phases: approach, stance, backswing, hip/trunk acceleration, instep contact, follow-through and landing/recovery. At every frame, `v107_motor_admission` checks 29 q/qdot/acceleration/torque limits, collision geometry/pair witness, stance support/slip and provenance; absent native pair evidence or unqualified human source => **fail closed**. The interface is executable and tested offline, but does **not** train, optimize or command a new kick on GPU.

## Reproducibility and boundaries

Frozen protocol: `docs/V117_FROZEN_PROTOCOL.md`, matrix: `configs/v117_frozen_matrix.json`, original write-once runner: `tools/v117_original_g1_batch.py`. Independent audit scripts: `tools/v117_causal_safety_statistics.py`, `tools/v117_contact_evidence_reconciliation.py`, `tools/v117_human_marker_temporal_eligibility.py`, `tools/v117_witnessed_horizon_admission.py`, `tools/v117_plot_physics_and_kinematics.py`.

The original 11 physics executions are already finished: the runner intentionally refuses to overwrite their JSON/log/audits; to re-execute, use a new, preregistered version and write-once output names under the same GPU lock. To **recompute without changing evidence**, use an interactive Python import of the three `audit()/inspect()` functions and compare returned JSON summaries against the existing write-once local `reports/v117_*.audit.json`. Tests are safe to rerun with `/home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests`.

SHA-pinned local evidence lives in `/home/you/projects/kinetickick-wt-v117/reports/v117_*.json`, and logs/figures adjacent to it. It is intentionally not published because the large raw files are in the Git ignore contract; do not imply remote report is a raw evidence repository.

## Clear research verdict and next work

**A: NOT CERTIFIED**: native wrist–hip actor/collider pair manifold remains missing despite meaningful read-only parity evidence. **B: PARTIAL**: 11 full true 200Hz 29DoF, support and work proxy windows acquired; force decomposition and native collision path remain proxy/incomplete. **C: OFFLINE SAFETY INTERFACE ONLY**: useful structural multi-phase screen, no high-quality human force data or trained validated whole-body motor policy. **D: COMPLETE FOR ORIGINAL-CONTROL FROZEN AJ/AK/AL MATRIX**: 11 real runs, 9 independent comparisons, 6/9 strict successes and 3 genuine failures preserved; not a new learned strategy. **E: CODE + TESTS + FROZEN PROTOCOL + REPORT**.

Next prioritize a PhysX-native positive collider/manifold witness with separated negative and original wrist–hip case; further qualify true human high-level instep motion with foot-ball contact timing and synchronized GRF/CoP. Then pre-register a genuinely new phase-conditioned whole-body controller and evaluate it on unseen geometries and seeds against v32/CMU without post-hoc tuning. Do not claim professional-style explosive shooting, naturalness or collision safety from this matrix alone.
