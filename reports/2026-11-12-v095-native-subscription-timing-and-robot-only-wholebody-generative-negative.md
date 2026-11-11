# KineticKick v0.1.95 — nine isolated PhysX calibration controls + real 29DoF robot-only generator ablations

Date: 2026-11-12 (Asia/Shanghai). Branch: version/v0.1.95. Clean parent version/v0.1.94 at 76ed1b1ce2d747471a5a700771a5d224027248bf. Frozen protocol commit BEFORE physical experiments: 6f6715714a761c4dbb76e7bc3c8df097ee43f744 (initial freeze 79f3edb748298deb92a643e9249cccec174e1797). Exact frozen matrix SHA256: e6d8e8c5720c5e3031e5188edd2ce87dc40266eda3ff2fcbd1e1c6adb71d3ed5. No original robot controller changed, original physical evidence/CMU/C3D and model checkpoints preserved.

## Preflight and safety
Tang Ubuntu RTX4090 online; initial and end measurements had no foreign CUDA compute jobs. ~37GiB free (>20GiB), unchanged shared experiment flock at /home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock. All GPU trials serialized within this existing exclusive lock. Fresh isolated v95 worktree branched from clean v94, all older branches/worktrees untouched and not force-pushed. Nine PhysX children each wrote new write-once local ignored JSON/log with independent SHA audit. All 9 processes returned 0 with 180/180 actual 5ms physics frames (no retries). App-close completion marker absent even when runtime physics evidence was complete; Kit/Python exit behavior remains unresolved and must not be confused with physical success. Local-only raw and PNG are **not** pushed.

## A: native contact timing positive/negative control
Pre-registered nine simulations with **actual** PhysX sphere versus static kinematic rigid box, total 1620 true 200Hz physics steps, plus separated-box negative control. These are isolated contact-instrument calibration runs, not G1 shooting tests.

| Frozen trial | Frames | Positive CUDA tensor contact | Native true collider pair+manifold | Physical condition |
|---|---:|---|---|---|
| cpu_pose_off | 180 | not instrumented | not instrumented | observed collision |
| cpu_native_on (subscribe pre-reset) | 180 | not instrumented | NO (0 callbacks/headers) | observed collision |
| cpu_native_after_on (subscribe post-reset) | 180 | not instrumented | NO (0 callbacks/headers) | observed collision |
| gpu_pose_off | 180 | not instrumented | not instrumented | observed collision |
| gpu_native_on (pre-reset) | 180 | not instrumented | NO (0 callbacks/headers) | observed collision |
| gpu_native_after_on (post-reset) | 180 | not instrumented | NO (0 callbacks/headers) | observed collision |
| gpu_tensor_on | 180 | YES, 20 sampled points, peak 667.0803N | not instrumented | observed collision |
| gpu_both_on | 180 | YES, 20 sampled points, peak 667.0803N | NO (0 callbacks/headers) | observed collision |
| gpu_separated_on | 180 | 0 (negative) | not instrumented | observed separation |

Native subscription timing before vs after reset did NOT recover native true actor/collider data in this configuration. The positive CUDA RigidContactView filtered sensor root (/World/ProbeBall) vs box root (/World/ProbeBox) is an observed PhysX pair filter, but does NOT give genuine native shape/collider IDs, full manifold, impulse ground truth or wrist--hip contact on G1. Native report remains **uncalibrated**, and zero reports CANNOT disprove self collision. CPU native and prior CPU tensor remain uncalibrated. This is an important reproducible negative.

Independent SHA-bound audit verified 9/9 physical runs, 2 independent tensor positives, separated negative, all 180 real physics steps, and **6/6 OFF/ON pair exact saved trajectories** (step index, sphere z/vz, box position; no claim to identical unrecorded internal physics or G1 state). Original G1 foot-ball or wrist--hip native collider contact certification: 0. Independent original G1 shooting physics added this batch: **0**. Denominator for new trained-model physical success: **none**.

Local JSON audit SHA256: 2c094d20bd0747bb0e0fc8e7fd874c6806801c01c3d47f6621a43108fb4ae57e (reports/v095_independent_evidence.audit.json).

## B: original G1 200Hz coupled dynamics and hard gate failures
Independently read v91 original CUDA PhysX G1 source SHA + original audit for four past E/F control episodes, including all actual29 qdot speed ratios, bilateral support loads, wholebody COM, true foot-ball contact time, nonfoot force, goal, ball-speed proxy, full +1.25s after impact where available, and 14s recovery. Actual result remains historical **0/4 strictly passing**, not added to this batch's new G1 count.

- E/v32: 736 true steps; impact ~1.580s; missed >=95% instep share, ball speed floor, nonfoot safety, full 14s and standing recovery.
- E/CMU10_02: 2800 true steps; impact ~1.575s; legal instep but ball-speed below 3.5m/s; no relaxing the speed threshold.
- F/v32 and F/CMU10_02: 320 steps each, impact ~1.565s; failed goal, nonfoot safety, 14s recovery; postimpact +1.25s **not observed** and is treated as censored (no invented postrollout).
- Wrist/hip opposite net link forces are indirect, not genuine verified native contact actor/collider pairs. Root tilt alone is not a hard fall certificate.

Plots produced locally ONLY from actual sources: v095_calibration_contact.png (sha256 3ff52ad7f0854fdfe99355ce3229be0edb909716f10bfaf71c57927d156a7022), v095_real_g1_nonfoot_failures.png (sha256 71dce15e591cf2b010612aeb0d7c2128df9e14f601c624b996c1164ee6f62759).

## C+D: actual 29DoF temporal generator and ablations (offline, NOT physical actuation)
Implemented actual trained 0.4s 50Hz future **all 29 normalized joint velocity trajectories**, and integrated qdot with the contemporaneous measured real29 joint position and velocity limits to generate 20x29 proposed positions. Multi-output ridge with fixed alpha10, 6 future temporal anchor knots, trained on SHA-validated original G1 PhysX v84 C/D trajectories only; independent *scene* E/F historical v91 original G1 trajectories evaluated read-only. Original joint group partition: support leg6, strike leg6, trunk3, counterarm7, other arm7. This is a genuine robot-motion prediction/generation experiment, NOT high-quality human kick imitation, a dynamic-contact controller, or physically tested new candidate. The same E/F cases were seen in earlier research, so they do not constitute a fresh publication-grade never-seen test.

| Retrospective original G1 heldout | Original frames | Candidate 0.4s windows | State-only MAE | State+support MAE | Reverse-phase negative MAE | Arms frozen MAE | Right-leg only MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| E_v32 | 736 | 32 | 0.030449 | 0.030426 | 0.074143 | 0.040220 | 0.047419 |
| E_cmu02 | 2800 | 135 | 0.017052 | 0.017015 | 0.028365 | 0.018763 | 0.019649 |
| F_v32 | 320 | 11 | 0.026321 | 0.026042 | 0.092582 | 0.042260 | 0.056934 |
| F_cmu02 | 320 | 11 | 0.026321 | 0.026042 | 0.092582 | 0.042261 | 0.056934 |

MAE is normalized to *actual G1 original joint speed limits*. Contact features bring tiny, inconclusive improvements. Shuffled-contact negative control is close to full model (E_v32=0.030470 vs 0.030426), while some group-peak timing worsens. F pair nearly identical pre-failure; not 2 independent human motions. Reverse or freeze-arm negative control worsens G1-trajectory error, but this does not prove that natural counterarm coordination has been learned. Any generated velocity/position must still pass actual angle/torque/speed, stance no-slip, original collider clearance, real GRF and original physical 14s recoverability gates. No interface commands G1 actuators.

Original exploratory fitted score artifact remains write-once locally, then deterministic additional per-proposal joint-position integration was added without changing alpha or physical data. Final improved candidate qualification JSON sha256 dd4c1d57b1a0c7fdd51668e541a04b9a6fc76c8a053856392e8b89906499fe48. First qdot-only score artifact preserved sha256 72396ad96195ebbe8ef3e02c0c80759c6b0cbd7f900d6c702ee3ff9c88eb9627 (source was NOT overwritten). Trained weights remain local NPZ. Robot-only generation plot local: v095_generator_ablation.png sha256 2da97febb3f4fd361e00501abf923aadd41e6d7fb11a9aa13aa850dda6e2683c.

## Data qualification + falsifiable method claim
No reviewed local CMU C3D is a verified professional instep impact with measured synced GRF/real foot-ball impulse (v92). Six clips belong mainly to a single subject and suffer marker dropout. GroundLink (https://csr.bu.edu/groundlink/) includes human motion with real left/right GRF/CoP but is not validated expert soccer foot-ball impulse; only possible support-learning input after provenance/license validation. CMU mocap (https://mocap.cs.cmu.edu/) may teach limited geometry style, not true observed instep impact. SportsPose (https://arxiv.org/abs/2304.01865) is geometry, not instep force truth.

A later properly preregistered model could learn phase-aligned temporal dependencies of planted stance, pelvis/torso lead, strike-leg distal energy, left counterarm and follow-through; test newly licensed human heldout subjects and independent original G1 unseen ball locations/seeds/friction. Compare full-body, foot-only, arms frozen, reverse phase and state-only across strict physics + naturalness reviews. The **high-velocity humanlike shot hypothesis remains unproven**. At this version native contact pair positive calibration fails, so v94 offline joint/stance/shape projection is still forbidden from motor deployment.

## Tests, scripts, immutable source and reproduction
Relevant code: tools/v095_contact_probe.py; tools/v095_contact_batch.py; tools/v095_independent_evidence.py; tools/v095_trajectory_generator.py; tools/v095_failure_figures.py; three new pytest files test_v095_contact_contract.py, test_v095_independent_evidence.py, test_v095_trajectory_generator.py. Frozen contract: configs/v095_frozen_matrix.json and docs/V095_FROZEN_PROTOCOL.md. Training and audit records in local ignored reports/v095*.json/npz/log/png. Re-run experiments only after creating a fresh branch and changing write-once destinations: DO NOT overwrite old raw evidence.

Initial full test on isolated checkout failed 27 due missing gitignored source assets; after adding ignored source URDF and data/third_party/experiments/report JSON read-only links, intermediate rerun failed 2 due additional missing original logs. Both failed test logs are preserved as negative environment-provisioning evidence. Links were set only where files were absent, target original bytes never rewritten. Final complete suite result is recorded in the final appendix after completed regression. All test metadata must refer to the final actual local log SHA, not to claimed hypothetical tests.

Prior art context: 2026-11-12 natively PhysX contact stream documentation https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html . Native callback zero has no evidential weight without positively certified actor/collider manifold. The next independent batch should examine native scene bindings/physics subscription lifecycle AND obtain 2-way actor+shape IDs in positive control first, then read-only actual G1 original failure pair OFF/ON full-state parity before any motor actuation. Seek real elite human GRF/impact data and freeze a genuinely unseen test set before training high-amplitude support/impact-aware whole-body action.

## Final clean-source regression and data-protection audit (completed)
After fixing only local ignored worktree references to immutable, original v94/v91 sources (3 original paid URDF files, 955 original JSON and 933 original logs, plus existing data, media, experiments, third-party and checkpoint trees), the full final suite passed **1066 passed, 2 skipped, 69 subtests passed, 0 failed** in 30.77s. Command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests . Final write-once local test log reports/v095_full_test_suite_final.log SHA256: 84d3e093c9563b1d09176baad4f668c359651dbad6be7659f58abc809de32622.

Preserved failed first provision (27 fails) and intermediate test run (2 fails); they were environment missing-source failures, not hidden as algorithm successes. Original ignored file symlinks refer to source data and logs; neither originals nor prior physics recorded files were overwritten. Final frozen config hash independently checked against version/v0.1.95 frozen Git commit and equals e6d8e8c5720c5e3031e5188edd2ce87dc40266eda3ff2fcbd1e1c6adb71d3ed5. Tang after experiments: GPU compute idle, root filesystem ~37GiB free. Output includes 9 new isolated PhysX probes (9/9 instrumented real 200Hz physics; not original G1), 0 new G1 trials, and 4 independently re-audited *historical* original G1 physical failures (0/4), never conflated.

No native original G1 collider manifold, no learned elite human GRF/impact target, no physical test of generated new 29DoF action, and no verified high-speed natural shot are claimed.
