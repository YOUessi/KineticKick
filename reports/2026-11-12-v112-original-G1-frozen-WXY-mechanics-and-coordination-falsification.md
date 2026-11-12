# KineticKick v0.1.112 — Original G1 W/X/Y frozen PhysX matrix and independent causal-motion falsification

**Research status:** observed original controllers, **not a new learned natural-kick policy**. Independent analysis completed 2026-11-12 after a previous execution was interrupted. The **10 physical simulations in this batch were already executed and write-once recorded on Tang before this audit resumed**; this completion stage did not create additional GPU physical episodes. The original source/protocol was frozen at `f9bbdb1af416b302d144d7ff3eb42c47ed8f50d2` before original batch execution, with parent `75efc4ddd88b65dd9d931e7d8b708d77b9d2d465`.

## Original, not reconstructed, physics evidence

- 10/10 original **free-root 29DoF G1, CUDA PhysX at 200Hz** scored and preserved, **28,000 certified steps**. Every run completed actual 14s simulation (2,800 × 0.005s) and the independently analyzed pre-impact 0.4s/post-impact 1.25s true sensor windows.
- Six **independent scene/controller combinations**: W/X/Y seeds 2333/2341/2357, each with unchanged original v32 and CMU10_02. Four extra OFF/ON tensor diagnostic repeats (W and X) are **not independent success trials**. Original batch: `reports/v112_original_g1_batch.audit.json`; coupled dynamics: `reports/v112_200hz_coupled.audit.json`; separate matched-control analysis: `reports/v112_coordination_falsification.audit.json`.
- 4/4 OFF/ON exact shared 200Hz trace, original shot record and all compared physical-event fields parity pass; maximum observed delta **0**. This says nothing about unrecorded states or a certified native contact manifold.
- **Strict success:** 2/10 executed repeats; **2/6 independent scene/controller conditions**, both Y only. No controller shows a scene-wise strict advantage: each succeeds on Y and fails on W/X. Every failure remains a failure even if the ball enters the goal.
- Per-episode original JSON ~30MB, logs, audit JSON/NPZ and failure PNG remain local and ignored by Git. Original SHA256 and path are recorded per run in the batch audit; do not compress, replace or fabricate any raw evidence.

| Frozen case | Native goal | Measured instep impulse | Ball speed proxy (m/s) | Full 14s | Strict |
|---|---:|---:|---:|---:|---|
| W/v32 OFF, ON | no | 32.59% | 5.0071 | yes | 2 FAIL (goal, instep) |
| W/CMU02 OFF, ON | no | 32.71% | 5.0083 | yes | 2 FAIL (goal, instep) |
| X/v32 OFF, ON | yes | 62.79% | 4.4748 | yes | 2 FAIL (instep) |
| X/CMU02 OFF, ON | yes | 63.10% | 4.4964 | yes | 2 FAIL (instep) |
| Y/v32 OFF | yes | 100% | 3.8420 | yes | PASS |
| Y/CMU02 OFF | yes | 100% | 3.8384 | yes | PASS |

All ten satisfy the *original 29DoF qdot envelope* (observed peak ratio ~0.909 of original limits), non-foot net-contact cutoff and complete 14s recovery. W loses both goal and instep legality; X enters goal but **fails the 95% instep impulse threshold**. A ~5m/s speed proxy is not evidence of elite human-style kicking or actual impact impulse transfer. Original ball-speed post-contact proxy is not ground-truth ball impulse.

## Independent 200Hz biomechanics and falsification

All frozen original raw JSON are independently SHA256-checked. Coupled dynamics computes each group's actual joint-speed RMS, COM displacement, root height, left/right support-force **proxies**, qdot ratios, non-foot events, and contact-relative phase windows. It neither fills censored data nor manufactures force-plate GRF/CoP or native actor/collider paths.

An additional post-hoc controller comparison reads only the **six OFF trials** for W/X/Y, checks identical 200Hz grid and identical joint order, and measures physical 29DoF qdot differences.

| Frozen scene | CMU-v32 ball speed proxy (m/s) | First actual 29DoF qdot delta >1e-7 (s) | Time before contact | Pre-0.4s qdot RMS delta (rad/s) |
|---|---:|---:|---:|---:|
| W | +0.001222 | 1.405 | 0.165 s | 0.003840 |
| X | +0.021601 | 1.405 | 0.175 s | 0.033240 |
| Y | −0.003668 | 1.405 | 0.170 s | 0.004313 |

These are small/heterogeneous effects of the original **late CMU phase prior**, not evidence that professional human whole-body coordination was transferred. The short pre-impact effect window is inconsistent with an experimentally demonstrated full approach/support/hip-trunk-leg-arm learned sequence. Simultaneous group-speed peaks alone cannot establish human coordination or causal power flow.

For W/v32, pre-impact left/right foot-force-proxy means were 253.37/96.69 N and COM horizontal displacement was 0.959 m across 0.4s. These are G1 *simulated force/speed* observations, not validated biological GRF/CoP or impact foot-ball power attribution. Their relevance to post-impact recovery is observable only within the recorded 14s. No elite human kick supervision is presently certified.

## Contact-pair truth gate stays closed

Tensor sensors explicitly matched USD wrist and hip *source/filter link paths*, but ON runs yielded **zero sensor-observed pair-data samples**, without a simultaneously calibrated native actor0/actor1/collider0/collider1 contact manifold positive control. A sensor match by path is not proof of collision; neither are equal-and-opposite net forces. Conversely zero readings here are **not proof that a wrist/hip collision cannot occur**. True native contact pair geometry, contact point, normal, impulse and shape-to-actor mapping are still missing.

## Independent test and publication guardrail

The provenance/frozen-command evidence (including previous v111 missing-motion failure) must remain immutable. v112 no longer depends on the dirty v108/v109/v110 worktrees. New output analysis files are exclusively additional post-hoc tooling and do not mutate the original experiment protocol/controller/configuration. End-to-end tests verified on Tang: **1260 passed, 2 skipped, 69 subtests passed, 0 failed (35.89s)** using the IsaacLab Python environment; all 7 new coordination analysis tests passed. Test logs remain local under `reports/v112_full_pytest_final.log`. This tests the software and evidence integrity, **not the learned human-style motor policy**, which remains disabled. Never convert these original-controller strict passes into a result for an unactioned learned 29DoF model.

**Paper hypothesis to test later:** multi-phase human movement data with measured ground reaction and foot-ball impact, retargeted to a support-aware contact-safe whole-body motor controller, should improve held-out approach timing and safe kick impulse without increasing self-contact or qdot violations. Falsifiers: no improvement vs original v32/CMU across new frozen ball/seed splits, high post-contact falls, or no verified human impact evidence. Freeze training data eligibility, all ablations, exclusion rules, and hardware checks *before* new policy training.

**Next dependent batch:** certified actor/collider contact positive/negative controls and original G1 OFF/ON; qualifying multi-subject elite human kick data; support-foot contact/slip and torque-aware 29DoF command execution. Do not actuate a learned candidate or claim pro-grade blast kicking until all native safety/physical gates and independent scene/seed tests pass.
