# KineticKick v0.1.96 — PhysX contact-report calibration and original G1 fail-closed audit

Date 2026-11-12. Clean new branch version/v0.1.96 from v95 parent ee7095fe4fc0ec2d121a5f38c485236c8e6e604f. **Frozen protocol commit before new physics: 3480862b90b7a391c0ae9976095b6aa10c4c10fc.** Frozen config SHA256: 68c6066c001713488e4c3f0fc5293073013d01fd61caedcaf7861312ff4610a7.

## Environment and source preservation

Tang online, RTX4090 initially idle, initial disk 37GiB (>20GiB). All nine new Isaac PhysX trials serialized via the preexisting shared flock, with unique write-once local-only raw JSON/log/audit names. Prior CMU/C3D, model weights, original G1 physics, old version branches and worktrees untouched. These are **isolated sphere-box PhysX experiments**, not original G1 shooting or learned-policy actuation.

## New physically executed matrix

Nine fully completed real 180-step, dt=0.005s PhysX runs (**1,620 actual frames**). All matched frozen config and passed SHA-backed independent physics-step verification.

| Case | Device | Reporter | Outcome |
|---|---|---|---|
| cuda_off | GPU | off | 180 real frames, observed contact dynamics |
| cuda_manual_native | GPU | body schema manually | 180 frames, 0 native contact header |
| cuda_spawn_native | GPU | spawn-time sensors | 180 frames, 0 native header |
| cuda_both_native | GPU | rigid body and collider schemas | 180 frames, 0 native header |
| cuda_spawn_dual | GPU | spawn-time sensor + tensor + native | 180 frames, tensor force peak **667.0803 N**, 0 native header |
| cuda_separated_dual | GPU | deliberately separated positive/negative | 180 frames, **0 filtered tensor contacts** |
| cpu_dynamic_off | CPU | dynamic box without report | 180 frames |
| cpu_dynamic_native | CPU | dynamic box, spawn-time report | 180 frames, 0 native header |
| cuda_dynamic_native | GPU | dynamic box, spawn-time report | 180 frames, 0 native header |

Five frozen saved-state observer OFF/ON pairs passed **5/5 exact parity** for recorded sphere z, vz and final box position, maximum saved difference 0. This is not complete internal-state parity on original G1. Genuine native actor0/1 and collider0/1, normal, contact point and impulse certification remains **0/9**. Neither zero native reports nor inverse net-force vectors prove a specific contact partner. The callback failure persists across schema authoring and dynamic-versus-kinematic partner; do not infer its root cause without runtime debugging. Each child's close-completed marker is missing although its independent physical trace and exit code completed; this lifecycle caveat is preserved.

## Original free-root 29DoF G1 dynamical failure audit

Independent SHA verification re-read four **historical** original G1 E/F episodes at real 200Hz, with all29 measured joint velocities, support load, group lag proxies, nonfoot events and strict physical gates. These four are not new physical trials.

| Historical G1 trial | Real frames | Independent strict result |
|---|---:|---|
| E / v32 | 736 | FAIL: instep share, speed, nonfoot, full 14s and recovery |
| E / CMU02 | 2800 | FAIL: low speed despite legal instep |
| F / v32 | 320 | FAIL: goal, nonfoot, 14s recovery |
| F / CMU02 | 320 | FAIL: goal, nonfoot, 14s recovery |

F rows terminate before impact+1.25s, so recovery and missing post-contact measurements are censored, never filled in. F event-window nonfoot net-force maxima 180.01 and 192.98N are **not** original collider manifold evidence. Original historical strict success **0/4**. No elite human motion was learned by the robot-only trajectory generator; group peak-time proxies are not human truth.

## Independent test, artifacts and safety status

Independent audit script tools/v096_independent_evidence.py revalidates all 9 raw/log/receipt SHA, all 180 real consecutive steps per case, actual calibrated positive CUDA tensor witness and separated negative, five parity pairs and four historical original G1 strict results. tools/v096_evidence_figures.py generated two local ignored evidence graphics (no invented frames).

Batch manifest SHA256: 63313fd85b2c0f26ccafed0c337376a246ace222109c0ebc73e8e1cb5a62ef6a.
Independent audit SHA256: 650a26d4ca195ad346bfb0dac7aa024761fd59901bdb446a12a578a0d26217b0.
Real calibration plot SHA256: 177e1b7719c727fe7c7cfabd7ef00a12995dd306f15b076b1269817b281476f1.
Historical G1 failure plot SHA256: 160d86a77eb31d69c27f91969674e33d20cab1344bdd4ce0a1cad02b43e5fc98.

**This batch: new real isolated PhysX runs 9, original G1 shots 0, learned-policy G1 trials 0, valid original G1 new success rate undefined.** Native witness and human natural-coordination learning remain blocked. See docs/V096_WHOLEBODY_LEARNING_AND_CONTACT_GATES.md.

## Reproducibility

Pre-registration: docs/V096_FROZEN_PROTOCOL.md and configs/v096_frozen_matrix.json.
Execution: tools/v096_contact_batch.py, one shared GPU mutex, unique local-only outputs, fail-closed hardware guards.
Independent read-only audit: tools/v096_independent_evidence.py.
Figures: tools/v096_evidence_figures.py.
All tools use write-once output paths; rerun requires new version/paths rather than overwriting prior physics.
Complete test results, source clean-state and final remote SHA verification are recorded below after execution.

## Final tests, resource verification and delivery
Final full project test scope was the actual project tests directory, with the original IsaacLab Python environment and one CPU BLAS thread:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
**1072 passed, 2 skipped, 69 subtests passed, zero failed, 30.02s.** Actual write-once test log at reports/v096_full_suite_tests_only.log SHA256 cbd39dc60bd860e4994b03bb2904ed01c1c22dca1b83f49295a6631ef1294720.

An initial broader invocation including both tests and the root directory stalled during collection with no output for more than five minutes. Only the v96-owned test PID was terminated; the empty failed-attempt log was preserved at reports/v096_full_suite_final.log and not misreported as a success. The clean scoped full tests-directory run above completed with exit code 0. 1933 missing local ignored test-data references were recreated as symlinks to immutable prior original assets, not rewritten source files. Full test suite and protocol files were not weakened after the physics evidence was viewed.
