# KineticKick v0.1.102 — CUDA PhysX positive and G1 pre-step shutdown blockage (2026-11-12)

## Source and safeguards
Clean parent version/v0.1.101 `fbfd3e35016538e6168af063b5d64f19451f7b9e`, separate `version/v0.1.102`; protocol pre-frozen `78fdc31062c022514f1e8f7aeb76dfd80e957bbe` (SHA256 `544e12608e2ce4bc30e1c61ac87e73202ed5e3cfa148f93a5e09ee6217f8c28a`). GPU exclusive flock, no foreign compute app, disk >=35GiB. Old source/CMU/C3D/PhysX JSON not modified. v101 invalid CLI is retained as a prior separate negative.

## CPU and actual PhysX: independent dependency gates
- Official 29DoF ONNX CPU metadata/session control: passed in 0.017324s, original model SHA256 `489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`. This does **not** certify running the policy inside Isaac Sim.
- Isolated CUDA PhysX ball/box **180 real 200Hz steps** (0.9s): validated ball-box dynamics, tensor pair contact positive, **113 nonzero contact rows**, peak **667.080322 N**. Full raw SHA256 `c54aa11e5af811081953e54bd73bcda91b843e2be0ca368de08df228f2bec4e3`. Native contact-pair witness STILL FALSE, not equivalent to a real actor/collider manifold.
- One unchanged original 29DoF G1 L/v32 OFF attempted, seed1601, frozen ball offset(-.095,-.035), exact v080 source. Process exceeded own 120s watchdog (120.571s); **0 verified real G1 physics frames, no raw trace**, log SHA256 `88397c045028558d3c189cac27c090b96b7e5e0a38f720a5181de5fa5843296f`. All five later L OFF/ON G1 experiments **not executed** under frozen fail-closed plan.

**Correct counting:** new isolated real PhysX experiments **1**, 180 steps; new certified original G1 shooting experiments **0**, strict shooting success **not applicable**, not 0/6.

## New diagnostic evidence — failure location narrowed, not solved
Eight independent periodic all-thread Python faulthandler samples show the main thread repeatedly inside `isaaclab.sim.simulation_context.SimulationContext._app_control_on_stop_handle_fn -> render -> SimulationApp.close`, invoked from `v076_eval_guarded.py:192`. The G1 log has the second actor MLP initialized (154 input features) but *no* `SEQUENCE_GROUND_COLLIDERS`, `FREE_SHOOTING_READY`, or actual 200Hz physics frames. This differs from earlier v99/v100 logs that ended before that MLP; the only positively demonstrated stuck location is the **shutdown callback**, not ONNX initialization or a proven C++ physics deadlock.

In original v076, `held_env` is assigned only once `run(...)` returns. If it exits with an exception during setup, `finally` calls `app.close()` without the post-physics shutdown callback guard (guard only executes when `held_env is not None`). This is a **plausible masking mechanism**, not proof of the initiating exception. Next startup diagnosis should capture `sys.exc_info()` *at the v076 finally / pre-app-close line* via a bounded read-only tracer; no model gains/actuator settings may be changed. If a missing ignored asset is the root cause, correct only the local evidence symlinks and rerun under a NEW frozen version, never overwrite this failure.

## Science and gating
Previous 15 G–K physical episodes remain historic only (5/15 strict), with censored failed recoveries. No new learned human shooting strategy or elite synced GRF/CoP data. A ball/box positive does not certify an original G1 collision pair. No wrist-hip collider path/manifold, no trained safe full-body controller motor validation, no paper claim of natural full-body ballistic instep strike. Raw logs/stack samples/JSON stay on Tang, not GitHub.

## Final regression and preserved failures
After adding only 2,006 read-only local symlinks to existing ignored historical artifacts (the missing original raw JSON caused 25 initial collection/test failures), full repository CPU tests under original IsaacLab Python: **1126 passed, 2 skipped, 69 subtests passed, 0 failed**, 33.02 seconds. Both initial failure and final logs are preserved locally. All original qdot/nonfoot/instep/14s recovery gates remain unchanged. New tracked worktree files unchanged from freeze except this text report.
