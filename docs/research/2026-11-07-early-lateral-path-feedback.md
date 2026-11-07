# v0.1.34 — one fixed early lateral path-feedback candidate

Written before GPU execution on `version/v0.1.34`, from v33 `d230be3a9c6e1f98fd7758d9d1b67464387f7c5a`. Frozen comparison is the exact same-condition v32 controller and learned recovery, not a new human coordination model. The original 19 physics/source pins, v32 checkpoint and old acceptance gates remain unchanged.

## Hypothesis and controller

The v33 left-near-slow diagnostic measured a nearly goal-aligned foot face but approximately 19.5 degrees of foot material-point velocity error. It also measured a leftward tangential ball impulse. This supports testing the motion path; it does not establish a single-capsule impact predictor, friction-free counterfactual or guaranteed controller gain.

Use the already declared fixed foot reference `[.06, 0, -.025]` m in the right ankle-roll link frame. It is a kinematic marker, not the dominant collision capsule. Compute its actual world position and rigid-point velocity `v_link + omega × R p_local`. With `g = normalize((goal-ball)_xy)` and `l = (-g_y,g_x)`, use `e = l·(p_ref-ball)_xy`, `v_cross = l·(v_ref-v_ball)_xy`, and `s = clamp(-.25 e - .04 v_cross, -.03, .03)` m. Relative velocity is the instantaneous transverse projection, not the derivative of a rotating goal line. Before touch the goal and ball are measured each command.

A single damped six-dimensional Jacobian solve at **measured** leg q/root pose requests XY offset `s l`, zero vertical offset and zero additional orientation. Uniformly scale the resulting six-joint increment to at most .06 rad/component. Add this increment to the unprojected original v32 PD reference; never clamp that entire reference to the soft position interval. DLS is a local approximation: zero angular request does not promise zero realized foot rotation or a chosen ball impulse.

Earliest onset is .60 s, with a .12 s quintic ramp. Multiply by the existing continuous measured-context gate, preserving its numerical zero dead zone, and by measured support ramps: left normal load 40→80 N and right load 40→0 N. Requested increments and support changes use the same uniform slew cap, .015 rad/component/20 ms. These are added-reference bounds, not new physical joint limits. The first ball-to-any-robot-body normal-force evidence above 2 N is latched at 5 ms; freeze the then-applied increment and fade it with a .20 s quintic. A 2.10 s no-contact timeout also exits. No re-aiming after first touch, no ball/root state writes, and no additional torque/velocity pulse. One unchanged `PostShotRecoveryTask.commands(None)` call per command retains the pinned v32 recovery network. Its trajectory is not guaranteed unchanged when the approach changes. Any overlap is logged.

## Fixed scope and stop rules

There is no PPO training, gain search or per-case table. `configs/early_path_study_v1.json` fixes one disabled-feedback 14 s replay, then at most four candidate 14 s episodes:

| Order | Observed development case | v32 speed m/s | v32 final continuous standing s |
|---|---|---:|---:|
| 1 | heldout_v028_left_near_slow | 4.296256512636222 | 7.670 |
| 2 | heldout_v028_near_slow_17 | 4.737232056067209 | 9.915 |
| 3 | center | 4.631403423758371 | 8.565 |
| 4 | combined_right_faster | 5.525366230485013 | 8.705 |

The disabled candidate must exactly retain the original physical traces, actual commands/kinematics, contacts and outcome on the first case. New diagnostic fields and file/source metadata are excluded from byte comparisons. This verifies the added read-only observation/gate path, not learned performance.

After enabling the single fixed candidate, stop immediately on physical termination, any actual qdot above original limits, illegal body-ball/instep surface, incomplete 14 s, original recovery failure, or outgoing-speed retention below 95% of that condition's v32 value. Both old misses must become goals before center/strong. A miss with otherwise legal physics may be followed by the second predeclared miss, but no fresh case or parameter retry is allowed if either remains a miss. Neither of the two unused fresh conditions is inspected or executed here. Old developmental conditions do not establish held-out generalization.

Record observed lateral path/velocity, original contact surface, shot speed/direction, goal and complete recovery. Do not substitute reduced IK error or code-test counts for physical acceptance. Preserve failures and exact acquisition sources. P2 qualified human 3D and P3 learned coordination remain unresolved; this is a bounded P0 engineering experiment.
