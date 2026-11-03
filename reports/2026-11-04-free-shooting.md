# Free-root shooting progress — 2026-11-04

Branch: `version/v0.1.1`. Three physical strike styles now meet the declared bounded acceptance gate. Rendered/live review is being completed before publication.

## Verified baseline and integration

- Existing Root-V2 baseline: genuine contact at control step 161, peak ball speed 1.06753 m/s; substantial sideways displacement and no verified aimed goal.
- Official PAiD recurrent policy integrated with the exact 160-observation contract, verified joint ordering and per-trial LSTM reset. Model remains local, SHA-256 `489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`.
- Physical PAiD G1 remains floating-base; no new foot geometry. Ball is 0.43 kg, radius 0.11 m. Trial code writes poses/velocities only at initialization and prevents reference-wrap teleports.
- Contact evidence uses actual PhysX contact points/normals at every 5 ms substep. The 1.6 m × 1.1 m goal test requires the entire sphere to cross. Standing also checks foot support and rejects non-foot support.

## Exploratory physical results

1. Single official-policy trial at ball (0.14, -0.93) m, goal (0, -3.7) m: approached, inside-foot contact at 2.335 s, peak normal force 244.82 N, peak ball speed 4.949 m/s, goal crossing, and 4.89 s of continuous measured standing by the 8.2 s trial end. Requested instep label was correctly rejected because the measured patch was inside-foot.
2. Twelve-position grid: x={0, 0.14, 0.28}, y={-0.65, -0.8, -0.95, -1.1} m. All 12 approached, contacted with the inside foot, scored and were standing at the 7.2 s endpoint. First-impact points/normals independently passed the audited PAiD inside-foot envelope check. These are exploratory cases under one deterministic reset configuration, not a frozen multi-seed robustness result.
3. Thirteen-reference survey: all ten standard motion references used inside-foot contact. The three stylized clips did not establish usable alternative surfaces. Some standard trajectories destabilized after longer final-frame holds. The policy's success with inside-foot contact does not establish instep or outside-foot control.

Local evidence (ignored): `reports/v0_1_1_baseline_contact.json`, `reports/v0_1_1_official_nominal.json`, `reports/v0_1_1_official_grid.json`, `reports/v0_1_1_reference_sweep.json` and their logs.

## Controller and frozen acceptance

- Inside-foot: the original ball/goal-conditioned recurrent policy.
- Instep: a smooth, limited hip-yaw / ankle-pitch correction during the strike, then return to the recurrent controller.
- Outside-foot: coordinated, torque-actuated hip/knee/ankle targets computed from the audited foot kinematics. Returning to normal control by frame 128 (2.56 s) was necessary for recovery. Long holds caused falls and were rejected.
- The frozen configuration is `configs/free_shooting_v1.json`, SHA-256 `eca34699ea712aec2df15d6d9e3e6222d937fbf0cf3c5ae400a99e03b083e9f1`.
- No new model training was performed. These are calibrated physical controllers around a pretrained policy, not a newly learned general-purpose soccer policy.

Each style was evaluated at the following five conditions for seeds 711 and 712. Each trial lasts 8.2 s. Seeds actually perturb initial root x/y by +/-1 cm, root yaw by +/-0.01 rad and joint positions by +/-0.005 rad. They are not duplicate deterministic resets.

| Condition | Ball x,y (m) | Goal x,y (m) |
|---|---|---|
| 0 | 0.14, -0.93 | 0.00, -3.70 |
| 1 | 0.08, -0.93 | 0.15, -3.70 |
| 2 | 0.20, -0.93 | -0.15, -3.70 |
| 3 | 0.14, -0.87 | 0.00, -3.85 |
| 4 | 0.14, -0.99 | 0.00, -3.55 |

| Style | Seed 711 | Seed 712 | Minimum final standing window | Peak ball speed range |
|---|---:|---:|---:|---:|
| Inside | 5/5 | 5/5 | 4.875 s | 4.67–4.95 m/s |
| Instep | 5/5 | 5/5 | 4.555 s | 1.39–3.25 m/s |
| Outside | 4/5 | 4/5 | 3.945 s | 1.69–1.94 m/s |

**28/30 full-chain successes.** Both failures are outside-foot shots in condition 3 that miss the goal; both robots still recover standing. No trial fell or triggered a reset. All 30 trials have the requested measured first-contact surface and 100% of the measured foot-ball normal impulse on that surface. The checker independently replays every claimed success from the 5 ms traces and rejects duplicated seeds, duplicated conditions, changed controllers and shortened successful traces.

The gate specified before tuning was at least 4/5 per style per seed; it is met. This does not mean 100% reliability. The tested ball/goal range is narrow, the approach follows one learned motion, and perception uses simulator state. Arbitrary starting positions, moving balls, physical robot deployment, and motion naturalness are not validated. The sphere has the declared sliding friction but no explicitly calibrated rolling resistance.

Local acceptance evidence: `reports/v0_1_1_frozen_seed711.json` and `reports/v0_1_1_frozen_seed712.json`. Earlier exploratory failures remain in the ignored reports directory. Raw assets, weights, traces and videos remain on Tang.

## Reproduction on Tang

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/eval_free_shooting.py --acceptance --seed 711 --output reports/recheck_seed711.json
/home/you/conda_env/env_isaaclab/bin/python tools/eval_free_shooting.py --acceptance --seed 712 --output reports/recheck_seed712.json
/home/you/conda_env/env_isaaclab/bin/python tools/check_free_shooting_acceptance.py reports/recheck_seed711.json reports/recheck_seed712.json
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style outside
```

Use `--style inside`, `--style instep`, or `--style outside`. Live and batch call the same runner, initialization, controller and success calculation. `--baseline` explicitly disables the new style controls. Custom `--trials` manifests carry their complete control settings. `--video-dir` writes actual camera frames for a single trial; it does not replay or reposition the robot.
