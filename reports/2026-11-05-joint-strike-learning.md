# v0.1.25: Joint residual learning for approach, strike and recovery

Status: rejected as a robustness improvement. Independent physical evaluation passed 0/3 executed development conditions; the frozen baseline passed 1/3 on those same conditions. Four ball-position conditions were deliberately not run after the nominal/speed gate failed.

The frozen v0.1.15 controller and v0.1.23 seven-case evaluation remain unchanged. This version starts from v0.1.24 (`3871e11`) on its own branch. The previous running-only fine-tuning changed the approach state delivered to an older shot controller. The new experiment learns one feedback correction throughout the physical episode instead of adding another handoff.

## Method and acceptance

A local reference stores the 700 actual position-action commands and velocity demands from the successful frozen center episode. A zero-initialized MLP observes actual joints, root motion, foot position/velocity/support, ball position/velocity, goal and reference errors. Its 30 outputs provide bounded corrections to 29 joints (up to 0.25 rad) and a bounded timing-rate adjustment (up to ±0.15 s accumulated phase shift, accumulation disabled after 2 s). The same controller and observation implementation are used in training and evaluation.

The recorded PD target may lie outside the kinematic joint interval; this is already true of the frozen controller. Residual composition preserves that existing target and allows only inward corrections there. The original physical joint limits, effort limits, velocity limits and actuators remain unchanged.

Training uses real ball dynamics through the whole 14 s episode. Half of reset conditions are nominal; the other half gradually reaches ball x ±5 cm, y ±10 cm and robot initial world-y velocity ±0.20 m/s. The reward includes reference tracking, right-foot contact, goal crossing and post-contact stability. The training contact-surface signal is only a filtered-force local-normal proxy; independent evaluation still requires the exact collision-point surface, legal body contact, running contact and final standing.

The nominal data are a successful robot rollout, not new human motion data. This method does not demonstrate autonomous human coordination discovery. Seven specified conditions are development cases; they are not independent held-out success-rate trials.

## Verified prerequisites and diagnosis

- Zero-residual independent physical replay exactly matched the frozen center kinematics: contact at 1.570 s, instep impulse fraction 100%, outgoing horizontal speed 4.6302 m/s, goal, and 8.385 s final standing. The full 14 s / 2800 physics samples were retained in `reports/v025_zero_reference.json`.
- A 16-environment one-update interface smoke completed in 1.784 training seconds. This was an interface check, not a task result.
- The first pilot was interrupted after the log reported approximately 94 training seconds and 67 completed updates. It exposed a discontinuity: the initial residual function clipped the entire nominal target on any nonzero correction. A 2.5e-6 rad intended correction could instead change a target by 1.5 rad. The targeted regression test failed before the fix and passed afterwards. The invalid pilot is excluded from learned-performance comparisons; `model_0.pt`, source snapshots and logs remain local. The Isaac interrupt handler did not produce a final checkpoint/result.
- A zero-residual 16-environment training probe (24.114 wall seconds) had real contacts and goals but did not remain stable to the horizon. The same entry point with one environment (20.941 wall seconds) completed 13.98 s with contact, goal, 8.38 s stable and no resets. The replayed open-loop command prior is numerically sensitive in parallel simulation and requires stabilizing feedback. This is distinct from a learned robustness improvement.
- The first independent zero replay predates reward-only stable-seconds bookkeeping and the bounded residual fix. Its historical source hashes are retained; the final single-environment training probe verified the corrected control path at zero correction. Later changes parameterize the same phase bounds and record applied rather than requested residuals; they do not change zero commands.

## Learned pilot and independent outcomes

The corrected pilot stopped at an observed performance plateau: **533.587 s, 348 PPO updates**, curriculum scale 0.9025. It did not reach the full prescribed perturbation envelope. Its completed stochastic training episodes included 51,466 nominal and 55,936 perturbed resets; neither group achieved the training full-task proxy. There were 9,607 right-foot contact episodes and 28 goals. These counts cannot establish generalization or independent success rates.

Saved checkpoint: `experiments/joint_strike_v025/joint_pilot_continuous/last_model.pt`, SHA256 `d9781eb9bf3f49ea974e76b69653b681b0fa50562facdc7f8e2d8622e5e9a4c0`. Manifest, exact training source copies, intermediate checkpoints and result remain in that directory. The result and final checkpoint were saved and hash-checked before terminating a simulator teardown that did not exit normally. No rollout or optimization was running at termination.

| Condition | Frozen full task | Learned full task | Learned termination | Valid foot contact | Final standing |
|---|---:|---:|---|---|---:|
| Original | pass | fail | non-foot contact, 1.740 s | none | 0 s |
| Robot initially slower, world vy +0.20 m/s | fail | fail | root height 0.425 m, 1.320 s | none | 0 s |
| Robot initially faster, world vy −0.20 m/s | fail | fail | root height 0.434 m, 1.200 s | none | 0 s |

The frozen unperturbed velocity was `[-0.14360835, -1.77231789, 0.28313002]` m/s; the perturbations affect the robot's initial forward velocity, not a moving ball. Every candidate starts with the matching frozen measured pose and requested velocity. All three outcomes passed provenance, physics-duration and controller-coverage audit, despite failing the task. Candidate reports are `reports/v025_joint_final_{center,initial_slower_02,initial_faster_02}.json`; audited comparison is `reports/v025_joint_comparison.json`. No missing ball-position outcome is counted as a failure or success.

The learned nominal episode applies a maximum absolute **0.1782 rad correction on its very first command**, despite the uncorrected nominal reference being successful. Over that episode, correction magnitude reaches 0.1933 rad and timing lags by up to 112.4 ms. These are within the configured bounds but materially change the approach. Nominal resets plus a soft residual reward penalty did **not** preserve nominal behavior. This is direct evidence against the current training setup; it is not evidence that G1 hardware makes the task impossible.

The recorded successful trajectory preserves the output of a closed-loop controller on one rollout, but does not retain that controller's stabilizing responses to new states. Parallel zero-replay failures and the trained policy's nominal regression therefore expose two separate weaknesses: the prior is fragile away from its exact path, and the current optimization lacks an effective constraint preserving that path. A more credible next experiment should retain the frozen closed-loop expert as the prior and verify nominal retention before optimizing residual timing/placement. Whether that improves perturbed strikes remains untested; simply extending this pilot is not justified by its plateau.

Evaluation uses the original running, continuous-contact, collision surface, goal and standing gates. A single policy has no recovery-controller handoff event, so that implementation-specific event flag is not required; actual physical recovery remains required.

## Verification and limits

- Six targeted tests and six subtests passed, covering zero/tiny residual behavior, bounded phase, measured feedback and rejection of jointly truncated command logs. They are code-contract checks, not robot-skill acceptance.
- One fresh read-only code review found an evidence coverage gap: truncating both feedback and kinematics together could pass while physics remained complete. The regression failed before repair, then passed; the final three-case audit uses the corrected validator. No second review or broad test suite was run.
- The 19 frozen files still match their stored hashes. All earlier version branches and local research assets are preserved. Upstream repositories and original simulator/actuator setup are unchanged.
- Approximate optimization cost: 94 s invalid aborted pilot + 1.784 s smoke + 533.587 s corrected pilot, about **10.5 minutes** total; separate zero training probes took 45.054 wall seconds. Simulator startup, teardown and physical evaluation wall time are not included in that optimization total.
- This version does not solve ball-position/initial-speed adaptation, learned recovery, or human-like explosive coordination. It preserves an auditable failed hypothesis and working training/evaluation entry points; it does not replace the successful frozen controller.

## Reproduction on Tang

Local simulator, frozen assets, baseline reports and generated weights are required; a fresh clone is not self-contained.

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/build_joint_strike_template.py
/home/you/conda_env/env_isaaclab/bin/python tools/run_joint_strike.py --zero --output reports/NEW_zero.json
/home/you/conda_env/env_isaaclab/bin/python tools/train_joint_strike.py --output experiments/NEW_joint_run --num-envs 1024 --iters 1000 --max-seconds 900
/home/you/conda_env/env_isaaclab/bin/python tools/run_joint_strike_matrix.py --checkpoint experiments/NEW_joint_run/last_model.pt --prefix NEW_eval --cases center initial_slower_02 initial_faster_02
/home/you/conda_env/env_isaaclab/bin/python tools/analyze_joint_strike.py --checkpoint experiments/NEW_joint_run/last_model.pt --prefix NEW_eval --cases center initial_slower_02 initial_faster_02 --output reports/NEW_comparison.json
```

The template builder refuses to overwrite the existing local template. Run it only when that generated asset is absent. Training and physical evaluation preserve existing outputs. Creating `STOP` in a live training output directory requests a checkpointed stop at the next optimizer boundary.
