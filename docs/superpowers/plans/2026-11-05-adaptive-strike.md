# Adaptive strike implementation plan

> Execute inline using superpowers:executing-plans. User has authorized implementation and version publishing.

**Goal:** measured-state adaptation to ball position and initial velocity with frozen baseline comparison.
**Architecture:** bounded robust-running PPO fine-tuning plus explicit online placement/phase controller; frozen low-level baseline retained.
**Tech stack:** Python, IsaacLab/PhysX, PyTorch/RSL-RL, existing PAiD ONNX.
**Spec:** docs/superpowers/specs/2026-11-05-adaptive-strike.md

## Constraints and review focus
Version v0.1.24; no frozen file modifications; same original physical limits and success gates. Test finite/bounded inputs, phase monotonicity, support loss, contact latch, and missing/truncated evidence. Keep training/selection/holdout provenance separate.

- [x] Add validated optional velocity jitter to training, unchanged defaults. Test argument rejection and range configuration, then run the 600 s pilot from frozen running weights.
- [x] Add separate adaptive runner and pure bounded feedback module, sensor diagnostics, input/output trace and hashes. Test state dependence and bounds before physical probes.
- [x] Evaluate nominal/initial-speed probes, diagnose failures, then run the seven development conditions if probes justify it. Retain baseline results without rerunning them unnecessarily.
- [ ] Record actual results and remaining limits, targeted tests, one fresh review, commit and push version branch.

## Execution ledger
Branch created from 31e477b (v0.1.23). Frozen baseline files unchanged. Source inspection: training reset velocities are exactly fixed; runtime running cues ignore actual ball; strike windows fixed. Frozen speed probes drift laterally ~0.4 m and terminate physical_nonfoot at 1.66 s while upright; collision body unknown.

Implementation checkpoint: optional initial planar velocity jitter added, defaults zero. New feedback planner and runner use measured ball/root/support; strike residual and velocity pulse share the shifted clock. Original files verified unchanged (19 hashes). Targeted checks: 35 passed plus 40 subtests. First pytest invocation was blocked by an unrelated ROS plugin missing lark; reran with plugin autoload disabled. PPO pilot active, session 74331, output experiments/running_v024/velocity_jitter_pilot, 600 s bound. Physical acceptance still pending.

Pilot A: 600.524 s / 472 PPO iterations. All three physical probes failed: nominal left-foot early contact 1.355s, faster right-knee contact at ~1.5s, slower pelvis/hip self contact at ~1.66s. Lateral drift reduced but learned gait changed timing/path. Next bounded diagnostic: synchronize locomotion reference phase as well as strike phase, condition approach anchor on calibrated nominal root path. No physics or acceptance changes. Pilot A source snapshot and plan preserved with weights.

Candidate B synchronized approach: center passed, both velocity cases made instep contact but failed later/at contact. Candidate C ankle-preserving translation + full nominal recovery path: center passed (9.795 s standing), slower terminated at 2.46 s, faster at 1.48 s. Candidate D contact handoff to frozen running policy did not fix either velocity failure; preserve D as rejected ablation, retain C for final characterization. Stop adding manual corrections. Complete four ball-position cases, plus explicitly exploratory ±0.10 m/s paired diagnostics (not a holdout/generalization estimate). Integrated strike/recovery training remains the architectural gap.

Final characterization: candidate C and frozen control both 1/7 prescribed conditions; extra exploratory ±0.10 m/s both 0/2. No independent holdout because development acceptance failed. One 600.524 s training job, 20 physical diagnostic episodes total. Final targeted suite 36 passed plus 40 subtests; 19 frozen hashes unchanged. Whole-branch read-only review running; no simulator process remains from this task. User capability goal is NOT achieved.

Final review: no Critical defects. Important runtime-mode/fingerprint audit gap fixed; missing feedback evidence also treated as important to the measured-state claim and fixed. New rejection tests RED→GREEN; targeted suite 38 passed / 45 subtests. Re-audited all nine existing pairs, outcomes unchanged. Reviewer excluded hardware/generalization efficacy and unimplemented integrated training; these remain explicitly unclaimed. Publication is a negative-result research version, not completion of the user capability goal.
