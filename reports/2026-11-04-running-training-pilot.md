# v0.1.10 — running-policy adaptation pilot

This version addresses the largest gap in the full task: a continuous flight-phase run-up. The existing reliable approach remains a roughly 1 m fast stepping sequence. The new work uses the same free 29DoF G1 and unchanged actuator gains, effort limits, velocity limits, mass and collision geometry.

## Remaining goal gaps

| Capability | Evidence before this pilot | Still required |
|---|---|---|
| Approach → kick → follow-through → stable recovery | v0.1.7/v0.1.8 support hip/inside, hip/instep, knee/instep and knee/outside in their frozen conditions | Standing start → repeated alternating-flight running → physical handoff to kick → recovery |
| Direction and power | Three discrete goal points for three actions at each action's native strength | Joint light/medium/heavy and direction control; outside-foot calibration |
| High balls | v0.1.9 inside-foot chips, actual centre apex about 0.32–0.42 m in previews | Other contact surfaces, independent height/power control, target height at the goal plane |
| Contact posture and human-like coordination | Legal surface and measured hip/knee dominance in selected actions | The requested near-vertical toe-down instep is not established by those successes; broader style/mode combinations remain incomplete |
| Generality | Small frozen perturbations of particular ball positions and references | Different approach distances, ball/goal positions and transitions without manually selected waypoints |

A successful locomotion pilot alone would not close the complete shooting task. Training resets below begin in a moving running pose; they do not demonstrate standing-to-running.

## Implementation

- `tools/build_running_training_reference.py` and `running_reference_geometry.py` map the local LAFAN RUN motion into the verified PAiD joint/body order, retain its first 2.2 s at 50 Hz, and remove the unvalidated braking tail. The final generator uses the simulator's actual capsule collision shapes, not the URDF cylinders. It applies a smooth floor to early left hip pitch (maximum correction 0.23145 rad / 13.26°), with its analytic velocity derivative, to separate the proximal thigh capsule from the pelvis sphere. Other 28 joints and playback timing are unchanged. A constant 10.51687 mm world-height correction gives 1 mm minimum foot-capsule clearance. At the 111 sampled frames, all nonadjacent capsule gaps are nonnegative; minimum pelvis gaps are 1.000 mm left / 0.383 mm right. This is kinematic screening, not dynamic balance evidence.
- The reference starts with horizontal speed about 1.78 m/s; mean forward reference speed is 2.65 m/s. Initial pelvis tilt is 37.64°, torso tilt 49.11°. Its endpoint remains moving. Training ends the episode there instead of freezing or teleporting the reference back to the start.
- `tools/paid_actor_warmstart.py` reconstructs the official 160-observation, two-layer 128-unit LSTM actor with its fixed observation normalization. The actor and exploration standard deviation are trainable; the first critic starts fresh and the second run resumes actor, critic and optimizer. CPU action/state parity over 24 recurrent steps differs by at most 2.63e-6.
- `tools/train_running_pilot.py` performs bounded PPO adaptation. The first pilot uses 96 parallel environments, up to 400 iterations and a 600 s training limit; a second bounded continuation uses the corrected capsule reference. Half of reset phases are frame 0; the remainder are uniform frames 0..60. These are ordinary episode initialization writes. Original tracking rewards are supplemented with forward reference velocity/position rewards, survival and fall terms. Root height <0.45 m or tilt >60° is a physical termination; the finite reference endpoint is a timeout.
- `tools/export_running_actor.py` replaces only actor/LSTM/normalization initializers in the original ONNX graph, preserving the strict evaluator schema and metadata. The graph's old reference outputs are retained for compatibility; the evaluator takes its actual running reference from the trial manifest, and requests only action/hidden-state outputs. CPU parity is checked before saving each export.

Existing shooting controllers and frozen v0.1.8/v0.1.9 configurations are not changed. Models, source motion, generated reference, traces and media remain local/ignored. A fresh clone still needs the Tang assets and simulator installation.

## First pilot and contact diagnosis

The initial cylinder-screened reference had been raised 10.46046 mm but still contained a runtime capsule self-overlap. On the same reference/seed1011, the original actor terminated at0.80s and the 400-update actor at1.46s. The trained actor travelled2.098m forward but drifted1.308m sideways, produced only one25ms qualifying flight and fell below0.45m root height. This is delayed failure, not continuous running. First training:313.89s,921,600 transitions, checkpoint SHA256 `92acd895b4a86c21b6fa3712b340fc02a1c9de107234174c0dc6311ceb7e82c9`. Export parity: max action/state error1.20e-6.

A targeted0.1s physical contact diagnostic located the initial force specifically between **pelvis and left_hip_roll_link**,6322N at5ms. The upstream spawn has `replace_cylinders_with_capsules=True`, and runtime USD confirms capsule segment height equals the URDF cylinder length. The adjacent elbow links are collision-filtered; the problem was not the arms. The capsule overlap is about0.228mm; hip roll/yaw do not move the nearest capsule endpoint, whereas reducing deep hip pitch separates it.

After the bounded hip-pitch correction, the same diagnostic reports zero nonfoot contact for the initial40ms. No collision filters, physical geometry, torque limits or joint hard limits were changed. These short diagnostics establish the corrected reset, not full running stability. Old artifacts are preserved alongside corrected reference SHA256 `45ab681492fa8141e1f34840502f153a60e7e1d839714efac8797e8871b7d380`.

## Corrected continuation

The corrected continuation ran another400 iterations in306.30s (two runs total620.20s /1,843,200 environment transitions). Final checkpoint SHA256 `e76fc8c1a011e5d59aac3824067d90c2b5f2cb9fea977cb5e15cc9e6b404e142`; exported action/state parity error<=1.20e-6.

The seed1013 fixed-start physical replay with the documented one-frame reference lead **still fails**. It terminates at1.46s on root orientation, after travelling2.2115m forward with0.0512m net lateral displacement. Peak tilt69.80°; no qualifying alternating flight interval after0.30s; nonfoot contact later reaches151.60N. The corrected reset removes the large initial self-collision but does not solve subsequent balance or continuous flight. The phase convention/reference/seed differ from the first pilot, so these are not a controlled numerical improvement claim.

Two bounded attempts now show that this small amount of adaptation does not establish sustained running. No third unchanged training batch or parameter sweep was launched. The next controller work must address dynamic balance and running contact timing on this capsule-safe reference, then separately demonstrate standing start, braking/foot placement before the ball, controller handoff and recovery. Increasing iterations alone or swapping reference animation is not yet justified as a solution. The existing shooting controller remains the default.

Artifacts: `experiments/running_v010/pilot_seed1010_n96_i400/`, `experiments/running_v010/capsule_resume_seed1012_n96_i400/`; raw evaluations `reports/v010_running_baseline_evaluation.json`, `v010_running_pilot_evaluation.json`, `v010_running_corrected_evaluation.json`; pair probes `v010_initial_contact_diagnostic.json`, `v010_corrected_contact_diagnostic.json`. Actual replay: `media/running_pilot_v0.1.10.mp4` (explicit failed-run label, normal-speed then0.25x repeated renderer frames; no synthesized motion).

The training environment's ordinary auto-reset convention leaves the reference one20ms frame ahead of physical phase. Its native step order preserves that lead throughout each episode; only the specially synchronized first reset is aligned. `eval_running_pilot.py` therefore gives the trained actor the next reference q/dq and torso world angular velocity, leaving measured gravity, body velocities, joint states, last actions and target points unchanged. It records this one-frame offset and its own source hash. This is an observation convention, not a physical state write. The shooting evaluator and its frozen source hashes remain unchanged.

## Focused checks and usage

Focused checks: CPU actor reconstruction parity; reproduction of each implemented reference version; two bounded training runs; trained-export parity; same-reference baseline and trained physical rollouts; two0.1s contact-pair diagnostics before/after the supported correction. No full test suite, large target matrix or repeated review cycle.

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/build_running_training_reference.py \
  --output experiments/running_v010/new_run_reference.npz
/home/you/conda_env/env_isaaclab/bin/python tools/train_running_pilot.py \
  --motion experiments/running_v010/new_run_reference.npz \
  --output experiments/running_v010/new_pilot \
  --num-envs 96 --iters 400 --max-seconds 600 --seed 1010
/home/you/conda_env/env_isaaclab/bin/python tools/export_running_actor.py \
  --checkpoint experiments/running_v010/new_pilot/last_model.pt \
  --output experiments/running_v010/new_pilot/policy.onnx
```

Do not overwrite old experiment directories. Resume requires a new output directory and `--resume PATH/last_model.pt`; continued training is still bounded by the supplied iteration and time limits.

Evaluate the final local research checkpoint (the result is not an accepted running controller):

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/eval_running_pilot.py \
  --trials experiments/running_v010/corrected_evaluation_trial.json \
  --model experiments/running_v010/capsule_resume_seed1012_n96_i400/policy.onnx \
  --running-diagnostic --sequence-evidence --reference-lead 1 --steps 110 \
  --output reports/new_running_pilot_evaluation.json
```

Existing runup/loft configuration fingerprints verified unchanged; changed sources parse, both training/export flows executed, and the wrapper was exercised by the final real-physics replay. No full regression suite was run.
