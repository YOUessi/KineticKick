# v0.1.11 — running feedback and physical transitions

## Outcome and remaining user goals

The new controller completes a moving-start running segment with repeated alternating flight. A longer replay runs, decelerates and transfers control to the original soccer actor at a measured support state, with no fall or nonfoot impact over6.94s. **It still fails the final standing gate. Standing start → running → shot → stable recovery is not complete.** These are individual deterministic physical replays, not robustness or arbitrary-distance acceptance.

| User requirement | Evidence in this version | Remaining gap |
|---|---|---|
| Continuous flight-phase running | Six alternating flights of40/100/65/40/40/65ms; flight-chain progress3.836m | Longer running, steering and different approach distances |
| Starting, shooting and recovery as one sequence | Running-to-braking improves from a2.72s fall to6.94s without a fall; gated actor handoff occurs at4.30s | Zero-speed start fails at1.44s; final recovery gate fails; no physical ball or shooting handoff in the running scene |
| Direction and light/medium/heavy power together | Two bounded three-request shooting probes record actual goal crossings and release speeds | Both probes0/3 joint successes; no new calibrated power table |
| More foot styles and goal-plane target heights | Existing inside-foot chip demonstrations preserved | New height requests all arrive at ground level; other styles have no new accepted high-ball controller |
| Near-vertical toe-down instep and varied ball positions | Existing posture and small-perturbation results preserved | No new≥75° instep acceptance or broad ball-position/distance adaptation |

All trials use the original free29DoF G1, original mass/collisions, actuator gains, torque limits and velocity limits. Evaluation suppresses automatic physical resets; no root or ball state writes occur after initialization. Shooting uses actual ball contacts without extra ball forces. The running scene has no physical ball: its two target-point observation inputs are explicit fixed placeholders, not goal-control evidence.

## Controller changes

`running_balance.py` expands the160-observation recurrent actor to176 observations while preserving its old weights, fixed normalization, actor/critic structure and initial outputs. The16 added inputs are reference-anchor position/orientation, measured base velocity, actual left/right ground support and reference contact timing. New input columns initially have zero weights. CPU migration parity across12 recurrent steps was exact; capsule foot-height computation agrees with independent URDF forward kinematics within2.79e-8m.

Actual support is measured with ground-filtered PhysX contact views. Reference foot heights use the simulator's imported capsule geometry, including the cylinder-to-capsule conversion diagnosed in v0.1.10. The training rewards now include support timing, clearance and angular-velocity tracking. A strict optional termination catches any nonfoot force≥20N across all four5ms physics samples, including wrist contacts. The same conservative running acceptance gate remains unchanged.

`train_running_balance.py` provides bounded training and direct recurrent physical evaluation. Resets initialize the reference phase; references remain finite and never wrap or teleport within an episode. The one-frame observation lead matches the upstream training convention. Later transition training samples half of resets across the complete reference, instead of restricting all random phases to its first1.8s.

`running_sequence_metrics.py` requires the final0.8s to have complete5ms evidence, planar speed≤0.25m/s, angular speed≤0.5rad/s, tilt≤15°, and both feet carrying≥30N for≥80% of that window. Any earlier fall or nonfoot impact fails the complete episode. These thresholds were defined before the transition replay and were not loosened after failure.

The optional `--stand-handoff` transfers to the original160-observation ONNX actor only after four consecutive20ms steps with reference speed<0.1m/s, actual speed<0.45m/s, tilt<15° and both feet≥30N. Actions blend over0.4s. Resetting this actor's recurrent memory does not reset physical robot state. The original ONNX SHA256 remains `489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`.

## Bounded local training

All three runs used1024 environments on Tang's existing GPU. Total measured training time was1812.31s (30.21minutes), excluding simulator startup and evaluation. No paid compute or external training service was used.

| Run | Stop / seconds | Checkpoint SHA256 |
|---|---|---|
| `balance_n1024_i800` | 800updates /850.82s | `6e3bbd275aa983fbf74c922390d433f790aaa5eea8bd67e3adba2f268dc610a6` |
| `contact_n1024_i300` | 360s wall-time bound /360.84s | `9f262a4e094ac2e22c2622c364f9cf1d3b2d90ef49ba7087c54cc7e29bfdec04` |
| `run_stop_n1024_i500` | 600s wall-time bound /600.64s | `5c4e4f5505f13501614c45a4cc51dd8bc6e81818544de88240c2b05e23c410fa` |

Directories are under ignored `experiments/running_v011/`, with manifests, checkpoints, logs and source snapshots. Earlier checkpoints and failures remain intact. A fresh clone needs these Tang-local assets, the original motion data and the recorded simulator/upstream installation.

## Actual locomotion replays

| Evidence file under `reports/` | Result |
|---|---|
| `v011_balance_evaluation.json` | Four flights and no fall in2.2s, but nonfoot contacts up to247.75N: strict failure |
| `v011_balance_armguard_evaluation.json` | A constant0.08rad outward shoulder offset destabilizes the policy: height failure at1.50s; rejected |
| `v011_contact_evaluation.json` | Strict short running pass:2.2s, four flights45/70/60/50ms,5.338m forward,−0.297m sideways, zero nonfoot force |
| `v011_run_stop_evaluation.json` | Extending the short model into unseen braking fails at2.72s on tilt; actual speed stays high while the reference slows |
| `v011_run_stop_trained_evaluation.json` | Full4.94s, six flights,6.481m forward, zero nonfoot force; final standing fails |
| `v011_run_stop_grounded_evaluation.json` | Grounded recovery reference, same model:4.94s without fall; final standing still fails |
| `v011_standing_start_probe.json` | Truly stationary reference initialization; falls below minimum root height at1.44s after1.261m forward; not a standing-start success |
| `v011_run_stop_handoff_evaluation.json` | Six flights; no fall/nonfoot impact through6.94s; handoff at4.30s; final standing still fails |

The last replay's net displacement is6.946m forward and0.579m sideways. At handoff, actual speed is0.119m/s, tilt7.22°, and left/right support223.46/97.54N. Over the final0.8s, maximum planar speed is0.296m/s, maximum angular speed0.656rad/s, maximum tilt4.62°, and double-support fraction61.49%. Its final single frame is slower (0.100m/s), but a single frame cannot establish a stable hold. The longer duration and controller change mean this is not a controlled A/B improvement measurement.

`build_run_stop_reference.py` extends the original RUN clip with its deceleration, a1s blend and a finite standing hold. The verified first111 frames remain bit-for-bit unchanged. The first candidate's standing q/root combination placed capsule feet40.4/38.1mm above ground. The corrected generator smoothly lowers only the recovery reference to1.0/3.34mm foot clearance; it does not reposition the real robot. Changing the reference alone did not fix the learned recovery behavior.

`build_standing_start_reference.py` adds a0.4s zero-velocity standing hold and1.2s acceleration candidate. Unconstrained cubic endpoint velocities initially folded the foot into the shin; bounded monotone tangents remove that kinematic overlap. A remaining joint-velocity seam of10.06rad/s is explicitly recorded. This candidate is **not dynamically validated** and its physical replay fails. It must not be used as a successful starting animation.

## Joint direction, power and height probes

`feedback_shot.py` is an explicitly experimental entry point. It converts a requested goal-plane point and horizontal release speed into a ballistic launch intent, then applies bounded joint targets through measured foot kinematics and actual support. The ball is0.11m radius; planned launch velocity is not substituted for measured ball velocity. The existing shooting evaluator and frozen defaults are unchanged.

Requests were `(x,z,speed)=(-0.30,0.35,4.5), (0,0.55,5.5), (+0.30,0.75,6.5)`, with goal plane y=−3.7m. These are three diagonal samples, not a full direction/power matrix.

| Probe | Measured horizontal release speeds | Goal-plane x | Goal-plane z | Joint success |
|---|---|---|---|---|
| Fixed full-pose foot path | 4.03 /4.10 /4.38m/s | −0.243 /−0.063 /+0.400m | approximately0.11m in all cases | 0/3 |
| Prepared path + contact-normal constraint | 4.33 /3.59 /4.64m/s | −0.578 /+0.176 /+0.335m | approximately0.11m in all cases | 0/3 |

The first path is mostly unreachable before contact: its IK gate often rejects the requested pose until only20–40ms remain. The second starts from measured foot position/velocity, prepares continuously and constrains contact point plus surface normal, leaving rotation around that normal free. It still fails all complete-sequence gates: the right knee touches the ball, with peak illegal forces514.83/2071.62/26.54N. Coarse “goal and recovered” counters therefore cannot certify these shots. This branch preserves both failures and does not publish either as an accepted controller.

Source and trial snapshots are in `experiments/running_v011/first_shot_*` and `prepared_shot_*`; results are `reports/v011_feedback_shots_probe.json`, its separate summary, and `v011_prepared_shots_probe.json`. The additional Cartesian approach was stopped after these failed probes. Whole-body placement and legal clearance must be solved alongside foot orientation and desired ball velocity before widening the foot-style or target matrix.

## Preview and reproduction

Local actual-renderer previews:

- `media/running_balance_v0.1.11.mp4`: accepted2.2s moving-start segment, normal speed then0.25x.
- `media/run_to_brake_v0.1.11.mp4`:4.94s running/braking replay, clearly labeled unsteady recovery.
- `media/run_to_handoff_follow_v0.1.11.mp4`:6.94s actual replay with measured-state controller handoff; recovery remains unaccepted. The follow-camera replay has identical physical traces to the fixed-camera evidence (`reports/v011_handoff_follow_camera.json`); it only improves visibility. The original fixed-camera video is preserved.

Replay the latter in the existing Tang environment, choosing unused output paths:

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/train_running_balance.py \
  --motion experiments/running_v011/run_stop_handoff_reference.npz \
  --output reports/new_run_stop_handoff_evaluation.json \
  --num-envs 1 --evaluate --strict-contact --stand-handoff --recovery-window .8 \
  --resume experiments/running_v011/run_stop_n1024_i500/last_model.pt \
  --seed 1121 --video-dir media/new_run_stop_handoff_frames
```

Focused verification includes actor migration/foot-geometry checks, actual bounded training and the listed physical probes, foot-path endpoint/velocity checks, and four recovery-evidence tests. Existing runup/loft configuration fingerprints match unchanged sources. No full suite or large perturbation/target matrix was run.

Next work needs a physically consistent standing-start trajectory and training across start/brake/controller transitions, then a handoff that includes the real ball and preserves balance and legal contact. Direction, power, target height and ball position must be conditioned together in that process. The present evidence does not justify claiming the complete user goal is achieved.
