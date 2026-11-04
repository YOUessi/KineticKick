# v0.1.8: outside-foot sequence and fixed-strength goal points

## Verified scope

The free original 29DoF G1, original actuator limits, 0.43 kg physical ball, and fixed goal at [0,-3.7] are retained. No post-initialization robot/ball state writes, reference looping, or extra ball impulses were introduced. This version does **not** complete sustained running or jointly calibrated low/medium/high aiming.

- **Outside-foot full sequence: 6/6.** Frozen knee-dominant control, seeds1811/1812, ball x offsets -0.02/0/+0.02 m, initial root xy +/-1 cm/yaw +/-0.01 rad/joints +/-0.005 rad. Every 10 s trial has legal first right-foot contact, outside surface, a goal, no illegal body-ball/nonfoot contact, and final continuous standing >=6.29 s.
- **Fixed native-strength aim: 18/18.** Hip/inside, hip/instep and knee/instep each hit x=-0.3/0/+0.3 on the fixed y=-3.7 plane, with seeds1813/1814 and the same reset perturbations. Maximum horizontal error0.14278 m (limit0.15 m); every trial also passes surface, requested dominance, goal and recovery gates. The reference strength is approximately5.0/1.4/3.8 m/s respectively; these are different motions' native speeds, **not** one motion's low/medium/high levels. Target interpolation and arbitrary ball/goal positions are not supported.
- **Prior controls regression: 5/6.** The three unmodified v0.1.7 trial descriptions were repeated with seeds1813/1814. One old knee/instep trial had mixed contact (instep impulse fraction0.5699) despite a goal and stable recovery. v0.1.8's default knee/instep command is the newly validated centre-point command. The earlier v0.1.7 18/18 claim remains limited to its recorded seeds/conditions, not all random initial states.

Acceptance sources are `reports/v018_outside_acceptance_seed1811.json`, `...1812.json`, `reports/v018_aim_acceptance_seed1813.json`, `...1814.json`. `reports/v018_release_summary.json` replays contacts, goal crossing and sequence gates from raw5ms traces; `configs/runup_shooting_v2.json` pins code/assets/references and evidence hashes. All generated JSON/media remain local per .gitignore.

## Physical changes and diagnosis

The initial outside pose failed because it rotated the right hip while that foot was still supporting the robot. The new control waits for left-foot support, flexes/lifts the kicking leg, rotates into the outside strike, and releases smoothly. The physical ball is placed later along the measured foot path while an explicit, logged virtual approach waypoint preserves planned support placement; the ball itself never moves by software after reset. Contact then occurs around2.56 s, after the outside surface is oriented. A small left-shoulder-roll recovery pulse avoids the measured wrist/thigh near collision. Measured knee contribution—not the candidate's requested label—determines the released mode.

Aiming changes the policy's virtual goal input, keeping the real goal and measurement plane fixed. The preview rejects targets not in the frozen discrete command set. Every target must also pass the entire motion sequence, not merely move the ball across the target plane.

## Unfinished capabilities and bounded evidence

**Sustained running is not achieved.** Same-morphology LAFAN RUN reference conversion preserves the29-joint order, body FK,50Hz schema, and source hashes. It is a kinematic input, starts already moving and is not a standing-start controller; its tail also has uncorrected ground penetration. The existing actor first terminated at0.28 s because right-wrist reference-height error exceeded0.25 m. A separate marked diagnostic replaced reference-error stops with actual root height0.45 m and tilt60 degrees, preserving actuator limits. It fell at0.88 s (tilt62.65 degrees), with only one60ms alternating flight interval. Neither initial free fall nor that interval is labelled running. Reports: `v018_running_probe.json`, `v018_running_absolute_probe.json`. Further work requires a same-asset closed-loop locomotion controller/training and a verified transition to the shooting controller; more static reference substitution is not supported by this evidence. No training was started in this version.

**Three-level aim/power is not accepted.** New support-gated joint-velocity feedforward changes ordinary PD velocity targets only, capped at12rad/s and then hardware velocity limits. It produced three distinct nominal hip/instep speeds in one batch (0.55/1.36/1.83m/s), but also changed strike direction. Repeating the same weak command changed goal-plane x from-0.328 to-0.048 m despite almost identical initial state: contact timing changed5ms and local contact height about2.9mm. The final9-node correction batch achieved9/9 full sequences but only6/9 joint aim/speed matches; the low/medium speed gap also collapsed at side targets. Knee gain probes did not preserve both surface/mode and separated speed levels. No further table sweep was performed, and **no low/medium/high calibration file is bundled**. `show_targeted_runup.py` therefore requires an explicit external frozen config; its validator requires18/18 per combination and paired measured speed separation>=0.30m/s.

These failures are evidence about the present controller/reference/contact combination, not proof that G1 hardware cannot run or kick harder.

## Reproduction

On the existing Tang installation:

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style outside
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style instep --target-x 0.3
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style instep --target-x -0.3
```

The default v0.1.8 config uses fixed native strength; no `--power` is advertised here. Reproduce old source fingerprints by checking out the corresponding old version branch. The complete CPU suite still requires local Tang assets; a fresh clone is not self-contained. Full Tang-local CPU suite: **197/197 passed** in14.63s. Independent whole-branch review found no Critical/Important issue and independently replayed all24 promoted trials. Minor deferred: the ordinary preview wrapper's last console line prints sequence status without aim error/combined success; complete evaluator output and JSON contain those fields. Do not use that last line alone to infer target success.

## Physical previews

`media/outside_sequence_v0.1.8.mp4`: complete10s physical outside trial, then actual contact frames repeated at0.25x and a final hold. `media/aim_native_v0.1.8.mp4`: left/centre/right fixed-native-strength knee/instep trials, identical camera and seed1815, each complete10s. Only renderer frames, crops, labels and repeated-frame slow replay/holds are used; no synthesized robot motion. Individual preview results and media hashes are in `reports/v018_preview_manifest.json`.
