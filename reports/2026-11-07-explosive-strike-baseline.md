# Explosive-strike offline baseline — 2026-11-07

This v0.1.29 analysis measures existing physical episodes. It does not change the controller, rerun a simulator, train a model, or assign a naturalness score. Frozen v0.1.15 is the seven-case v0.1.23 replay, including `reports/v023_baseline_center.json`; v0.1.27 and v0.1.28 are loaded from the paths and SHA-256 digests in `reports/v028_final_comparison.json`.

Full per-joint extrema, unavailable-evidence reasons, paired deltas, and all 40 episode digests: `reports/explosive_strike_baseline_v029.json`.

## Original task outcomes retained

| Version | Episodes | Full task passes | Actual foot contacts | Certified horizontal GRF | Thorax orientation |
|---|---:|---:|---:|---:|---:|
| v0.1.15 | 7 | 1 | 5 | 0 | 0 |
| v0.1.27 | 15 | 14 | 14 | 0 | 0 |
| v0.1.28 | 18 | 16 | 18 | 0 | 0 |

The case counts differ and are deterministic sensitivity results, not population success rates. The v0.1.28 matrix has 15 development cases and three later evaluation cases; all original goal, running, contact-surface, recovery, illegal-contact, and physical-integrity gates remain in force.

## Center case: speed, impulse, support and recovery

| Version | Preimpact fixed-foot-point speed (m/s) | Outgoing ball horizontal speed (m/s) | First normal impulse (N·s) | Loaded-support COM Δv forward (m/s) | Final sequence standing (s) |
|---|---:|---:|---:|---:|---:|
| v0.1.15 | 6.804 | 4.630 | 2.281 | -0.402 | 8.385 |
| v0.1.27 | 6.804 | 4.631 | 2.282 | -0.402 | 8.910 |
| v0.1.28 | 6.804 | 4.631 | 2.282 | -0.402 | 8.910 |

The COM quantity is a measured velocity change during the last continuous ≥80 N left-support episode intersecting the preceding 300 ms. A negative value means a loss of forward COM velocity along the fixed ball-to-goal direction. It does not identify the support foot’s horizontal force, a force path, or energy transferred to the kicking leg. Standing durations here use the original `sequence_records[0].recovery` criterion. The shot tracker has a separate standing counter (center v0.1.27/v0.1.28: 8.965 s versus 8.910 s for the sequence criterion); both are retained in the JSON, without changing either gate.

## Center case: signed joint timing

Both positive and negative peaks are retained because the largest absolute knee speed can occur in a different motion phase than the late hip-speed peak. Times below are relative to the first measured foot-ball normal-force sample; negative times precede contact.

| Version | Joint | Positive q̇ peak (rad/s @ ms) | Negative q̇ peak (rad/s @ ms) | Positive estimated-power peak (W @ ms) | Negative estimated-power peak (W @ ms) |
|---|---|---:|---:|---:|---:|
| v0.1.15 | right_hip_pitch | 6.15 @ -280.0 | -13.28 @ -40.0 | 248.06 @ -65.0 | -370.73 @ -5.0 |
| v0.1.15 | right_knee | 8.42 @ -220.0 | -5.40 @ -295.0 | 162.01 @ -245.0 | -184.11 @ -285.0 |
| v0.1.15 | right_ankle_pitch | 8.45 @ -250.0 | -2.06 @ -190.0 | 362.99 @ -275.0 | -37.28 @ -225.0 |
| v0.1.15 | waist_yaw | 1.48 @ -180.0 | -1.23 @ -270.0 | 8.21 @ -110.0 | -21.54 @ -180.0 |
| v0.1.15 | left_shoulder_pitch | 1.37 @ -295.0 | -4.25 @ -150.0 | 20.32 @ -105.0 | -16.83 @ -5.0 |
| v0.1.15 | right_shoulder_pitch | 4.62 @ -115.0 | -1.14 @ -295.0 | 27.11 @ -175.0 | -12.97 @ -30.0 |
| v0.1.27 | right_hip_pitch | 6.15 @ -280.0 | -13.28 @ -40.0 | 248.30 @ -65.0 | -370.79 @ -5.0 |
| v0.1.27 | right_knee | 8.42 @ -220.0 | -5.40 @ -295.0 | 162.01 @ -245.0 | -184.11 @ -285.0 |
| v0.1.27 | right_ankle_pitch | 8.45 @ -250.0 | -2.06 @ -190.0 | 362.99 @ -275.0 | -37.28 @ -225.0 |
| v0.1.27 | waist_yaw | 1.48 @ -180.0 | -1.23 @ -270.0 | 8.21 @ -110.0 | -21.54 @ -180.0 |
| v0.1.27 | left_shoulder_pitch | 1.37 @ -295.0 | -4.25 @ -150.0 | 20.32 @ -105.0 | -16.83 @ -5.0 |
| v0.1.27 | right_shoulder_pitch | 4.62 @ -115.0 | -1.14 @ -295.0 | 27.11 @ -175.0 | -12.96 @ -5.0 |
| v0.1.28 | right_hip_pitch | 6.15 @ -280.0 | -13.28 @ -40.0 | 248.30 @ -65.0 | -370.79 @ -5.0 |
| v0.1.28 | right_knee | 8.42 @ -220.0 | -5.40 @ -295.0 | 162.01 @ -245.0 | -184.11 @ -285.0 |
| v0.1.28 | right_ankle_pitch | 8.45 @ -250.0 | -2.06 @ -190.0 | 362.99 @ -275.0 | -37.28 @ -225.0 |
| v0.1.28 | waist_yaw | 1.48 @ -180.0 | -1.23 @ -270.0 | 8.21 @ -110.0 | -21.54 @ -180.0 |
| v0.1.28 | left_shoulder_pitch | 1.37 @ -295.0 | -4.25 @ -150.0 | 20.32 @ -105.0 | -16.83 @ -5.0 |
| v0.1.28 | right_shoulder_pitch | 4.62 @ -115.0 | -1.14 @ -295.0 | 27.11 @ -175.0 | -12.96 @ -5.0 |

The 300 ms window includes approach motion before the final plant. The following view uses only the loaded-support subset of that window, including the first loaded bin and excluding the collision bin. In the center case the first loaded sample is 1.395 s, with onset bracket 1.390–1.395 s; contact is 1.570 s.

| Version | Joint | Post-plant positive q̇ peak (rad/s @ ms) | Post-plant negative q̇ peak (rad/s @ ms) | Positive estimated work (J) | Negative estimated work (J) |
|---|---|---:|---:|---:|---:|
| v0.1.15 | right_hip_pitch | absent | -13.28 @ -40.0 | 20.773 | -2.474 |
| v0.1.15 | right_knee | 7.24 @ -165.0 | -0.35 @ -5.0 | 0.760 | -2.776 |
| v0.1.15 | right_ankle_pitch | 3.40 @ -105.0 | -1.71 @ -175.0 | 0.102 | -0.125 |
| v0.1.27 | right_hip_pitch | absent | -13.28 @ -40.0 | 20.772 | -2.475 |
| v0.1.27 | right_knee | 7.24 @ -165.0 | -0.35 @ -5.0 | 0.760 | -2.776 |
| v0.1.27 | right_ankle_pitch | 3.40 @ -105.0 | -1.71 @ -175.0 | 0.102 | -0.125 |
| v0.1.28 | right_hip_pitch | absent | -13.28 @ -40.0 | 20.772 | -2.475 |
| v0.1.28 | right_knee | 7.24 @ -165.0 | -0.35 @ -5.0 | 0.760 | -2.776 |
| v0.1.28 | right_ankle_pitch | 3.40 @ -105.0 | -1.71 @ -175.0 | 0.102 | -0.125 |

These signs are robot joint coordinates. Shoulder and waist values are relative-joint proxies, not world upper-arm/thorax angular velocities. Pelvis projected heading is available from 20 ms root orientations; thorax and upper-arm world orientations are unavailable in these frozen reports, so a full pelvis–thorax–arm sequence cannot be concluded.

| Version | Pelvis projected yaw net / range (deg) | Peak absolute projected yaw rate (rad/s @ ms) | Positive hip / knee / ankle forward integrals (m) |
|---|---:|---:|---:|
| v0.1.15 | -0.613 / 7.290 | 2.094 @ -260.0 | 0.833 / 0.038 / 0.004 |
| v0.1.27 | -0.616 / 7.293 | 2.094 @ -260.0 | 0.833 / 0.038 / 0.004 |
| v0.1.28 | -0.616 / 7.293 | 2.094 @ -260.0 | 0.833 / 0.038 / 0.004 |

The integrals are signed/positive/negative Jacobian foot-velocity components in the 300 ms window; the table shows positive components only. The JSON retains all signs, the floating-base remainder, and numerical closure against the measured fixed-point velocity. These are kinematic displacement contributions, not energy percentages or ball-work shares.

## Matched changes

| Comparison | Matched cases | Full passes before → after | Mean foot-speed change (m/s) | Mean normal-impulse change (N·s) | Mean final-sequence-standing change (s) |
|---|---:|---:|---:|---:|---:|
| v0.1.15 → v0.1.27 | 7 | 1 → 7 | -0.167 | 0.074 | 3.689 |
| v0.1.27 → v0.1.28 | 15 | 14 → 15 | -0.026 | 0.066 | 0.480 |

Means include only pairs with both measurements available; the JSON records pair counts and each case’s delta. No significance test or human-biomechanics match is implied.

## Per-case contact and follow-through

| Version | Case | Full task | Preimpact foot speed (m/s) | Normal impulse (N·s) | Initial forward follow-through (ms) | Final sequence standing (s) |
|---|---|---|---:|---:|---:|---:|
| v0.1.15 | center | pass | 6.804 | 2.281 | 195.000 | 8.385 |
| v0.1.15 | ball_x_minus_5cm | missed_goal | 6.804 | 2.612 | 190.000 | 8.435 |
| v0.1.15 | ball_x_plus_5cm | wrong_contact_surface | 6.645 | 1.844 | 210.000 | 9.110 |
| v0.1.15 | ball_near_10cm | recovery_failed, physical_failure, running_failed | 6.987 | 2.185 | 205.000 | 0.000 |
| v0.1.15 | ball_far_10cm | wrong_contact_surface | 6.283 | 1.883 | 185.000 | 8.490 |
| v0.1.15 | initial_slower_02 | no_foot_contact, wrong_contact_surface, missed_goal, recovery_failed, physical_failure, running_failed, running_strike_transition_failed | unavailable | unavailable | unavailable | 0.000 |
| v0.1.15 | initial_faster_02 | no_foot_contact, wrong_contact_surface, missed_goal, recovery_failed, physical_failure, running_failed, running_strike_transition_failed | unavailable | unavailable | unavailable | 0.000 |
| v0.1.27 | center | pass | 6.804 | 2.282 | 195.000 | 8.910 |
| v0.1.28 | center | pass | 6.804 | 2.282 | 195.000 | 8.910 |
| v0.1.27 | ball_x_minus_5cm | pass | 6.911 | 2.589 | 200.000 | 9.995 |
| v0.1.28 | ball_x_minus_5cm | pass | 6.911 | 2.589 | 200.000 | 9.995 |
| v0.1.27 | ball_x_plus_5cm | pass | 6.751 | 2.119 | 195.000 | 7.955 |
| v0.1.28 | ball_x_plus_5cm | pass | 6.751 | 2.119 | 195.000 | 7.955 |
| v0.1.27 | ball_near_10cm | pass | 6.703 | 2.131 | 190.000 | 7.495 |
| v0.1.28 | ball_near_10cm | pass | 6.703 | 2.131 | 190.000 | 7.495 |
| v0.1.27 | ball_far_10cm | pass | 5.519 | 2.053 | 215.000 | 9.710 |
| v0.1.28 | ball_far_10cm | pass | 5.519 | 2.053 | 215.000 | 9.710 |
| v0.1.27 | initial_slower_02 | pass | 5.510 | 1.688 | 190.000 | 10.190 |
| v0.1.28 | initial_slower_02 | pass | 5.294 | 1.773 | 160.000 | 10.160 |
| v0.1.27 | initial_faster_02 | pass | 6.796 | 2.139 | 210.000 | 5.990 |
| v0.1.28 | initial_faster_02 | pass | 6.796 | 2.139 | 210.000 | 5.990 |
| v0.1.27 | combined_left_slower | pass | 6.866 | 2.044 | 205.000 | 10.240 |
| v0.1.28 | combined_left_slower | pass | 6.846 | 2.092 | 195.000 | 7.540 |
| v0.1.27 | combined_right_faster | pass | 7.168 | 3.174 | 200.000 | 10.040 |
| v0.1.28 | combined_right_faster | pass | 7.168 | 3.174 | 200.000 | 10.040 |
| v0.1.27 | heldout_left_near_slower | pass | 6.752 | 1.579 | 200.000 | 9.920 |
| v0.1.28 | heldout_left_near_slower | pass | 6.720 | 2.000 | 195.000 | 8.835 |
| v0.1.27 | heldout_right_far_slower | pass | 6.760 | 2.088 | 205.000 | 10.010 |
| v0.1.28 | heldout_right_far_slower | pass | 6.505 | 2.008 | 195.000 | 8.845 |
| v0.1.27 | heldout_left_far_faster | pass | 7.180 | 2.672 | 195.000 | 8.615 |
| v0.1.28 | heldout_left_far_faster | pass | 7.180 | 2.668 | 195.000 | 9.545 |
| v0.1.27 | heldout_right_near_faster | pass | 7.224 | 1.906 | 210.000 | 5.285 |
| v0.1.28 | heldout_right_near_faster | pass | 7.226 | 1.897 | 210.000 | 7.095 |
| v0.1.27 | heldout_confirm_left_far_slower | pass | 6.855 | 1.827 | 195.000 | 7.615 |
| v0.1.28 | heldout_confirm_left_far_slower | pass | 7.006 | 2.291 | 190.000 | 10.045 |
| v0.1.27 | heldout_confirm_right_near_slower | no_foot_contact, wrong_contact_surface, missed_goal, recovery_failed, physical_failure, running_failed, running_strike_transition_failed | unavailable | unavailable | unavailable | 0.000 |
| v0.1.28 | heldout_confirm_right_near_slower | pass | 6.189 | 1.927 | 210.000 | 7.015 |
| v0.1.28 | heldout_v028_near_slow_13 | pass | 6.376 | 1.837 | 180.000 | 9.830 |
| v0.1.28 | heldout_v028_near_slow_17 | missed_goal | 6.218 | 2.403 | 200.000 | 10.395 |
| v0.1.28 | heldout_v028_left_near_slow | missed_goal | 6.062 | 2.424 | 200.000 | 7.165 |

Follow-through starts at the first zero-force sample after the primary contact episode and runs for 300 ms, unless the trace ends or another foot-ball contact occurs. Its continuous forward duration ends at the last positive sample before the first nonpositive forward velocity; reversal is descriptive, not a failure. Integrated path length is a fixed-reference-point kinematic estimate. Original recovery duration and acceptance are retained; final-standing onset is derived from the original final duration, not from a new recovery threshold.

## Methods and evidence boundaries

- Physical trace validation rejects gaps, nonfinite force/time, inconsistent stored contact, and unexplained early truncation. The legacy duplicated terminal marker is excluded from integration.
- Precontact windows are the 300 ms preceding the first positive measured foot-ball force. Samples at contact are excluded, including their entire collision bin. At 5 ms sampling, a complete 300 ms window consequently provides 295 ms of strictly preimpact rectangular bins. Fractional first bins are clipped; window coverage is recorded.
- Signed velocity and estimated-power extrema retain all exact global ties and every local positive/negative plateau, with boundary flags. Representative global timing is the earliest tie. There is no smoothing, so small local extrema may reflect numerical noise. Absent signs have no fabricated zero-valued peak time. A 5 ms observation is not a continuous-time optimum.
- Power is `joint_vel × joint_torque_estimate`; positive and negative work are integrated separately. The frozen producer calls the implicit-actuator applied torque an estimated clipped PD torque. These are approximate actuator quantities, not measured hardware power, inverse dynamics, muscle power, or intersegment energy-transfer shares.
- Support onset is bracketed between the first ≥80 N sample and its preceding physics boundary. The last continuous left support reaching the last precontact sample is used. COM acceleration uses finite differences between two loaded observations; it cannot include the collision bin. Clipped positive vertical ground force only determines the support gate.
- Optional horizontal GRF is admitted only when `dynamics_telemetry.support_ground_force` certifies world force ON the robot, normal plus friction, ground filtering, complete coverage, and a verified sign probe. Missing, partial, or uncertified vectors remain unavailable. No horizontal force is reconstructed from vertical support or COM deceleration.
- First-contact impulse integrates actual sampled contact-point normal-force magnitudes over the first contiguous contact episode. Separate episodes are counted independently. This scalar normal impulse is not the net vector impulse or the ball momentum change. Preimpact foot speed is the last noncontact sample of the fixed foot point `[0.06, 0, -0.025] m`, not the collision-point speed.
- Segment headings project each measured body X axis onto the world horizontal plane and unwrap yaw. Heading rates are finite differences at interval midpoints, distinct from world-Z angular velocity. Relative pelvis/thorax projected heading requires common observed timestamps. A measured segment sequence does not establish causal energy flow.
- Missing quantities carry `status: unavailable`, a reason, and evidence needed. Invalid mandatory physical traces or changed source digests stop analysis. Optional missing fields do not silently become zero. The JSON preserves the original outcome for every case.

## Reproduction and provenance

```sh
python3 tools/analyze_explosive_strike.py \
  --output reports/explosive_strike_baseline_v029.json \
  --markdown reports/2026-11-07-explosive-strike-baseline.md
```

Outputs are new analysis artifacts. Existing files require explicit `--overwrite`; physical source reports cannot be output destinations. No local motion/model assets are exported.

| Input or analysis source | SHA-256 |
|---|---|
| `configs/strike_baseline_v1.json` | `768569e20e42e3bcea04730780916ef601f7d1ce4c958648de32af11eed1ce8b` |
| `reports/v023_baseline_final_summary.json` | `563661c0219d6e40421406ebb8de5df2827dde51931c343baf454e3d1c8f92a0` |
| `reports/v028_final_comparison.json` | `60b4c5a4f742d0ac40efbc6c8bd2a134649c3a15e056d0db8e5459d9ea7341f9` |
| `tools/explosive_strike_metrics.py` | `4e80f28607c14c681f82e79f39be4476c5bfd063b57a80701a3dc9b9d9784135` |
| `tools/analyze_explosive_strike.py` | `5bfb7524104845a21e7551044e00a62c31ee424a3d116783a22cedc6d0a9e8bc` |
| `tools/strike_baseline_metrics.py` | `22a86d021cf358f9f6c3a18b13166fd8dea81b31b535b28b5870ee51146fb1e5` |
| `reports/v023_baseline_center.json` | `aeeaa46a762b4569cdcea0fc4160b4e04c490608ff9f098f7791516e463528c1` |
| `reports/v027_final_center.json` | `395337fa2e0b8bd2051d8f8a5f9f52cc3942b326e930650b3d7b494dae104b9c` |
| `reports/v028_final_center.json` | `1b76423d04bcf3e007e8e248ffece090369b952e928c0f32e0af80cc6a3db37d` |

All source episodes are digest-checked against the existing summaries before metrics are computed; original task outcomes are recomputed and checked against those summaries. The full per-episode digest manifest is in the JSON. This verifies consistency with the local recorded evidence; it is not independent validation of the simulator or a fresh replay.

## Three observed v28 replays: actual force and segment telemetry

These are additional observation-only replays of the v0.1.28 controller. They preserve the original limits and acceptance criteria. Their original task pass does not erase a landing shock, overspeed, or unaccounted momentum change. Each report retains its own probe and collector source SHA; probe files are allowed to differ. The original and clean-exit probe versions have the same force-sign result.

Independent instrumented-analysis artifact: `reports/explosive_instrumented_v029.json`.

| Case | Original full task | Records exact | Kinematics exact | All legacy trace fields exact |
|---|---|---|---|---|
| initial_slower_02 | pass | True | True | True |
| center | pass | True | True | True |
| combined_right_faster | pass | True | True | True |

The fixed diagnostic interval is **1.80–1.90 s**. Peak force below includes observations at both endpoints. Impulses integrate only the overlapping post-step bins: the force sample at 1.800 s belongs to the preceding 1.795–1.800 s bin and is excluded from the interval impulse. This distinction matters for center, whose 1.800 s sample is its largest observed force in this interval. These are fixed absolute-time windows, not identical phases of every case’s landing.

| Case | Total force peak N @ s (body weights) | Foot-point speed peak m/s @ s | Max observed speed/limit | Window momentum residual norm (N·s) |
|---|---:|---:|---:|---:|
| initial_slower_02 | 6406.781 @ 1.855 (19.588) | 17.111 @ 1.855 | 1.902592 | 10.8815 |
| center | 3169.407 @ 1.800 (9.690) | 0.676 @ 1.800 | 0.367091 | 0.000100507 |
| combined_right_faster | 2677.304 @ 1.800 (8.186) | 0.497 @ 1.800 | 0.355078 | 0.000206618 |

World-vector component impulses over the same interval; X/Y/Z axes are kept explicit. Normal and friction are separately measured, not reconstructed from vertical support or COM changes.

| Case | Foot | Normal impulse [X,Y,Z] N·s | Friction impulse [X,Y,Z] N·s | Total impulse [X,Y,Z] N·s |
|---|---|---|---|---|
| initial_slower_02 | left | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] |
| initial_slower_02 | right | [0.000005, 0.000000, 37.813789] | [4.631879, 19.762905, 0.000003] | [4.631884, 19.762905, 37.813792] |
| center | left | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] |
| center | right | [0.000007, 0.000000, 54.178602] | [6.255263, 7.086064, -0.000000] | [6.255271, 7.086064, 54.178602] |
| combined_right_faster | left | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] | [0.000000, 0.000000, 0.000000] |
| combined_right_faster | right | [0.000006, 0.000000, 45.602486] | [3.760002, 3.390653, -0.000000] | [3.760009, 3.390653, 45.602486] |

| Case | Pelvis net 3-D rotation / projected yaw (deg) | Thorax net 3-D rotation / projected yaw (deg) | Relative pelvis→thorax rotation / projected-yaw change (deg) |
|---|---:|---:|---:|
| initial_slower_02 | 12.552 / -1.882 | 6.035 / -1.093 | 17.911 / 0.789 |
| center | 19.501 / 4.915 | 12.560 / 8.546 | 27.600 / 3.632 |
| combined_right_faster | 18.452 / 3.485 | 7.149 / 5.666 | 21.149 / 2.181 |

Three-dimensional net rotation is the angle of the relative orientation between observed endpoints, not an Euler-angle sum. Projected yaw uses the world-horizontal projection of each body X axis. Rotation magnitudes, relative headings, and observed sequencing do not establish energy transfer.

| Case | Precontact pelvis / thorax / left-arm / right-arm peak projected yaw-rate times (ms) |
|---|---:|
| initial_slower_02 | -287.5 / -32.5 / -72.5 / -17.5 |
| center | -177.5 / -242.5 / -87.5 / -17.5 |
| combined_right_faster | -262.5 / -232.5 / -77.5 / -7.5 |

Precontact values use the earlier 300 ms protocol and finite-difference interval midpoints. They are descriptive peak timings; the JSON retains signed extrema, all ties, and observed world angular velocities separately.

### Momentum residual and speed-limit observations

The diagnostic residual is `m(v_COM,end − v_COM,start) − J_left − J_right − m g Δt`. Mass is the measured summed PhysX body mass. Gravity is assumed equal to the recorded probe gravity `[0,0,−9.81] m/s²`; it is not independently stored as robot-world telemetry. All per-bin residual vectors and every overspeed joint are retained. A residual is not reassigned to friction or removed from task results.

| Case | Largest residual bin (s) | Original residual [X,Y,Z] N·s | Overspeed joints at bin start → end |
|---|---|---|---|
| initial_slower_02 | 1.855–1.860 | [1.314282, -10.459705, 2.696866] | 1.855: right_knee_joint 38.05184/20.0 rad/s (1.902592×); max ratio 1.902592 → 1.860: none; max ratio 0.225971 |
| center | 1.860–1.865 | [-0.000000, 0.000050, 0.000001] | 1.860: none; max ratio 0.261506 → 1.865: none; max ratio 0.286066 |
| combined_right_faster | 1.810–1.815 | [-0.000002, 0.000041, -0.000004] | 1.810: none; max ratio 0.170757 → 1.815: none; max ratio 0.196445 |

The `initial_slower_02` replay retains a full task pass but shows a large landing shock, right-knee overspeed, and a subsequent non-closing momentum bin. The large impact bin itself matches the measured contact impulse closely; the large unexplained change occurs in the following 5 ms bin. Temporal coincidence supports investigation, not a causal conclusion.

### Published solver mechanism: supporting inference, not a confirmed local bug

An open, unmerged PhysX proposal describes a pre-solve mechanism in which one joint exceeding its velocity limit scales all articulation joint velocities by the smallest limit-to-speed ratio. The proposal changes that behavior to separate per-DOF clipping. The installed solver binary has not been matched to that source. [PhysX PR #534](https://github.com/NVIDIA-Omniverse/PhysX/pull/534) (status checked 2026-11-07).

The following model is evaluated from the observation at the start of the largest residual bin:

```text
gamma = min(1, min_j limit_j / |qdot_j|)
x_rootCOM = x_root + R_pelvis * pelvis_inertial_origin
v_internal = v_COM - v_rootCOM - omega_root × (x_COM - x_rootCOM)
predicted impulse = mass * (gamma - 1) * v_internal
```

| Case | gamma | Predicted impulse [X,Y,Z] N·s | Prediction vs original residual error norm (N·s) |
|---|---:|---|---:|
| initial_slower_02 | 0.525598738 | [1.314941, -10.460071, 2.696803] | 0.000756399 |
| center | 1.000000000 | [0.000000, -0.000000, -0.000000] | 4.97135e-05 |
| combined_right_faster | 1.000000000 | [0.000000, -0.000000, -0.000000] | 4.11455e-05 |

The pelvis inertial origin `[0,0,−0.07605] m` is read from the SHA-matched URDF, not queried directly from PhysX. The model treats recorded root linear velocity as root-COM velocity and assumes the proposed uniform scaling applies while the base twist is unchanged. Close agreement in the overspeed case is evidence consistent with that mechanism; it does not prove which local binary operation caused the residual. No engine version, force, joint limit, state, or acceptance result is changed by this analysis.

### Instrumented provenance

| Case | Physical report SHA-256 | Frozen v28 SHA-256 | Probe file and SHA-256 |
|---|---|---|---|
| initial_slower_02 | `c7f0c9a8b6aecf1a08ac3711dfcdbb23fd0e9a47abda0d1c655bf07d742e2928` | `4fe258c739d0a7739321f4a7e7975d53bd212fbd4193a61fcbbd20ee7033b155` | `reports/v029_force_probe.json` — `d6e585664fa6f1e7939d606db79837c150fe44eb57f7a98f90627ef373cb5a02` |
| center | `42103e7b4a19ff08b02e5c8f5da46090d7631271f5a16200be946dad6bad9976` | `1b76423d04bcf3e007e8e248ffece090369b952e928c0f32e0af80cc6a3db37d` | `reports/v029_force_probe_cleanexit.json` — `048cc33377338189d0940ad072bf72138f98064b4b3ae68726d79edfd818a316` |
| combined_right_faster | `f29c4ef98ef1b0900048857c775f23e74cc4c737bf27a42fe20df236b3508bab` | `e5e6c936f946fb8b38e5a42e22e475c0eb22e9f3730d7f37d7f82fe2e4180ebe` | `reports/v029_force_probe_cleanexit.json` — `048cc33377338189d0940ad072bf72138f98064b4b3ae68726d79edfd818a316` |

Collector and wrapper source hashes are copied unchanged from each physical report’s `dynamics_telemetry`; later wrapper cleanup does not retroactively change acquisition provenance. The JSON also records the URDF hash, original probe certificate, baseline-match results, and current analysis-source hashes.

Reproduce the baseline plus these three instrumented analyses:

```sh
python3 tools/analyze_explosive_strike.py \
  --output reports/explosive_strike_baseline_v029.json \
  --instrumented-output reports/explosive_instrumented_v029.json \
  --instrumented-reports reports/v029_instrumented_initial_slower_02.json reports/v029_instrumented_center.json reports/v029_instrumented_combined_right_faster.json \
  --markdown reports/2026-11-07-explosive-strike-baseline.md --overwrite
```
