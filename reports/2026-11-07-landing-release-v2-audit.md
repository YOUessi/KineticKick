# Landing-release audit — 2026-11-07

Independent CPU analysis for v0.1.30. Original task acceptance is preserved; lower landing peaks cannot conceal a failed goal, contact surface, running transition, recovery, or physical-integrity gate.

This v2 artifact preserves the earlier five-case v1 JSON/report unchanged and compares four new measured runs with frozen v28. The three representative runs also retain comparison with the already verified continuous-pose implementation. Intentional controller/config/source fingerprint changes are provenance, not physical-output differences. The preserved v1 JSON has SHA-256 `5d0e998b7cfaa2604af5ccec902a86db7e713be91aef49bd72c51c2e24cb6f0c`.

In `initial_slower_02`, trigger lag is 0.12 s and frozen release strength is one; physical traces, original records, contacts and kinematics are exactly equal to the earlier continuous-pose replay. The original slower overspeed/momentum improvement is retained. In center and combined-right-faster, all 700 lag observations are zero, no release triggers, and all 2,800 physical samples, contacts, 700 kinematic samples, 700 legacy feedback rows and original records are exactly equal to frozen v28. Their original 8.965 s and 10.040 s final standing durations are restored, rather than inheriting v1's recovery regressions.

The declared partial-strength diagnostic `heldout_confirm_right_near_slower` passes the full original task, protected prefix and primary strike checks. Its trigger lag is 0.04064528644 s, frozen strength 0.5080660582 and settled retention 0.4919339418. Final standing starts at 4.010 s instead of 6.985 s, a 2.975 s improvement. Whole-run maximum q̇/limit stays 0.90906248 with no overspeed. Event-window q̇/limit rises from 0.666109 to 0.690527 and foot-velocity jump from 2.317125 to 2.363098 m/s; this is not improvement of every metric. Original vector GRF/friction were not acquired for this case and remain unavailable; the new measured momentum residual is 0.0000902563 N·s.

At the partial case's original absolute-window exit (1.860 s decision), positive retention leaves an internally computed projection offset that the inactive original mask bypasses: right ankle pitch −0.00922071 rad and ankle roll +0.05008839 rad. The actual final target steps are −0.0264331 and −0.0600027 rad, compared with original +0.1512706 and −0.0802740 rad. These are separate observations: the whole final command is not certified C², and raw policy/residual changes remain part of its step. `short_relative_swing` remains in all four conditions.

The independent lag checks compare seconds with both `max(-phase_action, 0)` and `max(policy_time - phase_s, 0)`, verify strength from the declared `phase_lag_scale_s` at the first trigger, require exact freezing thereafter, and verify settled retention `1-strength`. The lag scale and release duration are independently read configuration quantities, both 0.08 s in these acquisitions. Zero-lag preservation requires complete original-run equality, not only absence of a trigger. These paragraphs interpret the reproducible tables below without changing any task gate or metric.

Each measured touchdown is the first right-foot ground-normal transition from zero to positive after ball contact. Its fixed diagnostic window is 50 ms before to 100 ms after that event. The legacy 1.80–1.90 s interval remains in the JSON. Endpoint-inclusive force peaks and overlapping-bin impulses are distinct, following the P1 definitions.

| Case / run | Mode | Release decision (s) | Protected prefix | Primary strike unchanged | Original full task | Audit pass |
|---|---|---:|---|---|---|---|
| v030_lag_pose_initial_slower_02 | pose | 1.82 | True | True | True | True |
| v030_lag_pose_center | zero-lag preservation | None | True | True | True | True |
| v030_lag_pose_combined_right_faster | zero-lag preservation | None | True | True | True | True |
| v030_lag_pose_heldout_confirm_right_near_slower | pose | 1.76 | True | True | True | True |

| Case / run | Completed physics s | Terminal markers | Final stable standing s | Whole-run max q̇/limit | Initial outgoing XY speed m/s | Termination reasons |
|---|---:|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 14 | 0 | 10.16 | 1.90259 | 3.92292 | [[]] |
| v30-continuous-pose/initial_slower_02 | 14 | 0 | 9.665 | 0.908253 | 3.92292 | [[]] |
| v030_lag_pose_initial_slower_02 | 14 | 0 | 9.665 | 0.908253 | 3.92292 | [[]] |
| v28/center | 14 | 0 | 8.91 | 0.908129 | 4.6314 | [[]] |
| v30-continuous-pose/center | 14 | 0 | 8.51 | 0.908129 | 4.6314 | [[]] |
| v030_lag_pose_center | 14 | 0 | 8.91 | 0.908129 | 4.6314 | [[]] |
| v28/combined_right_faster | 14 | 0 | 10.04 | 0.907301 | 5.52537 | [[]] |
| v30-continuous-pose/combined_right_faster | 14 | 0 | 6.355 | 0.907301 | 5.52537 | [[]] |
| v030_lag_pose_combined_right_faster | 14 | 0 | 10.04 | 0.907301 | 5.52537 | [[]] |
| v28/heldout_confirm_right_near_slower | 14 | 0 | 7.015 | 0.909062 | 4.22812 | [[]] |
| v030_lag_pose_heldout_confirm_right_near_slower | 14 | 0 | 9.99 | 0.909062 | 4.22812 | [[]] |

| Case / run | Touchdown s | Event force peak N | Event foot-speed jump m/s | Event max q̇/limit | Event momentum residual norm N·s | Max residual bin s |
|---|---:|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 1.855 | 6406.78 | 19.2934 | 1.90259 | 10.8815 | [1.855, 1.86] |
| v30-continuous-pose/initial_slower_02 | 1.855 | 2179.6 | 3.19945 | 0.579248 | 0.000198727 | [1.95, 1.955] |
| v030_lag_pose_initial_slower_02 | 1.855 | 2179.6 | 3.19945 | 0.579248 | 0.000198727 | [1.95, 1.955] |
| v28/center | 1.785 | 3169.41 | 1.35347 | 0.749054 | 9.20553e-05 | [1.79, 1.795] |
| v30-continuous-pose/center | 1.79 | 3310.86 | 1.84099 | 0.684313 | 0.000121191 | [1.7650000000000001, 1.77] |
| v030_lag_pose_center | 1.785 | 3169.41 | 1.35347 | 0.749054 | 9.20553e-05 | [1.79, 1.795] |
| v28/combined_right_faster | 1.78 | 2677.3 | 3.07787 | 0.639248 | 0.000146365 | [1.81, 1.815] |
| v30-continuous-pose/combined_right_faster | 1.785 | 3169.35 | 1.9308 | 0.66503 | 0.000224113 | [1.835, 1.84] |
| v030_lag_pose_combined_right_faster | 1.78 | 2677.3 | 3.07787 | 0.639248 | 0.000146365 | [1.81, 1.815] |
| v28/heldout_confirm_right_near_slower | 1.82 | unavailable | 2.31712 | 0.666109 | unavailable | unavailable |
| v030_lag_pose_heldout_confirm_right_near_slower | 1.82 | 2606.75 | 2.3631 | 0.690527 | 9.02563e-05 | [1.79, 1.795] |

The following impulses use the complete event window. Vectors are world [x, y, z], in N·s; normal and friction are independently recorded, then summed. They are bilateral foot-ground impulses, not ball impulse or inferred energy shares.

| Case / run | Normal impulse | Friction impulse | Total impulse |
|---|---|---|---|
| v28/initial_slower_02 | [7.1234e-06, 0, 53.0177] | [9.68939, 16.4837, 3.15121e-06] | [9.6894, 16.4837, 53.0177] |
| v30-continuous-pose/initial_slower_02 | [8.18963e-06, 0, 60.9534] | [9.15496, 6.92097, 6.59573e-07] | [9.15497, 6.92097, 60.9534] |
| v030_lag_pose_initial_slower_02 | [8.18963e-06, 0, 60.9534] | [9.15496, 6.92097, 6.59573e-07] | [9.15497, 6.92097, 60.9534] |
| v28/center | [9.97564e-06, 0, 74.2463] | [5.30249, 8.71009, -3.32725e-07] | [5.3025, 8.71009, 74.2463] |
| v30-continuous-pose/center | [1.02219e-05, 0, 76.0792] | [5.81151, 9.21565, 2.12048e-06] | [5.81152, 9.21565, 76.0792] |
| v030_lag_pose_center | [9.97564e-06, 0, 74.2463] | [5.30249, 8.71009, -3.32725e-07] | [5.3025, 8.71009, 74.2463] |
| v28/combined_right_faster | [8.61872e-06, 0, 64.1471] | [1.87704, 4.65738, 2.28608e-06] | [1.87705, 4.65738, 64.1471] |
| v30-continuous-pose/combined_right_faster | [8.85218e-06, 0, 65.8846] | [3.03613, 5.30785, 1.56597e-06] | [3.03614, 5.30785, 65.8846] |
| v030_lag_pose_combined_right_faster | [8.61872e-06, 0, 64.1471] | [1.87704, 4.65738, 2.28608e-06] | [1.87705, 4.65738, 64.1471] |
| v28/heldout_confirm_right_near_slower | unavailable | unavailable | unavailable |
| v030_lag_pose_heldout_confirm_right_near_slower | [8.96808e-06, 0, 66.7472] | [-2.70946, 7.01638, 1.9426e-06] | [-2.70945, 7.01638, 66.7472] |

| Case / run | Pelvis net rotation ° | Thorax net rotation ° | Pelvis projected yaw change ° | Thorax projected yaw change ° |
|---|---:|---:|---:|---:|
| v28/initial_slower_02 | 16.5961 | 15.0893 | 0.999118 | 3.97269 |
| v30-continuous-pose/initial_slower_02 | 16.7245 | 7.93151 | -2.46478 | 5.92022 |
| v030_lag_pose_initial_slower_02 | 16.7245 | 7.93151 | -2.46478 | 5.92022 |
| v28/center | 17.4071 | 12.7078 | 4.31379 | 10.4972 |
| v30-continuous-pose/center | 18.3726 | 13.2301 | 4.56375 | 11.0408 |
| v030_lag_pose_center | 17.4071 | 12.7078 | 4.31379 | 10.4972 |
| v28/combined_right_faster | 19.0757 | 10.7458 | 3.89949 | 8.6857 |
| v30-continuous-pose/combined_right_faster | 19.7842 | 11.6341 | 5.05222 | 10.1198 |
| v030_lag_pose_combined_right_faster | 19.0757 | 10.7458 | 3.89949 | 8.6857 |
| v28/heldout_confirm_right_near_slower | unavailable | unavailable | unavailable | unavailable |
| v030_lag_pose_heldout_confirm_right_near_slower | 16.674 | 12.8918 | 2.32002 | 8.23327 |

| Case / run | Pelvis-relative advance m | Relative displacement at 250 ms m | Relative speed at 120 ms m/s | Relative velocity step m/s | P0 diagnostic failures |
|---|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 0.083033 | -0.110178 | -0.412208 | 1.59443 | ['short_relative_swing'] |
| v30-continuous-pose/initial_slower_02 | 0.083033 | -0.0817865 | -0.412208 | 1.25302 | ['short_relative_swing'] |
| v030_lag_pose_initial_slower_02 | 0.083033 | -0.0817865 | -0.412208 | 1.25302 | ['short_relative_swing'] |
| v28/center | 0.13392 | 0.0394325 | 0.0849894 | 1.80661 | ['short_relative_swing'] |
| v30-continuous-pose/center | 0.134674 | 0.0444887 | 0.0849894 | 1.95573 | ['short_relative_swing'] |
| v030_lag_pose_center | 0.13392 | 0.0394325 | 0.0849894 | 1.80661 | ['short_relative_swing'] |
| v28/combined_right_faster | 0.127633 | 0.0261276 | -0.149385 | 1.70672 | ['short_relative_swing'] |
| v30-continuous-pose/combined_right_faster | 0.1291 | 0.0373455 | -0.149385 | 1.9473 | ['short_relative_swing'] |
| v030_lag_pose_combined_right_faster | 0.127633 | 0.0261276 | -0.149385 | 1.70672 | ['short_relative_swing'] |
| v28/heldout_confirm_right_near_slower | 0.0885055 | -0.0369604 | -0.487063 | 1.11705 | ['short_relative_swing'] |
| v030_lag_pose_heldout_confirm_right_near_slower | 0.0885055 | -0.031357 | -0.487063 | 1.38703 | ['short_relative_swing'] |

Every unavailable field keeps its evidence requirement in the JSON. The event-aligned window uses observed timestamps and can remain incomplete after early termination. Physical task failures and P0 follow-through diagnostic failures are retained separately; neither is removed because another number improved.

## Outcomes and evidence

- **v030_lag_pose_initial_slower_02**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 363/363, contacts 28/28, kinematics 91/91, feedback 91/91. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 1.11022e-16 rad, 697 exact endpoint checks.
  Lag/strength check: `True`; trigger lag 0.12 s, frozen strength 1, settled retention 0; maximum lag-unit disagreement 7.45058e-08 s. Verification mode: `active_release`.
  Compared with preserved v1 continuous-pose: physical outputs exact `True`; policy/config fingerprints intentionally checked as provenance, not motion equality. Full v28 comparison remains primary.
- **v030_lag_pose_center**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 2800/2800, contacts 26/26, kinematics 700/700, feedback 700/700. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 5.55112e-17 rad, 700 exact endpoint checks.
  Lag/strength check: `True`; trigger lag unavailable s, frozen strength unavailable, settled retention unavailable; maximum lag-unit disagreement 0 s. Verification mode: `inactive_zero_lag_preservation`.
  Compared with preserved v1 continuous-pose: physical outputs exact `False`; policy/config fingerprints intentionally checked as provenance, not motion equality. Full v28 comparison remains primary.
- **v030_lag_pose_combined_right_faster**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 2800/2800, contacts 30/30, kinematics 700/700, feedback 700/700. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 1.11022e-16 rad, 700 exact endpoint checks.
  Lag/strength check: `True`; trigger lag unavailable s, frozen strength unavailable, settled retention unavailable; maximum lag-unit disagreement 0 s. Verification mode: `inactive_zero_lag_preservation`.
  Compared with preserved v1 continuous-pose: physical outputs exact `False`; policy/config fingerprints intentionally checked as provenance, not motion equality. Full v28 comparison remains primary.
- **v030_lag_pose_heldout_confirm_right_near_slower**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 351/351, contacts 17/17, kinematics 88/88, feedback 88/88. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 5.55112e-17 rad, 88 exact endpoint checks.
  Lag/strength check: `True`; trigger lag 0.0406453 s, frozen strength 0.508066, settled retention 0.491934; maximum lag-unit disagreement 4.65661e-07 s. Verification mode: `active_release`.

## Zero-retention target boundary

This diagnostic checks the actual saved command immediately before and after the first zero-retention decision. The physical sample ending at the decision holds the previous command; the next 5 ms sample holds the new one. Soft bounds come only from the ten recorded `approach_soft_limits_rad` joints and the six right-leg `absolute_soft_limits_rad` joints where the later acquisition includes them (11 unique joints). Soft-target excess is distinct from actual joint-speed violation, and does not replace the frozen task gate.

| Case / run | Side | Decision s | Max target step rad | Joint with max step | Right ankle pitch target before → after rad | Right ankle pitch soft excess after rad | Right ankle roll soft excess after rad |
|---|---|---:|---:|---|---|---:|---:|
| v030_lag_pose_initial_slower_02 | reference | 1.88 | 0.217857 | left_hip_pitch_joint | 0.47124 → 0.461953 | 0.00816661 | 0 |
| v030_lag_pose_initial_slower_02 | candidate | 1.88 | 0.268799 | right_hip_pitch_joint | 0.521777 → 0.742474 | 0.288687 | 0.0910995 |
| v030_lag_pose_center | reference | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |
| v030_lag_pose_center | candidate | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |
| v030_lag_pose_combined_right_faster | reference | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |
| v030_lag_pose_combined_right_faster | candidate | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |
| v030_lag_pose_heldout_confirm_right_near_slower | reference | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |
| v030_lag_pose_heldout_confirm_right_near_slower | candidate | unavailable | unavailable | unavailable | unavailable → unavailable | unavailable | unavailable |

The earlier coefficient-retention implementation changed whether the absolute target was clamped when its weight became zero. Those acquisitions did not sample the raw expert target. The later whole-operator implementation records base, projected and blended targets and is checked separately against q_blend = q_base + r(q_projected − q_base), including exact r=0/1 endpoints. Downstream residuals and changing policy state still affect the saved final target. Remaining runtime soft bounds stay unavailable (19 joints in legacy acquisitions; 18 in the later acquisition). Original task success is separate from target continuity.

## Partial release at the original absolute-window exit

A positive final retention can leave a projected correction when the original absolute window becomes inactive. The internal blend then stops being applied by that original mask. The offset below is computed from recorded internal targets at the same decision; it is separate from the observed total target step across policy decisions.

| Run | Exit decision s | Strength | Retention | Bypassed projection max rad | Final target max step rad | Original v28 target max step rad |
|---|---:|---:|---:|---:|---:|---:|
| v030_lag_pose_heldout_confirm_right_near_slower | 1.86 | 0.508066 | 0.491934 | 0.0500884 | 0.274432 | 0.268663 |

These measurements do not establish whole-command C2 continuity. All per-joint targets, sample brackets and actually recorded soft bounds remain in the JSON.

## Methods and provenance

The protected prefix ends strictly before the first feedback decision whose recorded retention is less than one. It is not cut at the earlier trigger timestamp. Every original physical-trace, contact, kinematic and feedback value is compared exactly; added telemetry/debug fields are excluded. Policy timestamps are validated on the original 20 ms grid before selecting the prefix. The primary strike comparison covers only the first contiguous foot-ball force episode, its preimpact fixed-point velocity, and first zero-force outgoing ball state.

Recorded weight pairs are checked against the declared mode: all scales swing, absolute and placement; pose preserves the swing weight and scales absolute/placement. The saved float32 multiplication allows an absolute tolerance of 5×10⁻⁷. The audit also checks nonincreasing retention and a completed velocity pulse whenever retention is below one. It does not infer unseen trigger state.

P0 supplies pelvis-relative path and whole-run joint-limit diagnostics. P1 supplies force components, impulses, foot-velocity jumps, joint overspeed samples, segment motion and the unaccounted momentum residual. Probe certificates are independently recomputed from their stored samples. The release audit does not infer energy transfer, fix momentum residuals, or change any acceptance threshold.

| Source | SHA-256 |
|---|---|
| `tools/audit_landing_release.py` | `c435285c3de3976b6bfc6e1f2c1b01fa6a6f53cc31fedc4f052c429561c5bd7a` |
| `tools/audit_p0_dynamic_strike.py` | `f2dfa4ad64991c2be26275fb231e31ee938730d46b286e95046e82d79d361bcf` |
| `tools/explosive_strike_metrics.py` | `4e80f28607c14c681f82e79f39be4476c5bfd063b57a80701a3dc9b9d9784135` |
| `tools/strike_baseline_metrics.py` | `22a86d021cf358f9f6c3a18b13166fd8dea81b31b535b28b5870ee51146fb1e5` |
| `tools/strike_dynamics_telemetry.py` | `642150d97cabf728053f6d7eb7c4b943bf58f71a2ae876949b37dac186fbbb67` |
| `tools/analyze_explosive_strike.py` | `5bfb7524104845a21e7551044e00a62c31ee424a3d116783a22cedc6d0a9e8bc` |
| `tools/placement_strike.py` | `09863a12444d67d832afdce2115bc935a2225ebb9973e5820ba1857f0cea366a` |
| `tools/closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| `reports/v028_final_initial_slower_02.json` | `4fe258c739d0a7739321f4a7e7975d53bd212fbd4193a61fcbbd20ee7033b155` |
| `reports/v029_instrumented_initial_slower_02.json` | `c7f0c9a8b6aecf1a08ac3711dfcdbb23fd0e9a47abda0d1c655bf07d742e2928` |
| `reports/v030_lag_pose_initial_slower_02.json` | `d8be9ecf30cdc3b6cdf00edd4cbfe797a6c5252981160507e6b84458883a9818` |
| `reports/v030_continuous_pose_initial_slower_02.json` | `44864278094b276a88339cf14458651abf6fde1cc2311e79574a93e04172926b` |
| `reports/v028_final_center.json` | `1b76423d04bcf3e007e8e248ffece090369b952e928c0f32e0af80cc6a3db37d` |
| `reports/v029_instrumented_center.json` | `42103e7b4a19ff08b02e5c8f5da46090d7631271f5a16200be946dad6bad9976` |
| `reports/v030_lag_pose_center.json` | `52e40af73a607931a3ef07d0663e09ec2bde9b3b700858f8ce982c97374059ec` |
| `reports/v030_continuous_pose_center.json` | `6c6ef1fcc6a208cc955e66d30f24e783659d11d2791ca2e453b7b2583db04d2e` |
| `reports/v028_final_combined_right_faster.json` | `e5e6c936f946fb8b38e5a42e22e475c0eb22e9f3730d7f37d7f82fe2e4180ebe` |
| `reports/v029_instrumented_combined_right_faster.json` | `f29c4ef98ef1b0900048857c775f23e74cc4c737bf27a42fe20df236b3508bab` |
| `reports/v030_lag_pose_combined_right_faster.json` | `2b142605b7fa5fc19872a72e1fac839551e41dc0a71a31dad3fad8e957472ecf` |
| `reports/v030_continuous_pose_combined_right_faster.json` | `9e4635e908e0c75a77ed171329a2b215d1d34431f8effee108a09c7b8e6d99d0` |
| `reports/v028_final_heldout_confirm_right_near_slower.json` | `f27924003ce222ed9b80fa8d7dfc257445d12e9e82c38e1b12246365035c5f4c` |
| `reports/v030_lag_pose_heldout_confirm_right_near_slower.json` | `addfa8c9357347fdb3a3749851d80cf17151fc8df6cfb7948ef25ed139c1db43` |
| `reports/v029_force_probe.json` | `d6e585664fa6f1e7939d606db79837c150fe44eb57f7a98f90627ef373cb5a02` |
| `reports/v029_force_probe_cleanexit.json` | `048cc33377338189d0940ad072bf72138f98064b4b3ae68726d79edfd818a316` |

Controller hashes below are the values recorded by each acquisition, rather than claims that the current working files still have those contents. The raw-report hashes above anchor this metadata; the JSON also retains the resolved release configuration.

| Acquisition | Source | Recorded SHA-256 |
|---|---|---|
| v28/initial_slower_02 | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/initial_slower_02 | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/initial_slower_02 | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v30-continuous-pose/initial_slower_02 | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v30-continuous-pose/initial_slower_02 | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v30-continuous-pose/initial_slower_02 | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v30-continuous-pose/initial_slower_02 | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |
| v030_lag_pose_initial_slower_02 | `placement_strike.py` | `09863a12444d67d832afdce2115bc935a2225ebb9973e5820ba1857f0cea366a` |
| v030_lag_pose_initial_slower_02 | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_lag_pose_initial_slower_02 | `landing_release.py` | `1f6cf6a67ffbda866a875e038a81ede3565fe40d5c3a507b88e51ff48b70d363` |
| v030_lag_pose_initial_slower_02 | Resolved joint-strike configuration | `3e7647e05b071d3f257e16a046f63f674821d44f1489cc69b175e9ab40aeb39f` |
| v28/center | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/center | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/center | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v30-continuous-pose/center | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v30-continuous-pose/center | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v30-continuous-pose/center | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v30-continuous-pose/center | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |
| v030_lag_pose_center | `placement_strike.py` | `09863a12444d67d832afdce2115bc935a2225ebb9973e5820ba1857f0cea366a` |
| v030_lag_pose_center | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_lag_pose_center | `landing_release.py` | `1f6cf6a67ffbda866a875e038a81ede3565fe40d5c3a507b88e51ff48b70d363` |
| v030_lag_pose_center | Resolved joint-strike configuration | `3e7647e05b071d3f257e16a046f63f674821d44f1489cc69b175e9ab40aeb39f` |
| v28/combined_right_faster | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/combined_right_faster | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/combined_right_faster | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v30-continuous-pose/combined_right_faster | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v30-continuous-pose/combined_right_faster | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v30-continuous-pose/combined_right_faster | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v30-continuous-pose/combined_right_faster | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |
| v030_lag_pose_combined_right_faster | `placement_strike.py` | `09863a12444d67d832afdce2115bc935a2225ebb9973e5820ba1857f0cea366a` |
| v030_lag_pose_combined_right_faster | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_lag_pose_combined_right_faster | `landing_release.py` | `1f6cf6a67ffbda866a875e038a81ede3565fe40d5c3a507b88e51ff48b70d363` |
| v030_lag_pose_combined_right_faster | Resolved joint-strike configuration | `3e7647e05b071d3f257e16a046f63f674821d44f1489cc69b175e9ab40aeb39f` |
| v28/heldout_confirm_right_near_slower | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/heldout_confirm_right_near_slower | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/heldout_confirm_right_near_slower | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v030_lag_pose_heldout_confirm_right_near_slower | `placement_strike.py` | `09863a12444d67d832afdce2115bc935a2225ebb9973e5820ba1857f0cea366a` |
| v030_lag_pose_heldout_confirm_right_near_slower | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_lag_pose_heldout_confirm_right_near_slower | `landing_release.py` | `1f6cf6a67ffbda866a875e038a81ede3565fe40d5c3a507b88e51ff48b70d363` |
| v030_lag_pose_heldout_confirm_right_near_slower | Resolved joint-strike configuration | `3e7647e05b071d3f257e16a046f63f674821d44f1489cc69b175e9ab40aeb39f` |

Acquisition source/probe fingerprints, exact prefix counts and first differing keys, signed force-component impulses, every observed overspeed joint, and all per-bin residuals remain in the JSON artifact. No simulation, GPU operation, state write, or model training is performed by this command.

```sh
python3 tools/audit_landing_release.py --candidates reports/v030_lag_pose_initial_slower_02.json reports/v030_lag_pose_center.json reports/v030_lag_pose_combined_right_faster.json reports/v030_lag_pose_heldout_confirm_right_near_slower.json --output reports/landing_release_v030_v2_audit.json --markdown reports/2026-11-07-landing-release-v2-audit.md --overwrite
```
