# Landing-release audit — 2026-11-07

Independent CPU analysis for v0.1.30. Original task acceptance is preserved; lower landing peaks cannot conceal a failed goal, contact surface, running transition, recovery, or physical-integrity gate.

The shared `all` release failed the original task at 4.560 s despite lower impact metrics. The earlier coefficient-only `pose` release passed the original task but exposed the recorded zero-weight target boundary. The subsequent whole-operator `pose` implementation passed the protected-prefix, primary-strike and full-task checks for the three measured representative conditions. Its 2,100 recorded decisions passed the internal blend formula and projected-bound checks, including 2,091 exact retention-zero/one endpoints. These observations do not establish smoothness of the final actuator targets or success on the untested matrix.

The whole-operator change resolves the measured slower-case overspeed and large momentum residual in this replay: landing q̇/limit falls from 1.90259 to 0.579248, and event-window residual norm from 10.8815 to 0.000198727 N·s. Performance is mixed elsewhere. Center peak force increases from 3169.41 to 3310.86 N and its foot-speed jump increases from 1.35347 to 1.84099 m/s. Combined-right-faster peak force increases from 2677.30 to 3169.35 N; final stable standing starts at 7.645 s instead of 3.960 s, leaving 6.355 s instead of 10.040 s of final standing. Passing the original task therefore does **not** mean no regression.

The P0 `short_relative_swing` diagnostic remains in all three whole-operator conditions. In slower, maximum pelvis-relative advance stays 0.083033 m and the contact+120 ms relative speed stays −0.412208 m/s. This landing-release experiment has not resolved the separate follow-through deficit. The conclusions in these paragraphs interpret the reproducible tables below; they do not modify a gate or metric.

Each measured touchdown is the first right-foot ground-normal transition from zero to positive after ball contact. Its fixed diagnostic window is 50 ms before to 100 ms after that event. The legacy 1.80–1.90 s interval remains in the JSON. Endpoint-inclusive force peaks and overlapping-bin impulses are distinct, following the P1 definitions.

| Case / run | Mode | Release decision (s) | Protected prefix | Primary strike unchanged | Original full task | Audit pass |
|---|---|---:|---|---|---|---|
| v030_release_all_initial_slower_02 | all | 1.82 | True | True | False | False |
| v030_release_pose_initial_slower_02 | pose | 1.82 | True | True | True | True |
| v030_continuous_pose_initial_slower_02 | pose | 1.82 | True | True | True | True |
| v030_continuous_pose_center | pose | 1.74 | True | True | True | True |
| v030_continuous_pose_combined_right_faster | pose | 1.72 | True | True | True | True |

| Case / run | Completed physics s | Terminal markers | Final stable standing s | Whole-run max q̇/limit | Initial outgoing XY speed m/s | Termination reasons |
|---|---:|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 14 | 0 | 10.16 | 1.90259 | 3.92292 | [[]] |
| v030_release_all_initial_slower_02 | 4.56 | 1 | 0 | 1.24327 | 3.92292 | [['physical_height', 'physical_orientation']] |
| v030_release_pose_initial_slower_02 | 14 | 0 | 9.865 | 0.908253 | 3.92292 | [[]] |
| v030_continuous_pose_initial_slower_02 | 14 | 0 | 9.665 | 0.908253 | 3.92292 | [[]] |
| v28/center | 14 | 0 | 8.91 | 0.908129 | 4.6314 | [[]] |
| v030_continuous_pose_center | 14 | 0 | 8.51 | 0.908129 | 4.6314 | [[]] |
| v28/combined_right_faster | 14 | 0 | 10.04 | 0.907301 | 5.52537 | [[]] |
| v030_continuous_pose_combined_right_faster | 14 | 0 | 6.355 | 0.907301 | 5.52537 | [[]] |

| Case / run | Touchdown s | Event force peak N | Event foot-speed jump m/s | Event max q̇/limit | Event momentum residual norm N·s | Max residual bin s |
|---|---:|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 1.855 | 6406.78 | 19.2934 | 1.90259 | 10.8815 | [1.855, 1.86] |
| v030_release_all_initial_slower_02 | 1.855 | 6018.95 | 13.4913 | 1.24327 | 3.79965 | [1.855, 1.86] |
| v030_release_pose_initial_slower_02 | 1.855 | 2179.6 | 3.19945 | 0.579248 | 1.97923e-05 | [1.85, 1.855] |
| v030_continuous_pose_initial_slower_02 | 1.855 | 2179.6 | 3.19945 | 0.579248 | 0.000198727 | [1.95, 1.955] |
| v28/center | 1.785 | 3169.41 | 1.35347 | 0.749054 | 9.20553e-05 | [1.79, 1.795] |
| v030_continuous_pose_center | 1.79 | 3310.86 | 1.84099 | 0.684313 | 0.000121191 | [1.7650000000000001, 1.77] |
| v28/combined_right_faster | 1.78 | 2677.3 | 3.07787 | 0.639248 | 0.000146365 | [1.81, 1.815] |
| v030_continuous_pose_combined_right_faster | 1.785 | 3169.35 | 1.9308 | 0.66503 | 0.000224113 | [1.835, 1.84] |

The following impulses use the complete event window. Vectors are world [x, y, z], in N·s; normal and friction are independently recorded, then summed. They are bilateral foot-ground impulses, not ball impulse or inferred energy shares.

| Case / run | Normal impulse | Friction impulse | Total impulse |
|---|---|---|---|
| v28/initial_slower_02 | [7.1234e-06, 0, 53.0177] | [9.68939, 16.4837, 3.15121e-06] | [9.6894, 16.4837, 53.0177] |
| v030_release_all_initial_slower_02 | [7.32494e-06, 0, 54.5177] | [9.39091, 10.7979, 2.904e-06] | [9.39092, 10.7979, 54.5177] |
| v030_release_pose_initial_slower_02 | [8.26536e-06, 0, 61.517] | [9.30008, 6.90067, 8.16907e-07] | [9.30009, 6.90067, 61.517] |
| v030_continuous_pose_initial_slower_02 | [8.18963e-06, 0, 60.9534] | [9.15496, 6.92097, 6.59573e-07] | [9.15497, 6.92097, 60.9534] |
| v28/center | [9.97564e-06, 0, 74.2463] | [5.30249, 8.71009, -3.32725e-07] | [5.3025, 8.71009, 74.2463] |
| v030_continuous_pose_center | [1.02219e-05, 0, 76.0792] | [5.81151, 9.21565, 2.12048e-06] | [5.81152, 9.21565, 76.0792] |
| v28/combined_right_faster | [8.61872e-06, 0, 64.1471] | [1.87704, 4.65738, 2.28608e-06] | [1.87705, 4.65738, 64.1471] |
| v030_continuous_pose_combined_right_faster | [8.85218e-06, 0, 65.8846] | [3.03613, 5.30785, 1.56597e-06] | [3.03614, 5.30785, 65.8846] |

| Case / run | Pelvis net rotation ° | Thorax net rotation ° | Pelvis projected yaw change ° | Thorax projected yaw change ° |
|---|---:|---:|---:|---:|
| v28/initial_slower_02 | 16.5961 | 15.0893 | 0.999118 | 3.97269 |
| v030_release_all_initial_slower_02 | 16.9766 | 10.5193 | -4.85636 | 3.31533 |
| v030_release_pose_initial_slower_02 | 16.7961 | 7.81665 | -2.53654 | 5.82097 |
| v030_continuous_pose_initial_slower_02 | 16.7245 | 7.93151 | -2.46478 | 5.92022 |
| v28/center | 17.4071 | 12.7078 | 4.31379 | 10.4972 |
| v030_continuous_pose_center | 18.3726 | 13.2301 | 4.56375 | 11.0408 |
| v28/combined_right_faster | 19.0757 | 10.7458 | 3.89949 | 8.6857 |
| v030_continuous_pose_combined_right_faster | 19.7842 | 11.6341 | 5.05222 | 10.1198 |

| Case / run | Pelvis-relative advance m | Relative displacement at 250 ms m | Relative speed at 120 ms m/s | Relative velocity step m/s | P0 diagnostic failures |
|---|---:|---:|---:|---:|---|
| v28/initial_slower_02 | 0.083033 | -0.110178 | -0.412208 | 1.59443 | ['short_relative_swing'] |
| v030_release_all_initial_slower_02 | 0.083033 | -0.129242 | -0.412208 | 2.02806 | ['short_relative_swing', 'early_reversal'] |
| v030_release_pose_initial_slower_02 | 0.083033 | -0.0823504 | -0.412208 | 1.22731 | ['short_relative_swing'] |
| v030_continuous_pose_initial_slower_02 | 0.083033 | -0.0817865 | -0.412208 | 1.25302 | ['short_relative_swing'] |
| v28/center | 0.13392 | 0.0394325 | 0.0849894 | 1.80661 | ['short_relative_swing'] |
| v030_continuous_pose_center | 0.134674 | 0.0444887 | 0.0849894 | 1.95573 | ['short_relative_swing'] |
| v28/combined_right_faster | 0.127633 | 0.0261276 | -0.149385 | 1.70672 | ['short_relative_swing'] |
| v030_continuous_pose_combined_right_faster | 0.1291 | 0.0373455 | -0.149385 | 1.9473 | ['short_relative_swing'] |

Every unavailable field keeps its evidence requirement in the JSON. The event-aligned window uses observed timestamps and can remain incomplete after early termination. Physical task failures and P0 follow-through diagnostic failures are retained separately; neither is removed because another number improved.

## Outcomes and evidence

- **v030_release_all_initial_slower_02**: audit failures `['recovery_failed', 'physical_failure', 'running_failed']`; original task failures `['recovery_failed', 'physical_failure', 'running_failed']`. Protected sample counts (reference/candidate): physical 363/363, contacts 28/28, kinematics 91/91, feedback 91/91. Recorded weight-rule check: `True`.
- **v030_release_pose_initial_slower_02**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 363/363, contacts 28/28, kinematics 91/91, feedback 91/91. Recorded weight-rule check: `True`.
- **v030_continuous_pose_initial_slower_02**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 363/363, contacts 28/28, kinematics 91/91, feedback 91/91. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 1.11022e-16 rad, 697 exact endpoint checks.
- **v030_continuous_pose_center**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 347/347, contacts 26/26, kinematics 87/87, feedback 87/87. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 5.55112e-17 rad, 697 exact endpoint checks.
- **v030_continuous_pose_combined_right_faster**: audit failures `[]`; original task failures `[]`. Protected sample counts (reference/candidate): physical 343/343, contacts 30/30, kinematics 86/86, feedback 86/86. Recorded weight-rule check: `True`.
  Whole-operator formula/endpoint check: `True` over 700 decisions; max formula error 1.11022e-16 rad, 697 exact endpoint checks.

## Zero-retention target boundary

This diagnostic checks the actual saved command immediately before and after the first zero-retention decision. The physical sample ending at the decision holds the previous command; the next 5 ms sample holds the new one. Soft bounds come only from the ten recorded `approach_soft_limits_rad` joints and the six right-leg `absolute_soft_limits_rad` joints where the later acquisition includes them (11 unique joints). Soft-target excess is distinct from actual joint-speed violation, and does not replace the frozen task gate.

| Case / run | Side | Decision s | Max target step rad | Joint with max step | Right ankle pitch target before → after rad | Right ankle pitch soft excess after rad | Right ankle roll soft excess after rad |
|---|---|---:|---:|---|---|---:|---:|
| v030_release_all_initial_slower_02 | reference | 1.88 | 0.217857 | left_hip_pitch_joint | 0.47124 → 0.461953 | 0.00816661 | 0 |
| v030_release_all_initial_slower_02 | candidate | 1.88 | 0.240862 | right_ankle_pitch_joint | 0.444473 → 0.685335 | 0.231548 | 0.113396 |
| v030_release_pose_initial_slower_02 | reference | 1.88 | 0.217857 | left_hip_pitch_joint | 0.47124 → 0.461953 | 0.00816661 | 0 |
| v030_release_pose_initial_slower_02 | candidate | 1.88 | 0.278017 | right_hip_pitch_joint | 0.455593 → 0.718565 | 0.264779 | 0.0961607 |
| v030_continuous_pose_initial_slower_02 | reference | 1.88 | 0.217857 | left_hip_pitch_joint | 0.47124 → 0.461953 | 0.00816661 | 0 |
| v030_continuous_pose_initial_slower_02 | candidate | 1.88 | 0.268799 | right_hip_pitch_joint | 0.521777 → 0.742474 | 0.288687 | 0.0910995 |
| v030_continuous_pose_center | reference | 1.8 | 0.288589 | left_hip_pitch_joint | 0.0485741 → -0.114428 | 0 | 0.0923885 |
| v030_continuous_pose_center | candidate | 1.8 | 0.309507 | left_hip_pitch_joint | 0.00813812 → -0.164944 | 0 | 0.0849118 |
| v030_continuous_pose_combined_right_faster | reference | 1.78 | 0.251128 | right_ankle_pitch_joint | 0.105884 → 0.357013 | 0 | 0.000558212 |
| v030_continuous_pose_combined_right_faster | candidate | 1.78 | 0.360091 | right_hip_pitch_joint | -0.0251065 → 0.176553 | 0 | 0.0957289 |

The earlier coefficient-retention implementation changed whether the absolute target was clamped when its weight became zero. Those acquisitions did not sample the raw expert target. The later whole-operator implementation records base, projected and blended targets and is checked separately against q_blend = q_base + r(q_projected − q_base), including exact r=0/1 endpoints. Downstream residuals and changing policy state still affect the saved final target. Remaining runtime soft bounds stay unavailable (19 joints in legacy acquisitions; 18 in the later acquisition). Original task success is separate from target continuity.

## Methods and provenance

The protected prefix ends strictly before the first feedback decision whose recorded retention is less than one. It is not cut at the earlier trigger timestamp. Every original physical-trace, contact, kinematic and feedback value is compared exactly; added telemetry/debug fields are excluded. Policy timestamps are validated on the original 20 ms grid before selecting the prefix. The primary strike comparison covers only the first contiguous foot-ball force episode, its preimpact fixed-point velocity, and first zero-force outgoing ball state.

Recorded weight pairs are checked against the declared mode: all scales swing, absolute and placement; pose preserves the swing weight and scales absolute/placement. The saved float32 multiplication allows an absolute tolerance of 5×10⁻⁷. The audit also checks nonincreasing retention and a completed velocity pulse whenever retention is below one. It does not infer unseen trigger state.

P0 supplies pelvis-relative path and whole-run joint-limit diagnostics. P1 supplies force components, impulses, foot-velocity jumps, joint overspeed samples, segment motion and the unaccounted momentum residual. Probe certificates are independently recomputed from their stored samples. The release audit does not infer energy transfer, fix momentum residuals, or change any acceptance threshold.

| Source | SHA-256 |
|---|---|
| `tools/audit_landing_release.py` | `a76cdf4518bbda132f2cf4099a749fd6964e49736b9ff848cfc36e10d0b64fa0` |
| `tools/audit_p0_dynamic_strike.py` | `f2dfa4ad64991c2be26275fb231e31ee938730d46b286e95046e82d79d361bcf` |
| `tools/explosive_strike_metrics.py` | `4e80f28607c14c681f82e79f39be4476c5bfd063b57a80701a3dc9b9d9784135` |
| `tools/strike_baseline_metrics.py` | `22a86d021cf358f9f6c3a18b13166fd8dea81b31b535b28b5870ee51146fb1e5` |
| `tools/strike_dynamics_telemetry.py` | `642150d97cabf728053f6d7eb7c4b943bf58f71a2ae876949b37dac186fbbb67` |
| `tools/analyze_explosive_strike.py` | `5bfb7524104845a21e7551044e00a62c31ee424a3d116783a22cedc6d0a9e8bc` |
| `tools/placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| `tools/closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| `reports/v028_final_initial_slower_02.json` | `4fe258c739d0a7739321f4a7e7975d53bd212fbd4193a61fcbbd20ee7033b155` |
| `reports/v029_instrumented_initial_slower_02.json` | `c7f0c9a8b6aecf1a08ac3711dfcdbb23fd0e9a47abda0d1c655bf07d742e2928` |
| `reports/v030_release_all_initial_slower_02.json` | `ae7fd3b22c2238b187998eae93db299421ac917e7a2d43495dea015870521615` |
| `reports/v030_release_pose_initial_slower_02.json` | `2bc1774d05b34b1cee097e8d1b170f2052210a20ef5288efcdec09bd490e2f7d` |
| `reports/v030_continuous_pose_initial_slower_02.json` | `44864278094b276a88339cf14458651abf6fde1cc2311e79574a93e04172926b` |
| `reports/v028_final_center.json` | `1b76423d04bcf3e007e8e248ffece090369b952e928c0f32e0af80cc6a3db37d` |
| `reports/v029_instrumented_center.json` | `42103e7b4a19ff08b02e5c8f5da46090d7631271f5a16200be946dad6bad9976` |
| `reports/v030_continuous_pose_center.json` | `6c6ef1fcc6a208cc955e66d30f24e783659d11d2791ca2e453b7b2583db04d2e` |
| `reports/v028_final_combined_right_faster.json` | `e5e6c936f946fb8b38e5a42e22e475c0eb22e9f3730d7f37d7f82fe2e4180ebe` |
| `reports/v029_instrumented_combined_right_faster.json` | `f29c4ef98ef1b0900048857c775f23e74cc4c737bf27a42fe20df236b3508bab` |
| `reports/v030_continuous_pose_combined_right_faster.json` | `9e4635e908e0c75a77ed171329a2b215d1d34431f8effee108a09c7b8e6d99d0` |
| `reports/v029_force_probe.json` | `d6e585664fa6f1e7939d606db79837c150fe44eb57f7a98f90627ef373cb5a02` |
| `reports/v029_force_probe_cleanexit.json` | `048cc33377338189d0940ad072bf72138f98064b4b3ae68726d79edfd818a316` |

Controller hashes below are the values recorded by each acquisition, rather than claims that the current working files still have those contents. The raw-report hashes above anchor this metadata; the JSON also retains the resolved release configuration.

| Acquisition | Source | Recorded SHA-256 |
|---|---|---|
| v28/initial_slower_02 | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/initial_slower_02 | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/initial_slower_02 | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v030_release_all_initial_slower_02 | `placement_strike.py` | `ae89b72707fa58f21ec4985c564cafd8e502c0c20df54f29d373e8d92b6e53c6` |
| v030_release_all_initial_slower_02 | `closed_loop_expert.py` | `a5f0043ee9cd9cf10675791fa0bbfbd09025d76ab3181c1cb637fd54b2749fe1` |
| v030_release_all_initial_slower_02 | `landing_release.py` | `36ada39739f154b54f578e2b2a9d0b997a53499f3e10ef8795d198c59312a2ae` |
| v030_release_all_initial_slower_02 | Resolved joint-strike configuration | `92bcb8d2ebeec3d5c7ea927eae2dc67f0e657464197fe1bb6f6277fc96c28e7e` |
| v030_release_pose_initial_slower_02 | `placement_strike.py` | `ae89b72707fa58f21ec4985c564cafd8e502c0c20df54f29d373e8d92b6e53c6` |
| v030_release_pose_initial_slower_02 | `closed_loop_expert.py` | `a5f0043ee9cd9cf10675791fa0bbfbd09025d76ab3181c1cb637fd54b2749fe1` |
| v030_release_pose_initial_slower_02 | `landing_release.py` | `36ada39739f154b54f578e2b2a9d0b997a53499f3e10ef8795d198c59312a2ae` |
| v030_release_pose_initial_slower_02 | Resolved joint-strike configuration | `5053860689bcd0f6aae6e112e0b5b593daa9d8c46ed71a92a3202ff4a4caa3e0` |
| v030_continuous_pose_initial_slower_02 | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v030_continuous_pose_initial_slower_02 | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_continuous_pose_initial_slower_02 | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v030_continuous_pose_initial_slower_02 | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |
| v28/center | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/center | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/center | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v030_continuous_pose_center | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v030_continuous_pose_center | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_continuous_pose_center | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v030_continuous_pose_center | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |
| v28/combined_right_faster | `placement_strike.py` | `ceee326708a7415899e08972ea9e9341fe64824046848890bc3e1a8734b7f38e` |
| v28/combined_right_faster | `closed_loop_expert.py` | `e0f62b362a2b3acd6503d455eaa0e93a52423abfd50df81b07bd0a5d952f0710` |
| v28/combined_right_faster | Resolved joint-strike configuration | `523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e` |
| v030_continuous_pose_combined_right_faster | `placement_strike.py` | `f5e3b1696acb875e3a6e728c6fd52bbb7bb66aa27cb6e47ea19fb388378c4813` |
| v030_continuous_pose_combined_right_faster | `closed_loop_expert.py` | `0cf2c0243be3c73f912f5dd1985d12f5db08d1d3f4fbb2c40f58ee2a1e0b617a` |
| v030_continuous_pose_combined_right_faster | `landing_release.py` | `fbf5561597f2602ffadafc30c79bebd3a30439e7cbab66e5bf95852ebca97884` |
| v030_continuous_pose_combined_right_faster | Resolved joint-strike configuration | `d35f66d689e9a0b461d1fb0c5a7bd25eceb67537837bd9efb3c29fecbedd51e4` |

Acquisition source/probe fingerprints, exact prefix counts and first differing keys, signed force-component impulses, every observed overspeed joint, and all per-bin residuals remain in the JSON artifact. No simulation, GPU operation, state write, or model training is performed by this command.

```sh
python3 tools/audit_landing_release.py --candidates reports/v030_release_all_initial_slower_02.json reports/v030_release_pose_initial_slower_02.json reports/v030_continuous_pose_initial_slower_02.json reports/v030_continuous_pose_center.json reports/v030_continuous_pose_combined_right_faster.json --output reports/landing_release_v030_audit.json --markdown reports/2026-11-07-landing-release-audit.md --overwrite
```
