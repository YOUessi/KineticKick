# v30 landing matrix: saved evidence

Matrix complete: **false**. Unexecuted conditions remain unexecuted.

| Group | Planned | Executed | Original task pass / fail | Velocity-quality failures | Unexecuted |
|---|---:|---:|---:|---:|---:|
| development | 18 | 6 | 6 / 0 | 1 | 12 |
| fresh_evaluation | 3 | 0 | 0 / 0 | 0 | 3 |

**Declared velocity stop criterion observed:** `heldout_left_near_slower`, `right_knee_joint` at 2.250 s: 21.143692 / 20.0 rad/s, ratio 1.057184601; 1 sample-joint exceedance(s). Its original task result remains reported separately.

Acquisition source cohorts: `{"2": 4, "3": 2}`; these are different source hashes. The v2 saved finite-input checks support the subsequent finite-only trigger guard amendment: **true**. Initial t=0 ground force is unrecorded and trigger-ineligible.

The original 19 frozen files match. Original v28 failures remain historical failures: heldout_v028_near_slow_17, heldout_v028_left_near_slow.

| Case | State / source | Original outcome | Instep match | XY ball speed (m/s) | Standing (s) | Max actual qd / limit | Sampled shank gap (mm) | Δ speed / Δ standing vs v28 |
|---|---|---|---|---:|---:|---:|---:|---|
| center | executed / v2 | pass | True | 4.6314 | 8.965 | 0.908129 | 45.253 | 0.0000 / 0.000 |
| ball_x_minus_5cm | unexecuted | unexecuted | — | — | — | — | — | — |
| ball_x_plus_5cm | unexecuted | unexecuted | — | — | — | — | — | — |
| ball_near_10cm | unexecuted | unexecuted | — | — | — | — | — | — |
| ball_far_10cm | unexecuted | unexecuted | — | — | — | — | — | — |
| initial_slower_02 | executed / v2 | pass | True | 3.9229 | 9.665 | 0.908253 | 31.677 | 0.0000 / -0.495 |
| initial_faster_02 | unexecuted | unexecuted | — | — | — | — | — | — |
| combined_left_slower | executed / v3 | pass | True | 4.4535 | 9.480 | 0.909875 | 36.292 | 0.0000 / 1.940 |
| combined_right_faster | executed / v2 | pass | True | 5.5254 | 10.040 | 0.907301 | 37.513 | 0.0000 / 0.000 |
| heldout_left_near_slower | executed / v3 | pass | True | 4.0753 | 8.795 | 1.057185 | 29.064 | 0.0000 / -0.130 |
| heldout_right_far_slower | unexecuted | unexecuted | — | — | — | — | — | — |
| heldout_left_far_faster | unexecuted | unexecuted | — | — | — | — | — | — |
| heldout_right_near_faster | unexecuted | unexecuted | — | — | — | — | — | — |
| heldout_confirm_left_far_slower | unexecuted | unexecuted | — | — | — | — | — | — |
| heldout_confirm_right_near_slower | executed / v2 | pass | True | 4.2281 | 9.990 | 0.909062 | 13.010 | 0.0000 / 2.975 |
| heldout_v028_near_slow_13 | unexecuted | unexecuted | — | — | — | — | — | — |
| heldout_v028_near_slow_17 | unexecuted | unexecuted; v28: missed_goal | — | — | — | — | — | — |
| heldout_v028_left_near_slow | unexecuted | unexecuted; v28: missed_goal | — | — | — | — | — | — |
| landing_eval_left_near | unexecuted | unexecuted | — | — | — | — | — | — |
| landing_eval_right_far | unexecuted | unexecuted | — | — | — | — | — | — |
| landing_eval_faster | unexecuted | unexecuted | — | — | — | — | — | — |

## Measured follow-through diagnostics

These are additional P0 measurements, not original task gates or human-naturalness standards. Relative motion subtracts pelvis translation; world travel also includes body carry.

| Case | Relative forward advance after separation (m) | Relative displacement at 250 ms (m) | Relative speed at 120 ms (m/s) | Relative reversal (s) | Measured point velocity jump (m/s) |
|---|---:|---:|---:|---:|---:|
| center | 0.0989 | -0.0189 | 0.2281 | 0.1600 | 1.6715 |
| initial_slower_02 | 0.0464 | -0.1622 | -0.3162 | 0.0600 | 3.1995 |
| combined_left_slower | 0.0809 | -0.0504 | -0.1972 | 0.1600 | 2.2212 |
| combined_right_faster | 0.0918 | -0.0371 | -0.0631 | 0.1600 | 3.0779 |
| heldout_left_near_slower | 0.0785 | -0.0621 | -0.0825 | 0.1600 | 6.7063 |
| heldout_confirm_right_near_slower | 0.0792 | -0.0479 | -0.5010 | 0.0800 | 2.3631 |

## Additional measured differences versus the same v28 condition

Signed candidate-minus-v28 differences; these measurements do not define a new task pass or naturalness score.

| Case | Δ max qd ratio | Δ sampled shank gap (mm) | Δ relative forward advance (m) | Δ relative speed at 120 ms (m/s) | Δ measured velocity jump (m/s) |
|---|---:|---:|---:|---:|---:|
| center | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| initial_slower_02 | -0.9943 | 0.0000 | 0.0000 | 0.0000 | -16.0940 |
| combined_left_slower | -0.1142 | 0.0000 | 0.0000 | 0.0000 | 0.0097 |
| combined_right_faster | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heldout_left_near_slower | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.5273 |
| heldout_confirm_right_near_slower | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0460 |

## Scope and provenance

- Unexecuted rows are missing measurements, never successes or failures. Development and prespecified fresh evaluation are counted separately.
- v2 and v3 acquisitions use distinct source hashes. Saved finite-input evidence supports the finite-only guard amendment; it does not make the binaries or source files identical.
- P0/P1 follow-through, force, impulse and pose diagnostics are additional; no diagnostic threshold is promoted to original task acceptance or human naturalness.
- URDF capsule gaps are sampled at recorded joint positions and do not guarantee continuous clearance.
- A force probe verifies sensor semantics; it does not certify complete external robot forces or explain every COM momentum residual.
- Finite condition counts are deterministic observations, not estimated population success rates.

Actual-velocity failures use the declared additional ratio tolerance of 1.0001; task-success rows can still fail that quality check. No matrix-wide no-regression or acceptance-completion claim is made.

Machine-readable evidence retains each raw report path/hash, matched protocol revision, original failure reasons, P0 geometry/path diagnostics, and the full P1 audit.

## Reproduction and verification

This is the stopped six-episode snapshot: the execution queue stopped after the observed `heldout_left_near_slower` velocity-limit exceedance. The later `v030_late_contact_heldout_left_near_slower` observer acquisition is outside this matrix. No fresh evaluation report was read or run for this snapshot.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 tools/audit_landing_matrix.py \
  --candidates \
  reports/v030_lag_pose_center.json \
  reports/v030_lag_pose_initial_slower_02.json \
  reports/v030_lag_pose_combined_right_faster.json \
  reports/v030_lag_pose_heldout_confirm_right_near_slower.json \
  reports/v030_lag_pose_combined_left_slower.json \
  reports/v030_lag_pose_heldout_left_near_slower.json \
  --output /tmp/v030_landing_matrix.json \
  --markdown /tmp/v030_landing_matrix.md --overwrite
```

The preserved local JSON is `reports/landing_matrix_v030_audit.json`, copied byte-for-byte from the reproduction output above. It is local generated output, 14,188,357 bytes, SHA-256 `9fb25d719b58d9b05a8dde518c6e8c93e40633eccf4067ece37e54d157bdd4d6`. The current observer runner may differ from acquisition sources; classification uses each report's saved hashes and its own study pin, never today's runner hash.

Focused CPU verification: **37 passed** across `tests/test_landing_matrix_audit.py`, `tests/test_p0_dynamic_audit.py`, and `tests/test_landing_release_audit.py`. A separate CLI check confirmed `--require-complete` rejects absent planned cases without creating output. No GPU, training, or repeated physical validation was performed by this audit.
