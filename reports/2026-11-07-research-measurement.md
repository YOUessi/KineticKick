# v0.1.29 — research measurement and original-human data

This version adds evidence, not a stronger controller. It begins from published v28 `1b3e1998ef99f9ce6280662bc3308ec4b43ebf59`; all 19 frozen files and v28 controller/configuration files are unchanged. No ball impulses, new joint limits, collider edits, state resets between phases, policy training or simulator upgrade occurred.

## Actual execution

Three selected, complete 14 s runs used the original v28 controller with a read-only 5 ms observer. For each case, every original field in all 2800 trace samples, contacts, kinematics, initial state, limits and shot records matches the original report exactly. All three retain original full-task success. This is selected deterministic evidence, not a general success-rate estimate.

| Case | Horizontal ball speed, m/s | Preimpact foot reference speed, m/s | First normal ball impulse, N·s | Maximum actual joint speed / recorded limit |
|---|---:|---:|---:|---:|
| center | 4.631403 | 6.803527 | 2.281680 | 0.908129 |
| initial_slower_02 | 3.922918 | 5.294361 | 1.773413 | 1.902592 |
| combined_right_faster | 5.525366 | 7.167889 | 3.173558 | 0.907301 |

The additional observer records ground-filtered normal and friction forces separately, their sum, robot mass, segment quaternion/angular velocity, and root velocity. Existing torque remains the implicit actuator's clipped-PD estimate; it is not measured pure actuator effort. The isolated yawed 1 kg sliding-body probe establishes that the force matrix is normal-only and that separately aggregated friction uses the positive sign. Integrated momentum error is1.832e-6N·s. This probe establishes the measured API semantics, not universal full-robot momentum completeness.

The first probe completed valid physics but hung during application shutdown; its verified process was terminated after completion. Releasing the tensor observer and SimulationContext callbacks before closing Kit fixed cleanup. `v029_force_probe_cleanexit.json` exited normally with the same physical result. No old evidence was overwritten. Following independent review, certificate loading now recomputes the certificate from samples and rejects inconsistent signs/results; observer-construction failure closes the already-created environment.

## Finding that determines the next behavior experiment

At 1.855 s in `initial_slower_02`, the right knee reaches 38.05184 rad/s against the unchanged 20 rad/s limit. The landing has 4697.75 N normal force and approximately 4356.37 N horizontal friction; the measured foot point reverses from forward motion to a 16.82 m/s world-Y velocity. The original run still eventually stands. It therefore passes the old task criterion but exposes an important dynamical regression.

The impulse itself closes mass-weighted COM momentum. The following 1.860 s bin has an unexplained external-impulse residual of approximately[1.31428,-10.45971,2.69687]N·s. Scaling all joint velocities by 20/38.05184 reproduces this residual within about 0.00076 N·s using the recorded root/COM state and the URDF pelvis inertial origin. This is a strongly supported mechanism inference, not direct verification of the installed binary. A public [PhysX change proposal](https://github.com/NVIDIA-Omniverse/PhysX/pull/534) describes such pre-solve all-DOF scaling; it was open/unmerged when inspected. The original measured forces and residual remain unchanged in the analysis.

The next behavior version will test a shared measured-state release of the swing, absolute-strike and placement-IK envelopes after actual foot-ball separation, pulse completion and observed ground approach. It will check actual overspeed, subsequent momentum residual, follow-through and full recovery together. This version does not claim that correction has already succeeded.

Detailed 40-episode offline baseline plus instrumented analyses: [P1 dynamics](2026-11-07-explosive-strike-baseline.md). Additional follow-through thresholds are diagnostics, not newly invented human naturalness ground truth: [P0 audit](2026-11-07-p0-acceptance-audit.md).

## Human source progress and remaining gap

CMU subject 10 trials 01/02/03/05/06 provide 2761 frames of original surface-marker motion. The pipeline produces 20 landmark proxies, 13 segment directions and 14 cross-part relation cosines, retaining source validity, zero-residual flags and any new interpolation separately. Independent decoding agrees. Original data, downloaded provenance and generated arrays remain local and ignored.

Skill level and instep subtype remain unknown; true strike/contact labels remain null. Candidate phases and four ball-marker-centroid motion onsets are explicitly uncertain, not contact ground truth. This enables real-data pipeline development but does not complete P2. The task-conditioned trained generator, coordination-transfer evidence, matched multiseed physical comparisons and actual blinded ratings are still outstanding.

See [source audit](../docs/research/human-kick-sources.md) and [coordination schema/results](../docs/research/human-kick-coordination.md).

## Reproduction on Tang

```bash
# New output names are required; original evidence is preserved.
/home/you/conda_env/env_isaaclab/bin/python tools/probe_contact_force_semantics.py --output reports/new_force_probe.json
/home/you/conda_env/env_isaaclab/bin/python tools/run_instrumented_strike.py --controller v28 --case initial_slower_02 --probe reports/new_force_probe.json --output reports/new_instrumented_slower.json
/home/you/conda_env/env_isaaclab/bin/python tools/build_human_kick_coordination.py --help
```

Only one GPU job runs at a time. The repository retains code, configuration and textual reports; raw reports, weights, motion assets and media are Tang-local dependencies. The sustained P0–P4 goal remains active.


## Focused verification

Parent integration checks: **81 tests and 5 subtests passed** across the new telemetry, explosive metrics, P0 audit, actual human-data pipeline, original baseline metrics and follow-through tests. All 19 frozen fingerprints remain unchanged. Independent review confirmed exact original trace equality and identified the certificate/initialization cleanup issues fixed above. The full asset-dependent project suite was not run. P0–P4 completion is not inferred from these software checks.
