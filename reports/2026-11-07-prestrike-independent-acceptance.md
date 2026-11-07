# v0.1.36 — v35 frozen prestrike checkpoint: independent PhysX acceptance

**Outcome: one of two previously observed goal misses was repaired. The candidate did NOT pass the frozen acceptance protocol. No controller promotion.**

Parent: version/v0.1.35, commit 07ddb6ad6e05624f7e69dcacf69e5c5b93c7e31c.
Trial config: configs/state_conditioned_strike_v12.json (v35 source plus final model path and SHA256, tagged evaluation-only).
Actor: experiments/prestrike_v035/ppo_s3501/last_model.pt, SHA256 bf51f9cc9aead37fe1a474a1eed77fd7cf2cc5d6b833087eb616541f4dead9f0.
Post-shot v32 recovery: SHA256 ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326.
Frozen study: configs/prestrike_study_v1.json; all 12 source/config hashes rechecked before evaluation.

## Research validity and protocol

These are independent **post-training physical rollouts of two previously observed development conditions**, not unseen state/generalization tests: both historically named heldout_v028 conditions were included in the 18-case v35 training resets. Use the fixed order: left-near-slow, then near-slow-17, then center and combined-right-faster **only if both misses become valid goals with no physical regression**. The two fresh prespecified conditions were not accessed. Do not rename deterministic development success into an estimated population rate.

Original free-root PAiD G1 29DoF, original physical ball, joints, foot colliders, speed/effort bounds and full-sphere goal acceptance. No new ball impulse or simulator state writes after initialization. Saved simulation has 2,800 physical 5ms samples / 700 control 20ms steps per 14-second episode. Fixed sources are kept byte-identical.

## Observed outcomes (original strict PhysX gates)

| Quantity | Previous miss A: left-near-slow | Previous miss B: near-slow-17 |
|---|---:|---:|
| v32 goal | no | no |
| v36 goal | **yes** | **no** |
| Legal contact | 100% right instep | 100% right instep |
| Contact time | 1.600s | 1.605s |
| v32 -> candidate outgoing XY ball speed | 4.296257 -> **4.313390**m/s | 4.737232 -> **4.757934**m/s |
| Speed retention | 100.399% | 100.437% |
| All-run actual joint-speed excess samples | **0** | **0** |
| Maximum actual joint speed / original limit | 0.909074 | 0.907745 |
| Final continuous stable stance | 9.510s | 9.690s |
| First actual nonzero learned correction | 0.620s | 0.620s |
| Maximum applied learned joint delta | 0.001937rad | 0.002308rad |
| Maximum 20ms delta step | 0.000789rad | 0.000957rad |
| Independent original task | **PASS** | **FAIL (missed_goal)** |

Both cases have full-duration physical data, strict foot-surface classification, real contacts and no nonfoot termination. The new 29-joint residual actually contributes, but its very small amplitudes do not establish high-power natural human coordination or reliable contact-direction control. The first 0.6s root, ball, joint angles and velocities match each v32 baseline **exactly** in all physical samples. Both original scorer outcomes were independently replayed from raw 5ms telemetry and reproduced identically.

## Precise cause of remaining miss

The goal has center (0.7,-6.52)m, width 1.6m, and ball radius 0.11m. The unchanged original goal checker credits only when the **trailing edge** crosses and the **entire sphere** is within the opening; valid ball-center lateral x is [0.01,1.39]m. Interpolated raw physical telemetry:

| Geometry | A, successful | B, missed |
|---|---:|---:|
| Ball center at nominal y=-6.52m | x=+0.175505m | x=-0.032061m |
| Ball center at trailing-edge crossing y=-6.63m | x=**+0.190165m** | x=**-0.027527m** |
| Full-sphere clearance relative to side of opening | **+18.0165cm** | **-3.7527cm** |
| Instantaneous outgoing heading error to target-center | -11.93deg | -16.86deg |

The second shot is strong enough and physically recovered, but is laterally too far left to count as a legal full-sphere goal. The difference is *consistent with* prior v33 foot-point velocity/tangential-impulse direction findings, not proof of one unique cause: ball and initial speed differ across the two examples. No simulated contact-vector direction or learned human kinetics is inferred from foot yaw alone.

## Frozen-gate decision: STOP and archive both outcomes

The new audit tools/audit_prestrike_v036.py independently verifies: (1) source/config/weight SHA256; (2) exact registered case inputs and 14s raw physics; (3) true instep and physical integrity; (4) original task and replayed goal state; (5) all realized qdot against original limits; (6) 95% ball-speed retention; (7) first-600ms identity; (8) actual 29-joint residual amplitude and slew. First case passes every gate. Second fails **goal** and **original_task_outcome** only. 21 focused CPU tests passed; the whole project suite was not run.

Ignored, Tang-local evidence: reports/v036_prestrike_left_near_slow.json, reports/v036_prestrike_near_slow_17.json, both matching .log files, reports/v036_prestrike_independent_audit.json and reports/v036_focused_tests.log.
Final corrected independent audit SHA256: 0d4fe9064f3da8004f0b7ce2467b6737f7d56bf484e987ea3a3bae8a22909ed6.

No additional center/strong/fresh-condition run, retraining, checkpoint selection, hyperparameter sweep or promotional video occurred. Frozen old versions and the default v5 control remain unchanged. The v36 branch publishes only source, the isolated evaluation configuration, tests, and this explicit positive/negative result. Raw data and weights stay local to Tang under .gitignore.

## Next independent research protocol (not executed as an extension of v35)

1. Target the observed signed foot-contact velocity/normal/tangential impulse and full-ball goal margin, rather than increasing swing power or tuning a single known ball position. Retain legal right instep, original power/speed limits, full recovery and the frozen comparisons.
2. Design a **new prespecified** training/evaluation split and compute budget for a shared mechanics-grounded precontact trajectory and support policy; archive all failures and avoid repeatedly spending fresh conditions on tuning.
3. Prioritize qualifying original human professional instep data: true full-body 3D, verified impact phase and surface, foot speed, proximal/distal time course, and admissible use/license. Current CMU markers and MUVY videos do not yet satisfy this standard.
4. Build and ablate a contact-phase human-coordination generator against generic tracking/style priors on matched task functionality. Do not claim P2/P3/P4 complete because a local controller fixes one deterministic miss.

**Conclusion: P0 full gate fails (1/2 old misses repaired); P2 professional instep qualification and P3 human-coordination model remain incomplete.**
