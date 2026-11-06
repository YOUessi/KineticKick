# Predictive Shank Path Implementation Plan

> Execute inline with superpowers:executing-plans. User has authorized continued research, bounded local GPU experiments, version branch publication and actual preview.

**Goal:** Remove the known precontact two-shank collision while retaining legal instep contact, goal and stable recovery.
**Architecture:** Existing frozen experts and measured placement/phase control plus an isolated original-URDF swept-clearance helper. Short-horizon predictions select a bounded swing-hip/ankle target adjustment; no physical state writes.
**Tech Stack:** Existing Torch, NumPy, original PAiD G1 URDF, IsaacLab/PhysX.
**Spec:** docs/superpowers/specs/2026-11-07-shank-path-design.md

## Global constraints and review focus

- version/v0.1.28 from a0dfb47, existing checkout; preserve main and all previous versions,19 frozen files and existing configurations.
- Exact capsule segment geometry including endpoint/interior/parallel cases; compare independent ReferenceGeometry on actual failed state.
- Minimum over future samples, not final horizon alone; finite gradients near closest-feature switches.
- Original nominal protection, left-support/right-unloaded and precontact gates, bounded targets/rates; no collision filters/acceptance relaxation.
- Independent full physical acceptance; no rewarding a miss or a changed contact surface as success.
- New3 conditions specified before tuning; old failed confirmation now development. One reviewer after implementation, one fix pass if necessary.

### Task 1: Diagnosis and predictive geometry/control
- [x] Read paired-body evidence and compute actual/forecast capsule distances and sensitivities using existing original FK.
- [x] Add tests/test_shank_clearance.py: real recorded states agree with independent geometry; swept path detects an intermediate close pass whose endpoint is clear; inactive/support/nominal and rate bounds.
- [x] RED→GREEN implement tools/shank_clearance.py, integrate tools/placement_strike.py optional control and diagnostic fields, create configs/state_conditioned_strike_v5.json without modifying previous configs.
- [x] Run one targeted physical failed-case replay. If it fails, use actual trace/forces before another trial. Preserve every attempt.

### Task 2: Original acceptance, comparison and preview
- [x] Extend source/config provenance and v27 comparison in tools/run_placement_strike.py/tools/analyze_placement_strike.py; add declared cases to tools/strike_cases.py.
- [x] Targeted CPU checks, fresh focused code review; fix important findings once with appropriate regressions.
- [x] Freeze final model/source; run affected old conditions, then remaining old/new cases serially; actual preview independently accepted. One learned-adapter-off ablation under the final source and physics, with original acceptance unchanged.

### Task 3: Publish
- [x] Record all outcomes, local raw evidence, costs, source fingerprints, limits in reports/2026-11-07-shank-path-adaptation.md, README and VERSION_CONTROL.
- [x] Version commit and non-forced upstream push; show actual preview and outstanding limitations.

## Progress ledger

Pre-flight: geometry/control produce diagnostics consumed by original-criterion audit; recorder/auditor must agree on new source list and gate timestamps. Other original interfaces unchanged. CUDA probe0; no new physical run yet. Existing failed v27 replay already reproduced twice, so do not rerun it just for reproduction.

Task1 CPU geometry/control:14 targeted tests pass. Trial1 (60ms/12mm) fails the same shank collision; correction begins1.48s, actual gap improved only about2mm near closest passage. Existing recorded state at1.46s predicts32.83mm at60ms but10.28mm at80ms. Trial2 changes anticipation to80ms/20mm, retaining0.10rad magnitude and0.04rad/control-step bound; this addresses actuator response delay and positive-gap contact. Preserve trial1 config locally. Do not run a broad matrix unless real contact/goal/recovery pass.

Trial2 separates the shanks (no knee-pair forces), but misses the ball and then collides pelvis/right thigh at1.67s because post-contact braking correctly never activated. This is a different phase failure, not unresolved knee collision. At1.60s the body's x≈−.244m and striking ankle x≈−.300m while ballx=−.105m; the approach is laterally misplaced. Switch from late swing adjustments to an early-controller hypothesis: ablate the inherited negative lateral anchor proportional gain(−1.8338→0) in configv6, leaving all other fitted coefficients and v5 path guard unchanged. This diagnoses whether the feedback amplifies lateral approach error; do not claim cause before replay.

Fresh read-only code review identified two P2 issues; one fix pass completed: independently remember/rate-limit/release actual ankle compensation after clipping, and explicitly label evaluation case membership by candidate version. New tests went RED→GREEN. Actuator bounds take priority over requested rate if their intersection is empty; record pre-target/soft limits/requests and independently audit projection of both channels. Current v28 held-out set is only the three newly declared cases; prior15 are development.

Early lateral-anchor P ablation does not solve approach and reintroduces knee collision; reject it. Trace comparison places lateral deviation before1s and last support foot about19cm left of nominal at1.42s. The stance roll formula is −kp*body_position_error−kd*body_velocity_error, where error=reference−actual, but fittedkd=−1.348 gives an amplifying velocity term. Next single-parameter hypothesis restores damping sign(+1.348) while keeping the original anchor P term and v5 guard; verify actual world motion rather than infer success from sign alone. If it fails, stop gain edits and investigate approach trajectory generation.

Stance damping sign trial(v7) misses and collides hip/pelvis at1.72s; reject. Stop single-gain ablations. New bounded hypothesis: learn an early actuator response to measured lateral capture error(reference+ball shift−root+0.18s velocity error), active only for slower initial running, with30 coefficients,0.2rad target bound and0.04rad/step requested rate. Original phase/IK/guards retained. Train on three already-known development conditions(failed right-near+.15, originalslower+.20, leftslower+.10). Up to96 candidates×3cases, at most300s of optimizer time and6generations; independent new3 remain unseen. Require a full physical contact/goal/recovery replay before broader evaluation.

Approach fit6generations215.95s produced proxy3/3, but N1 replay missed/hip collision1.68s. One33.59s diagnostic batch recorded identical initial joint pose/velocity, root differences<1e−8m, first-step action differences~0.0008, and accumulated joint divergence0.13rad by0.5s. This is sensitive closed-loop batch/runtime numerical behavior, not evidence of a different requested initial condition; exact underlying numeric source remains unconfirmed. Do not increase blind training. Test the diagnostic batch's all3 candidate once in the authoritative N1 runtime. Fresh focused review found feature audit accepted self-consistent stale values; fixed by independent recorder velocity samples plus frozen-reference/physical-position recomputation and negative test.

Exact-runtime selection from already generated candidates (maximum4, stop at first original full success) passed at the first candidate, generation3,44.68s. Actual instep impulse1.927Ns, goal, full14s, finalstanding7.015s. Earlier generation5 missed; diagnostic batch best touched ambiguous surface then collided. Freeze generation3 bytes as finalv5; copy identical accepted raw report into final matrix prefix, retaining SHA/source evidence. Continue remaining development and prespecified evaluation; do not retune on new3.

Final evidence:18/18 complete14s + actual instep + running contact + stable recovery, original full-goal acceptance16/18. Matched old15 improve14→15; newevaluation1/3 with both failures onlymissed_goal. Zero learned coefficients revert repaired case to no-contact physical failure. Actual preview independently passes with4.23m/s/6.10sstanding;20focusedtests pass,19frozenhashes unchanged. All owned GPU jobs finished; no training/deployment job left running. Code commit382e69f published to origin/version/v0.1.28 with upstream tracking; original previews and both held-out goal misses reported.
