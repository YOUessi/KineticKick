# KineticKick P0–P4 Implementation Plan

> User authorized continuous execution and parallel agents. Use superpowers:executing-plans inline and delegate independent bounded tasks under dispatching-parallel-agents; no repeat approval pause.

**Goal:** Complete the requested scientific route from audited physical strong instep baseline to demonstrated task-conditioned human coordination transfer and paper-level comparisons.
**Architecture:** Frozen physics/controller baselines, independent event-aligned measurements, provenance-aware human coordination data, continuous conditional generation and matched physical evaluation.
**Tech Stack:** Existing Python/NumPy/Torch/IsaacLab/PhysX; source formats verified before choosing parsers.
**Spec:** docs/superpowers/specs/2026-11-07-thesis-roadmap-design.md

## Global constraints
- Preserve all19 frozen files, original geometry/limits, free root and real ball contact. Physical writes only at episode initialization.
- No goals/naturalness/elite-human claims from proxies. Keep goal misses and numerical sensitivity visible.
- Independent version branches; local assets/media ignored; no force-push or main changes.
- No simultaneous GPU tasks. Costly searches need explicit bounded hypotheses and recorded actual use.
- Existing v28 new3 cases have been observed; future tuning on them makes them development, requiring fresh evaluation cases.

## Review focus
- Contact normal-only force versus total normal+friction GRF; direction and timing.
- Implicit actuator estimated PD effort versus measured generalized reaction; no unsupported energy-transfer claims.
- Pelvis-relative leg swing versus whole-body transport; discrete samples versus continuous guarantees.
- Human skill/instep/3D provenance, missing markers, coordinate units, event confidence and licensing.
- Unseen data leakage, unequal budgets, invalid trials and fabricated or unblinded ratings.

### Task A: P0 independent acceptance audit (agent p0_final_audit)
Own tools/audit_p0_dynamic_strike.py, tests/test_p0_dynamic_audit.py, report2026-11-07-p0-acceptance-audit.md; generatedp0_v028_acceptance.json.
- [x] Original18 physical conditions plus v27 regression and follow-through audit; distinguish existing thresholds from extra diagnostics.
- [x] Focused tests and actual frozen-trace output; no new simulations or controller changes.

### Task B: P1 dynamics metrics (agent p1_dynamics_metrics)
Own tools/explosive_strike_metrics.py, tools/analyze_explosive_strike.py, tests/test_explosive_strike_metrics.py, report2026-11-07-explosive-strike-baseline.md.
- [x] Signed event-aligned power/velocity timing, support braking proxies or qualified measured GRF, pelvis/thorax/arm relations and follow-through.
- [x] Analytic tests for windows/signs/missing data; actual v15/v27/v28 report with limitations.

### Task C: P1 missing telemetry (parent)
Own tools/strike_dynamics_telemetry.py, tools/probe_contact_force_semantics.py, tools/run_instrumented_strike.py, tests/test_strike_dynamics_telemetry.py.
- [x] Isolated sliding-body probe establishes force-matrix/normal/friction semantics and sign by momentum balance; no robot/ball control mutation.
- [x] Optional observer wraps unchanged original runners, records5ms support normal/friction/total forces, segment quaternion/angular velocity, mass and estimated drive metadata. First real abnormal-landing case has2800/2800 rows and exactly unchanged original records/kinematics; other selected cases pending.
- [x] One nominal and two informative strong/adapted conditions, then independent metric aggregation; do not rerun full18 merely to add absent telemetry.

### Task D: P2 original-human source/data (agent p2_human_sources)
Own docs/research/human-kick-sources.md, configs/human_kick_sources_v1.json, optionalinspect_human_kick_assets.py and ignoreddata/research_human_kicks.
- [x] Source/license/skill/instep/3D catalog; small permitted original sample acquisition with hashes and quality inspection.
- [ ] Search for genuinely skilled/elite evidence; unknown qualification stays unknown. No author messages or spending.
- [ ] Following source selection: implement event annotations and relational features with observed/inferred confidence; obtain adequate human data before claiming P2 complete.

### Task E: P3 generator
- [ ] Freeze input/output/conditioning contract and coordination feature schema from actual human data.
- [ ] Implement and train continuous whole-body conditional generation under common tracking backend; use limited smoke runs before substantial compute.
- [ ] Validate novel task responses, coordination transfer, physical execution and human-reference contribution; do not count scaffolding as complete.

### Task F: P4 experimental protocol and execution
- [ ] Prespecify splits, task distributions, seeds, budget accounting and metrics; ordinary tracking/style-prior/coordination methods share robot and data.
- [ ] Execute independent multiseed matched experiments and mechanism ablations.
- [ ] Obtain real blinded naturalness ratings; never substitute simulated raters or metrics for people.

## Live ledger

Start: active sustained P0–P4 goal. v28 published1b3e199, clean; createdv29 before edits. Three independent agents launched with disjoint ownership; parent owns allGPU. No activeGPU run yet. User has no private motion and instructs continuing public-source search. Existing follow-through extra diagnostic suggests limited pelvis-relative motion; not yet a changed acceptance criterion. P2 originalCMU source found but skill/instep unknown. Current focus is completed A–D concrete outputs and physically validated measurement; later research stages remain open.


Update — measurement execution:
- Parent owns the only GPU process. `v029_instrumented_initial_slower_02.json` completed cleanly: original task passes,2800 physics bins, originalrecords andkinematics exactly equal tov28. New normal+friction observer makes no physical writes.
- Force probe proves `get_contact_force_matrix` is normal-only; add separately aggregated `get_friction_data` with positive sign. Isolated momentum error1.832e-6Ns. Initial probe finished validphysics but hung atKit shutdown; only its verified PID62094 was terminated. Callback/instance cleanup was fixed; `v029_force_probe_cleanexit.json` then completed normally with identical physical result. Old evidence remains preserved.
- P0 audit reveals dynamics regressions despite15/15 old-case task success: initial_slower_02 has right-knee38.05rad/s against recorded20rad/s limit;1.855s right landing normal4697.75N and friction[159.84,4353.44,0]N. Neither configured limits nor collision geometry were changed. Original task gates do not establish naturalness or acceptable impacts.
- The1.855s impact bin closes COM momentum, but the next1.860s bin has a substantial unexplained residual. P1 will retain this limitation; the isolated probe does not certify a complete robot momentum budget. Do not attribute this to a specific solver mechanism without evidence.
- Earlier read-onlyFK identifies greater toe-down orientation and residual strike envelopes at similar foot height; it does not prove causality. Next behavior version will test one measured-state, smooth, shared release of swing/absolute-strike/placement-IK after real contact separation and pulse completion. No ball-position table or physical parameter change.
- `v029_instrumented_center` is the current GPU run; after it exits, run only combined_right_faster for informative high-speed measurement. Do not overlap jobs or rerun full18 merely for telemetry.
- P2 agent acquired CMU10 trials01/02/03/05/06 (04walking excluded); quality/independent decoder validation and relational features ongoing. Skill and instep qualification remain unknown. P2–P4 completion claims remain open.


Measurement release checkpoint:
- All GPU jobs are closed. Three14s observer runs passed their original tasks and8400 original trace rows/contacts/kinematics are exactly unchanged fromv28. Source snapshots retained underignored reports/v029_instrumentation_source. Later collector changes only recompute saved certificates and clean up initialization failures; force computation/control unchanged.
- Parent integration verification:81 targeted tests and5subtests passed in1.18s. A first command named a nonexistent legacy test file and ran no tests; corrected to tests/test_strike_baseline.py. Full project suite not run. All19 frozen fingerprints verified unchanged.
- P1 report and CLI include40 historical episodes, three instrumented runs, exact provenance and the single-bin residual. The all-joint-velocity scaling hypothesis predicts the abnormal residual within0.000756Ns, but local binary behavior remains an inference. Do not add the inferred correction to measured forces.
- P2 CMU pipeline complete:five realclips2761frames,10 tests, original/source-zero-residual/new-fill masks, candidateevents only. Skilled/professional instepqualification remainsfalse. A new official SkillCorner professional-match continuous pose/event sample is being investigated as reconstructed3D evidence; it is not yet part of this release.
- P0 full completion remainsopen because dynamic landing regression andtwo new-case goal misses are not solved. P2–P4 research requirements remainopen. Next is a new version branch for the measured-state landing-release experiment; do not stop the sustained goal after publishing this measurement release.
