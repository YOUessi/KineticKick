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


Live update — v30 ongoing, no GPU currently active:
- v29 measurement release published793ed63; createdversion/v0.1.30 before behavior changes.
- Parent completed five causal runs. All-release falls4.56s and is rejected. Pose-only succeeds; whole-operator retention fixes introduced projection discontinuity. Slower no longer exceeds recorded speed limits, but unconditional release delays stronger-case recovery3.685s.
- Next bounded hypothesis freezes release amplitude from already recorded control-phase lag; zero lag stays identical. 18 reused cases remain development, fresh3 declared before tuning are not yet run. All19 frozen assets remain required.
- P1 independent landing audit preserves first five outcomes; agent extends only audit code forv2. P0 agent selects one existing small-positive-lag diagnostic.
- P2 professional SkillCorner full one-match scan:153252frames,24non-header windows,1817shooterframes. No continuous full coordination passes unchanged uncertainty screen.629275569B within750MB ceiling; no second match. P2–P4 remain incomplete.


V30 phase-lag representatives checkpoint:
- Matrix session2369 exited0 after initial_slower_02/center/combined_right_faster. Session18656 exited0 after selectedpartial heldout_confirm_right_near_slower. No GPU active at this checkpoint.
- Slower complete original physics identical to continuous-posev1; center/strong complete original physics identical tov28, eliminating3.685s strong recovery regression. Intermediate-lag case retains4.228121m/s and stands9.99s (old7.015). All four original tasks pass with no observed joint overspeed.
- Studyv3 fixes a demonstrated nonfinite-trigger edge case only, no numeric parameter change. Prior four finite-input traces remain valid evidence with their original source hashes.12 helper tests pass. P1 independently auditingpartial projection exit before remaining14 reused cases andfresh3.


V30 matrix failure and changed diagnostic route:
- Session15971 exited2 after two of remaining14: combined_left_slower passed/nooverspeed; heldout_left_near_slower passes originaltask but rightknee2.25s21.143692/20rad/s, vsoriginal20.988663. Current6executed/15unexecuted; nofresh3read/run. Matrix stopped, no tuning onnewcase or pretending full acceptance.
- The failure is the second right-foot post-shot touchdown2.245s, with allstrike/swing/IK weights andvelocitypulse0. It is outside the solved first-landing retained-pose mechanism. Existing late anomaly slightly worsens; blanket release and single-joint schedules are not a complete solution.
- One new read-only ground-contact observer replay session66181 exited0. Reportv030_late_contact_heldout_left_near_slower.json contains17bins2.20–2.28s of independentnormal/frictionanchor data andpre/postlinkorigin footvelocities. All2800oldtracefields,records,kinematics,sequence exactly unchanged. Existing generic comparator mistook added ground_contact_detail aslegacy; direct old-key equality confirms no physical changes.
- P1 analyzes actual contact leverage/approach velocities; P0 independently compares actual support/body state to fixed world running references to inform an earlier whole-body reference handoff. No GPU currently active. No new behavior parameters chosen yet.
- All originalstudy_v3 runtime sources/config archived underexperiments/landing_v030/lag_v3_sources. Optionalnewobserver wrapper is a later source change, not retroactively assigned to the6matrix acquisitions.
- P2 now has175frames25Hz KTH professional-match multiview joint reconstruction, original14joints+3camera annotations; units arbitrary, instep/contact/independent3Daccuracyunknown. Adapter5tests passed. Bounded CPU retriangulation and<=7sourceimage inspection underway; no training orpaid/APIcost.
