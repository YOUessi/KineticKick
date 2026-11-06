# v0.1.30 measured landing-release experiment

Created on version/v0.1.30 after publishing793ed63. Parent owns the onlyGPU; P1 agent owns independent comparison, P2 agent owns the separate professional-match reconstruction adapter. This plan is declared before behavior changes or candidate simulation.

## Hypothesis and one shared mechanism

The slower initial-state run keeps delayed swing/strike/IK references while its real foot is already descending toward the ground. A shared monotonic C2 retention factor can release these references toward the already running feedback expert, preventing the observed landing overspeed. It is not a new collision body, actuator limit, ball impulse, pose reset or ball-position lookup.

Use exact original capsule endpoints/radii and measured foot world pose, linear/angular velocity and ball position. Confirm at least15ms of true separation: latched original contact, ball normal force<=2N, capsule-ball surface gap>=5mm and increasing. Require original velocity pulse to have ended. Trigger once when an endpoint is descending faster than0.30m/s, lowest capsule height<=0.16m, right ground normal force<40N, and minimum endpoint height+0.16s*min(endpoint vertical velocity,0)<=0.015m. The16cm/160ms geometry lookahead and80ms smooth release are common, declared parameters, not per-case fits.

At trigger retention is1; over80ms use1-(6u^5-15u^4+10u^3), then retain0 for the rest of the episode. Apply it to swing mixture, absolute-strike pose and placement-IK weights. Do not rerun recurrent actors to compute counterfactual outputs. Original phase and velocity pulse, hip/shank guards, running/standing experts and all19 frozen files remain. Record original/effective weights and gate inputs. Check actual command changes; smooth weighting alone is not proof of smooth joint targets.

## Bounded causal experiments

1. First full14s initial_slower_02 with all three weights released, compared to original instrumentedv28. Exact pre-change replay and unchanged first ball output required. Inspect foot geometry, actual joint speed/limit, force components, COM residual, pelvis-relative follow-through, full original task and recovery. Do not accept from one improved peak.
2. One same-condition pose-only ablation (absolute+IK release; swing untouched) using identical gate parameters, to distinguish remaining swing contribution. No parameter sweep. If neither changes the measured risk as predicted, inspect the mechanism and earlier whole-body/stance timing rather than repeatedly accelerating release.
3. If a candidate improves the causal endpoints while preserving task, run center, combined_right_faster and the original repaired right-near-slower case. Freeze parameters before the original18-case acceptance. Existing18 are all development by now; report the original3 as historically prespecified, not newly unseen.
4. Three fresh evaluation conditions declared now: landing_eval_left_near=[-0.012,+0.018],vy+0.165; landing_eval_right_far=[+0.018,-0.018],vy+0.110; landing_eval_faster=[-0.012,-0.022],vy-0.110. Run only after selecting/fixing the shared mechanism, never use these to select gains and call them heldout. Keep original goal misses explicit.

No whole-suite reruns for every candidate. New state/geometry tests, focused existing controller tests and actual causal physics are sufficient initially. Preserve failed runs and snapshots; make a preview only after a candidate has actual physical evidence. P0 and the sustainedP0–P4 goal remainopen until their stated requirements are met.


## First causal evidence and semantic correction

The all-three release preserves initial shot but still overspeeds24.865/20rad/s and falls at4.56s. The prespecified pose-only ablation retains swing, completes14s with9.865s standing and no observed overspeed; fixed-window COM residual falls from10.8815 to9.85e-5Ns. This supports retaining the learned swing mixture. Neither result is hidden or relabeled.

Independent review found an actual new discontinuity atretention0: canceling the original projection branch makes ankle targets jump. The next single pose-only confirmation therefore retains the whole original absolute operator, including its projection: q_out=q_base+r*(q_old_projected-q_base), with exactr=1/r=0 endpoints. This is algebraically identical to weight attenuation where projection is inactive, and smoothly fades projection where it is active. Gate thresholds and80ms timing are unchanged; no new position/velocity clamp or simulator change. Missing ball-force data now invalidate separation rather than count as0N. Original/source snapshots remain underexperiments/landing_v030/first_mechanism.


## v2 control-phase-lag amplitude hypothesis

Whole-operator pose-only release removes observed overspeed in initial_slower_02, but unnecessarily modifies zero-lag center and combined_right_faster. The latter retains5.525m/s but recovery starts3.685s later. This is a recorded regression, not acceptable no-regression evidence. Before new evaluation, condition only the amplitude on existing recorded control-phase lag: lag=max(-phase_shift,0), checked against max((elapsed-frame)*.02,0). Trigger also requires finite lag>1e-6s. At the first existing measured gate, freeze strength=clamp(lag/phase_lag_scale_s,0,1), with separately declared phase_lag_scale_s=.08s. Retention=1-strength*C2(u) over the unchanged80ms duration. This is a falsifiable heuristic scale, not a mechanically proved equality or a sensed human phase.

Expected tests: zero-lag center/strong retain complete original trajectories; .12s delayed slower retains the already verified continuous-pose trajectory; one previously observed small-positive-lag development case probes partial amplitude. All geometry/separation/pulse gates and inherited gains stay fixed. Partial amplitude leaves retention>0 and the original finite absolute window can still remove the remaining projected contribution at its original boundary; inspect it explicitly and do not claim the full command is C2. Preservev1 sources and evidence underexperiments/landing_v030/continuous_pose_v1 and preserve landing_release_study_v1.json. New studyv2 pins revised code/config before fresh3.


The fourv2 representatives completed: slower exactly matches verified continuous-pose physics; center/strong exactly matchv28; intermediate-lag historical failure completes14s and stands9.99s. A finite-observation guard identified in review is fixed before remaining/fresh cases: -inf geometry or ground force must not satisfy a new trigger. Four reproducing CPU tests failed before and pass after the guard. No threshold or model parameter changes; studyv3 records the source-only guard revision and priorv2 sources stay archived.
