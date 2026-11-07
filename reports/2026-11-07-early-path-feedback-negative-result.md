# v0.1.34 — shared lateral path feedback: candidate rejected

The single fixed candidate did not repair either of two previously observed goal misses. Both still made legal instep contact, completed the original 14 s episode and recovered, but the second also exceeded the original left-ankle pitch speed limit. The candidate is rejected; center, stronger-shot and fresh-condition evaluation stopped. Frozen v32 and default v5 remain available. No physical limit, collider, ball force, termination or acceptance threshold was relaxed.

## What changed and why

The v33 impulse diagnostic measured a foot face close to the goal direction but a roughly 19.5-degree material-point velocity error. v34 tests a shared Cartesian lateral position/velocity feedback from actual foot, ball and goal states. A fixed foot reference point, original kinematic Jacobian, existing context gate and measured ground support produce bounded six-joint references after 0.6 s. The first real robot-ball force is independently latched every 5 ms; the applied correction then fades over 200 ms. Original v32 commands and learned recovery are each evaluated normally, exactly once per command.

The protocol and all gains were fixed before simulation. This is a local motion-control hypothesis, not a selected contact capsule, fitted impulse predictor or learned human coordination generator. See `docs/research/2026-11-07-early-lateral-path-feedback.md` and `configs/early_path_study_v1.json`.

## Actual results

| Case | v32 → v34 outgoing XY speed, m/s | v32 → v34 final standing, s | Legal instep | Goal | Original actual-speed limits |
|---|---:|---:|---|---|---|
| left-near-slow | 4.296257 → 4.451893 | 7.670 → 9.145 | yes | no → no | pass |
| near-slow-17 | 4.737232 → 4.660955 | 9.915 → 8.275 | yes | no → no | **fail** |

On near-slow-17, left ankle pitch reached **37.056236 rad/s at 1.405 s**, against its unchanged **37 rad/s** limit. One 5 ms sample exceeded the limit, maximum ratio **1.001519899**. The small excess is retained as failure; a bounded PD reference does not guarantee bounded realized joint velocity.

Both candidate episodes had no physical termination or illegal body-ball contact, no extra state writes, one right-instep strike and complete recovery coverage. The normal-force separation convention measured outgoing speeds above 95% of each matched baseline. These facts do not override the miss or overspeed failure. The v33 observation that friction anchors can outlast the last normal-force bin remains applicable; these are the original frozen outgoing-speed semantics, not a new claim about the final friction-free ball state.

The new correction first became nonzero at 0.76 s and last remained nonzero at 1.78 s in both conditions. Maximum amplitudes were 0.048094/0.048006 rad; the maximum sampled step was 0.0150000155 rad, within the declared 0.015 rad plus float32 roundoff. No overlap with nonzero v32 post-shot residuals was observed. First-touch relative timestamps and simulator contact timestamps use their explicitly recorded clocks; they are not interchangeable without the initialization offset.

After normal separation, unsmoothed fixed-point forward motion continued for 0.215/0.195 s, with estimated forward displacements of 0.392/0.475 m over the 0.30 s analysis window. This is descriptive follow-through evidence, not a naturalness score or ball work estimate.

At the last 20 ms pre-contact decision, requested lateral translations saturated at 3 cm. The bounded Jacobian increments left approximately 1.45/1.23 cm of nonlinear FK target error. This measures limited requested reach; it is **not** proof that actual foot motion followed the requested translation. A separate measured-velocity comparison is retained with the local diagnostics. The first case's pre-impact fixed-point lateral velocity improved only from about −1.922 to −1.823 m/s. Hip-group kinematic contribution remained dominant; this does not prove that floating-base motion alone caused the miss. The second case's support-ankle excess demonstrates that local swing changes can affect the rest of the physical trajectory.

## Verification and fixed stopping

The disabled-feedback replay exactly matched its v32 reference in **2,800 physical samples, 700 kinematic/action samples, all original feedback, contact events, outcomes and recovery evidence**. It actually dispatched through the new controller, retained source hashes, and every new applied delta was zero. Thus this was a meaningful zero-path comparison, not an accidental call to the old runner.

Fifteen targeted CPU tests passed: lateral sign/frame invariance/input preservation/bounds, first-touch latch, support changes, slew/fade and incremental IK. One implementation review found the missing runner dispatch during development; it was integrated before any simulation. Subsequent runtime review found no additional blocking issue. The full suite was not run. Three physical episodes ran and exited 0: one disabled control and two fixed candidates. **No training, parameter sweep, new render or fresh-case run occurred.** An offline analysis initially supplied the wrong argument type to the existing follow-through function; it was corrected to the existing contact-metrics contract without rerunning physics or changing measurements.

The protocol allowed at most four old conditions after a zero check, but the second candidate triggered its hard stop. Center and combined-right-faster are unexecuted, not assumed passes; both old misses also remained unresolved. No result is represented as an unseen success rate. P0 remains incomplete.

## Evidence and reproducibility

Local generated evidence stays on Tang under `.gitignore`:

- `reports/v034_zero_left_near_slow.json`, SHA256 `969ffcfa121abbec2a6df184869fd4d95c59f6a2e0d4a8195b087ee7865a795d`;
- `reports/v034_zero_parity.json`, SHA256 `b2c1276544fce3aec4a06b2b2b571561cc3a13f973f0b2efe5d44251630e500c`;
- `reports/v034_candidate_left_near_slow.json`, SHA256 `520413861fe5a97fe2cd677194fc573bf9c3aa738bc7184d84d28c35b48a4b37`;
- `reports/v034_candidate_near_slow_17.json`, SHA256 `9a649c098f7ab250c4a4168a1b8e0180d97b0b173b1a0403d8459e2b4f2be6f6`;
- corresponding `*_analysis.json`, `reports/v034_final_result.json`, source copies/logs and analysis script in `experiments/early_path_v034/`.

Pre-run study SHA256: `21d7012c39853a20a66179502378f106fed288e2309bdb603dfb090e9856326f`. It pins both candidate/zero configurations and runtime sources. The v32 recovery checkpoint remains `ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326`.

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_instrumented_strike.py \
  --controller placement --config configs/state_conditioned_strike_v10.json \
  --case heldout_v028_left_near_slow --output reports/NEW_path_case.json
```

Tang-local frozen assets and simulator are required. Existing outputs must not be overwritten. No default controller is upgraded.

Parallel P2/P4 read-only work also recorded newly screened public sources and concrete controller-interface/budget differences. No qualified full-body skilled-human instep dataset was established and no BeyondMimic/SuperMimic soccer baseline was trained. See the version's controller-compatibility and public-source notes; these preparatory findings do not complete P2, P3 or P4.
