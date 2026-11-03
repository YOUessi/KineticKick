# Free-root shooting progress — 2026-11-04

Branch: `version/v0.1.1`. Overall three-surface task is **not complete**.

## Verified baseline and integration

- Existing Root-V2 baseline: genuine contact at control step 161, peak ball speed 1.06753 m/s; substantial sideways displacement and no verified aimed goal.
- Official PAiD recurrent policy integrated with the exact 160-observation contract, verified joint ordering and per-trial LSTM reset. Model remains local, SHA-256 `489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`.
- Physical PAiD G1 remains floating-base; no new foot geometry. Ball is 0.43 kg, radius 0.11 m. Trial code writes poses/velocities only at initialization and prevents reference-wrap teleports.
- Contact evidence uses actual PhysX contact points/normals at every 5 ms substep. The 1.6 m × 1.1 m goal test requires the entire sphere to cross. Standing also checks foot support and rejects non-foot support.

## Exploratory physical results

1. Single official-policy trial at ball (0.14, -0.93) m, goal (0, -3.7) m: approached, inside-foot contact at 2.335 s, peak normal force 244.82 N, peak ball speed 4.949 m/s, goal crossing, and 4.89 s of continuous measured standing by the 8.2 s trial end. Requested instep label was correctly rejected because the measured patch was inside-foot.
2. Twelve-position grid: x={0, 0.14, 0.28}, y={-0.65, -0.8, -0.95, -1.1} m. All 12 approached, contacted with the inside foot, scored and were standing at the 7.2 s endpoint. First-impact points/normals independently passed the audited PAiD inside-foot envelope check. These are exploratory cases under one deterministic reset configuration, not a frozen multi-seed robustness result.
3. Thirteen-reference survey: all ten standard motion references used inside-foot contact. The three stylized clips did not establish usable alternative surfaces. Some standard trajectories destabilized after longer final-frame holds. The policy's success with inside-foot contact does not establish instep or outside-foot control.

Local evidence (ignored): `reports/v0_1_1_baseline_contact.json`, `reports/v0_1_1_official_nominal.json`, `reports/v0_1_1_official_grid.json`, `reports/v0_1_1_reference_sweep.json` and their logs.

## Remaining work

- Develop and verify explicit instep and outside-foot control with actual physical surface evidence.
- Freeze candidates and evaluate the declared 30-trial ball/goal perturbation matrix.
- Exercise repeated live trials through the same evaluation logic and inspect all three modes visually.
- Preserve honest failure accounting and report the supported range.
