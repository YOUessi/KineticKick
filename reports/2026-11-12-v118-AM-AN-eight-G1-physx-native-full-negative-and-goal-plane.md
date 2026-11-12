# KineticKick v0.1.118 — full verified research batch: real G1, failed goal geometry, native contact negative, phase/failure attribution

**Executed 2026-11-12, Tang Ubuntu RTX4090.** Independent branch `version/v0.1.118` sourced from validated v117 SHA `24857ba8e5544dcf72ce904d38b2d5699e6426fb`; all previous worktrees and previous evidence untouched. Original G1 protocol committed and remotely pushed **before** physics at `9aced6b768b2c7e611e98e71118f685ccc3f87af`; local ignored results bound to frozen matrix SHA256 `d035ec1a03cfe6d09ba73e8e85967d34d4d5e84d5aaec805c6a9fbfae99c5dd0`. Further native instrument source committed and pushed before native probe at `d468129343615b9f89c7630ce9d2b29a2bed9b6a`.

## Machine safety and reproducibility

At start Tang online, original root worktree `version/v0.1.64` had unrelated untracked file (left untouched). v117 source clean. GPU had no compute process; disk 31GiB >20GiB minimum. New independent clean v118 worktree and unique outputs used. Historical gitignored C3D, ONNX, third_party and 2345 ignored report references were linked to existing original files without copying/mutating them. Every GPU child ran serially under existing shared nonblocking GPU flock and per-child 140s original G1 / 120s isolated native watchdog; no unrelated GPU process killed. No source was retuned to the observed test shots. Raw per-shot JSON is ~30MB and remains gitignored locally with SHA audit, as requested.

## D. Eight original free-root 29DoF G1 genuine 200Hz PhysX runs

Frozen 2 new ball offsets x 3 original untouched controllers (6 independent conditions), with one extra v32 read-only OFF/ON exact-parity repeat per offset (2 non-independent replicates). All 8 completed actual 14s at 200Hz, **22,400 genuine physics frames**, all original 29 source joint velocity limits checked. Strict PASS requires original goal, instep impulse share >=95%, ball speed proxy >=3.5m/s, all original qdot, nonfoot <20N and observed 14s recovery. This is a robot proxy, not an elite human metric.

| Ball scene (seed) | Original controller | Read-only observer | Real goal | Instep | Ball speed proxy | Worst original qdot / limit | Real 14s recovery | Strict |
|---|---|---|---|---|---|---|---|---|
| AM (3001) | v32 | OFF / ON duplicate | yes / yes | 100% | 4.718638m/s | 0.9083 | yes | PASS / PASS |
| AM | CMU10_02 | OFF | yes | 100% | 4.718213m/s | 0.9083 | yes | PASS |
| AM | CMU10_05 | OFF | yes | 100% | 4.718791m/s | 0.9083 | yes | PASS |
| AN (3049) | v32 | OFF / ON duplicate | no / no | 100% | 5.415615m/s | 0.9080 | yes | FAIL(goal) / FAIL(goal) |
| AN | CMU10_02 | OFF | no | 100% | 5.419221m/s | 0.9080 | yes | FAIL(goal) |
| AN | CMU10_05 | OFF | no | 100% | 5.421816m/s | 0.9080 | yes | FAIL(goal) |

**Independent success 3/6; duplicate-inclusive actual G1 executions success 4/8. AM 3/3, AN 0/3; neither reference makes AN pass.** All current measured sensor net nonfoot contact peaks 0N; lack of native collider identity makes that *not* self-collision exclusion.

### New mechanically verified cause of the AN failure

Second independent SHA verified parser `tools/v118_independent_200hz_audit.py` confirmed every qdot ratio against original gates, full 0.4s precontact, 1.25s postcontact and full14s real observed windows **8/8**. New `v118_goal_plane_evidence.py` interpolates **only between adjacent measured 0.005s 3D ball positions** where the actual original goal plane y=-6.52m was crossed; never creates a future frame or overrides original engine goal truth. Original goal center x=0.7m, width=1.6m (nominal lateral aperture x=-0.1 to 1.5m). Ball center lateral positions on that plane:

* AM: v32 **+0.026695m**, CMU02 +0.027916m, CMU05 +0.027967m — inside original goal lateral width and recorded original goal PASS.
* AN: v32 **-0.222593m**, CMU02 -0.231507m, CMU05 -0.232498m — **ball center outside original left post by 0.1226–0.1325m**, and recorded original goal FAIL.

Thus AN is demonstrably **lateral miss of goal** at the actual simulated goal plane despite legal instep contact, high original ball speed and recovery; the new CMU references marginally increase ball speed while moving the ball center *further left*, not solving accuracy. The original engine, not this geometric interpolation, remains the decisive goal PASS/FAIL evaluator. Figures of SHA-pinned true paths `reports/v118_AM_real_ball_goal_path.png` and `v118_AN_real_ball_goal_path.png` stay local. No cherry-picked success-only protocol.

## A. Actual isolated native PhysX full contact positive/negative test (independent 3 runs)

The native probe source was frozen independently before GPU. Three additional genuine CUDA PhysX **isolated ball/box calibration** runs x180 steps = **540 real 200Hz physics steps**, not G1 shots; original G1/collision configuration untouched:

| Frozen case | True dynamics | Tensor contact | Native actor/collider, contact point/normal/impulse |
|---|---|---|---|
| Positive OFF colliding | collided | OFF | not reported |
| Positive ON colliding | collided | 113 Tensor data counts, peak **667.0803N** | **0 regular callback headers, 0 full callback headers, 0 native poll/full poll headers** |
| Separated ON | separated as intended | 0 contacts | 0 native headers |

The v118 API `subscribe_full_contact_report_events` existed and was subscribed; `get_full_contact_report` was called 180 times on the positive collision, but **0 native headers**, no API error. OFF/ON actual recorded ball z/vz physics **exactly equal**. This is a meaningful instrument-negative control: a real Tensor-positive physical collision does **not** furnish an identifiable native actor/collider manifold on this setup. We cannot prove either that left wrist directly contacts left hip or that it does not. A follow-up must isolate authored USD/body and simulation-backend native buffering requirements with a positive certified collider-pair before modifying any real G1 motor controls. Native admission remains **NOT CERTIFIED**; no fake collider identities are filled in.

## B. Measured 29DoF entire body phase and power evidence

`v118_phase_power_chain_audit.py` separately SHA verifies all eight actual 200Hz raw data, joint velocities and estimated original joint torques, COM/foot support, groups waist, left/right legs and left/right arms; partitions precontact [-0.4,-0.2],[-0.2,0], postcontact [0,+0.2] and [+0.2,+1.25] without extrapolating. A sustained positive motor-power onset (>=3 consecutive 200Hz frames and relative peak threshold) in AM/v32: waist 1.170s, right leg 1.170s, left leg 1.205s, right arm 1.170s, left arm 1.305s. AN/v32: waist/right leg/right arm 1.165s, left leg 1.205s, left arm 1.305s. **Those onset values were identical for all three original controllers within each scene**, which is not evidence of learned human whole-body timing. Reversing the real precontact power order is a *negative-control analysis only*, never a physical rollout or professionally labeled motion. Estimated motor work and support force are robot proxies, **not** human muscle work, GRF, CoP or synchronized elite-ball impact.

All four source-SHA matched CMU vs v32 actual 29DoF qdot comparisons first show physical difference at **1.405s**, 0.160–0.165s before measured contact (AM impact 1.570s, AN impact 1.565s); this late-reference perturbation is not complete natural human approach, support, pelvis, swing, opposite-arm and recovery learned motion. Real qdot peaks stayed below limit in this batch; past v116/v117 real violations remain valid negative evidence.

`v118_plot_failure.py` produces actual measured 200Hz qdot/COM/support plots and true ball-goal XY paths; generated PNGs stay on Tang; plots are not synthetic motion and are never used to claim native collider-level safety.

## C. Whole-body control and human data admission verdict

Original motor controller remains unchanged; **0 newly trained full-body policies, 0 learned-policy motor actuations**. Existing `v117_witnessed_horizon_admission.py` and `v107_motor_admission.py` continue to *refuse* any 29DoF proposed motor plan without verified collision geometry and foot support/slip certification. Full KineticChain-29 phase-coupled torque/whole-body/stance design and honest 4-tier human evidence admission are specified in `docs/V118_HUMAN_FORCE_ELIGIBILITY_AND_KINETICCHAIN29.md`; no new human contact/GRF truth was invented. Available local 6 genuine C3Ds (2 subjects, 3,358 frames, 120Hz) do not certify elite instep contact or analog GRF/CoP. External GroundLink provides promising synchronized motion+forceplate for general support dynamics **but is not yet verified elite instep soccer data**. Avoid downloads into Tang with only ~30GiB free, and never use already observed AM/AN as novel held-out test shots.

## E. Independent code, proof and caveats

Frozen original matrix `configs/v118_frozen_matrix.json`, committed controller `tools/v118_original_g1_batch.py`; frozen native positive/negative `tools/v118_native_full_probe.py`, `tools/v118_native_full_batch.py`; SHA independent scoring `tools/v118_independent_200hz_audit.py`; phase-power `tools/v118_phase_power_chain_audit.py`; paired onset `tools/v118_controller_earliest_effect.py`; true ball crossing `tools/v118_goal_plane_evidence.py`; actual trace figures `tools/v118_plot_failure.py`; dedicated tests.

Local source evidence `reports/v118_original_g1_batch.audit.json`, `v118_native_full_batch.audit.json`, `v118_independent_200hz.audit.json`, `v118_phase_power_chain.audit.json`, `v118_earliest_controller_effect.audit.json`, `v118_goal_plane_evidence.audit.json` and all 11 native/G1 raw JSON/logs are **write-once ignored**, and their hashes must be checked. GitHub contains reproducible scripts/config/test/text reports rather than 30MB JSON or media. Complete project test count and final source SHA are recorded in the verified Git final commit; no unexecuted batch is claimed as real.

**Research conclusion:** This batch falsifies the idea that valid instep impulse plus faster ball necessarily implies useful shot: AN achieves ~5.42m/s but all three controls miss left of the goal. Existing late CMU references remain insufficient to establish learned natural full-body coordination. Native collider path/impulse certification and qualified human force/ball-impact data remain the highest-priority blockers. All negative outcomes retained; no policy promoted.
