# KineticKick v0.1.144 — six genuine CMU C3D whole-body timing proxies, independent falsification, 12+3 historical original-G1 PhysX checks

Date: 2026-11-13. Parallel independent source branch from **clean v142 SHA** `a0fe4eb41ef026d89e233f4443621a7008b79bf5`, because v143 contains an active untracked implementation and was **not touched**. Frozen preimplementation v144 protocol: **`4c220f2967fa6d9440e5159f4e4b6d30b911f537`**; see `docs/V144_FROZEN_HUMAN_TEMPORAL_CHAIN_AND_PHYSICS_PROVENANCE.md` and `configs/v144_frozen_temporal_human_and_physical_audit.json`.

**Scope/boundary:** The user's paper goal remains naturally coordinated original29 G1 instep striking learned from genuinely excellent footballers and validated in 200Hz rigid-body physics. Today's work is original human-marker observation and frozen *historical* physics negative audit only. **New real PhysX episodes: 0. New policies trained: 0. Qualified professional instep impulse/GRF labels: 0. True native wrist-hip PhysX pair/manifold: UNKNOWN.** Neither synthetic QP admission nor marker velocity is physical kicking success.

## 1. Actual protected device and source inspection

Tang online via authorized Remote Desktop Commander, `/home/you/projects/kinetickick`. Root is still dirty on `version/v0.1.64` (untracked `=100===`); v143 is its own active untracked worktree, no edits were made to either. `df -h` shows only about **30GiB free**, above the hard stop of 20GiB but 97% occupied. Original `v120` pytest PID **242644** has held RTX4090 context (~260MiB) for >21 hours, so no new Isaac Sim or CUDA competing work, no kill, no original source deletions, no weights/CMU/PhysX overwrites. New independent `/home/you/projects/kinetickick-wt-v144` from locked v142 commit; branch `version/v0.1.144` created only after checking local and remote branch name unused. All output created under ignored local `reports/` with write-once semantics.

## 2. Real source marker temporal dynamics (NOT impact labels)

Actual SHA256 verified for **all 6 original C3D and matching 6 AMC files**, plus v50 quality audit and original manifest. Real CMU original marker frames total **3,358 at 120Hz**, across subjects **10 and 11**. Sources are original 3D mocap, not 2D extracted poses. No new ground truth was generated. Original C3D coordinate units are **mm**, converted to **meters** with per-marker finite/zero missingness masks; C3D has 333 raw marker slots but only 255 labeled markers. The first implementation incorrectly required all slots labeled, and the original C3D includes nonfinite tracking dropouts; both were caught by running on real data and fixed without interpolating. This was a **source handling** fix, not result tuning.

Predeclared 12 real 3D body-segment **direction vectors**, including pelvis, trunk, bilateral thigh/shank/feet and upper arms/forearms. The unsigned rate proxy is the angle between normalized vectors at t−2 and t+2 divided by 4/120s. These are **marker segment orientation-change proxies**, not anatomical joint velocities/torques/muscle power. All clips were retained in the denominator; only the 2 already-prequalified strong *uncalibrated ball triad marker-motion* onsets, **10_02** (frame 353, 2.941667s) and **10_05** (frame 303, 2.525s), entered a 0.6s pre-onset rate-peaking descriptor. The other 4 clips had **no qualified event anchor**, including all subject 11 clips.

| Segment (real observed marker direction rate) | 10_02 peak lead before ball marker onset | 10_05 peak lead | 10_02 / 10_05 peak unsigned rate (rad/s) |
|---|---:|---:|---|
| Pelvis left-to-right hip vector | 0.1333s | 0.1417s | 3.672 / 4.434 |
| Right thigh hip-to-knee | 0.0833s | 0.0917s | 9.610 / 9.838 |
| Right shank knee-to-ankle | 0.0250s | 0.0250s | 20.651 / 19.742 |
| Right foot heel-to-toe | 0.0917s | 0.0167s | 13.187 / 12.663 |
| Trunk left hip-to-sternum | 0.1083s | 0.2333s | 2.458 / 2.253 |

**Descriptive finding:** pelvis→right thigh→right shank peak lead ordering holds in **2/2 selected real marker sequences**, each pre-onset window has 72 directly valid samples for these segments. It is consistent with a possible proximal-to-distal segment phase pattern, **NOT evidence of causal force transfer, correct instep contact, true event time or elite technique**. Right-foot and trunk ordering differs across sequences, so a single hand-authored global chain is not justified. Both selected sequences come from **subject 10**. A true subject-disjoint event-model training/test validation **cannot** be run with these labels.

Independent reconstruction computes angle via `atan2(norm(cross),dot)` rather than primary `arccos(dot)`; across observed 120Hz segment-rate frames, maximum difference **3.8175e-10 rad/s**. Predeclared reverse-window counterfactual reverses the selected ordering in **both 2/2** (not plausible human movement, only a time-order sanity check). An additional *exploratory, non-preregistered* 4,096-draw deterministic independent circular-phase-shift diagnostic produced both-sequence order frequency **104/4096=2.539%**; **this is NOT a valid population p-value, independent subject test, or proof of biomechanics** and must not enter claims of learned power.

C3D files expose no qualified true right-instep impact, synchronization to force plate/CoP, professional athlete certification, exact ball contact, or calibrated ball marker cluster center. Original CMU download manifest reports research permission but TLS issuer verification failed; local SHA256 cannot substitute an independently signed upstream provenance. Any G1 torque optimization against these would remain weak marker-pose prior only, not full power-chain learning.

## 3. Recheck real original G1 physics without new GPU execution

Reauthenticated the **actual 12 original v53 G1 PhysX rollout JSON sources by their locked SHA256**, reconstructed all four frozen ball conditions × three policies from immutable original v53 strict audit and required all 12 original strict verdicts to match the gate conjunction. Also reran v142's SHA-locked independent historical v54 source requalification: **3 original episodes / 6,628 actual 5ms physical states**. Neither work constitutes new physics. The 12 and 3 are different original archival experiment cohorts, not 15 newly executed shots.

| Frozen original v53 controller | Original physical conditions SHA-verified | Strict success | Instep legality rejected | Other important failures |
|---|---:|---:|---:|---|
| frozen_v32 | 4 | **0/4** | 4/4 | goal 2/4, ball speed 3/4, early stop/recovery 1/4 |
| CMU10_02 | 4 | **0/4** | 4/4 | ball speed 3/4, original 29 qdot overspeed 1/4 |
| CMU10_05 | 4 | **0/4** | 4/4 | ball speed 3/4, early stop/recovery 1/4 |

Failure counts within a row **overlap** and cannot be added as distinct failed trials. In v54, last authentic failed `cmu_10_05` 29DoF step remains at **5.140s**, named contact event at **5.145s**, event nonfoot net force at **5.135s**. No same-step 5.145 q/dq; no postevent 1.25s or 14s recovery record. **Original suspected pair remains left_wrist_yaw_link vs left_hip_roll_link**; these were named sensor candidates, not certified native actor/collider pair. **Direct native manifold remains UNKNOWN**, and mere equal/opposite contact force does not prove self collision.

Original strict gates preserve instep impulse>=95%, valid goal, outgoing ball >=3.5m/s, entire 29 qdot, no nonfoot contact and full recovery; nothing was promoted by goal alone.

## 4. Implementation, independent tests, negative controls, figures

- `tools/v144_human_temporal_source_and_physical_audit.py` — frozen SHA source provenance, original 6×C3D/AMC parsing, 12 markers segments, missingness-guarded real 120Hz angle rates, exactly 0.6s real pre-onset weak-anchor analysis, and immutable 12 original PhysX SHA checks plus original v54 requalification; output `open('x')`, validate all JSON before file creation.
- `tools/v144_independent_temporal_falsification.py` — independent atan2 full-segment numerical review, reverse-time negative and exploratory phase-shift sanity controls, original 12 trial gate validation, reject fabricated force/impact, subject-independent generalization and fake new PhysX.
- `tools/v144_plot_real_human_and_physx_failures.py` — two source-faithful local figures: real pelvis/thigh/shank/trunk/feet observed time leads and the original physical strict task-gate failure matrix.
- `tests/test_v144_human_temporal_sources.py` and `tests/test_v144_independent_human_and_physics.py` — **40 passed** targeted tests covering actual six clips, missing markers, synthetic known angular rates, 120Hz temporal windows, disallowed event reclassification, false native/training/force evidence, 12 original SHA traces, document tampering, overwrite prevention. The first extended output was replaced by an actual full rate-array independent audit; a too-tight 1e-10 floating-point check failed on a near-zero arccos sample (3.8e-10 difference) and was corrected to independently justified 1e-8 numerical tolerance. Failure log retained locally.
- Complete selected cross-version CPU regression v124–v144: **788 passed / 2 skipped / 0 failed**, 2 existing SciPy SLSQP bound-clipping warnings. Not a full GPU/PhysX integration test.

Ignored local evidence outputs include: `reports/v144_six_real_CMU_12_v53_3_v54_first_pass.json`, `reports/v144_independent_full_allframe_atan2_second_audit.json`, two PNG figures, first failed / final regression logs. Raw C3D, AMC, NPZ, large physics JSON, weights, images **were not committed**.

## 5. Follow-on dependency, without exaggeration

A: v143 is developing additive poststep native callback integration; v144 deliberately does not rewrite it. Still need positive/negative **real isolated PhysX** contact control, runtime USD actor/collider identity truth, and OFF/ON original 3 policies × 2 = 6 *new actual G1 physical* paired replays, plus exact physics invariance. Don't infer pair from named net sensors.

B: Work towards a physically observed true 200Hz original29, COM, feet GRF/CoP, slip, arm-hip clearance and measured recovery state stream. Historical nonfoot named contact alone cannot fill missing states.

C: Before dynamics training, source additional legally licensed **professional** high-fidelity human fullbody kick mocap, independently annotated real contact side/surface/time, synchronized force plate and ball kinematics, with multiple subjects and subject-heldout split. Freeze phase-aligned multi-body synergy/retargeting and physical 29DoF torque/stance constraints **before** unseen experiments. v144's timing order is a weak testable *hypothesis*, not learned policy or torque interface.

D/E: Preserve v53 negative failures and strict all-gate pass; once GPU safely free, independently calibrate and evaluate physically. Original GPU PID is deliberately protected. With 30GiB free and a 20GiB hard floor do not launch memory-hungry experiments until availability and storage verified.

**New physical attempts 0; new trained policies 0; actual high-speed natural instep physical success unknown; real 6 human clips measured; 12 v53 rollouts SHA verified + v54 6,628 authentic states requalified.**
