# KineticKick v0.1.56 — Original G1 200 Hz PhysX, effective seeded trials, and contact-reporter backend boundary

**Date:** 2026-11-11 (Asia/Singapore)
**Base:** `version/v0.1.55` at `052925e14031917f5bfad33705116d2429fecfab`
**Branch:** `version/v0.1.56` (created on clean tree before v56 source creation)
**Frozen pre-evaluation config SHA-256:** `4778dd8e93250b4515adbf75ac2086c55389737eb476f7cf01e758109d73b1bf`
**Instrumentation:** original free-root 29DoF G1; original 200 Hz / 0.005 s PhysX; 50 Hz policy; one IsaacLab GPU simulation at a time.
**Claim level:** physical negative and positive outcome analysis; **NOT** professional human kicking skill, a certified self-collision pair, a learned coordination model, or hardware safety.

## Preflight and provenance

Tang was reachable and had 45 GiB disk free, above the absolute 20 GiB stop threshold. Its RTX 4090 Laptop GPU was idle, with no foreign GPU compute jobs. The base v55 Git commit was clean before checkout to the previously unused local `version/v0.1.56`. Do not rebase/amend/push-force, overwrite historical physics JSON, or upload local motion archives, media or model parameters.

The pre-execution protocol enumerated 15 uniquely named reports; source SHA-256 checks lock the unchanged original controller, three original policy configurations, and new simulation/reader entrypoints. The runner checks free disk, GPU process occupancy, a filesystem lock, sources and existing reports; output reports/logs are write-once, and progress preserves hashes.

Local original inputs/outputs **intentionally not in Git**: `reports/v056_*.json` and `*.log` (including the 15 raw physical episodes, frozen progress and complete analysis), source motion data / original weights, and generated `media/v056_*.png`. Git tracks code, protocol, tests and this report. See `tools/run_frozen_batch_v056.py`; `configs/physx_contact_safety_v056.json`.

## I. Fifteen original G1/PhysX episodes completed

Exactly **12 new seeded physical evaluations** = 2 new ball scenarios (C/D) × 2 distinct seeds (2111/2333) × 3 frozen policies (v32, CMU 10_02, CMU 10_05). The remaining **3** are unchanged center-ball observer parity replays (one per policy). The random seed now changes the **initial robot state and original measured 200-Hz trace** for all six same-policy/same-scenario seed comparisons. This corrects the v55 effective-seed failure. New scenarios are *exploratory*, not a pristine unseen holdout.

All 15 simulations produced admissible raw free-root G1 physical records and independent strict audit; **15/15 physical evaluations**, not 15/15 task successes. No controller fitting occurred on the newly inspected failures.

| Exploratory scene / seed | Frozen v32 | CMU 10_02 | CMU 10_05 | Main observed failure |
|---|---|---|---|---|
| C / 2111 | FAIL | FAIL | FAIL | Three misses; v32 additionally exceeds an original joint-velocity limit (max ratio 1.021). |
| C / 2333 | PASS | PASS | PASS | All three satisfy the original multi-gate contract, under this *one easier condition*. |
| D / 2111 | FAIL | FAIL | FAIL | All three non-foot stops; CMU policies stop at 1.56 s, **before foot-ball contact**. v32 touches ball but later terminates at 3.72 s. |
| D / 2333 | FAIL | FAIL | FAIL | All three stop around 1.68 s after poorly directed/non-legal contact around 1.615 s; ball speed ~1.83–1.84 m/s and instep fraction 0. |

Strict exploratory **success per policy: 1/4 = 25%**, pooled **3/12 = 25%**, but pooling cannot establish learning benefit: all three policies pass the same C/2333 scene, and all three fail the other three. In particular, the 1/4 CMU success is *not* evidence that the weak human proxy improved over frozen v32. The result also does not prove natural human coordination.

The three center comparisons reproduce v55: v32 PASS, CMU 10_02 PASS, CMU 10_05 FAIL on `physical_nonfoot` at ~5.14 s, despite its ball crossing the goal. All nine audited base physical fields (`traces`, `contacts`, `kinematics`, `records`, `sequence_records`, `running_evidence`, `stride_shot_evidence`, `stride_shot_success`, `termination_reasons`) are **exactly equal** to the original three v55 runs, so the read-only callback did not perturb the shared measured physics fields. This does *not* verify that the callback is producing contacts.

Failure gates **cannot be substituted**: legal right instep impulse >= 95%, entire ball inside goal, ball speed >= 3.5 m/s, no actual 29DoF qdot above original limits, genuine run-up, termination-free 14 s and post-shot stable recovery. A shot crossing the goal is not full-task success.

## II. 200-Hz coupled dynamics and censoring

The frozen 15 G1 runs contain actual 29-joint velocity telemetry and two-foot support normal-force observations. `tools/audit_real_batch_v056.py` uses unchanged v53 strict scoring and v55 measured pair-dynamics auditing. `tools/analyze_measured_support_phase_v056.py` calculates five body-group activation envelopes, timed group correlations, stance-force fractions and pre/post impact completeness without fitting any policy.

Across 15 episodes, 13/15 have a complete observed pre-contact 0.4 s window and 10/15 have a complete post-contact >=1.25 s window. The two D/2111 CMU crashes have **no recorded foot-ball contact**, therefore no contact-centered 0.4/1.25 s windows may be manufactured. Three D/2333 episodes stop too early to show the full follow-through/recovery. The initial 14-second complete paths are not proof of *human* ground reaction or human timing.

`media/v056_measured_support_unseenC_s2111_frozen_v32.png` and `media/v056_measured_support_unseenD_s2333_frozen_v32.png` were genuinely generated from original physical forces, remain on Tang per ignore rules. Their creation is reproducible from the tracked analysis script and excluded raw source reports.

Important physical observability limits: link-origin separation is **not** collider-surface clearance, net joint/body force is **not** a uniquely attributed contact partner, foot sensor normals are **not** human force-plate GRF, and calculated temporal activation correlation is **not** human causal coordination or professional force transfer. Center-of-pressure and true ground support polygon confidence require separate genuine observation; avoid invented truth.

## III. Native PhysX actor/collider contact reporting: critical backend qualification

The three new center G1 callback replays have `PhysxContactReportAPI` **already present** on the left wrist, left hip, pelvis and right hip, but all three return **zero callbacks and zero native headers**. `rigid_pair_certified=false` for the hypothesized left-wrist/left-hip contact. There is still no directly validated G1 actor/collider paths or manifold. A zero report is **UNKNOWN**, not evidence that the bodies did not touch. The earlier 25.890153 N equal/opposite net-force coincidence is a candidate, not certified self-collision.

To isolate **whether the reporting API itself is observable**, four independent 200-Hz disposable **two-box** toy tests were recorded (two CUDA PhysX + two CPU PhysX, positive overlap/fall vs separated negative), without editing G1.

| Isolated PhysX toy | Filtered PhysX force (N, peak) | Callback headers | Native actor + collider paths + points |
|---|---:|---:|---|
| CUDA positive (actual box contact) | 626.339 | 0 | **NOT observed** |
| CUDA separated negative | 0 | 0 | Not observed |
| CPU positive (actual box contact) | 627.058 | 239 | **Observed; 4 point samples in first header** |
| CPU separated negative | 0 | 0 | Not observed |

CPU toy positive returned, for example, `/World/Probe/B`, `/World/Probe/A`, and collider paths `/World/Probe/B/geometry/mesh`, `/World/Probe/A/geometry/mesh`; it contained numerical position, normal, impulse and separation information. **This authentic manifold is for toy boxes only**, not G1, not wrist/hip. It demonstrates that the reporting mechanism can work under the local CPU configuration while the local GPU configuration reports no native contact headers **despite positive filtered collision force**. It does not establish that GPU PhysX *never* supports contact reports; engine/config/backend diagnosis remains unresolved. Importantly, G1 native empty callbacks cannot be used to declare the wrist/hip pair collision-free.

The CPU toy adaptation was a temporary copy of the tracked isolated script with `SimulationCfg(device="cpu")`; original code remained untouched. Both CPU simulations completed the full 300×0.005 s steps and flushed the write-once JSON, but the Isaac app shutdown stalled after result emission; we sent SIGINT **only to our own already-finished disposable probe processes**. This is not a claim of clean application shutdown. The tracked `tools/qualify_backend_contacts_v056.py` independently hashes all four actual toy JSON and all three G1 reports and emits a fail-closed, write-once local audit; CPU toy does not transfer truth to GPU G1.

Docs: [Omni Physics Contact Reports](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html), [Isaac Lab ContactSensor](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/contact_sensor.html).

## IV. Research-method and implementation decision

**Reject claimed success of naturalness or fast instep striking from this batch.** Pure frozen existing policies are controls, not new learned skills. v55's 29DoF fail-closed `wholebody_safety_interface_v055.py` remains a prototype and must not be activated on G1 without independently observed collision pairs/support boundaries. It permits body-wide multi-joint synthetic phase intent and bounded actuator actions in code, but this alone is **not** demonstration-trained human movement, a physics-validated collision barrier or a new football policy.

Next phase-conditioned joint/whole-body coordination method **proposed, not trained**: estimate phase `phi(t)` from actual human running/planting/strike sequences, retarget with contact- and support-aware inverse dynamics, jointly optimize pelvis/hip rotation, torso counter-twist, swing knee and coordinated opposite arm trajectory with actuator velocity/torque, foothold/COM and per-pair verified collision constraints. Train on qualified human motion with explicit time ordering and identity-disjoint train/val/test. Require independent original G1 physics with whole-trajectory naturalness metrics **and** unchanged physical shot/safety gates. Reject unqualified 120-Hz CMU phase proxies as ground-truth professional ball-impact force or GRF. No seen C/D v56 failure can be silently moved into an unseen evaluation set after tuning.

**Next experiment requirement:** isolate what changes between CPU/GPU contact-report dispatch, then preregister a G1 contact-pair validation protocol. If a CPU G1 alternative is attempted, it is a *different physics backend* and must pass new trace-by-trace parity; CPU toy positivity cannot replace original GPU G1 manifold evidence. Do not create self-articulation contact tensor views that previously crashed; separate scratch scenes only. Additional evidence: actual contact sensors with independently verified per-pair G1 collider paths, high-rate foot/COM support, honest missing data qualification and strict multi-seed comparisons after controller training. Never edit the v56 frozen evaluation config or old physical output.

## V. Reproduction and checks

Relevant versioned source:
- `configs/physx_contact_safety_v056.json`: registered 15-case protocol and frozen source hashes.
- `tools/run_placement_strike_v056.py`: original physics 29DoF runner; seeded true reset and read-only callback.
- `tools/run_frozen_batch_v056.py`: 15 sequential GPU cases, process locking and no-overwrite policies.
- `tools/native_contact_reader_v056.py`: actor/collider path and manifold recorder, with explicit lack-of-proof semantics.
- `tools/probe_physx_pair_isolated_v056.py`: disposable isolated GPU toy contact positive/negative with net/filtered force reference.
- `tools/audit_real_batch_v056.py`, `tools/analyze_measured_support_phase_v056.py`: independent complete audit and temporal dynamics.
- `tools/qualify_backend_contacts_v056.py`: CPU-vs-GPU reporting false-negative qualification using exact source hashes.
- `tests/test_native_contacts_and_seeds_v056.py` and `tests/test_backend_contacts_v056.py`: CPU-only unit and source-contract validation.

With **original raw assets available only on Tang**:
```bash
cd /home/you/projects/kinetickick
PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking \
  /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
# On a clean workspace with valid original configs, raw inputs and no existing reports:
# /home/you/conda_env/env_isaaclab/bin/python tools/run_frozen_batch_v056.py
# The driver refuses to replace existing write-once reports; existing Tang data are immutable.
# To independently re-audit existing reports WITHOUT overwriting the stored audits,
# make a new analysis output path after separately registering it.
```

**Full regression completed:** 691 passed + 69 subtests passed, no errors under `env_isaaclab`. Using the wrong system Python yields missing `rsl_rl` imports, **not** a regression; the reported passing run used the explicit original IsaacLab environment.

**Scientific conclusion:** This batch fixes real seed-independent physical evaluation, preserves strict failure evidence, and isolates a *GPU-native contact-report observability gap*. It does **not** prove left-wrist–left-hip self-collision, human-like explosive kicking or full-body motion learning.
