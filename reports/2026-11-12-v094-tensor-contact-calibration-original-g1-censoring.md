# KineticKick v0.1.94 — CUDA PhysX pair-filter contact calibrated; original 29DoF actuation remains fail-closed

**Date:** 2026-11-12 (Tang local, UTC+08). **Branch:** version/v0.1.94. **Protocol frozen BEFORE all physics:** 10d7bfd8a003ae5dee7f9c705ccb76acd3c2c824, parent committed v93 d69904df2bd2c0f833eb384520a63f4867334623. Protocol matrix SHA256 2655bf5b6ead092756a2df430fb7ebb4cefa83dcf81d1c1464dd8664e0e9cbae.

## 1. Environment and protection
Tang Ubuntu/RTX4090 Laptop GPU online. At first preflight 37GiB free (>=20GiB threshold), no foreign CUDA compute process. Preserved base branch v64, v90 untracked work and v93 frozen/dirty work without touching. v94 was created with git worktree add -b version/v0.1.94 from the clean *committed* v93 protocol SHA. Shared original GPU flock was held across all six isolated 180-step physical probes. Each child writes new local JSON, console log and SHA-bound audit only. Original robot PhysX/CMU/C3D/weights/earlier reports never overwritten. GitHub contains only re-runnable scripts, frozen config/protocol, tests and this text; large raw trajectories and PNG kept on Tang.

## 2. Repaired actual API mismatch and six new PhysX simulations

v93 aborted the CPU tensor probe before its first step because its code read RigidContactView.count, which does not exist. NVIDIA's RigidContactView API documents **sensor_count**, **filter_count**, sensor_paths/filter_paths and get_contact_data. v94 fixed the API, used the actual rigid-body sensor/filter path inventory, retained OFF stage without PhysxContactReportAPI and ON stage with explicit API, removed all kinematic box velocity writes, and did not change sphere/box mechanics.

Independent serial 200Hz Isaac Sim/PhysX (180 real physics steps per trial, 0.9 simulated seconds):

| Frozen case | Real physics | Z at end | Tensor force peak | Tensor pair points | Native collider/manifold |
|---|---:|---:|---:|---:|---|
| CPU pose OFF / colliding | 180/180 | 0.3800m | Not instrumented | Not instrumented | Not instrumented |
| CPU tensor ON / colliding | 180/180 | 0.3800m | **0N** | **0** | Not instrumented |
| CUDA pose OFF / colliding | 180/180 | 0.3800m | Not instrumented | Not instrumented | Not instrumented |
| CUDA tensor ON / colliding | 180/180 | 0.3800m | **667.0803N** | **113 filtered contact counts** | Not native |
| CUDA native ON / colliding | 180/180 | 0.3800m | Not instrumented | Not instrumented | **0 header, 0 callback** |
| CUDA tensor ON / box +5m apart | 180/180 | -3.0451m | **0N** | **0** | Not instrumented |

Actual first CUDA tensor pair observation at physics step 68: approximate world point (0,0,0.20)m, world normal (0,0,1), peak contact force magnitude approximately 667.0804N; subsequent settling sample has about 9.81N. These are **actual CUDA PhysX tensor outputs** with sensor root /World/ProbeBall and filter root /World/ProbeBox, not hand-annotated G1 contacts or force-ground-truth for elite soccer. The USD collider children are /World/ProbeBall/geometry/mesh and /World/ProbeBox/geometry/mesh, but tensor path roots alone do not deliver the original native colliding shape IDs. Filter count=1, sensor count=1. This is the first certified positive pair-filter witness in this line, **not** an original G1 self-collision manifold certification.

The CPU sphere physically settles on the box but tensor reads zero force/count: **CPU tensor is uncalibrated/false-negative in this setup**; treat it as negative instrumentation result, not no collision. CUDA separated control falls freely and yields zero pair force: no spurious filtered pair event in this control. The v94 native reporter still returns zero headers in 180 polls and zero callbacks in the same true-collision trajectory, so the direct native actor/collider/path/point-normal-impulse truth remains unverified. No proof for or against left wrist–hip or right wrist–hip direct self contact.

All six child processes returned code 0 and saved true physical trajectories. No V094_APP_CLOSED marker in any run; independent Kit/app.close exit behavior is an unresolved lifecycle problem and not proof that dynamics did not execute. No case was silently retried or dropped.

## 3. Three OFF/ON physical nonperturbation certificates

Independent SHA verification of raw JSON/logs and exact z/vz over 180 frames shows:
- CPU pose OFF vs CPU tensor ON: maximum recorded absolute z or vz difference **0.0**.
- CUDA pose OFF vs CUDA tensor ON: **0.0**.
- CUDA pose OFF vs CUDA native ON: **0.0**.

Thus 3/3 instrument comparisons preserve these **specified sphere position/velocity observables** at saved 9-digit precision. This does NOT prove equality for every unrecorded PhysX internal state or original G1 state. No control or collision geometry modifications were introduced between OFF/ON cases. Frozen source file hashes, log SHA256 and output SHA256 recorded by tools/v094_independent_audit.py.

**Physical accounting:** 6 new isolated real PhysX runs, 1080 verified 5ms steps, 5 real colliding landings plus one separated negative, 1 positive CUDA tensor sensor/filter witness, 0 native collider manifold witnesses. **Original 29DoF G1 shooting trials newly executed: 0. New learned-human-policy trials: 0. No new football success rate denominator.**

## 4. Independent historical 200Hz G1 dynamics and honest failure labels

Re-ran the v91 independent auditor against the unchanged original four E/F original 29DoF G1 real GPU PhysX JSONs. Each source raw SHA was checked independently, measured all29 velocities and speed limits checked, actual COM/left/right support/nonfoot contact evaluated, terminal +1us markers separated from real 5ms physics frames, and pre0.4/post1.25 impact windows audited.

| Historical original G1 case | True 200Hz frames | Instep share | Postimpact +1.25s complete? | Strict gate |
|---|---:|---:|---|---|
| E/v32 | 736 | 0% | Yes | FAIL: instep, ball speed, nonfoot, recovery |
| E/CMU02 | 2800 | 100% | Yes | FAIL: ball speed <3.5m/s |
| F/v32 | 320 | 100% | No; censored | FAIL: goal, nonfoot, recovery |
| F/CMU02 | 320 | 100% | No; censored | FAIL: goal, nonfoot, recovery |

Historical score stays **0/4**; these are v91's original controller episodes, not new runs and NOT the offline v94 safety filter's policy scores. The left/right wrist/hip opposite net force remains an indirect hypothesis. Even a 100% instep share and speed >5m/s is insufficient without actual goal, nonfoot safety and postimpact standing recovery.

The v92 source-verified six CMU 120Hz optical C3D clips yielded 0 fully qualified expert instep-impact/GRF examples and 0 qualifying geometric phase examples under the retrospectively adopted strict 90% visibility criterion; five clips share subject-folder10, one folder11. No professional identity, true instep impulse or human GRF was invented.

## 5. Actual implemented future wholebody control interface (not used for G1 actuation)

New tools/v094_safe_wholebody_projection.py exposes a 29DoF CPU-executable constrained velocity-projection operator, using actual measured 29-joint speeds/limits, measured bilateral support loads, 3×29 stance foot Jacobians, and source-SHA-verified collider signed distance plus 29D gradient. It hard-constrains speed, acceleration, loaded-foot no-slip and first-order collision clearance, and rechecks constraints post-QP. No support or collider evidence -> fail-closed. It returns an **offline candidate only**, NEVER a motor command, because original G1 native manifold, calibrated torque dynamics and qualified human action training are still missing. Synthetic unit tests validate limit and slip/clearance rejection; they are NOT motor tests.

docs/V094_WHOLEBODY_GENERATION_READINESS.md defines the falsifiable approach/plant/pelvis/trunk/swing/counterarm/recovery phase distribution, full-29DoF learned action and support/geometry/torque constraints, data qualification and frozen arms/leg-only/reversed-phase ablations. There is **no trained expert human wholebody controller this run**, and no publication claim of humanlike high-speed impact.

## 6. Code, raw evidence and reproducibility

Committed frozen (pre-physics): configs/v094_frozen_matrix.json, docs/V094_FROZEN_PROTOCOL.md, tools/v094_box_pair_probe.py, tools/v094_box_batch.py, tests/test_v094_contact_contract.py.

Post-physics independently verified tools/v094_independent_audit.py and tests/test_v094_independent_audit.py, offline control filter tools/v094_safe_wholebody_projection.py and tests/test_v094_safe_wholebody_projection.py, plus this report and readiness document. All new raw local JSON/log/figure/weights are gitignored and remain on Tang. Exact raw/log SHA and source SHA in local reports/v094_frozen_contact_batch.audit.json and reports/v094_independent_evidence.audit.json. Read-only historical v91 G1 SHA also recorded. Local true physical parities figure: reports/v094_independent_physx_parity.png (SHA256 81a886f7ee7fd1511c6e5b51371f97a5b7dfc166493a23544b2b36557fce11cb).

Targeted new CPU tests: 19 passed. Full-project regression: see separately archived local reports/v094_full_test_suite.log and final appended test count.

Run reproduction:
1. checkout this branch and ensure IsaacLab/Isaac Sim 4.5 original environment; keep original CMU and G1 physics data on Tang (no public release of large local assets).
2. Verify >20GiB available and no external CUDA jobs, and existing original global GPU flock path.
3. Freeze git HEAD and protocol SHA; run python3 tools/v094_box_batch.py once in a clean environment, with fresh write-once output targets. A rerun must use a new version, not overwrite this output.
4. Run IsaacLab env python tools/v094_independent_audit.py --v91-root ORIGINAL_V91_LOCAL_SOURCE --output reports/v094_independent_evidence.audit.json, and pytest with the same Python. A second audit needs another output target; do not overwrite.

Official reference for tensor contact views: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html ; note IsaacLab Issue 2282 addresses GroundPlane GPU filter warnings, but v94 uses an actual kinematic rigid box, not GroundPlane.

## 7. Next version (new frozen protocol required)
(1) Debug why original native contact report returns no headers even on the calibrated CUDA sphere-box positive case; inspect PhysX simulation scene/actor subscription lifecycle and callback timing, not by changing original G1 control. (2) Cross-validate CUDA filtered tensor contact with an additional positive shape and negative ground topology, ideally retrieve genuine collider IDs via native contact stream. (3) Once native positive calibration passes, run read-only original-G1 wrist/hip and foot/ball off/on with original-state bitwise/nonperturbation audit; **do not start new action before that gate**. (4) Seek licensed verified multibeat expert kick 3D+GRF/CoP data, then preregister and train a wholebody action generator with true 29DoF support/pelvis/trunk/arm/leg coordination; test honest holdout strict PhysX gates and naturalness. Preserve all negative trials, original raw PhysX and weights.
