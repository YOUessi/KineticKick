# KineticKick v0.1.86 — native contact positive-control negative result and independent wholebody evidence

**Date:** 2026-11-12 +08. **Parent:** v85 `3822151c8b86850bce2ca40daef3b56c932c5e4e`. **Frozen preregistration commit:** `2e509f928dc5bbece7316b3a40b6047eb39bb53c`, protocol SHA256 `9829f5b3956a8d522e6f112819d5dc467f83d4ed9e82d7dbc772c63068ee669b`. **Status:** negative instrumentation / physical-scene qualification failure, **NOT** trained natural human-like kick.

## Device, safety and independent workspace

Tang online, original 29DoF physical source preserved; ~37 GiB available (>=20 GiB safety guard), RTX4090 compute idle before calibration, existing global GPU lock acquired by v86 runner. The new version was created from v85 in a clean independent v86 worktree. Existing dirty *other* version worktrees were explicitly left untouched. The 6 original v85 G1 PhysX sources, CMU C3D and neural models were never overwritten or deleted. Runtime JSON/logs and 200Hz visualization remain Tang-local per gitignore.

## Frozen A contact positive-control matrix: strict negative

The isolated calibration probes were designed as two newly authored USD actors (falling sphere and static floor), **not the original G1 simulation**. The frozen 2×2 device/reportPairs matrix was CPU wildcard, CUDA wildcard, CPU authored-empty and CUDA authored-empty. Four exact cases were frozen, and the execution runner was programmed to abort the remainder on an uncertified child.

- **Executed:** 1 CPU/wildcard child launched Isaac Sim, created the two actors and report API **before reset**, completed 180 calls of 200Hz `SimulationContext.step()`, and recorded 180 post-step native polls. Its USD inventory had `reportPairs_authored=false`, threshold=0, dynamic-ball RigidBodyAPI and two collision bodies.
- **Native evidence:** **0 callbacks, 0 native headers, 0 native collider-pair witnesses**; no errors returned by poll. Crucially, we did **not** independently measure falling sphere position or sphere-floor physical impulse, so cannot certify that this isolated scene actually had the intended physical contact.
- **Lifecycle failure:** after the JSON was written, the child blocked **before `V086_PRE_CLOSE`**, consistent with the code's `sim.stop()` stage. Its 185-second watchdog terminated **only its own process group**. Even though the final OS return status was 0, watchdog=true and no `V086_APP_CLOSED` therefore correctly failed source certification. The standalone raw JSON does **not** make this a successful physical calibration.
- **Certified isolated contact trials:** 0; **uncertified isolated attempts:** 1; **unexecuted frozen cases:** 3. **New actual original G1 football simulations:** **0**, hence new kick success rate is **not measurable**. No new left wrist–left hip direct manifold truth.

This is a useful falsifiable negative: the first diagnostic program's lifecycle and ground-impact proof need repair *before* claiming contact-report blindness on the physics backend. Do not repeat the historical mistake of interpreting an empty callback as noncollision.

## B: independently re-read the real 200Hz v85 trajectories

Using the **original immutable raw JSON** for three unique policy conditions (v32, CMU10_02, CMU10_05) rather than counting native-off/on sensor duplicates as extra scenes, independently checked the raw SHA against v85 batch evidence. These three original G1 controls contain **2,800 + 2,800 + 1,028 = 6,628 real 200Hz physical steps**; CMU10_05 includes **one separate termination annotation** that must not be miscounted as a physical step.

Validated actual 29 joint names, measured q/qdot, estimated joint torques, whole-body COM, dual-foot support forces, pre-impact 0.4 seconds through post-impact 1.25 seconds (331 samples), nonfoot events and recovery censoring. The original CMU10_05 peak nonfoot load was **25.890153N** with first >=20N occurrence at ~5.135s and no real 14s recovery. It is not proof of specific wrist/hip manifold.

The **preimpact kinematic group-speed peak times were identical across all three policies at the frozen center/seed:** waist/trunk -0.215s, left counter-arm -0.225s, right swing leg -0.040s, and left support-leg group -0.175s relative to observed original foot-ball contact. Group RMS differs only slightly (e.g. swing right ~4.18149, 4.18178, 4.18186 rad/s). This is evidence that the weak existing CMU proxy **does not meaningfully reorganize the early kinetic chain at this scene**, not evidence that these timings match a professional footballer's naturalness. These are *measured robotic joint-velocity proxies*, not human ground-reaction force, angular momentum transport or calibrated motor work.

Source-witnessed audit is local `reports/v086_v85_three_policy_coupling_retry1.audit.json`. Its first write attempt failed on a NumPy boolean JSON type and left an incomplete, preserved retry0 file; the source code was corrected, and fresh **write-once retry1** completed. A three-policy 200Hz original trace comparison PNG is local `media/v086_phase_chain_3control.png` (SHA256 `3d5f304f032b07eb56ce1b10525773efff29caea71aa26b9bb20f457b35a958e`).

## C: safe wholebody control design — untrusted execution still blocked

Implemented `v086_wholebody_safety_gate.py`: structured **29DoF** q/qdot + actual two-foot normal forces + signed-distance collider pair/Jacobian/geometry SHA + native reporter positive-calibration freshness. The interface fails closed for missing or stale contact geometry, unsupported foot, near-limited qdot, nonfinite data or impending clearance violation. It never emits a motor command and explicitly returns `original_G1_motor_command_authorized=false` even for a well-formed offline prototype-ready state. This is an executable **evidence gate**, **not** an inverse-dynamics QP implementation, collision guarantee, learned policy, or deployment certification.

Professional football instep/GRF truth remains absent from the current six CMU optical clips. No force target was synthesized and no v85 failure case was used for test-set tuning.

## D/E tests and strict limitations

Targeted v86 regression: **17 passed**. Full project suite: **979 passed, 6 skipped, 69 subtests passed, 0 reported test failures** (run log `reports/v086_full_suite.log`). A confirmation rerun explicitly captured the shell exit code as **0** and independently reproduced **979 passed / 6 skipped / 69 subtests passed**. No new original G1 hard-gate score is asserted. All previous v85 nonfoot negative results retained.

**Follow-on dependency chain:** (1) in a *new* version reproduce or falsify `sim.stop()` teardown hang using timed markers; (2) record independent actual falling sphere z/impulse to prove a physical positive control; (3) only when demonstrably contact-positive, test CPU vs CUDA native actor/collider reporting and empty/wildcard relationship semantics; (4) then original G1 read-only contact pair with exact off/on physics parity; (5) only after qualified human data and safety geometry, train and test support→hip→torso→swing→arms→landing on held-out physical resets. **No new human-like high-power football capability is claimed.**
