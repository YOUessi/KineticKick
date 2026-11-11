# KineticKick v0.1.85 — 6 original G1 CUDA PhysX paired native contact probes, genuine wrist–hip failure reproduction, independent 200Hz audit

**Date:** November 12, 2026 (+08). **Branch:** `version/v0.1.85`. **Parent:** v84 `5baf99191a28df1632f9dee0256a7c0e8952bdf7`. **Physical protocol freeze after zero-GPU preflight correction:** `a2865fc277e57e9d7433f75710597094e6865d6a`, protocol `SHA256 d80c56e1cc1c6bdb811d047602cef20e61d748bae58ea4b57869b76972acb664`. The first preflight incorrectly retained the string v0.1.84 and aborted before any GPU child launch. That genuine negative is preserved as `reports/v085_main_batch_console.log`, and a separate fix was committed **before** the first physical run; no hidden changes to the frozen batch occurred after physics began.

## Executive scientific results

**6 new original free-root 29DoF G1 CUDA PhysX physical executions at 200Hz**, all six independently source-certified; all six had genuine contact-0.4s to contact+1.25s observations. The 3 frozen original-center ball conditions are original v32, original CMU10_02 proxy, original CMU10_05 proxy, each with existing native-report callback/poll OFF vs ON at seed 1201 (no true reset perturbation). **Physical hard-gate PASS 4/6, FAIL 2/6**; these six are **three distinct policy conditions**, not six independent resets. Four full 14s episodes passed, two CMU10_05 episodes failed at a nonfoot event and had **no observed 14s recovery**. All physical on/off pairs had **EXACT original 200Hz shared-state, contact, recorded outcome and termination parity 3/3**, including both 5.145s failures.

The v85 observer **did not** certify actor0/actor1, collider0/collider1, contact-point normal and positive impulse for either wrist–hip OR known foot–ball positive-control contact. Across the three native ON episodes, there were **0 callback invocations, 0 native report headers** despite **1100 + 1100 + 890 read-only polls** and three measured positive original foot-ball contact events. Thus **the native reporter is uncalibrated in the original CUDA G1 context**. Neither absence of self-collision nor a precise wrist–hip direct pair is proved. Stage inventory showed `PhysxContactReportAPI=True` and threshold 1N on wrist, hip, ankle and ball rigid actors; `reportPairs` target lists were read as empty, but this read alone does **not** resolve authored-empty-vs-unauthored semantics, registration at actor creation, or DirectGPU readback.

## Frozen A/D experiment matrix and strict independent hard gates

| Center policy / existing sensor | Real 200Hz physical frames | Hard gates | Ball speed 5–110ms proxy m/s | Instep impact | Max original 29 qdot/limit | Peak nonfoot |
|---|---:|---|---:|---:|---:|---:|
| Original v32 OFF | 2800 | PASS | 4.7653 | 100% | 0.9081 | 0N |
| Original v32 ON | 2800 | PASS | 4.7653 | 100% | 0.9081 | 0N |
| CMU10_02 OFF | 2800 | PASS | 4.7669 | 100% | 0.9081 | 0N |
| CMU10_02 ON | 2800 | PASS | 4.7669 | 100% | 0.9081 | 0N |
| CMU10_05 OFF | 1028 + termination marker | **FAIL** nonfoot / 14s / recovery | 4.7737 | 100% | 0.9081 | 25.890153N |
| CMU10_05 ON | 1028 + termination marker | **FAIL** nonfoot / 14s / recovery | 4.7737 | 100% | 0.9081 | 25.890153N |

Original full g1 scoring includes *actual measured* all-29 joint velocities, real G1 free-root source certification, original contact surface, >95% instep fraction, goal, ≥3.5m/s near-contact ball speed proxy, nonfoot <20N, actual 14s and recovered flag. No score was changed, no physics episode counted as successful solely for scoring a goal. Ball speed is **not** a measured human strike impulse or verified high-speed shot. All three center scene controls produce <5m/s. v84 C and D had 6 unique scene-policy conditions (7 physical executions including a duplicate ON), 6/6 unique strict passes; only 3/6 exceeded 5m/s and 0/6 exceeded 10m/s. The CMU-style proxy moves center ball speed by only +0.00163m/s (CMU10_02) or +0.00838m/s (CMU10_05, **nonfoot FAIL**). These are not meaningful demonstrations of a learned stronger human-like kick.

### Failure provenance and precise contact uncertainty

The original CMU10_05 off/on have identical real 200Hz physical contact events. At approximately **5.145 seconds**, the v85 recorder again identifies **left_wrist_yaw_link** and **left_hip_roll_link** equal/opposite net force vectors of magnitude **25.890153N**: one approximately `[-21.220844, -14.821829, -0.537769]` N, the other its opposite. This repeat is strong indirect mechanistic evidence, NOT same-manifold direct actor/collider proof. A root height around 0.786m at termination and finite tilt do not establish a fall. The original simulator terminated for nonfoot safety.

### New 200Hz recorder subtlety: final duplicate terminal row

CMU10_05 stores **1029 original JSON rows**: 1028 real step-spaced 200Hz states through 5.140s plus a separate `time_s=5.140001`, `terminated=true` annotation whose physical state equals the preceding row. The frozen independent tool correctly flagged a NONCONTIGUOUS_PHYSX failure, retained in `reports/v085_independent_200hz_console.log`. A **postregistered, source-preserving** `v085_evidence_termination_shim.py` permits removing only that exact duplicate annotation **from an in-memory audit view**, and separately records the marker and nonfoot extrema. Original JSON SHA and the hard-gate outcome are never changed. Thus there are **13,256 genuine 200Hz physical step states** (4×2800+2×1028) and 2 non-physical terminal annotations, not 13,258 physical steps. All six actual contact ±windows remain complete; no post-5.145s recovery is invented. The 6 physical-data figures are SHA-bound and kept Tang-local.

## B / E independent controls, source SHA and tests

The frozen 6-case source validator rehashes original .npz, running assets, URDF, policy ONNX, controller dependencies and reports no drift, uses the global GPU `fcntl` exclusive lock, refuses existing outputs and requires disk≥20GiB and idle GPU before each bounded child. Six child simulations completed, with zero uncertified GPU attempts. Original raw `reports/v085_center_*.json` and media `media/v085_*.png` are **not** pushed. Old CMU C3D, neural weights and PhysX files were not overwritten or deleted.

The independent postregistered audit recomputed all-29 qdot, support-foot load fractions, whole-body COM z, arm/waist/swing/support leg velocity-energy timing proxies, real nonfoot and 14s completion; all six contact pre/post windows qualify. The native positive-control validator independently requires same actor/collider pair, finite point/normal and positive impulse, not coincident net forces. v85 produced **three uncalibrated ON observers**. In the full trace, center v32 support is ~64.25% bilateral, ~14.43% left-only, ~17.18% right-only and ~4.14% flight; these *descriptive* support fractions are not anthropomorphic quality labels.

**Final full-project CPU regression: 966 passed, 2 skipped, 69 subtests passed, 0 failed**, after adding 12 frozen/postregistered tests plus 3 termination-marker regressions. Two initial analysis failures were preserved: frozen preflight version guard (before physics) and independent 200Hz timestamp detector finding the physical/terminal-marker distinction. All pass/fail results, event names, and sample counts are real and were not fabricated.

Read-only evidence provenance:
- v85 protocol: `d80c56e1cc1c6bdb811d047602cef20e61d748bae58ea4b57869b76972acb664`.
- Frozen v85 batch audit: `b779c351b43764140e13821a605d2d3ff02f4f18abb8e4837e66791ee8d9829e`.
- Independent 200Hz audit: `c7e5ba25cbb14d94a039ecc7c25330311e28a8bf2dca3730d0501b0d18d1d779`.
- Independent native witness audit: `fb6589c7ea4020b5eba9489b7277418b32b0737013e22048cc44a6f049d8dcaf`.
- v84/v85 shot margin comparison: `8fbc09b2fdce804fb45097e899af1c182f926323ad0ec45894d93a490673134a`.

## Science: what is solved, what is still blocked

A: **No direct native actor/collider contact witness**. Exact read-only parity is established for all three original-center cases; the reporter is blind even at known original foot-ball contact. The current probe must not be used to certify safe wrist clearance. The official NVIDIA Omni Physics native reporting guide describes the prerequisite report API, callback and header/point fields: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html .

B: **New independent original-G1 all-29 200Hz contact-coupled dynamics**, including real censored safety failure, COM/feet/qdot/energy proxies, 6 figures and marker-aware audit. Shape-level clearance and direct contact manifold remain missing.

C: **No learned professional full-body kick**. The existing v83 ~0.154rad support→hip→torso→swing→opposite-arm candidate is synthetic and physically blocked; it was not tuned on or tested against v85. See `docs/V085_WHOLEBODY_LEARNING_GATE.md` for event-conditioned 29DoF observation/action schema, contact-safe QP proposal, required human qualification, complete preregistered policy ablation and blinded naturalness evaluation. All six known CMU clips are optical geometric-phase proxies without synchronized GRF or verified professional instep impulse. Do not manufacture labels.

D: Frozen original controller on/off matrix completed: **6 real physical runs, strict 4/6, paired nondisturbance 3/3**, but no learned kinetic-chain physical success. Do not pool observer duplicates as independent policy generalization.

E: frozen source, postregistered analysis scripts and regression tests + this report on a unique GitHub branch; raw heavy evidence remains Tang-local. GPU was idle and ~37GiB free at closeout. No modification to other version worktrees or train jobs.

## Next executable research batch (requires fresh freeze)

1. Create an **isolated, bounded PhysX two-body foot-ball positive-control scene** with clearly authored `PhysxContactReportAPI` before actor initialization; verify callbacks and native header paths/contact point/normal/impulse work in the local GPU/CPU-readback mode. This isolated positive control does *not* certify original G1. Inspect whether `reportPairs` relationship is authored empty or genuinely unset, and the original GPU DirectGPU readback lifecycle, **without changing original G1 USD**. If changed reporting is needed, isolate it as a separate experimental condition and demand fresh exact parity before any safety claim.
2. Reuse frozen original center CMU10_05 failure scene and original 3-controller off/on control; require **calibrated positive control first**, direct wrist/hip collider-path witness later. Preserve 1028+marker semantics and all pre/post windows.
3. Only after calibrated geometry/support instrumentation and high-quality licensed human data: implement/train event-conditioned full-body phase policy + constrained inverse-dynamics QP. Frozen ablations leg-only, arms-frozen, phase-reversed, no-support barrier, original v32, human-only kinematic. Evaluate held-out perturbed resets and naturalness separately from strict goal/ball-speed/instep/qdot/nonfoot/recovery gates.

This v85 research work supplies real reproducible dynamics and a falsifiable negative instrumentation result, **not** the final natural high-power humanoid football paper claim.
