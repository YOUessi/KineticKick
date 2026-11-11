# KineticKick v0.1.97 — Six new original-G1 PhysX shots, G/H safety falsification and genuinely fresh robot-only 29DoF holdout

**Date:** 2026-11-12; **branch:** `version/v0.1.97`. Parent Git SHA `f143a1cc70f5e5c92c6ed2c803c16b1a1934faea` (v96). Pre-physics preregistration commits `b4a8c55b85b1a6e9824452987c2d584dcee5fb15` and `4bc1c2f8e66bfd4b54d0620ecf43cbc2ae4090a4`; latter corrected preflight version/count strings **before any physics episode**. Protocol JSON SHA-256 `befd7be5b0da3f6519b10d6d817100d4cbd1cb2297808e03f05c971b49a8ebd8`.

## Research question and predeclared comparison

Re-test three **UNCHANGED**, historically frozen original 29DoF G1 PhysX control conditions—v32, CMU10_02 and CMU10_05—under two **new and predeclared** football locations G/H. CMU's optical phase proxy comes from limited single-subject motion, with no verified professional-football identity or synchronized actual instep/ground-reaction-force supervision. Nothing in this batch learns or actuates a new motor policy. Comparing three fixed conditions at two ball placements with fixed seeds **does not establish high-powered human-like shooting**, especially when seeds do not cause independent initial-state physical perturbations.

Frozen G = ball XY displacement (+0.035, -0.095) m, lateral velocity condition +0.015, seed1409; H = (-0.045, +0.095) m, lateral -0.015, seed1423. Fixed order v32/CMU10_02/CMU10_05 within each scene. All six isolated shots ran on actual RTX4090 CUDA PhysX with original free-root 29DoF G1 and **200Hz physics**, original contact/friction, motor control and hard termination rules. Six/6 certified; zero unscored timeouts; no overwrite. **Full strict gate result 3/6**. This is an episode outcome for this fixed pair of placements, **not generalization accuracy of a learned policy**.

## Six new certified G1 physics outcomes (all strict gates independently recomputed)

| Fixed scenario | True 200Hz steps* | Impact s | Instep fraction | Ball speed proxy m/s | Peak non-foot net N | Max original 29 qdot/limit | Original 14s recovered? | Strict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| G / v32 | 776 | 1.630 | 70.60% | 3.8413 | 56.610 | 0.9973 | No; `physical_nonfoot` | **FAIL** |
| G / CMU10_02 | 844 | 1.630 | 70.38% | 3.8418 | 0.000 | 0.9246 | No; `physical_height` | **FAIL** |
| G / CMU10_05 | 784 | 1.630 | 70.41% | 3.8406 | 73.258 | 0.9086 | No; `physical_nonfoot` | **FAIL** |
| H / v32 | 2,800 | 1.555 | 100% | 5.1240 | 0.000 | 0.9078 | Yes | **PASS** |
| H / CMU10_02 | 2,800 | 1.555 | 100% | 5.1261 | 0.000 | 0.9078 | Yes | **PASS** |
| H / CMU10_05 | 2,800 | 1.555 | 100% | 5.1258 | 0.000 | 0.9078 | Yes | **PASS** |

*Three terminated G cases contain **one additional +1 microsecond terminal annotation**, which is **not a real 200Hz PhysX step**; raw JSON has 777/845/785 rows, independently excluded from the true-step denominator after verifying duplicated physical state. H rows are 2,800 real physical steps each. All six have physically observed first foot-ball contact and actual contiguous pre-contact ≥0.4s and post-contact ≥1.25s windows; G **does not** have later 14s recovery. No interpolation or fabricated dynamics beyond the stops.

All G shots go through the goal, have ball-speed proxy >3.5m/s and have the original 29 measured qdot ratios within limits. **They still FAIL because only ~70% of contact impulse is legal instep**, not ≥95%; two also trigger non-foot events and another stops on minimum root height. G/CMU02's 0N non-foot observation explicitly **does not** imply recovery or absence of instability; the source terminates via `physical_height`. The minimum whole-body COM height observed for G/CMU02 is approximately 0.404m, which is not itself a contact manifold. H positive physical controls all pass unchanged; differences of only ~0.002m/s in their ball-speed proxies **do not establish more natural movement, stronger impact technique or human motion transfer**.

At G/v32 physical t≈3.880s, left-wrist and left-hip receive simultaneously reported equal/opposite-ish net force magnitudes ~56.610N; at G/CMU10_05 t≈3.925s, analogous ~73.258N. These are named-body net forces from real PhysX G1—not authenticated native `actor0/1`, `collider0/1`, impact points, normals or contact manifold, and cannot by themselves prove self-collision. The earlier v54 native stream remains uncalibrated despite known physical contact and v94/v96 valid isolated CUDA tensor-positive ball-box contact witnesses. The direct native G1 wrist/hip pair evidence requirement remains **OPEN/UNKNOWN**.

### Reproducible acquisition and independent audit

- **Pre-physics strict freeze:** `configs/v097_frozen_matrix.json`, `docs/V097_FROZEN_PROTOCOL.md`, `tools/v097_gpu_batch.py`; Git frozen bytes checked before GPU run. Shared `flock` across all GPU physics, disk >=20GiB, foreign CUDA computation check before every episode, only own child watchdog, 180s bound; immutable local JSON/log/audit filenames.
- **Independent post-physics audit:** `tools/v097_independent_dynamics.py` SHA-authenticates every new raw source, checks all29 actual qdot and physical timeline, independent instep impulse/goal/ball-speed/terminal/root-height/non-foot/full14s/recovery gates, COM, foot normal loads, left-leg stance, trunk/hip/right-leg/counterarm kinematic peak-time proxies, and censored event windows. Six source-grounded 4-panel diagrams saved as ignored local `reports/v097_physx_*.png`. Original sources remain unchanged; no pictures or 30MB+ raw data pushed.
- Local aggregate `reports/v097_g1_physics_final.audit.json` is the frozen-run manifest. Independent `reports/v097_independent_dynamics.audit.json` SHA-256 `cf4703abf8477e347b0b963bf1d36ef0bd8bd945ca479defe407f60da8b9fc39`. No native collider-pair manifold certified.

## Fresh G/H heldout full29DoF generation: useful **negative** result, not a physical controller

`tools/v097_coordination_observer.py` and `tools/v097_trajectory_generator.py` implement a **real trained** 29DoF robot-trajectory proposal model, using SHA-verified **historical v84 C/D robot PhysX traces only** to fit a deterministic ridge predictor (alpha=10), then **G/H never-fed-to-training G1 PhysX episodes** as held-out test. Current measured 29 q and qdot and current bilateral stance features condition six future temporal anchors; output is full `20 × 29` predicted qdot at 50Hz over next 0.4s, integrated from **current** measured q0 into 29 positions. This is not professional-human imitation, impact-groundtruth learning, or real motor control. The G/H validation is fresh only relative to the training in this version, not a fully blinded paper benchmark.

| G/H original G1 trajectory | State-only normalized qdot MAE | State + contemporaneous support MAE | Shuffled support negative MAE | Reverse-time candidate MAE |
|---|---:|---:|---:|---:|
| G/v32 | 0.043015 | 0.042945 | 0.043043 | 0.077296 |
| G/CMU02 | 0.048852 | 0.048614 | 0.048854 | 0.081029 |
| G/CMU05 | 0.043605 | 0.043643 | 0.043638 | 0.077570 |
| H/v32 | 0.012549 | 0.012276 | 0.012615 | 0.024292 |
| H/CMU02 | 0.013458 | 0.013137 | 0.013498 | 0.025359 |
| H/CMU05 | 0.012619 | 0.012243 | 0.012665 | 0.024370 |

Time reversal, arms frozen and only right strike leg form negative control proposals with independently scored 29DoF physical *trajectory prediction errors*, not actual G1 motor rollouts. Support features weakly improve many cases but harm G/CMU05; all improvements are small, with correlated 50Hz sliding windows **not six independent human subjects**. H is more predictable largely because H lasts full 14s with different time distribution, not because it demonstrates learned football biomechanics. Fitting uses **robot data**, with no verified professional stance/contact/GRF or impulse transfer.

### Critical newly implemented 29-joint *actual URDF angle* screen

`tools/v097_candidate_safety_audit.py` independently reloads the frozen robot-only training G1 evidence, refits the identical model, SHA-binds the new G/H evaluation, and compares generated full29 q positions **against actual per-joint URDF lower/upper limits** from the original source, in joint-name order. It rejects all motor authority even when proposed speeds alone are within limits, because original 200Hz support Jacobians, native collider distance, inverse dynamics and torque, friction/GRF, and human motion certification are absent. Actual position-limit-violating proposal *windows*:

| G/H candidate | Windows examined | Any true G1 URDF angle violation | Peak exceeding angle bound |
|---|---:|---:|---:|
| G/v32 | 34 | 30 | 0.503 rad |
| G/CMU02 | 38 | 34 | 1.833 rad |
| G/CMU05 | 35 | 30 | 0.512 rad |
| H/v32 | 135 | 25 | 0.456 rad |
| H/CMU02 | 135 | 27 | 0.456 rad |
| H/CMU05 | 135 | 27 | 0.456 rad |

All generated normalized qdot ratio samples were ≤1, but many generated **position** trajectories violate real G1 URDF joint limits: this is precisely why offline velocity or future-motion MAE **cannot certify safe physical actuation**. The original trained candidates were **not** passed to motors; their G/H true physical scores above belong to the **unchanged frozen v32/CMU controllers**, never to the candidate. Screen evidence in ignored local `reports/v097_candidate_safety_urdf.audit.json`; recomputed MAE matches frozen generated report byte-consistently. The original URDF source has not been modified.

## Testing and original evidence protection

Focused v97 tests cover frozen G/H matrix, strict terminal marker accounting and gate conjunction, 29 joint group partition and generated 20x29 shape, disjoint training/heldout, support/phase/arm negative controls, URDF source/name bound and actual joint position rejection; **15 new dedicated tests passed** (3+4+4+4). Complete project suite run using original IsaacLab Python: **1,087 passed, 2 skipped, 69 subtests passed, 0 failed**. Initial full-suite attempt failed 25 earlier tests on **missing ignored historical JSON test files in the new worktree**, not data-content drift or new scientific code; preserved `reports/v097_full_suite_initial.log`. Reconstructed **1,967 ignored *symlinks only*** referencing already existing unchanged v96 original archives (997 JSON, 963 logs, 5 PNG, 2 NPZ) after verifying `git check-ignore` for each and never touching original content. Retest passed; `reports/v097_full_suite_final.log` SHA retained locally.

No original G1 controller gains, collision/friction, hardware, original C3D/human data, URDF, NPZ or saved PhysX evidence changed. No force pushes or overwrites of any preexisting version. Current code results do **not** prove a novel style/coordination controller or human-like explosive kick.

## Scientific conclusion and next causal experiments

(1) The **same old controls** all failed scene G due **foot contact legality** and often recovery/nonfoot collision, yet all three passed scene H. This shows **ball-placement fragility**, not evidence for a superior human-motion-conditioned algorithm. Measured original ball-speed proxies 3.84 and 5.12m/s still do not represent powerful human-style striking. (2) Temporal stance-conditioned robot-only 29DoF ridge generation gives weak mixed error gains; real angle-limit screen decisively rejects many trajectories. (3) Native G1 collision manifold and professional human foot-ball/GRF truth remain unqualified.

Next full batch must (a) instrument one **isolated** physically witnessed positive PhysX contact report *within the correct post-step dispatch stage* until actor0/1/collider0/1 and positive manifold are actually obtained, validate that this channel works on native CPU/GPU and when probing the original G1 without physical drift (off/on full-state parity); (b) source/license and time-align **real elite instep-kick** full-body trajectories with explicit human foot-ground-force, contact and impact validity or clearly mark missing truth; (c) implement a physically **projected time-aligned whole-body contact/impact-conditioned generator** with support foot/COM stability, bilateral arm-leg torso phase, inverse-dynamics torque/angle/speed, collision clearance and stable landing; then preregister new blind G1 seeds/placements and compare no-arm, reversed phase, foot-only and baseline under the entire original hard-gate conjunction.

Source of API caution: NVIDIA PhysX native report documentation https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html . A zero native callback without a positive paired-contact calibration **does not mean no self-collision**.
