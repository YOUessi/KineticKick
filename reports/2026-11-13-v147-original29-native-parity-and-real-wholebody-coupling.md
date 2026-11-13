# KineticKick v0.1.147 — Frozen native OFF/ON parity gate and real original29 whole-body dynamics

**Executed:** 2026-11-13, Tang (Ubuntu / RTX4090). **Parent:** v146 SHA `2b0c0cab7f42c15564b0ddd945b5f86ed1d49109`. **Protocol frozen independently before code:** `3ecf094137d9f1d82e94a9dba3b9bf7d215c97c8`, in `docs/V147_FROZEN_PARITY_AND_COUPLED_DYNAMICS.md` and `configs/v147_frozen_physics_parity_and_coupled_dynamics.json`.

**Main outcome:** real previously executed original G1 traces are fully re-analyzed at 200Hz; strict exact per-physics-field OFF/ON comparison has now been implemented for future real callbacks; six actual native OFF/ON episodes remain **blocked by a foreign GPU process**. No new PhysX episode, no newly certified left-wrist↔left-hip manifold and no new physical goal/speed/safety success.

## 1. Machine/source and non-interference guard

At batch start, the authorized Tang device was online. The original checkout `/home/you/projects/kinetickick` remained on dirty `version/v0.1.64` (untracked `=100===`), deliberately untouched. We checked the local version worktrees through v146 and remote branches through v146; v147 did not yet exist. A separate clean `/home/you/projects/kinetickick-wt-v147` worktree was created **from verified v146**, then committed the protocol alone, before implementation. Old CMU/C3D, PhysX JSON, 30–40MB derived arrays, models and media were neither deleted nor overwritten. This worktree received only seven safe gitignored symlinks to existing Tang-local asset folders, plus read-only symlinks to five v146 ignored derived test artifacts. The v146 `reports/v146_original_real_6628_coupled_locked.json` source checksum remained `fbe74cc06d809821fa0593079a986b208c146598b12e054dbeefd90705eded0b`.

Filesystem available: about **30 GiB** (97% used); RTX4090 already occupied by long-running **v120 pytest PID 242644**, ~260 MiB of CUDA context memory at inspection. `v147_frozen_physics_matrix_guard.py` returned `BLOCKED_V146_FOREIGN_GPU_CONTEXT_NO_PHYSICS`. Neither the existing process nor the exclusive GPU lock was interfered with. Idle GPU utilization would not authorize a launch in the presence of an existing context. No speculative PhysX experiment was started.

## 2. Actual code and falsifiable tests

- `tools/v147_exact_off_on_parity.py`: checks original 29-position, 29-velocity and 29-simulated-torque arrays; floating-base pose/twist, whole-body COM, both foot sensor net force vectors, nonfoot force and ball pose/twist **at every 5ms step**, byte-level equivalent JSON numeric value (zero tolerance) across OFF/ON. Rejects missing/duplicated physical state, censored-episode length mismatch, forged native certification, corrupt counters, invalid callback indices, and bad state hashes. Even exactly equal **input JSON** remains labeled `runtime_source_attested=false` and `live_noninterference_certified=false`, never a PhysX contact result.
- `tools/v147_real_wholebody_dynamics.py`: SHA-verifies the actual three original v54 PhysX files, independently reads all **6,628** real 5ms samples and groups every one of the original G1's 29 joints into left/right leg, waist, left/right arm by real original joint-name order. Computes sampled simulator torque×qdot proxies and their 5ms integral, all original qdot-limit ratios, central COM-velocity finite differences, bilateral *net sensor force* ratios, nonfoot peaks and early-termination boundary.
- `tools/v147_independent_dynamics_recheck.py`: a second, Python `math.fsum` implementation starts again at SHA-verified original files rather than calling the vectorized research output, checking **12 policy-window combinations** and **7,705 overlapping row visits** (not 7,705 unique real states).
- `tools/v147_frozen_physics_matrix_guard.py`: re-verifies the three frozen v52 source policy configs by their original v54 recorded SHA values, freezes the six OFF/ON cases at the original center ball and seed **1201**, rejects any occupied output path, checks read-only asset links and actual nvidia-smi/disk stop gates. This creates only an ignored write-once *manifest*, and deliberately never launches physics.
- `tools/v147_plot_real_coupling.py`: reproducible figures of group mechanical proxies and 5.14/5.145s unmatched contact evidence boundary. Both PNGs are local and ignored.
- `tests/test_v147_parity_and_real_dynamics.py` and `tests/test_v147_frozen_physics_matrix_guard.py`: positive and adversarial cases including micro-perturbed q/tau, all core tensor fields, unknown native claims, lost/repeated steps, nonfinite data, false callbacks, source config tampering, output collisions and 19GiB disk hard-stop.

The independent recheck has maximum group integrated torque×speed proxy disagreement **1.1368683772161603×10⁻¹³ J**. This is *algorithm agreement over the same genuine source*, not evidence that torque or contact was measured by hardware.

## 3. Real, SHA-qualified 200Hz original29 dynamics

The actual original v54 center-ball cases provide **2,800 + 2,800 + 1,028 = 6,628** complete, aligned 200Hz states. The known terminal annotation after the last CMU10_05 state is **not** counted as an additional physics frame. Nominal contact event `t=1.570s` is an old robot foot-ball event/proxy, **not an authenticated right-instep collider manifold or a human impact timestamp**. Frozen source analysis windows are `[1.170,1.565]s` (80 states), `[1.575,2.820]s` (250 states), and `[5.000,5.140]s` (29 states).

### Original simulated torque×joint-velocity absolute integral (descriptive proxy J)

| Original policy | Before [1.170,1.565] | After [1.575,2.820] | Late [5.000,5.140] |
| --- | ---: | ---: | ---: |
| frozen_v32 | 231.2046 | 481.9457 | 6.4320 |
| CMU10_02 | 231.3281 | 484.9972 | 3.8739 |
| CMU10_05 | 231.3288 | 485.8452 | **15.7533** |

These windows have different durations. The numbers are time integrals of **sum of absolute values of reported simulator-estimated torque × measured joint angular velocity**, and do **not** establish positive mechanical work, true actuator electrical power, professional explosive force transfer or comparable task performance. The two CMU variants differ from frozen_v32 by only about +0.05% in this pre-event proxy. Their slightly larger after-event proxies do not imply better instep impact.

Group breakdown in the 80-state pre-event window (left-leg/right-leg/waist/left-arm/right-arm; proxy J): frozen_v32 **104.13164/101.27019/10.73694/7.25565/7.81020**; CMU10_02 **104.26284/101.26157/10.72906/7.26196/7.81265**; CMU10_05 **104.25260/101.26248/10.73782/7.25868/7.81726**. This is compatible with the previously observed milliradian-level style activation and does not validate human-like generation.

### COM and contact negative evidence

The maximum **central finite-difference acceleration proxy** from recorded COM velocities in the 80-state pre-event window was **40.0005 / 40.0106 / 40.0106 m/s²** (frozen/10_02/10_05); in the 250-state post-event window **53.6532 / 53.6036 / 53.6955 m/s²**. These can reflect nonsmooth simulated contact and derivative noise; they are neither force-plate GRF nor independently certified dynamic accelerations.

The median ratio of the left-foot *net force magnitude* to bilateral net force in valid samples is **1.0 pre-event / 0.0 post-event** for all three histories. The source sensors aggregate contacts, not ground-only contact pairs, so this is **not** a proof of physical support transfer, ground CoP, foot friction or stable base momentum.

The aborted CMU10_05 still reaches a **25.890153884887695 N** nonfoot force peak at **5.135s**, and its last complete physical state is at **5.140s**. A distinct named left-wrist/left-hip force observation appears around **5.145s**, without a same-tick full original29 state or true PhysX collider paths. Do not call this a certified wrist↔hip collision or an observed fall. The 5.145s step was **not interpolated or invented**. The other two historical cases have zero reported peak nonfoot force, but their original legacy success flags do not override the v53 strict **0/4 per policy** failures (instep fraction, goal, ball speed, joint/safety/recovery conjunctive gates).

## 4. Reproducibility, tests and genuine failures

The initial expanded CPU test recorded **8 failures**, all from a new clean worktree missing v146's ignored 41MB derived JSON and local figures. No code or evidence was modified to hide them; five **read-only symlinks** to the original v146 results were bound (SHA checked). The exact same 36-module regression was then rerun and finished:

**915 passed, 2 skipped, 0 failed, 2 pre-existing SciPy SLSQP bound-clipping warnings.** v147-specific test suite: **44 passed**. Real source-dependent 6,628-row and independent 12-window analyses completed. These are CPU regression results, **not** an IsaacLab/GPU test suite, and we did not alter the protected v120 GPU test process.

Write-once **Tang-local, Git-ignored** outputs with SHA256:

- `reports/v147_original_real6628_dynamic_summary.json`: `f3fe036fcb5657915a98326147fa14e4d0f25e78d41d45a2182aa3d4cef07e60`
- `reports/v147_independent_real6628_recheck.json`: `0afd961ae3e090c5053fcafe4bd17d58c73c5721ba6098b0fe2e5a79c1f6062e`
- `reports/v147_six_run_guard_manifest.json`: `b80d0b0905bd49214f376740029b6821fee8f94a123d54c53449c241b903c597`
- `reports/v147_history_original29_body_proxy.png`, `reports/v147_history_original29_nonfoot_censor.png`, plus test logs.

To reproduce in an unoccupied new local filename (do **not** overwrite the listed original artifacts):

```bash
cd /home/you/projects/kinetickick-wt-v147
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v147_real_wholebody_dynamics.py --output reports/v147_reaudit_UNIQUE.json
$PY tools/v147_independent_dynamics_recheck.py --input reports/v147_reaudit_UNIQUE.json --output reports/v147_independent_UNIQUE.json
$PY tools/v147_plot_real_coupling.py --input reports/v147_reaudit_UNIQUE.json --prefix reports/v147_plot_UNIQUE
$PY tools/v147_frozen_physics_matrix_guard.py --output reports/v147_matrix_guard_UNIQUE.json
$PY -m pytest -q tests/test_v147_parity_and_real_dynamics.py tests/test_v147_frozen_physics_matrix_guard.py
```

All original large data remain under Tang's gitignored paths. GitHub includes only frozen config/protocol, small reusable analysis and validation code, tests, and this report.

## 5. Scientific falsification and next required physical batch

We have **not** learned an elite human full-body shot. Existing same-subject CMU clips lack synchronized actual instep contact/GRF and cannot support causal peak-power claims. Stronger whole-body learning must operate on **qualified multi-subject football mocap, trustworthy contact phase labels and force information**, and test subject-disjoint generalization. A candidate method is a shared phase-conditioned full-body latent for approach, plant-foot loading, pelvis and trunk angular momentum, swing-leg acceleration, counter-swing arms, impact, and landing, projected through a 6-DoF floating-base rigid-body/contact dynamics constrained optimizer:

`M(q) qdd + h(q,qd) = S^T tau + J_ground(q)^T lambda_ground + J_ball(q)^T lambda_ball`

with a friction cone, unilaterality, foot slip, CoM/support margins, original 29 velocity/torque/power limits, physically measured wrist–hip clearance, and recovery to 14 seconds. Human timing is a **learned hypothesis** rather than a fixed milliradian joint correction. **This v147 report proposes that controller; v147 did NOT implement it, train it or run it in PhysX**. Frozen thresholds and held-out tests must precede any such optimization.

Physical priority, when GPU is no longer held and disk remains above 20GiB: (1) real positive and negative contact-pair calibration on the native event bus with existing USD collider paths and exact step-clock verification; (2) six **sequential** original controller × {OFF,ON} true PhysX runs using the frozen v147 manifest (seed 1201), with complete real per-step parity and independent native pair review; (3) more varied unseen-ball, seed and recovery evaluations using the strict conjunction of original G1 goals, ≥95% *legal right instep impulse*, ≥3.5m/s ball speed, 29 qdot/actuator limits, zero nonfoot collisions, human-like run-up and complete14s landing.

**Strict new batch results:** real PhysX **0**; new strict original-G1 physical shot success **not measurable**; left-wrist↔left-hip native collider manifold **UNKNOWN**; live OFF/ON parity **NOT TESTED**; professional human movement-to-contact/GRF learning **NOT AVAILABLE**. No proxy gets promoted into a positive physics result.
