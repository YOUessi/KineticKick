# KineticKick v0.1.149 — Native callback clock-aliasing falsification and contact-report eligibility

**Executed:** 2026-11-13, Tang Ubuntu / RTX4090, safe CPU only. **Parent:** clean v148 `4505e095ec5e13e9226d35adcfb9ba368e053156`. **Protocol pre-frozen independently:** `8f6670c5942c330cc91679e798e1911e41e0606e`. This is a genuine implementation, source audit and batch experiment, **not** a new PhysX episode or a trained natural kick.

## Scientific root-cause finding

The v146 original-runner sidecar surrounds `env.scene.update(dt)` with its copy-only contact bracket. Its callback *rejects* native contact events outside that scene-update bracket (`tests/test_v146_native_original_and_real_dynamics.py::test_callback_outside_scene_update_is_rejected`). But the real PhysX step and scene update are distinct; NVIDIA documents that contact-report callbacks arrive after simulation steps, while IsaacLab's step lifecycle performs physics stepping before scene update. Therefore **a real event may be discarded by the v146 observer merely because it arrives outside that bracket**. This is verified as a **source-level instrumentation bug risk**, not measured loss of a true Tang wrist–hip collision.

Separately, NVIDIA's native contact-report subscription requires pre-existing `PhysxContactReportAPI` on a participating reporting actor or articulation root. Merely calling `subscribe_contact_report_events` does not enable reports from an otherwise unregistered actor. The actual original G1 stage's report eligibility remains **UNKNOWN** without a new live read-only USD inspection; silence can never prove absence.

Official documentation checked on November 13, 2026:
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html
- https://docs-prod.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html
- https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.html

## Actual work in this batch

1. `tools/v149_native_callback_capture.py`: immediate, copy-only capture of original raw Vec3 contact impulses, manifold points and collider/actor paths **before, inside, between and after** scene update; store callback count separately from original 5ms state count, and keep only **uncalibrated adjacent candidate steps**. Orphans are preserved, not promoted into fabricated 29DoF rows. Malformed callbacks produce explicit errors rather than silently suppressing physical state sampling.
2. `existing_schema_census` and `live_stage_with_schema_census`: read-only `HasAPI` queries of exact collider-owner actors and candidate reporting articulation root. Missing/ambiguous owner or absent schema yields `UNKNOWN/NOT_OBSERVABLE`; a child collider attribute alone does not establish that the owning rigid actor is reporting. The implementation never calls `PhysxContactReportAPI.Apply`, modifies USD prims or changes collision filters.
3. `tools/v149_run_original_anyphase_readonly.py`: an exact observer-only transformation of the existing v146 copied original runner. The original v080 physics/action/termination path, exclusive GPU mutex, preflight and write-once evidence guards are unchanged; source-code equality is tested against the explicit rename transform. New optional CLI `--v149-observer-mode` / `--v149-sidecar`.
4. `tools/v149_phase_matrix.py`: predeclared 3 frozen policy **names only** × 4 callback arrival phases × 2 callback payload types × 2 mocked pre-existing schema states, 48 synthetic CPU cases. Also SHA-rechecks three historical real v54 PhysX files.
5. `tools/v149_independent_phase_audit.py`: independent, source-separated 48-case denominator/provenance/clock/negative-control verification (without calling the primary recorder or matrix generator).
6. `tools/v149_plot_callback_hazard.py`: standalone source-CPU callback-loss figure, explicitly marked **synthetic**, saved local. Two new test modules cover 48 matrix cases, adversarial mutations, manifold buffer copy, unobservable schema, duplicate callback, orphan, no native tick certification, GPU guard, source byte-equivalence and plot write-once behavior.

## Recorded source-only batch results

| Experiment | Verified outcome | Scientific status |
|---|---:|---|
| Preregistered CPU callback conditions | **48/48** | Synthetic, not PhysX |
| Valid Vec3/native-shaped callbacks preserved | **24/24** | Source-only copy tests |
| Invalid scalar impulse payloads rejected | **24/24** | No fake Vec3 coercion |
| v146 bracket would retain (timing-only fixture) | **6/24** | 18 illustrate lost-event risk, *not* 18 real collisions |
| v149 any-phase copied (timing-only fixture) | **24/24** | True native step still uncalibrated |
| Independent matrix audit | **48/48** | Cannot attest runtime source |
| New original G1 PhysX episodes | **0** | GPU occupied |
| New legally successful high-speed shots | Not measurable | Do not infer success |

The 4 callback phases are named `BEFORE_FIRST_SCENE_UPDATE`, `INSIDE_SCENE_UPDATE`, `BETWEEN_SCENE_UPDATES` and `AFTER_TERMINAL_SCENE_UPDATE`. Every retained packet carries raw native-shaped paths plus candidate 5ms steps, and every result still explicitly says `native_pair_certified=false`. A callback from inside the scene-update bracket is **not** inherently a calibrated native step either.

Raw synthetic/historical SHA-bound report kept Tang-local: `reports/v149_phase_48_synthetic_and_historical6628.json`, SHA256 `6237dd1404d7d419a0ece0dd00709c969530af2983627b34ad27cf3654a16243`. Independently verified result: `reports/v149_independent_48_audit.json`, SHA256 `f311c0827302506fa06f61e08a2fbdc23acf371e4f7afcf52f8b80a2cb7226fd`. Diagnostic chart `reports/v149_synthetic_callback_arrival_censor.png` remains local and ignored.

## Real historical physical and original29 data custody

Original v54 PhysX files were actually read, source-SHA verified and reconciled: **2800 frozen_v32 + 2800 CMU10_02 + 1028 CMU10_05 = 6628 genuine historical 200Hz G1 states**, not newly executed rounds. The failed CMU10_05 record ends at the last complete original29 state at 5.140s. The separately named left wrist/left hip net-force observation at ~5.145s does not contain same-tick original29 or actual native manifold. The ~25.890153N equal/opposite net forces do **not** prove the two collider owners touched. Root tilt/height alone cannot certify a fall.

Original v53 strict 4-ball-position results remain frozen **0/4** for each of frozen_v32, CMU10_02 and CMU10_05. No synthetic contact, score, figure or CPU test may bypass ≥95% legal right instep impulse, ≥3.5m/s ball velocity, actual goal, all 29 qdot/effort limits, zero nonfoot collision and complete 14s recovery. No professional soccer impulse/GRF truth is available in the two CMU same-subject 120Hz clips.

## Safety audit, test results, reproducibility

At execution Tang was online, disk ~**30GiB** free (above 20GiB minimum); existing v120 pytest **PID 242644** retained ~**260MiB** on RTX4090 and was not terminated/signaled. No new IsaacLab task, no GPU collision, no old C3D/NPZ/model/PhysX overwrite, no force push and no modification to the dirty root v64 or other worktrees.

The *first* expanded CPU test found **23 setup errors** caused by missing old *gitignored* v148 JSON in the fresh branch. All errors were preserved in `reports/v149_expanded_cpu_first.log`. Re-established **read-only ignored symlinks** to the unchanged v148 evidence and verified the original v148 SHA256 `cf081b225bf8d247321fb76a1bb21e1424d7e433611525ea0611dd2706f8d152`; the rerun finished **1084 passed, 2 skipped, 2 existing SciPy warnings, 0 failed**, with log `reports/v149_expanded_cpu_verified.log`. Additional frozen source/plot adversarial tests were added afterward: **88 v149-specific tests passed** in `reports/v149_specialized_final.log`. The final expanded CPU suite including the two additional adversarial tests passed: **1086 passed, 2 skipped, 2 pre-existing SciPy warnings, 0 failed** (40.55s; `reports/v149_expanded_cpu_final.log`). These are CPU-only results.

To reproduce without overwriting, always change the output path suffix:
```bash
cd /home/you/projects/kinetickick-wt-v149
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v149_phase_matrix.py --output reports/v149_matrix_NEW_UNIQUE.json
$PY tools/v149_independent_phase_audit.py --input reports/v149_matrix_NEW_UNIQUE.json --output reports/v149_audit_NEW_UNIQUE.json
$PY tools/v149_plot_callback_hazard.py --input reports/v149_matrix_NEW_UNIQUE.json --output reports/v149_plot_NEW_UNIQUE.png
$PY -m pytest -q tests/test_v149_callback_phase_and_schema.py tests/test_v149_independent_evidence_and_runner.py
$PY -m pytest -q tests/test_v12*.py tests/test_v13*.py tests/test_v14*.py
```

## Gate to the next genuine physical batch and full-body research

**A unresolved:** the real wrist–hip PhysX manifold is still **UNKNOWN**. First validate **existing** report API coverage read-only at runtime, exact callback/native-step association via true controlled positive+negative cases, then original three controllers × OFF/ON (6 serial real runs), 200Hz original29-state strict parity, all report-API and collider paths, missing-step censorship. If the unmodified original stage has no report API, a true native event read-only witness is impossible on it; a separately calibrated USD world is a **different condition**, not non-invasive evidence.

**B partially historical only:** 6628 original full-body samples are authentic; no missing 5.145s row or ground-only GRF may be invented. **C/D blocked:** full-body floating-base, unilateral contact/friction, torque- and collision-constrained coordination and a multi-subject professional-human motion latent remain proposed **not trained or physically tested**. Freeze held-out ball/seed/subject tests prior to any optimization; maintain all original strict instep/velocity/goal/safety/recovery gates.

**Final scientific claim:** v149 removes a proven *source-level* callback discard path and adds an observability gate, but **does not prove the original wrist–hip contact, a natural high-velocity instep kick, or physically learned human power sequencing**.
