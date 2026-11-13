# KineticKick v0.1.146 — Original G1 read-only PhysX evidence bridge and actual 200 Hz coupled dynamics

**Research date:** 2026-11-13. **Frozen parent:** `b8146bb4c6f4e06284def2bfdfd20671ebfa96f4` (v145). **Pre-registered protocol SHA:** `7847215203ef936332dd98b5a597a72d2bf1fb72`; see `docs/V146_FROZEN_NATIVE_RUNNER_200HZ_AND_CPU_FALSIFICATION.md` and `configs/v146_frozen_native_original_runner_bridge.json`. **Status:** CPU implementation and independent historical evidence audit completed; native original PhysX OFF/ON GPU experiment **not executed**. Nothing here establishes a verified wrist–hip collider pair or high-speed human-like instep skill.

## 1. Device, provenance, and stop conditions

Tang Ubuntu device `6d184e3c-bff8-4e01-ab01-139e27bf329a` was online. The main working directory `/home/you/projects/kinetickick` is on dirty `version/v0.1.64` (untracked `=100===`), left untouched; v143 is also an independent active worktree. We used `/home/you/projects/kinetickick-wt-v146`, initially a separate, clean worktree with its own previously committed v146 protocol. 147 local version branches (v0.1.0–v0.1.146) were enumerated; v146 did not yet exist as a GitHub remote branch when checked via the connected repository interface. No old branch was reset or overwritten.

Filesystem: about **30 GiB free**, volume 97% utilized, above the frozen hard stop of 20 GiB but dangerously tight. Existing v120 `pytest` PID **242644**, elapsed >23 hours, still owns **260 MiB RTX4090**; physical GPU utilization is not a measure of exclusive availability. The direct `gpu_preflight()` returned `V146_FOREIGN_GPU_CONTEXT_NO_PHYSICS`. No attempt was made to kill the process, start Isaac Sim, change collision filters, or take the GPU. Shared lock file `experiments/kinetickick_physics_gpu.lock` is used only by the future runner with non-blocking flock. Original large C3D/AMC, ONNX, NPZ, PhysX JSON, and images are protected; original Tang-local evidence is stored under gitignored `reports/`.

## 2. Executable engineering delivered

- `tools/v146_run_original_with_readonly.py`: a reviewable copy of the actual `tools/v080_run_poll_strike.py` runner, not an invented controller. Its diff is limited to opt-in OFF/ON observer CLI, GPU/disk/lock guard, post-update capture, and write-once evidence path. Native action generation, scene configuration, original 20 N nonfoot termination, joint policy, ball/goal, 700×20 ms macro-step schedule, and physical recorder logic remain as originally authored. **Actual trajectory equality is not certified by a source diff.**
- `tools/v146_native_original_bridge.py`: reads **all 29 joint angles, velocities and simulator torque estimates**, root pose/twist, mass-weighted body COM, both foot-link net contact forces, peak nonfoot-link net force, and ball pose/twist after each nominal 5 ms `scene.update`. ON mode copies existing PhysX headers/points, stage ID and actual USD collider-to-rigid-owner map. It does **not** apply `PhysxContactReportAPI` or other stage changes. OFF mode never registers a native callback. The bracket gives only an **uncalibrated scene-update interval hint**, not the exact backend physics substep. It may capture no callbacks if the existing stage does not report them. All trust claims fail closed.
- `tools/v146_real_200hz_coupled_audit.py`: SHA-locks original 3 v54 PhysX result files; extracts exactly 2,800 + 2,800 + 1,028 verified 5 ms samples. Checks all original29 q/dq/torque, mass COM, ball, both feet net forces, nonfoot event; preserves 1 duplicate terminal annotation separately.
- `tools/v146_independent_real_dynamics_check.py`: independent source re-read and row-level numerical comparison against the locked real original JSON; independent qdot ratio and event-window checks; independent first interpolicy divergence in original 29DoF samples.
- `tools/v146_bind_local_assets.py`: source-preserving symlink checks for model, data and experiment assets; current actual paths already resolve to verified original links, without copying or overwriting.
- `tools/v146_plot_real_200hz_failure.py`: produces three factual PNG diagnostics from verified originals, locally only.
- Two v146 test modules exercise positive CPU fakes, absent callbacks, malformed manifolds, wrong USD ownership and stage ID, max packets, 5 ms clock, missing 29 joints, SHA drift, write-once, guarded GPU startup, and real original 6,628 samples.

**Important limitation:** The OFF/ON tests with a deterministic CPU fake only show software non-mutation. They are **not** the six required true PhysX OFF/ON parity episodes, and they cannot certify collision identity or non-invasiveness of a live callback subscription.

## 3. Frozen real-history observations

| Historical v54 policy | Actual complete 200 Hz states | Contact proxy time | Last physical state | 14 s completed | Peak nonfoot net force | Full original29 qdot exceeds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| frozen_v32 | 2,800 | 1.570 s | 14.000 s | yes | 0 N | 0 samples |
| CMU10_02 | 2,800 | 1.570 s | 14.000 s | yes | 0 N | 0 samples |
| CMU10_05 | 1,028 | 1.570 s | 5.140 s | **no** | **25.8901539 N** | 0 samples |

All three peaked at about **0.908129** times the recorded original29 qdot limits; simulator torque estimate peak/recorded effort limit was about 1.0. This is **not** hardware thermal/peak power certification, nor a substitute for the strict v53 multi-ball negative results (all three policies **0/4**). Two 14 s trajectories have old *legacy center-case* success flags, not a new legal high-speed instep PASS.

Around the old CMU10_05 failure, the available original 5 ms frame sequence stops at **5.140 s**; the old named left-wrist/left-hip forces appear in an observation at roughly **5.145 s**, without an aligned complete original29 state there. Opposite 25.89 N net forces are compatible with self-collision, **not proof of an actual PhysX collider pair, path or manifold**. There are **zero newly captured native pairs**. Do not label a late sensor annotation as a 1,029th physical step.

The frozen contact-centered history window is 80 original frames from **1.170–1.565 s** and 250 original frames from **1.575–2.820 s** for each policy. The genuine 1.570 s legacy contact timestamp is a recorded event/proxy, **not** certified instep manifold truth. Net foot sensor force may include foot–ball, foot–robot, and foot–ground contacts; without filtered ground-collider pairs, **neither foot's force is isolated GRF**. Original source also has no calibrated bilateral arm–leg closest distance at the same 5 ms samples; do not invent it.

## 4. Actual interpolicy dynamics; an important negative

All three original center-ball comparisons first diverge in a measured original29 q/dq/COM/foot-force row at **1.405 s**, i.e. original complete 5 ms sample **281**. From that point through the just-before-contact interval (33 shared 200 Hz samples), maximum differences relative to the original frozen_v32 were:

| Compared policy vs frozen_v32 | Body group | Max |Δq| (rad) | Max |Δqdot| (rad/s) | Max |Δ estimated torque| (N m) |
| --- | --- | ---: | ---: | ---: |
| CMU10_02 | both legs | 0.001323 | 0.460273 | 1.176391 |
| CMU10_02 | waist | 0.000648 | 0.048877 | 0.133013 |
| CMU10_02 | both arms | 0.000919 | 0.019598 | 0.024968 |
| CMU10_05 | both legs | 0.001217 | 0.466307 | 1.188902 |
| CMU10_05 | waist | 0.000452 | 0.048802 | 0.130391 |
| CMU10_05 | both arms | 0.001168 | 0.025190 | 0.024930 |

This independently confirms actual physical activation of the old CMU wholebody style reference, **but only roughly milliradian-sized joint-position differences** around original contact. It is neither meaningful elite motion imitation nor an effective human proximal-to-distal force transfer strategy. No adaptation/tuning was performed against these already examined negative histories. No new PhysX outcome was generated.

## 5. Independent experimental audit and negatives

Actual SHA-qualified original source re-read: **6,628 / 6,628** complete 200 Hz rows (2,800 + 2,800 + 1,028), with joint q/dq/torque/COM/ball/foot-net fields independently matched. Exactly **one** terminal metadata row excluded. Independent original interpolicy first-divergence samples all **281 = 1.405 s**; all three frozen windows independently checked at 80 pre + 250 post. Historical original 5 ms sources are not new control experiments; no OFF/ON real parity claims.

Write-once local, ignored data with SHA256:
- `reports/v146_real_200hz_coupled.json`: `fbe74cc06d809821fa0593079a986b208c146598b12e054dbeefd90705eded0b`.
- `reports/v146_independent_original_200hz.json`: `0eb6343b16e477b031e905ad7a1709d2e0c2e0fa1e7cff26dfdd2dfbe19d2091`.
- `reports/v146_kinetic_diagnostics_full14_dynamics.png`: full 14 s or censored coupled quantities.
- `reports/v146_kinetic_diagnostics_last_contact_evidence_gap.png`: failure at 5.135/5.140 vs unmatched named event at 5.145 s.
- `reports/v146_kinetic_diagnostics_precontact_body_activation.png`: weak actual 29DoF pre-contact activation.

For a clean new local output name (the current files must **never** be overwritten):
```bash
cd /home/you/projects/kinetickick-wt-v146
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools MPLBACKEND=Agg
PY=/home/you/conda_env/env_isaaclab/bin/python
$PY tools/v146_real_200hz_coupled_audit.py --output /home/you/projects/kinetickick/reports/v146_reaudit_NEW.json
$PY tools/v146_independent_real_dynamics_check.py --input /home/you/projects/kinetickick/reports/v146_reaudit_NEW.json --output /home/you/projects/kinetickick/reports/v146_independent_NEW.json
$PY tools/v146_plot_real_200hz_failure.py --input /home/you/projects/kinetickick/reports/v146_reaudit_NEW.json --prefix /home/you/projects/kinetickick/reports/v146_plot_NEW
$PY -m pytest -q tests/test_v146_native_original_and_real_dynamics.py tests/test_v146_independent_and_asset_bindings.py
```

New v146 unit and historical real-source tests: **47 passed**. Expanded CPU regression: **34 modules (v124–v146), 871 passed / 2 skipped / 0 failed** with 2 existing SciPy SLSQP bound-clipping warnings. This is a **CPU-only subset**, not the full IsaacLab suite or real native backend integration. The existing v120 full pytest should remain protected until the resource owner resolves it.

## 6. PhysX go/no-go and falsifiable research plan

The scientific objective remains high-quality human-derived **full-body timing and force-chain coordination** on the original 29DoF G1, not single-foot residuals or a proxy MSE. A future support-aware approach should learn compact time-varying fullbody latent coordination from qualified professional instep footage/mocap and optimize it under rigid floating-base equations, unilateral ground contacts, friction cones, actuator torque/power and qdot, safe arm–hip geometry and landing balance. The human-derived terms must be *causal hypotheses* verified with held-out subjects and target-ball scenarios, not retrofitted to these known failures. Existing two CMU samples from one subject contain no verified impact time or synchronized GRF, so this batch **does not train** a professional fullbody force model.

Before claiming left wrist–left hip PhysX contact **or** read-only noninterference, run a separate isolated positive/negative native-report calibration using trusted collider paths (including a known contact and a deliberately non-contacting pair). Then, only after v120 exits and disk remains ≥20 GiB, execute pre-frozen 3 original policies × {OFF, ON}, **six sequential true PhysX episodes**, same seed/ball/config, with all 200 Hz states including 29 q/dq/torque, foot net force and independent ground-only GRF labels if measured, base/COM/ball, and original termination. Run exact OFF/ON common-state equality before trusting observations. Do **not** silently mutate USD report APIs to make events visible; if native reporting absent, label native truth UNKNOWN and stop promotion.

For every new candidate require conjunctive gates: at least 95% legal instep impulse, valid goal, ball exit speed ≥3.5 m/s, all original 29 velocity and actuator limits, no nonfoot collision, legitimate continuous run-up and follow-through, and complete safe 14 s landing and recovery. Count early censored episodes as failed/unknown according to their missing gate, never as a successful recovery. All negative examples and original untouched PhysX sources remain preserved.

**Strict v146 status:** new PhysX episodes = **0**; new valid high-speed instep success = **not measurable**; real native wrist–hip collider manifold = **UNKNOWN**; real ON/OFF physical equality = **not evaluated**; qualified elite human instep contact/GRF supervised training = **not available**. Those are hard experimental dependencies, not narrative omissions.
