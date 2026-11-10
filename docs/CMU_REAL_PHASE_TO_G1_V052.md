# KineticKick v0.1.52 | Real 120Hz CMU kick → differentiable G1 phase reference → real PhysX

This is an **executed research prototype** on Tang, not a verified human-like explosive kick controller or elite-player biomechanics. It keeps the previous free-root G1, PhysX sphere/foot collision, PD actuators, original v32 recovery PPO, and physical pass/fail criteria untouched. Offline IK/phase fitting uses real CMU optical-body mocap, but the human time origin is only inferred outgoing ball-marker motion, **not** measured first contact, ball impulse, muscle power, foot surface, or synchronized human-to-robot impact.

## Actual hardware / environment

- Tang, Ubuntu, NVIDIA RTX 4090 Laptop GPU 16GB. Project directory: `/home/you/projects/kinetickick`.
- Python: `/home/you/conda_env/env_isaaclab/bin/python`, under the existing IsaacLab environment. Dependencies include PyTorch, NumPy, SciPy, pytest, and pinned CMU package in `requirements/cmu-mocap-v050.txt`.
- Use the same frozen original G1 29DoF URDF, physical soccer assets and postshot checkpoint as `v0.1.51`; **do not install a conflicting IsaacLab or retrain/rewrite old checkpoints**.
- CMU C3D/AMC, the two `v48` original complete G1/PhysX traces, reference NPZ, and long 5ms physical records all live **only on Tang** and are intentionally Git-ignored. A fresh Git clone alone is not a self-contained simulation/data package.

## Input contract and comparison

The frozen protocol `configs/cmu_g1_phase_projection_v052.json` SHA-verifies 19 exact sources. Eligible CMU original source clips are **10_02** and **10_05** (both from one person, not two independent subjects). Of six pose features, only approximate shared segment-direction cosines are transferred (left/right knee, elbow, thigh relative to trunk); the anatomically unqualified ankle-toe human feature is excluded.

Both human and G1 physical pose are represented on 20 intervals of 20ms spanning the pre-event 0.40s window. They have **different events**: human optical ball-marker launch onset vs real G1 PhysX first contact. This is a phase alignment proxy with explicit limitations, not certified matched motion/contact.

The differentiable 29DoF URDF forward kinematics uses six pose channels with gradients; a pre-fixed 96-step optimizer adds *bounded, relative* human phase change to the original G1 pose while regularizing joint motion, support-foot motion, swing-foot displacement and time smoothness. Export `700 × 29` 50Hz reference with amplitude at most `0.006 rad`, slew at most `0.0012 rad / 20ms`, under original URDF joint limits, and gate out sufficiently before robot contact. These software envelopes do **not** prove actual qdot, foot/ball action or recovery safety; those require true PhysX.

## Commands, in safe original order

All commands run from Tang project directory. Start with frozen source verification and do not overwrite existing artifacts. Scripts intentionally refuse to replace experimental sources/results.

```bash
cd /home/you/projects/kinetickick
export PYTHONPATH=tools:$PWD/third_party/HumanoidSoccer/source/whole_body_tracking
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/you/conda_env/env_isaaclab/bin/python

# Offline: real optical CMU relative-phase → physical G1 URDF 29DoF
$PY tools/build_cmu_g1_candidate_v052.py --clip 10_02
$PY tools/build_cmu_g1_candidate_v052.py --clip 10_05

# Real original 14s PhysX: run only one GPU IsaacLab process at a time.
$PY -u tools/run_placement_strike.py \
  --config configs/cmu_real_120hz_10_02_physx_v052.json \
  --output reports/v052_cmu_10_02_center_physx.json
$PY -u tools/run_placement_strike.py \
  --config configs/cmu_real_120hz_10_05_physx_v052.json \
  --output reports/v052_cmu_10_05_center_physx.json

# Physical negative control: same input reference present but disabled;
# physical variables shared with frozen v32 source must match exactly.
$PY -u tools/run_placement_strike.py \
  --config configs/cmu_real_120hz_10_02_physx_zero_v052.json \
  --output reports/v052_cmu_10_02_center_zero_physx.json

# Read-only signed original PhysX sample inspection and full strict gating:
$PY tools/audit_real_cmu_physx_v052.py --clip 10_02
$PY tools/audit_real_cmu_physx_v052.py --clip 10_05
$PY tools/write_cmu_real_physics_report_v052.py
$PY -m pytest -q tests
```

The generation and reporting commands abort when their intended outputs already exist. The original v52 pass/fail results and report have already been saved and hashed on Tang; **do not rerun over these files**. For a reproducibility replay, use a fresh isolated checkout and new output namespace with the same frozen assets; no original source or previous result should be overwritten.

## Observed original run results (2026-11-11)

| Check | CMU 10_02 reference | CMU 10_05 reference |
| --- | ---: | ---: |
| Offline six-pose proxy objective reduction | 26.627% | 23.441% |
| Nonzero 50Hz reference-generation frames | 17 | 17 |
| Actually applied new 50Hz G1 joint-reference frames | 5 | 5 |
| Real peak change in observed G1 joint angles | 0.001303 rad | 0.001207 rad |
| Real PhysX right-instep impact, ball into goal | Yes | Yes |
| Ball horizontal speed | 4.6341 m/s | 4.6391 m/s |
| Real 5ms joint speed-limit exceedances | 0 | 0 observed before termination |
| Real 5ms physics samples | 2800 (14s) | 1029 (~5.14s) |
| Final recovered standing | Yes | **No**; `physical_nonfoot` termination |
| Full task + safety acceptance | **PASS** | **FAIL** |

A ball-in-goal without stable recovery **is a failed complete humanoid kick**. The 10_05 trial is retained as a valuable immutable negative example. Do not tune it and call it a new held-out result. Only the original center-ball configuration and one rollout per human clip were tested; no generalization, human naturalness or training improvement may be claimed.

The disabled ZERO run and previous v32 share exactly identical recorded physical data in all common fields, but the previous report includes seven extra telemetry fields absent from this new output. This is a verified shared-field parity, **not byte-for-byte parity of the two full JSON files**.

## Files and provenance

- Research report and experimental verdict: `reports/2026-11-11-v052-real-CMU-phase-controlled-G1-PhysX.md`
- SHA-frozen original data/source configuration: `configs/cmu_g1_phase_projection_v052.json`
- Differentiable generation: `tools/build_cmu_g1_candidate_v052.py`
- Read-only physical auditor: `tools/audit_real_cmu_physx_v052.py`
- Text-only report generator: `tools/write_cmu_real_physics_report_v052.py`
- Original 10_02, 10_05 physical configs and disabled ZERO config: `configs/cmu_real_120hz_*_physx*_v052.json`
- Tests: `tests/test_real_cmu_phase_v052.py`, `tests/test_real_cmu_physx_v052.py`
- Git-ignored true evidence: `experiments/cmu_real_120hz_phase_v052/`, `reports/v052_cmu_*.json`. Their full SHA and per-case conclusions are in the committed human-readable report.

## Next hard gates

Get properly labeled **real human foot–ball contact, instep surface, GRF/support, pelvis/torso/swing-leg/arm phase, outgoing ball speed/spin** with sufficient diversity, then learn or optimize motion including **follow-through and landing** rather than adding 1-milliradian precontact variations alone. Design matched physics pairwise trials with untouched G1 and task envelope and held-out player/ball cases, measure control-phase naturalness by blinded human raters **only among physically valid complete kicks**. Use the saved 10_05 `physical_nonfoot` result as a regression test, not an unreported exclusion.
