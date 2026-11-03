# Free-root shooting implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Approach, shoot with three verified foot surfaces, and recover standing in free-root simulation.
**Architecture:** Evaluate the official ball-conditioned policy through a strict adapter, establish a shared physical evaluator, then adapt control only from recorded failures. Reuse the same runner for batch and live demonstration.
**Tech Stack:** Python, NumPy, PyTorch, ONNX Runtime, existing Isaac Lab/PhysX.
**Spec:** `docs/superpowers/specs/2026-11-04-free-shooting-design.md`.

## Global constraints

- Free root; original foot; ball mass 0.43 kg and radius 0.11 m.
- No robot/ball pose or velocity writes after trial initialization.
- Goal aperture 1.6 m wide and 1.1 m high; goal initially >=1.5 m beyond ball.
- Recovery: continuous 1.0 s, root >=0.55 m, tilt <=25 degrees, planar speed <=0.30 m/s, no termination.
- Each style must pass >=4/5 distinct conditions per seed over two seeds, after tuning parameters are frozen.
- One GPU simulator at a time; no global dependency upgrades; local assets remain ignored.

## Review focus

- RNN state retained between trials can change an identical initial condition: adapter reset determinism test plus repeated live trials.
- Near misses or partial aperture crossings can look like goals: interpolation and ball-radius boundary tests.
- Auto-reset can hide a fall: first termination is a permanent trial failure, tested in the evaluator.
- Style labels can conceal toe/sole contact: test known foot-frame normals and reject ambiguous normals.
- Observation order can silently mismatch checkpoint dimensions: verify metadata and joint ordering before physical evaluation.

### Task 1: Measured evaluation and policy integration

**Files:** `tools/soccer_onnx_policy.py`, `tools/free_shooting_metrics.py`, `tests/test_free_shooting.py`, `tools/eval_free_shooting.py`.
**Interfaces:** `OnnxSoccerPolicy(path, num_envs)` with `reset()` and `step(observations, frame)`; `goal_crossing(previous, current, goal, forward, width, height, radius)`; `ShotTracker.update(sample)` and `.summary()`.

- [ ] Write failing numerical tests for aperture crossing, wrong direction, reset/fall failures, standing duration, and three surface directions.
- [ ] Implement the evaluator and strict recurrent adapter; verify deterministic reset against the real local model when available.
- [ ] Add one physical evaluation command with ball and goal positions; match the checkpoint's 160 observations and recompute observations after initializing the reference and ball.
- [ ] Run the existing 43 tests plus new tests, then one nominal official-policy trial and a small position grid. Record actual outcomes.
- [ ] Commit the measured baseline and verified integration.

### Task 2: Surface control and recovery

**Files:** extend `tools/eval_free_shooting.py`; add `configs/free_shooting_v1.json` and bounded training/control modules only as evidence requires.
**Interfaces:** style command `instep`, `inside`, `outside`; frozen configuration identifies policy, reference, adaptation parameters, and initial-condition range.

- [ ] Diagnose observed contact, direction, stability and recovery failures from Task 1.
- [ ] Add regression tests for the identified failure before modifying control.
- [ ] Run bounded candidate experiments, preserving each candidate manifest and comparing measured complete-chain success.
- [ ] Freeze the best candidate for each style only after actual surface contact and recovery are verified.
- [ ] Commit the implementation and honest progress evidence; do not mark the goal complete if any style remains unverified.

### Task 3: Frozen acceptance and live demonstration

**Files:** `tools/show_free_shooting_live.py`, shared runner, `reports/2026-11-04-free-shooting.md`, README.
**Interfaces:** live and batch use the same control/state reset and `ShotTracker` semantics.

- [ ] Freeze the 30-trial matrix and evaluate every trial without discarding failures.
- [ ] Run repeated live trials and inspect a rendered sequence for each passing style.
- [ ] Run appropriate CPU regression checks, review the whole branch, resolve material findings, and push the version branch with recorded evidence.
- [ ] Report only the verified success range, remaining limitations, and local artifacts.
