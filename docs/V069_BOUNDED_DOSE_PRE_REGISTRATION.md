# KineticKick v0.1.69 | frozen dose-route falsification — NOT human-motion learning

Freeze parent: ae25bb053e2f672a57cc1ce950d1c91161b0081c (v0.1.67). The independent version/v0.1.69 worktree was created clean after checking existing branches, v65/v66/v68 unfinished worktrees, GPU/disk. Historical CMU, C3D, ONNX, checkpoints, PhysX JSON never modified. v68 remains dirty in its own worktree and is not altered.

## Why this is a real causal-fidelity test

The v68 trial of nominal 2x dose has a confirmed silent-route defect (the high task was never activated) and its static requested amplitude 2 × 0.0562073514 rad exceeds the 0.075 rad safety envelope. A missing route means zero valid high-dose physical observations even if PhysX emitted 2800 frames. We do not retrospectively edit v68 evidence. v69 tests whether a visibly routed, predeclared 1.25x waveform enters original free-root 29DoF G1 physical actuation, without exceeding original 75mrad amplitude or changing the 6mrad/20ms slew.

This is SYNTHETIC closed-loop temporal falsification, not optimization from v68 results, not trained from professional footballers. 1.25 chosen solely from existing 75mrad actuator envelope: 0.0562073514 × 1.25 = ~0.0702592 rad. Time reversal of the same amplitude samples is a directionality negative control, not a human truth label. Original v32 baseline and v63 weak synthetic phase are the other controls.

## Exact source-locked twelve-case experiment

Full order, original config hashes, script hashes, seed, ball offsets and velocity perturbations frozen in configs/g1_bounded_dose_batch_v069.json. Three NEW independent physics conditions/seeds each with four policies v32/low/dose/dose_reverse = twelve original GPU G1 runs. Conditions are the only intended new test cases, not validation for choosing 1.25 or tuning waveforms. Within-context paired comparisons are meaningful; three seed-context pairs do not make a randomized population.

Original 29DoF G1 CUDA PhysX, unchanged expert/PD, unchanged ball/ground/friction/nonfoot threshold; 200Hz original state, 50Hz policy, perturbed true seeded reset, safe_full observer. Canonical root GPU flock plus GPU compute idle check, ≥20GiB disk gate, tracked tree clean, existing raw/log write-once, SHA256 tracked.

Hard PASS: original wrapper source verified, original 29DoF true 200Hz, requested measured instep impulse ≥95%, ball actually enters goal, real 5–110ms post-contact ball speed peak proxy ≥3.5m/s, all original 29 qdot limits, no ≥20N nonfoot terminal event, real post-contact ≥1.25s, complete 14s and recovered. Original wrapper alone is NOT adequate: v67 documented illegal 72.98% instep wrapper PASS. Early termination must not produce fabricated recovery frames.

Routing proof: static route/marker tests, exact DoseTask class/module, 1.25x original reference amplitude, static 75mrad bound, source SHA, runtime tag from actual task after native PhysX execution, nonzero actual action steps and measured independent 29q contrasts. Named dose JSON without verified runtime tag is INVALID even if ball reaches goal. Identical measured trajectories reported rather than disguised as an improvement.

## Research question and constraints

H0: increasing this SYNTHETIC wholebody pre-contact waveform does not reliably increase legal instep ball velocity or natural 29DoF kinetic-chain sequence on new scenes. Twelve runs can falsify gross silent-route/no-op software claims, NOT establish learned human naturalness. Qualified human instep contact timing, synchronized ground reaction force, multisubject professional C3D, and original GPU wrist-hip real manifold remain missing. v68 read-only USD collider paths/net-force symmetry are not direct pair evidence. No collision filter or physical model changed to turn FAIL into PASS.

Frozen code: tools/phase_dose_v069.py, tools/run_placement_strike_v069.py, tools/run_v069_dose_batch.py, tools/audit_v069_dose_and_hard_gates.py, tests/test_v069_dose_route.py. The final report/plots are read-only post-registration artifacts: frozen runner/protocol will not be edited after source lock.
