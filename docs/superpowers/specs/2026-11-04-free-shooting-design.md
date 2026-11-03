# Free-root multi-surface shooting

## User outcome

The user confirmed the full simulated sequence: approach the ball, shoot using instep, inside, and outside foot surfaces, then recover standing. Continue improving toward this outcome. Use Tang's existing Isaac Sim 4.5 / Isaac Lab installation and the unmodified physical G1 foot; preserve local research assets. This work is on `version/v0.1.1`.

## Observed starting point

- Root-V2 tracks a 260-frame reference with 166 actor observations; it observes neither ball nor goal.
- Fresh baseline on 2026-11-04: one nominal trial contacted the ball at step 161, peaked at 1.06753 m/s, and survived 310 steps. Ball displacement was (-1.28112, -1.49127) m, not an aimed goal.
- Existing live health last recorded no contact. A nominal headless contact is not evidence of reliable live repeatability.
- PAiD's public `ckp/policy_30000.onnx` at upstream commit `e72e470230047dedaf66df0983f1d0ab746faeb5` has a 160-dimensional ball/goal-conditioned actor and two 128-unit LSTM layers. Its exported motion matches standard-001 exactly at frame 0 and contains 260 frames.
- The ONNX SHA-256 is `489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`. Model stays local. ONNX Runtime 1.23.2 was installed only into ignored `experiments/runtime/onnxruntime`; no existing simulator packages were changed.

## Approaches and builder decisions

1. Continue local Root-V2 training with ball observations and rewards: compatible with existing training but requires learning the missing task.
2. Evaluate and adapt the upstream ball-conditioned policy first: selected engineering approach, using existing training knowledge before spending GPU time. A successful upstream baseline still needs surface-specific control and verified recovery.
3. Overlay old fixed-root joint curves on walking: rejected based on previous 37DoF experiments which contacted the ball but fell.

This is a simulation task; onboard camera perception and physical robot deployment are not part of this iteration. Simulator state supplies ball and goal positions. New training or bounded residual/trajectory adaptation is allowed when measured failures justify it. Do not invent naturalness or robustness claims.

## Physical and evaluation contract

- Free root, original G1 collision geometry, one 0.43 kg / 0.11 m radius football per environment.
- Robot and ball pose/velocity writes only during trial initialization. Never inject a kick impulse, teleport during a trial, or hide termination behind resets.
- Approach must reduce root-to-ball distance and move the robot at least 0.25 m before valid impact. Record all early ball contacts, falls, and wrong surfaces.
- Use a declared goal plane and aperture, initially 1.6 m wide and 1.1 m high, at least 1.5 m beyond the initial ball. Count a goal only when the complete ball crosses the aperture after verified foot contact, using segment interpolation.
- A valid recovery requires 1.0 s of continuous upright, low-speed standing following the shot: root height >=0.55 m, tilt <=25 degrees, planar root speed <=0.30 m/s, foot support >=80 N, non-foot contact <20 N, no termination. Measure this explicitly; mere survival of a reference is insufficient.
- Confirm contact through PhysX foot-ball contact data every 5 ms, and log actual contact points and normals in the corresponding foot frame. Pin the seven-cylinder PAiD URDF SHA-256 to `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`. A requested style label is never evidence of actual instep/inside/outside contact. Ambiguous first contact or toe/sole impact does not pass a requested surface. Preserve every simultaneous contact point separately and integrate all positive normal forces; the requested surface must also supply at least 50% of the measured contact impulse.
- First demonstrate each style at a nominal start, then freeze parameters and evaluate five distinct ball/goal offsets per style, repeated with two reset seeds: 30 trials. Initial target is >=4/5 successful conditions per style in each seed. Record every failure and the exact evaluated range.
- Use the same initialization, physics, sampling, policy state reset, and success calculation in headless and live paths.

## Components

- A strict ONNX adapter verifies observation size/order, joint names, recurrent state reset, finite values, and reference outputs.
- A pure geometry/episode evaluator handles goal crossing, contact classification, approach, and standing windows, with numerical regression tests.
- A shared simulation harness initializes scenes and measures per-trial trajectories without rollover teleports or evaluation-time writes.
- Style parameters or policies are chosen from measured physics. If the upstream baseline is inadequate, add explicit style, ball and target conditioning to the existing training route, preserving frozen baselines.
- A live viewer and a reproducible batch command expose only measured capabilities. No final-success announcement before all three styles pass the complete sequence.

## Bounded execution

Run one GPU simulator at a time. Start with a short baseline, then a small candidate grid. A training run must declare its hypothesis, seed, horizon and output path before launch, and be evaluated against the previous frozen candidate. After two ineffective attempts at the same failure class, diagnose before trying another. Keep checkpoints and raw traces local; commit code and written evidence on the version branch.
