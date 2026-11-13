# V149 frozen protocol: reject callback clock aliasing and false contact absence

Parent branch `version/v0.1.148`, exact SHA `4505e095ec5e13e9226d35adcfb9ba368e053156`. This protocol must be committed independently **before** implementing v149 tools or running outcome-dependent experiments.

## Problem / pre-registered hypothesis
v146 hooks `env.scene.update(dt)`, not the native physics step, and its callback handler rejects events outside that narrow bracket. Isaac Lab explicitly performs physics stepping before scene update; NVIDIA documents that contact-report callbacks are dispatched after a simulation step. Thus physically real events could arrive when v146 has no pending step. Its callback loss would invalidate pair absence inference even if the collider/actor paths are available.

Further, NVIDIA's Contact Reports API requires **existing** `PhysxContactReportAPI` on the relevant USD prim(s). Subscribing does not itself enable reporting. There must be runtime read-only schema inspection and positive/negative calibration. This batch must NOT add this API, change collision filters, or perturb the original control policy. No event in an unobservable stream is **UNKNOWN**, not a negative contact label.

## Actual engineering protocol
1. New independent worktree and branch, immutable v148 parent; no editing dirty root or other branches. Audit GPU contexts and minimum 20 GiB free space.
2. Implement a copy-only callback observer which accepts events BEFORE, INSIDE, BETWEEN and AFTER scene-update bracket, copies borrowed native PhysX vectors and USD paths immediately, stores a conservative adjacency step-candidate set, and never upgrades this set to exact native step. Store an event even if no matching physical state exists; mark it orphan, never interpolate a fake 200Hz state.
3. Implement a read-only USD stage `PhysxContactReportAPI` **eligibility census** by actual prim path; report missing actor/owner as `NOT_OBSERVABLE`, never apply schema. Keep post-contact 29DoF and full 14s coverage separately.
4. Test 3 frozen policy identifiers × 4 callback phases × 2 header outcomes × 2 schema eligibility states = 48 CPU simulations **strictly synthetic and never physical episodes**, plus edge cases (aliasing, duplicate callbacks, empty events, zero impulse, orphan, immutable data, memory caps, nonfinite fields).
5. Independently re-SHA-lock historical v54 2,800+2,800+1,028 real 200Hz states. This is REANALYSIS, not new PhysX.
6. Verify OFF/ON controls and exact 200Hz parity remain mandatory **for real future hardware**, with source-only tests unable to mark runtime calibrated. Full original29 speed, joint, contact, legality, goal, ball speed and recovery gates remain untouched.
7. Generate GitHub-only reproducible source, tests, small config, report; large JSON and images remain Tang-local and gitignored; validate by full available CPU suite. Record negative results unedited.

## Go/no-go
GPU remains exclusive. If foreign PID still holds RTX4090, do **zero new** IsaacLab/PhysX and avoid terminating it. Runtime-native contact pair remains UNKNOWN until a truly calibrated existing-report API, live positive+negative controls, exact OFF/ON state parity and 3 original controller runs succeed. Two same-subject CMU clips without force plate or true professional instep timestamps do not authorize training claims.

References consulted 2026-11-13: NVIDIA Omniverse Contact Reports documentation (https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html); IsaacLab ManagerBasedRLEnv stepping documentation (https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.html). These describe API semantics, not a live Tang result.
