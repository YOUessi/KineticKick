# v0.1.31 isolated support-reference experiment

Parent: published v0.1.30 `6420c4a5d24983690e6b543331ffbd629dff5abd`. This protocol is declared before the first v31 physical run. The version branch was created before implementation. It is a falsifiable spatial-reference experiment, not acceptance of P0 or a trained coordination generator.

## Evidence and competing explanation

The known `heldout_left_near_slower` case has a second right-foot landing at 2.245 s followed by knee speed 21.143692 rad/s against the unchanged 20 rad/s limit. The prior support had unloaded. There is substantial ground-relative foot motion and a large asymmetric friction-anchor moment. Strike envelopes and velocity pulses are already zero. The original v28 also has this late failure.

The existing offline reference diagnostic does NOT show uniquely larger reference error in this failure: center has similar errors, and the strong case lands even later without overspeed. Therefore reference-frame mismatch is only a candidate mechanism. Measured support duration, incoming foot approach and pelvis trajectory remain competing mechanisms; do not sweep per-ball coordinates to rescue this hypothesis.

## Fixed intervention

Keep all v6 numbers, policy checkpoints, reference arrays, phase computation, collision geometry and actuator limits. Add only `support_reference_frame` with force 80 N, confirmation 15 ms, transition 200 ms. A foot is force-confirmed, not certified stationary.

Latch the first eligible measured support only after real ball contact, at least 15 ms of recorded separation, the original velocity pulse and absolute/placement/swing windows have ended, and before standing-recovery entry. Select the largest-force eligible foot. Freeze its reference position (including the existing placement anchor shift), actual position and wrapped actual-minus-reference torso-anchor yaw. Do not reanchor until reset.

Apply a quintic C2 interpolation of a common horizontal rigid transform: rotate about the frozen reference support pivot and translate that pivot toward actual support. Preserve Z. Transform only the running actor's virtual waypoints, world anchor orientation/position and world anchor angular velocity. During interpolation include the added yaw derivative in the latter. Joint references, measured state, contact labels, actual football/goal and recovery actor observations remain unchanged. No extra recurrent invocation and no writes to physical state. Apply the transform after the existing placement offset. Untriggered and alpha-zero observation rows must be exactly unchanged.

The smooth transform does not imply a C2 final neural action or a dynamically feasible robot trajectory. No target anchor linear velocity is present in the running observation; this does not authorize relabeling measured root velocity.

## Bounded decision sequence

1. Focused geometry/observation contracts plus existing closed-loop/placement checks; verify all 19 frozen hashes.
2. One 14 s instrumented `heldout_left_near_slower` episode, seed 1201, original 5 ms physics / 20 ms control. Capture ground-detail bins from 2.0 to 2.6 s, including potential timing shifts. Compare against `v030_lag_pose_heldout_left_near_slower.json` and the read-only contact extension. Preserve output and source digests.
3. Check exact physical equality until the first nonzero transform, legal instep/ball speed/follow-through/full recovery, actual velocity limits over the whole episode, handoff force/slip/rotation, and original/effective anchor errors. Reduced reference error alone is not success. If overspeed remains, a fall appears, or the task degrades, stop the spatial-only trial; diagnose once, then use a materially different measured support/trajectory hypothesis.
4. Only if the failing case improves without physical/task regression, run center, the strong combination and the slow case before expanding to the old 18 and the already declared three fresh perturbations. Do not inspect or tune on fresh outcomes early. Keep previously unexecuted entries explicit.

No training or parameter search is part of this first trial. Public-human source work remains separately qualified; incomplete professional-motion data do not satisfy P2. Raw outputs/media/source snapshots remain ignored Tang assets.
