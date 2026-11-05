# Frozen strong-instep research baseline

The user selected a shift to strike research: freeze v0.1.15, add force/coordination measurements and a small cross-position evaluation. v0.1.23 is an evaluation addition atop v0.1.22; the v0.1.15 controller, policies, assets and calibration stay unchanged.

## Approach and bounded scope

Reuse historical 5ms dynamics and 20ms kinematics offline, then execute seven preregistered 14s cases through the unchanged original runner. A thin evaluator can change only the initial ball position or a deterministic initial root velocity offset using the existing reset configuration. It does not alter actions or post-initialization physical states. Re-running everything with new sensors would spend GPU time without adding necessary evidence; tuning during the matrix would invalidate the fixed baseline comparison.

Cases: original center; ball x +/-5cm; ball y +/-10cm; initial world-y root speed +/-0.20m/s. Same seed1201, goal, timing, policies and actuator limits. These are deterministic sensitivity cases, not a statistical estimate of generalization. Start with exact center replay and compare the stored historical trace. Stop only for infrastructure/provenance/measurement errors, not genuine failed kicks.

## Measurements

Use the first actual positive foot-ball force episode, its scalar normal impulse and first subsequent zero-force ball velocity. Keep missing contact/outgoing samples null. Relative to actual contact, analyze the preceding300ms: signed hip/knee/ankle velocities and peak times, fixed foot reference-point velocity, estimated joint power tau*dq and positive/negative work, pelvis yaw and relative waist yaw, left-foot support and ankle-link movement during loaded samples. The contact bin is excluded from preparatory joint peaks/work. Root/foot orientations are measured at20ms, not interpolated into invented5ms observations.

Retain original legal-surface, goal, physical integrity and recovery acceptance. Report all failures and partial components. Estimated actuator work is not hardware measurement or muscle work. Scalar normal impulse is not net vector momentum change; kinematic contribution fractions are not energy fractions. No naturalness score or presumed human peak ordering is introduced. The archived normal-strength run supplies a descriptive single-case comparison only.

## Deliverables

Frozen manifest with source/model/data hashes; isolated evaluation wrapper; tested offline metrics; raw cases, reproducible compact results and research report; scientific plots and selected real preview if useful. Existing assets remain local/ignored. Publish version/v0.1.23 with upstream. No training, human-data acquisition or benchmark comparisons in this baseline task.

## Observed replay refinement

Exact equality is established against an additional original-entry headless run. Historical rendered recovery diverges numerically after the shot, so it is retained as a descriptive archived comparison rather than called bitwise full-episode reproduction. Physical failure remains a measured result; metadata-only terminal status is excluded from integration. Standalone analysis enforces the same source/initial-state checks as matrix execution.
