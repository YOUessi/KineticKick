# v0.1.38 — measured instep-point sweep-direction feedback: rejected physical trial

**Disposition: REJECTED.** A physically measured foot-material-point lateral sweep-velocity controller was implemented over the exact existing v32 frozen robot/ball expert and post-shot recovery. CPU kinematics/tests pass; corrected disabled path reproduces old v32 physics and action history exactly. But the single prespecified active candidate has an actual left-knee velocity-limit violation, then early physical nonfoot termination, no legal full-sphere goal and no post-shot recovery. Original v32/frozen v35/v36/default v5 are never promoted or overwritten.

## Goal and what was actually changed

The prior v0.1.37 signed-ball impulse audit showed that a learned tiny precontact action improved the original left-side miss A by moving the real full contact-force direction toward the goal, but on old miss B it moved the contact direction the wrong way. v0.1.34 had tried target FOOT POSITION feedback and was rejected. We tested the separate, explicitly limited hypothesis that a finite-horizon correction for **MEASURED SWEEP VELOCITY of a defined right-instep material point**, rather than terminal foot position or foot-face yaw, can improve direction.

The controller uses the measured right ankle-roll link position/linear/angular velocity, a fixed .115/0/-.025m local point on the instep proxy, actual ball velocity and measured ball-to-goal direction. It subtracts lateral material-point motion to generate a world-frame displacement over .012s, bounded by .022m. The original G1 six-joint leg Jacobian maps that displacement into at most .022rad per-joint reference corrections with orientation regularization, with at most .004rad/20ms realized reference update. It is enabled only after .85s with the original context gate, measured supporting left-foot normal load >=80N, right-foot load <=40N, ball distance <=.95m and forward relative foot speed >=.5m/s. First real physical foot-ball contact freezes/fades correction over .20s. Original 29DoF floating G1/PhysX, robot self-collision/foot geometry, actuator qdot/torque limits, recovered v32 PPO and complete 14s full-ball scorer are unchanged. **This is a fixed low-dimensional engineering controller, not human-coordination learning, SuperMimic or a learned contact-dynamics model.**

Frozen, independently audited source and trial registration: configs/foot_sweep_direction_study_v038.json. Paired candidate/zero configs: configs/state_conditioned_strike_v13.json and state_conditioned_strike_v13_zero.json. Current code: tools/foot_sweep_direction_v038.py. Frozen parent source: f84c5d78e3fe171e57e9f98ba98f8eab823c720d.

## Zero-path bug: found, documented, corrected before testing candidate

An initial invalid trial set enabled=false in config but the first implementation failed to apply that flag to the feedback gate. At approximately 1.42s the 29-joint reference started changing, peak 20ms correction reached .004rad, the robot later terminated at 4.52s from nonfoot collision. Although the run returned an OS exit code of zero, the disabled-control physics gate **failed**. Its 12MB early-terminated raw result, initial protocol bytes, and old module SHA remain in ignored local experiments/foot_sweep_v038 and reports/v038_zero_near_slow_17.json; it is not silently discarded or counted as a valid baseline.

The implementation was corrected so enabled=false **directly returns the exact original parent commands** without running new IK/CUDA math. One new CPU regression checks return-object identity and exactly one parent-controller call. The revised source SHA and the invalid initial protocol SHA were explicitly recorded in the final frozen study before additional simulation. There was NO active candidate run before this fix.

The corrected zero case did complete all 14 seconds (2800 original PhysX substeps, 700 decisions), with unchanged full-body physics, contacts, original controller feedback, kinematics and scoring relative to recorded v32 for the same near-slow-17 condition. All measured per-joint extra deltas were exactly zero. The original frozen v32 miss was preserved: 4.737232m/s outgoing ball and 9.915s stable standing, not a newly achieved goal. Independent audit status PASS means **zero parity**, not task success.

## Single prespecified active PhysX test — hard failure

The registered order was: old miss B near-slow-17 first, then old miss A only if B passed, then nominal center only if both passes held. All fresh/unseen cases were prohibited. On the first candidate only:

| Metric | Original v32 reference | New v38 fixed direction feedback |
| --- | ---: | ---: |
| Total planned physics duration | 14.0s | 14.0s |
| Actual executed physics | 14.0s /2800 ×5ms | **4.52s /904 ×5ms; early stopped** |
| True physical termination | None | **physical_nonfoot** |
| Realized joint-speed max ratio | below original limit | **1.071969 × original limit** |
| Specific violating qdot | none | **left_knee_joint 21.439383rad/s at 2.645s; bound 20rad/s** |
| Right-instep contact | yes | yes but only 77.78% classified instep impulse, remaining 22.22% ambiguous |
| First right-foot-ball contact | 1.605s | 1.605s |
| Horizontal ball exit speed | 4.737232m/s | 4.729290m/s (99.832% retained) |
| Entire soccer ball within goal aperture | no | **no** |
| Full-ball left lateral margin | -0.00525m | **-0.02143m** |
| Final stable recovery | 9.915s | **0s** |
| Nonzero directional-feedback decisions | 0 | 20 |
| First applied new correction | none | 1.420s |
| Peak added joint correction | 0 | 0.020784rad (prespecified max .022rad) |
| Max applied change per decision | 0 | .004000003rad (max .004rad + GPU roundoff) |

The controller's *reference-level* amplitude and slew bounds did NOT guarantee original realized joint-speed limits. It also changed actual impact surface evidence and produced an unsafe later left-knee event. The original extra-correction model was never updated or tuned. No independent active physics test exists for second old miss, center, stronger shot or fresh conditions, as required by the fixed hard stop. Never extrapolate this one failed deterministic development condition to a statistically measured success probability.

The active candidate's 4.52s duration and the zero-bug's similar earlier outcome show how easy it is for a small velocity-based adjustment to destabilize later landing and support. They do **not** establish a unique anatomical cause: the added correction changes coupled whole-body dynamics and there was no controlled isolated left-knee mechanism intervention. The local DLS / kinematic Jacobian is a small-displacement approximation, not a certified physical response or accurate future foot contact velocity.

## Validity and future scientific work

- Original model/checkpoint/expert and URDF assets retain their SHA256; existing default v5 remains intact.
- All source/config hashes, zero parity and bad initial run are preserved. Code tests include actual resolved G1 leg URDF FK/Jacobian comparison to an independent solver, rigid contact-point velocity, rotation covariance, bounds, fade and disabled-path identity.
- The independent physical audit tools/audit_foot_sweep_v038.py records zero-path original missed goal separately from PASS parity, and returns FAIL for early physical termination, actual qdot over limit, missed goal, and absent recovery. Raw reports are ignored by GitHub, but are preserved on Tang.
- No candidate selection, no hyperparameter search, no retraining and no data from fresh original P0 test cases. CPU tests are separate from physical pass; an OS exit-zero does not equate to success.

**Research decision:** Reject the fixed sweep-direction-only controller. More reliable high-dynamic kicking requires explicit *whole-body coordination* over swing, landing/support and dynamics, not merely optimizing one foot material point. The next worthwhile independent P3/P2 work is a qualified contact-event human-motion dataset and an actual conditional multi-part coordination generator, with the frozen robot/ball physical gate as a guardrail. The current available CMU source is a one-subject marker proxy with unknown professional level/instep contact, not a qualified human professional explosive shot prior.

Read-only numeric evidence (Tang local):
- invalid first zero: reports/v038_zero_near_slow_17.json; original study: experiments/foot_sweep_v038/initial_failed_protocol.json
- corrected exact zero: reports/v038_zero_corrected_near_slow_17.json; independent parity: reports/v038_zero_corrected_independent_audit_v3.json
- active negative candidate: reports/v038_candidate_near_slow_17.json; independent physical audit: reports/v038_candidate_near_slow_17_audit.json
- primary method: tools/foot_sweep_direction_v038.py; strict registered study: configs/foot_sweep_direction_study_v038.json

**P0 full task not improved; P2 professional data, P3 coordination learning and P4 cross-method fairness not completed by this stage.**
