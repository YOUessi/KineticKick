# KineticKick v0.1.116 — human force qualification and KineticChain-29 hypothesis

## Real source eligibility (not synthetic human power labels)

A read-only reproducible scanner loads six unique local CMU C3D recordings from subjects 10/11 with ezc3d, verifies file SHA256, the actual 120Hz marker arrays and native analog-channel counts. The repeated subject-10 copies are not independent subjects. These recordings have **zero native analog channels**, so they cannot measure calibrated human GRF, CoP or true foot–ball impact impulse. No provided data certifies professional skill, actual high-speed instep strikes or subject-disjoint force transfer. These files are usable for exploratory kinematic phase hypotheses only, not biological power or ball-impact ground truth.

External candidates, **not certified for elite instep impulse**: [GroundLink](https://csr.bu.edu/groundlink/) has synchronized real motion and force-platform GRF/CoP in other movements; [SoccerKicks](https://github.com/nayariml/SoccerKicks) offers 3D movement guides estimated from reference videos. Neither, based on verified evidence, supplies synchronized forceplate-and-instep-impact elite kicking truth.

## Hypothesis for real full-body generative coordination

KineticChain-29 should generate *time-coherent* whole-body actions across seven phases: approach, support-foot establishment, backswing, pelvis/trunk-driven acceleration, legal instep contact, follow-through and stable recovery. The generative state is the original 29DoF q/qdot, root pose/velocity, ball/goal state, COM, phase and **validated** dual-foot support, slip and native interbody collision witnesses. Predicted outputs are time-indexed 29-joint action targets and support/swing role states, not a late 1mrad foot residual. True human dynamics supervision would require multiple skill-qualified athletes, synchronously measured 3D motion, pressure/force plates, ball impact/velocity and recovery, with calibration and participant-disjoint splits.

A physically executable controller must satisfy all original 29 individual q/qdot/effort bounds, support and slip constraints, pair-specific self-collision certification, ball/foot contact geometry and robust recovery. If certified witness is missing, refuse learned-policy motor actuation. All historical failed scenes are development-only, not a fresh blind test. Pre-register scene/seed and participant heldout splits before new training.

## What must be falsifiable

Compare original v32, weak late-CMU, robot-only trajectory imitation, qualified-human phase-conditioned generator; ablate hip–trunk, contralateral arm, support context and collision layer. Report early approach (not only 0.17s before contact) temporal coordination, actual original 29DoF qdot and estimated power, contact-point witnesses, body collisions, goal, speed and standing recovery. Strict physical success is the logical AND of instep impulse share >=95%, goal, ball-speed proxy >=3.5m/s, all qdot, no nonfoot >=20N and observed 14s recovery. The speed floor is a minimum task gate and **not an elite-kick speed claim**. OFF/ON repeated observations are not independent trials.

ContactSensor net forces, filtered tensor zero and opposed wrist/hip net forces are **not proof** of native actor/collider identity. A positive native collider-pair/manifold validation with point/normal/impulse, separated negative control and no-perturbation OFF/ON is still a hard prerequisite. Native PhysX zero callbacks do not prove no self collision. No true qualified human instep/GRF source or new whole-body motor policy is claimed here.
