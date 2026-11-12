# KineticKick v0.1.113 — Frozen multi-scene full-body biomechanics and physical safety matrix

**Parent** a389b819f2959179c8edcb8a6f4e427a949103ab, clean v112. Every original asset unchanged, evidence SHA256 checked. Freeze source/config/test protocol before launching GPU.

## A. Physical contact and read-only parity
Original free-root 29DoF G1 in genuine CUDA PhysX 200Hz. Original v080 OFF and v098 Tensor ON observers leave the original controller unchanged. Compare all common states, event records, forces, 29 joint positions/velocities; require exact parity. Filtered Tensor zeros and equal/opposite net forces do NOT certify or refute native wrist–hip collider pair. Actor/collider paths, direct manifold, normal, point and impulse remain unqualified; learned motor actuation disabled.

## D. Preregistered held-out matrix
AA: seed2477, offset [-0.041,0.028]m, delta+0.012; original v32 OFF/ON, CMU10_02 OFF, CMU10_05 OFF.
AB: seed2503, offset [0.037,-0.058]m, delta+0.019; original v32/CMU02/CMU05 OFF.
AC: seed2531, offset [0.015,0.067]m, delta+0.027; original v32/CMU02/CMU05 OFF.
**10 actual rollouts, 9 independent scene/controller conditions; 1 OFF/ON repeat is not independent.** No seed or ball-position changes after result. Strict AND: goal, >=95% measured instep impulse, >=3.5m/s measured ball speed proxy, all 29 observed qdot within original limit, no >=20N non-foot hazard, observed 14s recovery. Unscored crash/timeouts not counted as failed shots. 20GiB free threshold, nonblocking global GPU lock, reject foreign GPU processes, serial GPU, child-only 140s watchdog, stop on first noncertification or parity breach.

## B. Frozen independent dynamics/causal contrasts
For original SHA256-pinned real 200Hz traces: exact 29DoF/time-grid; contact-relative approach, final 0.4s, contact+1.25s and recovery phase; COM displacement and speed; simulated support-foot normal-force proxies; per-body-group joint RMS and approximate motor work derived from simulated actuator torque estimates. Label observations censored if simulator stops early. Compare matched original controller against CMU on common physically observed prefix, preserve all negative metrics. Joint power from simulation is NOT calibrated human muscle work, and support proxy is NOT athlete GRF/CoP.

## C. Research hypothesis (not yet demonstrated)
Future action generator must learn full approach-to-recovery stance-foot / hips / waist / kick leg / contralateral arm phase coupling from synchronized **real qualified multi-subject human** motion with force plates, actual ball impulse, and contact labels. Train/validation/test split by participants and trials; all evaluation, constraints and safety gates preregistered. Full 29DoF q/dq/tau commands gated by true collision actor/collider manifold, measured support/slip, position, velocity, acceleration, torque, and recovered standing. No candidate controller is actuated this batch. Robot-only trajectory MAE and tiny reference perturbations are not human motor-learning evidence.

## E. Evidence, tests, publishing
Immutable write-once original JSON (about 30 MB/shot), logs/NPZ/figures remain on Tang and gitignored. GitHub receives reproducible sources, configs, frozen protocol, tests and text. Original SHA256 for every physical run, independent audit, failed gates, calibration parity, tests and remote SHA required; no original data deletion, branch overwrite or force push.
