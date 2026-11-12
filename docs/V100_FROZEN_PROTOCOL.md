# v0.1.100 | Frozen before any new GPU experiment

Parent committed SHA: `70e3f2a7b743c7c6c04df54c981b4313726bb603`. Previous v99 L/v32 OFF attempt **timed out at 180 seconds** at actor initialization, produced **no** scored G1 physical trajectory. Its original write-once failed log remains untouched. v99 untracked files remain untouched.

## Pre-registered recovery experiment
Use exactly `configs/v100_frozen_matrix.json`, 12 entries, L/M/N, 3 controls with L OFF/ON pairs. These are diagnostic follow-on conditions, **not a new blind test**. Only difference between physical OFF and tensor ON: OFF runs previously successful committed `tools/v080_run_poll_strike.py`; ON runs independently copied `tools/v098_run_pair_strike.py --tensor-pair-v098`. Both preserve their verified matching old physical controller logic. Never change policy gains, joint settings, contact safety, goal or ball placement after seeing results. For first run stop if no original PhysX data; do NOT retry same write-once path or count a timeout as 0% physics success. Hold global `flock` during GPU batch, no foreign CUDA apps and disk >=20GiB; every own child <=110s and child-only stop. All large JSON and logs local only.

Exact 200Hz original all29 states, shots, termination OFF/ON must match **zero difference**; compare raw trace keys and records, fail closed if mismatch. Not even positive filtered tensor forces are native actor/collider manifold. Native contact path certification remains missing until true isolated positive control.

## Strict independent gates
No shortcuts: instep impulse >=95%, true goal, speed >=3.5m/s (not human-level power), every actual qdot original G1 bound, non-foot net <20N, full 14s recovery; terminated frames stay right-censored. Include failures and make explicit denominator of certified CUDA G1 episodes. No new learned policy control. Candidate human optical sequences cannot supply force plate GRF or true elite instep impulse.

## Downstream and abort logic
Complete safe v097/v098 historical 200Hz SHA audit, coupled trunk/stance/arm/leg timing and separately measured physics outcomes; post-hoc descriptions cannot prove human naturalness or causal coordination. Use v99 negative log as distinct unsuccessful **attempt**. If L OFF freezes again, stop GPU; perform CPU research/test/report/push without relaunching watchdog and without touching any old evidence.
