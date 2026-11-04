# v0.1.9 — quick physical loft/chip

User requested a fast working high-ball example with different contact/swing settings, without a large validation cycle. This version adds three **inside-foot** presets, using the existing free29DoF G1, physical0.43kg ball, unchanged actuator limits and ordinary position/velocity targets. No ball impulse or state write is injected.

| Quick preset | First-flight ball-centre apex | Ball-bottom clearance | First flight | Release speed |
|---|---:|---:|---:|---:|
| `low_arc` — light chip |0.307m|0.197m|0.400s|5.24m/s|
| `medium_arc` — more upward follow-through |0.355m|0.245m|0.445s|5.56m/s|
| `loft` — stronger post-contact lift |0.440m|0.330m|0.520s|5.84m/s|

These are single quick-trial observations, not calibrated power levels or a robustness claim. All three selected trials had direct foot-induced lift, no knee/body-ball contact, a goal, no fall and final stable standing. They demonstrate flight rather than a ground roll. This is a modest chip; arbitrary-height lobs, other contact styles' high balls, and3-D target accuracy have not been established.

The first batch tested inner-foot tilt and upward hip motion. Earlier/larger ankle-pitch correction lowered the intended contact surface but delayed contact until the shank reached the ball; those four candidates were rejected. The successful strong preset preserves initial contact and adds upward hip/ankle velocity targets **after contact begins**, so the ball separates with about2.20m/s vertical velocity. The contact is approximately4.8cm below the ball centre, estimated from the sphere contact normal. Increased follow-through changes both launch height and speed; these are coupled example presets, not independent controls.

Scope of checks:12 exploratory physical trials of7s each, one grounded-ball negative check of the flight summary, and the three physical preview recordings. No full regression suite, large seeded matrix, model training or repeated review cycle was run. Existing controller modules are unchanged. Raw traces, frames and generated media remain ignored/local.

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py --profile loft
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py --profile medium_arc
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py --profile low_arc
```

Inputs/evidence: `configs/lofted_shooting_v1.json`, `reports/v019_loft_probe.json`, `reports/v019_lower_contact_probe.json`, `reports/v019_follow_lift_probe.json`. Preview: `media/lofted_shooting_v0.1.9.mp4`, assembled only from actual renderer frames with explicit slow replay; preview measurements/hashes are in `reports/v019_preview_manifest.json`. The full Tang-local simulator/assets are still required.
