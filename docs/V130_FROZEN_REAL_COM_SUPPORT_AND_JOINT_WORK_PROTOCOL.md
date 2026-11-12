# KineticKick v0.1.130 — real 200Hz COM/support/joint-work falsification protocol

Frozen before analysis. Branch parent cf1fa1ac824c97492e9738b4b94e12dc1d959482.

## Targets and nonclaims

Use ONLY SHA-verified untouched original v54 PhysX G1 traces at 200 Hz (same three original controllers and ball pose/seed), not fresh or fabricated human force labels. Audit (1) all 29 robot joint mechanical-power estimates tau*qdot, signed/positive/negative work and per-joint extrema, (2) reported whole-body COM displacement and sampled COM velocity, (3) original left/right foot normal-force proxies, foot-load dominance and interval transfers, (4) phase-conditioned nonfoot net forces and missing late-recovery coverage. Pre-impact window [impact-0.4s,impact), post [impact,impact+1.25s), and remaining observed late recovery; the failure is censored, never impute through 14s.

Predeclared foot-load dominance >0.8 only when total proxy exceeds 50N; no contact-path/GRF certification, force-proxy load share is not a support polygon, no global foot CoP since no synced contact positions. Show load-share and measured COM-vertical-acceleration *association* without claiming Newtonian force attribution or cause. Use circular time-shift negative controls lag 0/40/80 frames and never optimize lag afterward. No human training or robot control in this batch.

All source JSON SHA must match preexisting frozen v128 qualified input_sha256; same original 29DoF order; v54 failed run terminal 5.140001s is not a physical sample. Drop it only after identity check. Reject missing 0.4s/1.25s impact windows (do not silently shorten); late recovery must report actual end time, complete 14s status. Save no original large JSON to GitHub. Quantitative results before any interpretation; negative values are retained.

## Causal hypothesis (falsifiable)

If old CMU weak references create meaningful whole-body energy-transfer coordination, the original robot-only joint-work and load-transfer chronology should differ measurably beyond the final 0.18s before impact, and those signatures must survive foot support/velocity gate scrutiny. Null hypothesis: apparently natural-looking sequences are baseline running-controller dynamics unrelated to the weak reference. This is a descriptive test, not a pass/fail policy optimization, and cannot establish human-like motor learning without qualified soccer-player data.

## Physical safety and concurrency

At freeze, active existing v120 pytest PID 242644 owns GPU; do not launch GPU/IsaacLab or terminate it. Files and existing physical evidence are read-only. Any new collision/contact certification remains UNKNOWN until native collider pair/manifold with step-matched witnesses and OFF/ON invariance. Record honest 0 new physical episodes and no new strict physical success rate.
