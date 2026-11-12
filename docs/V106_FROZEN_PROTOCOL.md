# v0.1.106 | Freeze-before-physics original G1 recovery and collision instrumentation

Parent clean commit `2994b5ddca83337c3bdda223d92720bfdc12472c` (v105); new `version/v0.1.106` clean worktree. Root v105 already demonstrated G1 progression through 650 *20ms* policy steps with a **global Python sys.settrace**, but no complete raw output before 75s timeout; the tracer is NOT representative of native performance or physical parity, and does not establish 14s recovery.

## Unchanged physical controllers, two frozen seeds
Exact 8 ordered episodes: L seed1601 (-.095,-.035), velocity delta0, v32/CMU10_02/CMU10_05 OFF/ON; M seed1607 (+.035,-.105), delta=-.015, v32 OFF/ON. Original free-root G1 29DoF, original v080 OFF, v098 ON filtered tensor only, same configuration, policy, asset/CMU SHA, command, physics step 0.005s. OFF/ON introduces **only** previously documented v098 observer; no gains, height, friction, policy parameters, or safety thresholds modified. ON tensor partner view is not native PhysX actor/collider manifold truth. Never infer no wrist–hip collision from an uncalibrated zero G1 filter.

## Preflight/guards
Machine online, no external CUDA app, exclusive existing global flock, disk >=20GiB; all controllers and human proxy assets SHA checked, source files must equal Git HEAD; no GPU runs if v103 concurrent job. Each child <=140s, SIGTERM then own-process-group SIGKILL if necessary. All output write-once in ignored Tang reports; never overwrite CMU/C3D, weights, historical JSON/NPZ, or other branch.

**Fail closed:** if first OFF does not produce a true 200Hz 29DoF raw trace, STOP remaining 7; do not count unattempted runs as failures or make up a shooting success rate. If any OFF/ON changes any common physical trace field, event record, or safety gate, STOP further runs and report parity failure; do not silently accept. No tuning on the L/M known scenes. Do not run learned candidate control.

## Independent gates and attribution
Raw logs alone not enough to certify physics. A completed original 200Hz raw JSON must pass source seed, root freedom, all original 29 qdot and dt certification. Independently audit original instep impulse >=95%, goal, ball velocity >=3.5m/s, all source29 joint qdot limits, <20N nonfoot and full 14s recovery; earlier termination right-censored and failed. All passing/negative outcomes retained. Exact field-by-field trace and original shot records OFF/ON comparison uses zero tolerance; if source field schema differs, fail closed rather than dropping fields. All final summaries declare native manifold unavailable. Any A collision pair claim needs separately qualified positive isolated native actor/collider test.

## Dependency gates
Only after a true original G1 fully scored trial can multi-episode batch proceed. Historical v105 human coordination onset result cannot establish natural style. Professional human soccer synchronized GRF/CoP and foot contact impulse remain unavailable; this version addresses infrastructure, not learns real human ballistic strike. Report and push SHA after tests.
