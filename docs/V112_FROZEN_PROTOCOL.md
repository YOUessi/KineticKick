# v0.1.112 — Preregistered whole-body coordination falsification and collision instrumentation

Frozen source parent: `62f4d1f3715cc6e74352cdfdd6d40ce10e3b5584` (GitHub v107). The v108–v110 working trees are dirty; **do not edit them**. v112 is a clean branch from GitHub-confirmed v107; v110-only local experiments are read-only context, never new v112 data.

## Prospective real PhysX matrix
W seed 2333 offset [-0.022,+0.034] m, delta +0.016; X seed 2341 offset [+0.025,-0.034] m, delta +0.021; Y seed 2357 offset [+0.006,+0.059] m, delta +0.011. Each runs unchanged v32 and CMU10_02 controllers (six independent physical configurations). W and X each use OFF/ON read-only wrist/hip filtered tensor views for exact full-physical-state parity (four extra repeats). Total **10** raw 200Hz G1 free-root runs; GPU serial, single global lock; stop on unqualified run or parity mismatch. Strict: goal, >=95% instep impulse, peak ball speed >=3.5m/s, all 29 actual qdot under original limit, no non-foot safety collision, measured 14s stand/recovery. We report all failures even if a ball entered the goal. Positive controls for contact-pair sensor require actual native collider/actor paths, normal, point, impulse. Tensor zero and net-opposing forces are not native proof.

## Freeze and machine rules
Source/protocol/test frozen by git SHA **before running**; report SHA256 of each original JSON/log. 20GiB disk threshold; reject any foreign GPU process, nonblocking shared fcntl lock; 140s child watchdog; no forced kill outside own child; write-once raw/NPZ/figures remain Tang-only, GitHub commit only scripts, tests, configs, reports. No source drift, model change, force tweaking, seed/boundary selection on held-out results. Original CMU/C3D, weights and pre-existing worktree untouched.

## Research decision
All learned-policy actuation remains **disabled**. This is a *falsification/control* batch, not proof of human-like force transfer or a new policy. CMU marker/C3D data with no calibrated GRF, CoP, and measured instep impulse cannot certify athlete-grade shooting; require multi-subject kick data, calibrated force plates/contact, ball impulse, and unambiguous impact-phase labels before learning/claiming elite human coordination. Evaluate timing and energy transfer jointly, not solely a scalar qdot/pose MAE.

### v112 missing source asset recovery
One v111 attempt failed before physics due a missing ignored motion. This branch repeats the **same unmodified prospective seed/ball/controller matrix** after read-only linking provenance assets from verified v107 source, with preflight coverage extended to original controller target motion. No GPU work may start if SHA/asset absent; the v111 failure is immutable.
