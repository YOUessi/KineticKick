# v0.1.84 — Before-experiment protocol and safety decision tree

Parent `9f51c09f2cbf6c87ca4ffe882d440c639046490d`. v83 local untracked code remains untouched; v84 is an isolated clean worktree. All protocol and runner files must be committed **before** physics. Freeze SHA captured from committed files; no model/physics knobs changed after first rollout.

**A**: one paired original-v32 scene C with identical seed, ball offset, source and 200Hz settings; off uses unchanged original, on invokes preexisting v080 native readonly poll during 4.7–5.5s. Any differences in shared 200Hz physical fields mean instrumentation FAIL. Absence of native event headers means *unknown*, not no self-collision; opposite wrist-hip net forces are not a collider pair. No shape creation, no collision-filter changes, no GPU tensor view.

**B**: For all genuine complete/early-censored real runs, gather all original measured 29 qdot per 5ms, COM, bilateral foot supports, body angular surrogates, nonfoot diagnostics and 0.4s before/1.25s after confirmed contact; never impute censored recovery. Compute coupling lags only descriptively; do not mislabel them learned human timing.

**C**: retain v83 pre-registered fullbody support→hip→torso→swing→counter-arm candidate as a *blocked control interface*, not a new success. Its arm counter-swing/phase reversal ablations are conceptual until real collider clearance/GRF and safe G1 execution are established. No hidden tuning on observed v83 outcomes.

**D**: New independent positions C: (+0.085,-0.040)m, +0.035m/s, seed1219; D: (-0.075,+0.055)m, -0.035m/s, seed1223. Three original controllers v32, CMU10_02, CMU10_05, plus readonly ON matched v32 C: **7 total PhysX trials**. Seed is *not* a joint/root reset perturbation; no fake independence claim. Preserve early nonfoot terminations as true failure, but GPU boot/output errors as unscored, and stop further attempts on uncertified physics.

**E**: Independent strict instep≥95%, ball speed≥3.5m/s, goal, every measured qdot, nonfoot≤20N, 14s real recovery; never accept “goal” alone. Write-once local JSON/log/audit, submit only code, tests, protocol and textual report to isolated v84 branch. Do not touch original CMU/C3D/PhysX files, other worktrees, model weights or GPU jobs. Abort if GPU in use, lock owned or free disk <20GiB; GPU serialized. No professional-shot force/GRF truth asserted.
