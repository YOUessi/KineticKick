# Joint strike learning implementation plan

> Execute inline using superpowers:executing-plans. User has authorized continuing implementation and version publishing.

**Goal:** learn one measured-state controller through approach, actual kick and recovery.
**Architecture:** frozen successful actuator trajectory plus zero-initialized bounded PPO residual/phase control; identical execution code in training and physical evaluation.
**Spec:** docs/superpowers/specs/2026-11-05-joint-strike-learning.md
**Stack:** IsaacLab/PhysX, Torch/RSL-RL, existing frozen evaluator.

Constraints: no frozen-file edits; original physical limits/success criteria; all generated references/weights/raw telemetry ignored locally. Review: exact zero residual semantics, no hidden post-reset physical writes, training/evaluation feature parity, contact/goal reward bookkeeping, preservation of failed/truncated evidence.

- [x] Build a hash-checked local nominal actuator/state template; test bounded continuous phase and zero-correction identity.
- [x] Implement shared state features/controller/rewards and isolated physical runner; verify zero-residual nominal replay.
- [x] Add joint-task PPO trainer with nominal reset preservation and real ball dynamics; run one bounded pilot and diagnostic probes.
- [x] Record actual comparisons and limitations, targeted tests and one fresh review; commit/push independent version.

Ledger: started from 3871e11 (v0.1.24), clean checkout. version/v0.1.25 unused locally/remotely and created before edits. No simulator jobs active at start. Prior failure: isolated running fine-tuning changes the state delivered to the old shot; nominal-reference residual learning will first prove preservation before optimization.


Evidence: zero-reference physical evaluation reproduced baseline exactly (1.570 s instep, 4.6302 m/s outgoing, 8.385 s final standing). The zero run predates reward-only stable-seconds bookkeeping; its recorded historical source hashes are retained. 16-env/1-update interface smoke passed (1.784 s), not an outcome test.

Diagnostic checkpoint: first 1024-env pilot stopped after approximately 94 training seconds / 67 updates because episodes ended at about 0.6 s. Measured nominal PD targets include ankle demands outside kinematic joint limits; the original new residual function clipped the entire target on any nonzero correction. A targeted test failed with a 1.5 rad jump from a 2.5e-6 rad intended correction. Revised action composition preserves original PD requests and bounds only the residual, allowing only inward corrections when the original target is outside the interval. Physical joint/actuator limits remain unchanged. Old training sources and checkpoint retained locally; it is invalid for learned-performance comparison. Next gate: zero residual in the parallel training environment before restarting the bounded pilot.

Parallel zero probe: 16 environments all eventually failed; same training entry with one environment reproduced real contact, goal and 8.38 s stable at 13.98 s with no resets. This isolates sensitivity of the recorded open-loop command prior to parallel numerical/initial-state differences; source commands cannot substitute for a stabilizing policy. The revised pilot trains that feedback instead of assuming replay is robust. Probe wall time: 24.114 s + 20.941 s. Initial aborted pilot retained model_0.pt only; no terminal checkpoint or result was written by the Isaac interrupt handler.

Corrected pilot stopped for a performance plateau after 533.587 s / 348 updates, checkpoint d9781eb9bf3f49ea974e76b69653b681b0fa50562facdc7f8e2d8622e5e9a4c0. 51,466 nominal and 55,936 perturbed completed episodes, zero training proxy full successes; 9,607 right contacts and 28 goals. These are stochastic training statistics, not independent task acceptance.

Final fresh read-only review (gpt-6-astra): one P2, jointly truncated feedback and kinematics could pass audit against complete physics. Reproduced with a red test; validator now requires command coverage through every executed physical interval using existing validated_rows handling of terminal markers. No other Critical/Important/Minor findings. Reviewer deferred learned robustness/convergence to independent physical evaluation, which remains the implementer's responsibility.

Independent acceptance: center fails physical_nonfoot at1.74 s; slower fails physical_height at1.32 s; faster at1.20 s. No foot-ball contact in any; 0/3 versus frozen1/3. All three provenance/full-coverage audits pass. Four ball-position cases not executed following the planned gate. Reviewer-deferred robustness is explicitly rejected; convergence is not claimed. The original user goal remains unresolved; this branch records a failed joint-learning hypothesis without replacing baseline.
Final evidence fix verified: six targeted tests + six subtests passed. Frozen19 hashes verified during independent analysis. Version publication is the final step; exact commit and remote equality are verified in the session after creating this record.
