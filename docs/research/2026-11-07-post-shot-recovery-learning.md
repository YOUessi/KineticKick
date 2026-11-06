# v0.1.32 shared post-shot whole-body residual: fixed first trial

Declared before training or evaluating the new model. Parent branch v31 records the rejected spatial-reference experiment; physical control starts from v30 config v6. The spatial transform stays disabled. This is a P0 recovery-control experiment, not the P3 human-coordination generator.

## Mechanism and scope

The known late touchdown combines an already unloaded outgoing support with appreciable incoming ground-relative velocity. v31 lowered one impact peak but increased the unsupported gap and later terminated on non-foot contact. A single reference-frame shift or phase scalar does not independently control the incoming foot, outgoing support and lateral body state. The next experiment therefore exposes 29 shared joint-reference corrections conditioned on measured state; no case IDs or per-ball gain table are inputs.

The frozen placement/running/swing/recovery expert remains the baseline and is invoked once per control step. No modification to its reference arrays, phase, checkpoints, collision geometry, actuator limits, or ball forces. The new correction is structurally zero until real ball contact, 15 ms measured separation, and the original velocity pulse / absolute pose / placement IK / swing windows are all over. The first eligible decision locks activation. A 120 ms quintic onset opens the correction; standing-recovery entry triggers a 200 ms fade. Corrected command is the original final command plus a residual of at most 0.12 rad per joint, changing by at most 0.02 rad per 20 ms decision.

These are correction amplitude/smoothing bounds, **not new physical joint limits**. There is no additional projection of the total PD target into a soft position interval: the original PD reference can itself exceed that interval, and simultaneously enforcing that projection and correction-rate bounds can have an empty feasible intersection. Actual physical limits remain original and actual violations remain failures. The module must preserve exact zero output and prefix equality instead of silently clipping the original command.

Inputs (154 numbers) include actual joint state, root attitude/velocity, both foot positions and link-origin velocities, ground support, original-capsule clearance/descent, phase/activation/recovery age, previous correction and the previous original baseline command. Pelvis-local and world-frame values are named separately. The delayed baseline-command feature is read before invoking the expert, identically in training and evaluation. No future solver contact point is an input.

## Objective fixed before optimization

`post_shot_objective.DEFAULT_OBJECTIVE_CONFIG` is copied verbatim into v8 configuration. It jointly scores upright height, gradual planar deceleration, near-ground incoming-foot normal/tangent speed, foot angular speed, loaded slip, outgoing support availability, original-shank clearance, measured joint overspeed, non-foot force, and residual magnitude/change. Desired planar velocity decays from measured activation velocity to zero over 2 s. Engineering normalizers and weights are hypotheses, not biomechanical standards.

Evaluate the physical terms at the original 5 ms rate and accumulate their time integral; return a window average reward rate to the 20 ms RewardManager, which applies its own dt once. Handle termination once after it is actually known. Do not charge all airborne running as failure, use only prospective low-foot landing/support risk. Do not maximize low impact alone or relabel net contact as independently measured ground friction. Original physical task acceptance remains separate from training rewards.

## Budget, data and selection

One seed 3201, 256 environments, 7 s training episodes, at most 600 s of rollout/optimization or 400 PPO updates, whichever comes first. The deadline is checked before each policy step and optimizer update; an already executing call cannot be preempted, and any measured overrun is recorded. Initialization is outside that timer; final serialization is included in the reported elapsed time. Invalid/exceptional runs preserve a separately named failed state and never publish it as the final candidate. 32 control steps per rollout, 4 epochs/4 minibatches, fixed learning rate 1e-4, gamma .995, lambda .97, entropy .001, initial raw action std .25. MLP 256/128/64 ELU has zero initial output mean.

One quarter of environment slots reset to center; the rest uniformly sample the 17 already observed nonnominal conditions among the original 18. The three `landing_eval_*` perturbations are excluded. These 18 are development data, even when their historic IDs contain `heldout`. Robot/ball state initialization happens only at episode resets. No snapshots are injected within an evaluated physical round.

Select the **final finite checkpoint** at the declared budget, not a per-condition best checkpoint. Do not run a reward/hyperparameter sweep. Batch training uses the existing Torch experts; independent one-environment evaluation uses the original auto/ONNX soccer backend. This backend distinction is recorded, not assumed to be exact cross-backend parity.

## Required decision sequence

1. Focused controller/objective contracts and the 19 frozen hashes; no broad redundant suite.
2. One zero-model, 14 s independent physical replay of `heldout_left_near_slower`. Require all original trace/kinematic/sequence values equal v30, not merely a task flag. If it differs, repair the integration before training.
3. Run the single bounded training trial. Nonfinite values, altered frozen hashes or an integration exception stop the run; preserve its evidence. Proxy reset success is not acceptance.
4. One independent original 14 s evaluation on the known failure. Require protected-prefix/first-strike equality, legal instep, original output speed/goal/follow-through, original actual joint limits, no non-foot termination, and full balance recovery. Examine touchdown approach and support gap as well as impact peak. Any original physical failure or actual overspeed rejects this candidate and stops its matrix.
5. Only a passing candidate proceeds to center, strong and slower representatives and then the complete old18+fresh3 matrix. The two old directional misses remain separate unresolved P0 requirements; a post-shot-only method cannot correct a ball that already missed the goal. No P0 completion claim before all required outcomes.

Raw training/evaluation outputs and weights remain Tang-local ignored assets. P2 public-data qualification proceeds independently; this learned robot controller does not itself demonstrate human coordination transfer or paper-level P3/P4 results.
