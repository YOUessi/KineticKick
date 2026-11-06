# KineticKick P0–P4 research design

The user explicitly requests continuous execution of this five-stage research route, authorizes parallel agents, and rejects adding unrelated football skills. The existing G1/PhysX asset, actuator limits, real contact, free root and frozen baselines remain authoritative. No post-initialization robot/ball state manipulation or extra ball impulse is permitted. The choice of implementation details below is the agent's engineering decision within that authorization; no repeated approval menu is required.

## Current evidence

Published v0.1.28 is 1b3e199, with configv5 SHA523aac86743b0054c5d158f7594d839c06646582d947aa93aed4635a21b6e58e. Old15 conditions fully pass15/15 versusv27's14/15. Prespecified new3 legally contact/recover3/3, but full goal success1/3. Neither goal miss may be relabeled as success. v15 remains the frozen original strong-instep baseline. Current approximate4–5.5m/s legal shots must be retained. Existing estimated-PD power is not hardware or measured pure actuator power; kinematic foot velocity contributions are not energy-transfer shares.

## Approach and alternatives

Selected: establish event-aligned physical measurements and human-source provenance before modifying explosive coordination; build a relational whole-body coordination representation and task-conditioned continuous generator, then test through a common frozen low-level controller. This isolates generation from tracking quality. More hand-tuned ball-specific tables would not address the paper objective. Immediate end-to-end training before validating data and force semantics risks optimizing a proxy; it is deferred until measured baselines and source quality are known.

P0 supplements the existing actual traces with follow-through and regression audit. Positive sampled geometric clearance is not a continuous collision guarantee. Existing optional follow-through distance thresholds are explicitly extra diagnostics, not silently inserted frozen acceptance. P1 adds truly observed ground normal/friction forces and segment motion where current telemetry is insufficient. No gain/pose changes are bundled with the measurement release.

P2 separates original human motion, multiview reconstructions, monocular estimates and robot-retargeted motion. A soccer filename does not prove skill level or instep contact. Known data licenses, quality limits, acquisition hashes and source URLs are mandatory. The user has no private authorized data and instructs continued public-source search. No author messages, paid downloads or cloud spending are implied.

P3's planned interface is ball position, target direction, requested outgoing speed, and measured robot body/support state -> a newly generated whole-body trajectory with event timing and uncertainty. The learned representation includes support-leg/pelvis/thorax/swing-leg/arm relations, event offsets and feasible relative motion. Joint-angle reconstruction alone is insufficient evidence of coordination transfer. Synthetic fixtures may test software only; they cannot fulfill human-data or learning claims. Existing PAiD/running experts provide the first common tracking backend; official BeyondMimic/SuperMimic implementation availability and assets must be verified before comparisons.

P4 compares ordinary tracking, a motion-style prior and the coordination generator with common robot/assets/train data and matched adaptation budgets. Fixed low-level controller comparisons isolate generation; different low-level controllers are a separate sensitivity axis. Unseen conditions and human subjects/actions are declared before training, multiple seeds are retained, and failed/invalid runs are not silently discarded. Human naturalness requires actual blinded raters; numerical smoothness is not a substitute.

## Completion evidence

- P0: immutable18-case evidence audit, original failure repaired, legal instep/output/follow-through/recovery and performance regression quantified; remaining direction/follow-through issues explicit.
- P1: reliable event/measurement schema and real frozen-baseline report; measured support braking, pelvis/trunk timing, signed hip/knee/ankle peaks, explicitly qualified drive power, precontact foot speed, collision impulse and continuous follow-through. Effective speed improvement only after matched naturalness/balance/physical checks.
- P2: usable original or quality-controlled reconstructed human instep motions with actual skill provenance, event annotations and cross-body relations; any unmet qualification remains open.
- P3: trained model generates unseen task-conditioned whole-body trajectories, physically executes and exhibits measurable human relation transfer; no static-clip lookup presented as generation.
- P4: completed matched comparisons, independent tests, multiple seeds, contact/coordinative ablations, dynamics metrics and real blind ratings. An evaluation script or study form alone does not complete this stage.

## Version and execution boundaries

New version/v0.1.29 was created fromv28 before edits, after checking local/remote version branches. main and previous versions remain intact. The first v29 deliverable is measurement/acceptance infrastructure and verified data provenance. All19 original frozen files and previous controllers/configs remain unchanged. Raw motion, weights, generated reports and media remain local per.gitignore. Single GPU execution owner is the parent; independent agents perform CPU/source/data work. Each GPU action has a hypothesis, recorded configuration, expected evidence and a bounded stopping rule. Do not repeat unsuccessful tuning without changing the supported hypothesis.
