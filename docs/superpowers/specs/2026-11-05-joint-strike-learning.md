# v0.1.25 Joint strike and recovery learning

User outcome: measured-state adaptation to ball position and initial speed, appropriate foot placement/contact time, and recovery. Preserve frozen v0.1.15/v0.1.23 control and v0.1.24 negative results.

Implementation choice: a single learned residual controller throughout the 14 s physical episode, over actuator commands captured from the successful frozen center rollout. Start with exactly zero learned mean correction, verify the physical replay before training, then learn bounded 29-joint corrections and one bounded phase-rate action from actual ball/body/support state. No separate new-running/old-strike handoff. This is a nominal trajectory prior with learned feedback, not human-motion synthesis or an already generalized policy.

The alternative of training a new unconstrained full policy risks losing the existing successful motion; repeating the v0.1.24 hand corrections has already failed. The residual choice is an implementation decision within the authorized research scope.

Same original free G1, 29 joints, actuator limits, ball mass/material and physical acceptance. Robot/ball writes only on training episode reset or evaluation initialization. Goal, actual foot surface, running contact and final standing use the frozen evaluator. Training can use force-normal contact shaping, but this proxy cannot replace collision-point acceptance.

Training includes actual ball collision and all post-contact dynamics. Half the reset conditions stay nominal; other conditions initially sample ball offsets ±0.05m x / ±0.10m y and initial world vy ±0.20m/s. Reference tracking and a residual penalty preserve the prior; task terms reward right-instep-like contact, goal crossing and post-contact standing. These rewards are hypotheses to test, not claims of success.

One zero-residual 14 s replay must pass before expensive training. Initial PPO pilot at most 900 s and 1000 updates, one GPU process. Evaluate original and two speed conditions first, then ball positions if warranted. Preserve failed outputs; diagnose before any additional pilot. Existing seven conditions are development comparisons, not a held-out success probability. Publish honest outcomes on version/v0.1.25 with upstream; do not replace baseline unless physically justified.
