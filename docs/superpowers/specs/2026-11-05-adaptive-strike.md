# v0.1.24 Adaptive strike research

User outcome: adapt to measured ball position and initial root velocity, place feet and time contact, then recover. Preserve v0.1.15/v0.1.23 as frozen controls.

## Scope and approach
Use a hybrid state-feedback controller over the existing learned running and soccer policies. First fine-tune running balance under uniform initial planar velocity perturbations (±0.25 m/s), then add bounded measured-state reference/foot-target and phase corrections. Compare against the frozen seven cases; these are development cases, not an unbiased generalization estimate. If a candidate passes, freeze it before testing a small separately declared combined perturbation set.

Alternatives: a fully ball-conditioned end-to-end PPO policy needs a new reward/curriculum and significantly more training; per-case offline retuning does not meet the request. The hybrid approach is the implementation choice for this bounded research increment, not a claim that human coordination has been learned.

Original robot, free root, collision, actuators, force limits and acceptance criteria stay unchanged. No robot/ball state writes after initialization. Control may use measured body velocity, orientation, foot support and ball state. Never use a case label to choose controller parameters. No hidden state reset between approach, strike and recovery.

## Evidence and stopping rules
One initial 600 s PPO pilot, one simulator at a time. Inspect nominal and ±0.20 m/s cases before a matrix. A failed pilot is evidence, not acceptance; diagnose before any further bounded attempt. Preserve all outputs and hashes. Successful task requires actual right-instep contact, goal, no prohibited contact/fall and stable recovery over 14 s, reported separately from unit tests. If not converging, save the failed candidate and explain the unresolved mechanism.
