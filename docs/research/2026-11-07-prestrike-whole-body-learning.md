# v0.1.35 — one bounded pre-strike whole-body feedback trial

This is a P0 control experiment over frozen v32, after the rejected v34 local Cartesian trial. It is not human-motion training or a task-conditioned human-coordination generator. The v34 candidate's wrong directions and one support-ankle limit exceedance remain failures. No collider, physical joint/effort/velocity limit, ball impulse, acceptance threshold or upstream installation changes.

## Protected controller and observations

A new zero-mean 29-action feed-forward actor adds at most .04 rad per joint, with a uniformly scaled .008 rad/20 ms slew bound. No phase action is learned. The first .60 s and the original measured-context numerical dead zone are structurally zero. After .60 s the actor enters through a .12 s quintic. It reads the existing 154-dimensional recovery-state representation plus actual ball/root relative position and velocity, goal direction/distance, a fixed 4.75 m/s training-band center, permanent eligibility/first-touch age, its actual previous 29-joint increment and the original context gate: 198 values total. The constant speed input is not evidence of learned multi-level force control.

Each command calls `PostShotRecoveryTask.commands(None)` exactly once, retaining its pinned recovery checkpoint `ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326`. New external training actions are never passed to that old post-shot actor. Inference caches new features before advancing the frozen expert, matching external training observations. Added references do not project the entire existing PD target into the soft position interval.

At each 5 ms physics sample, the ball-to-all-robot-body force matrix identifies the first normal-force magnitude above 2 N. Freeze the then-applied new increment and fade it over .20 s; after first touch there is no fresh action/aiming decision. A 2.10 s no-contact timeout also exits. The old recovery remains active under its own original gates. Any overlap is recorded; preserving its weights does not guarantee the new incoming trajectory recovers. The actor is shared across cases; neither case identity nor fitted per-case outputs are inputs.

## Fixed reward semantics

Permanent credit eligibility starts at .60 s and remains true until episode reset, including after the new action fades. New tracking terms are restricted to the pre-control window: reference height (1.5), upright (1.5), reference running velocity (1), and joint tracking (0.5). Actual new-reference magnitude/change costs have weights .15/.10. The existing post-shot objective is **not activated early**: it retains its own original deceleration and support semantics. A near-limit actual speed cost starts at ratio .98 with .03 scale and weight .5, capped at 12.5 per second, throughout permanent eligibility.

At first touch +80 ms, read real ball COM velocity once. This is `v@+80ms`, not certified separation speed; v33 showed that normal force can end before tangential anchors. With current horizontal ball-to-goal direction `g` and its perpendicular `l`, one bonus is

`10 exp(-(v·l/.5)^2 - ((max(4-v·g,0)+max(v·g-5.5,0))/.5)^2)`.

It requires first contact only on the right foot and the original first-force local-normal quality proxy >=.75, with no wrong-body contact. This is a training proxy, not the independent original collision-point instep classifier. The 4–5.5 m/s band is a shared **soft training objective**, not a redefinition of baseline speed acceptance.

Original goal-plane crossing adds +20 once. The first measured |qdot| > original limit +1e-6 rad/s adds −20 once; invalid-contact proxy or no contact by 2.4 s adds −20 once for each distinct event. Physical termination adds −20 only before original post activation; after activation the frozen post objective owns terminal cost. No event is repeatedly paid. Five-ms rates are integrated with dt and exposed as a 20 ms mean; event values are divided by .02 once for RewardManager's own dt integration. Old post reward buffers and new applied-delta buffers stay independent.

## Budget, model selection and cases

Exactly one PPO trial: seed3501, 256 environments, 7 s episodes, 64-step rollouts, <=400 completed updates and <=600 rollout/optimizer wall seconds. Initialization/teardown are reported separately; one in-progress operation cannot be preempted at an exact wall-time boundary. Actor/critic 256/128/64 ELU, zero final actor mean, initial std .25, fixed LR1e-4, four epochs/four minibatches, entropy .001, gamma .995 and lambda .97. Other settings inherit the v32 trainer. The longer rollout increases the credit horizon; it does not guarantee convergence.

One quarter of reset slots use the original center; other slots uniformly sample the other 17 already observed development conditions. No jitter or fresh case enters resets. All 18 are training/development now, including historically named `heldout_*` cases. Batched frozen experts use Torch; one-environment evaluation uses the original auto/ONNX path. Backend equivalence is not assumed. Initial/episode resets are the only robot/ball state writes during training.

Use only the final finite checkpoint at the declared limit. No best-reward selection, extra seed, extension, hyperparameter sweep or per-case checkpoint. Nonfinite data/model, failed frozen-source pins or action-protection violation stop the run and preserve a failed state; fix an interface fault only with its cause identified, without relabelling an invalid run as completed training.

Before training, a zero-new-actor 14 s replay of `heldout_v028_left_near_slow` must exactly reproduce its v32 physics, original feedback and actions. The fixed subsequent independent order is the two old misses, then center and combined-right-faster, with the same baseline speeds as `configs/prestrike_study_v1.json`. Original instep/body contact, all-run actual velocity limits, complete14s/recovery and >=95% baseline outgoing XY speed are hard gates. Stop on physical/quality regression. Both misses must become goals before center/strong; otherwise retain the failed candidate and do not open fresh conditions. The two unused fresh conditions stay untouched.

Training reset/goal/stability counts and code tests are proxies, not independent acceptance. P0/P1 task completeness, naturalness, P2 qualified data, P3 coordination transfer and P4 fair multiseed comparison remain separate requirements.
