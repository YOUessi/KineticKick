# v0.1.35 — training saved; independent shot acceptance pending

**Status: paused at the user's explicit request, then published as a research checkpoint.** The single declared training run had already completed its 600 s wall-time limit and exited 0 when the stop request was handled. All subagents had finished. No trained-candidate shot evaluation or preview has been run. P0 remains incomplete; this version does not establish improved shooting, human coordination transfer or generalization.

## Saved implementation and verified preparation

A separate 29-action pre-strike feedback actor observes 198 actual-state/task features over the complete frozen v32 expert and learned recovery. It preserves the initial .60 s and the original nominal context dead zone, adds bounded references without projecting original PD targets, and freezes/fades its increment after measured first touch. A permanent reward-credit window captures later shot, goal, limit and recovery outcomes. The v34 local Cartesian candidate is not enabled.

Eighteen focused CPU tests passed before the run. A real disabled-new-actor replay exactly matched v32 in 2,800 physical samples, 700 action/kinematic records, original feedback, contacts and task/recovery outcomes. Its result is a zero-path equivalence check, not an improvement result. The recorded original case still misses the goal and recovers.

Pre-run protocol: `docs/research/2026-11-07-prestrike-whole-body-learning.md`; study: `configs/prestrike_study_v1.json`, SHA256 `cea23e5839ee0775ae3af5c014f582b5b5095bad715ebeaedc7e3476c0b1b895`. All source/configuration pins still matched when preparing this publication. During implementation, a ratio-versus-absolute speed-tolerance mismatch was corrected and tested before the protocol was fixed. The reward event now matches the independent evaluator's original absolute 1e-6 rad/s tolerance.

## Training record

- Seed3501, 256 environments, 7 s episodes, 64-step rollouts; the declared 18 previously observed development conditions only, with one-quarter nominal reset slots.
- Completed **64 updates in 600.027363 s**; measured budget overrun **0.027364 s**, within one in-progress call/final serialization. No extension or checkpoint selection by reward.
- Final model: `experiments/prestrike_v035/ppo_s3501/last_model.pt`, **2,254,517 bytes**, SHA256 **`bf51f9cc9aead37fe1a474a1eed77fd7cf2cc5d6b833087eb616541f4dead9f0`**. The published digest was independently checked against the actual local file. The trainer's finite-policy/positive-std guard passed before saving; no post-stop inference test was performed.
- Result SHA256: `9218c19205893d5e42a736e3325e2f8b4762f4d26cd5e6c8c7702c7147b32f56`; manifest SHA256: `0a7eb9429a731b8e426e4b5c409c745111976aa369bd47a4a6af5cc2a515ec36`.

Training reset counters: 3,436 episodes, 974 nominal episodes, 2,988 contact episodes, 2,477 goal episodes, 819 physical terminations, 346 observed overspeed episodes and 2,490 instantaneously stable terminal samples. **These are stochastic training proxies, not independent task passes or a success rate.** Inherited reset-statistics overspeed uses ratio >1+1e-6; the new reward event and original independent evaluator use absolute limit+1e-6 rad/s. The counters therefore must not replace the independent all-sample audit. Neither the fixed speed input nor the common reward band demonstrates multiple force levels or new target directions.

## Remaining acceptance and local artifacts

The two old goal misses, center and stronger-shot conditions have not been evaluated with the saved new model. Their fixed order and original physical/instep/recovery/95%-speed-retention gates remain in the study file. Neither unused fresh case was accessed for training or evaluation. There is no new preview and no default-controller promotion. Resumption requires a new user instruction; this publication does not restart work.

`configs/state_conditioned_strike_v11.json` is the **zero-new-actor/training configuration**: its new checkpoint fields intentionally remain null. It must not be mistaken for a validated deployment configuration for the saved model. The pinned v32 post-shot model is still included. Future candidate configuration and independent physical evaluation remain pending.

Tang-local generated artifacts, retained under `.gitignore`:

- `reports/v035_zero_left_near_slow.json`, SHA256 `4fb7bc5533d6ceae15b9953658c0ecc286fe16662b7a1b63ef7e093ab62d6177`;
- `reports/v035_zero_parity.json`;
- `experiments/prestrike_v035/protocol_sources/`, `zero.log`, `training.log` and `analyze_v035.py`;
- `experiments/prestrike_v035/ppo_s3501/` containing exact run-source snapshots, manifest, progress, result and saved weights.

GitHub contains code, configuration, tests and this progress record. Weights, raw simulation data and media remain local under the project agreement. The original physical assets, previous version branches and frozen baselines are preserved.
