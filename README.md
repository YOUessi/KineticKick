# KineticKick

## v0.1.15：正脚背加力射门

新增正脚背加力档：同球位、同球门，水平出球速度由3.77提高到4.63m/s（+22.8%），球进门，14秒回合末段连续站稳8.19秒。沿用原关节限制，不给球添加外部冲量。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py \
  --plan configs/stride_instep_strong_v1.json \
  --output reports/new_strong.json --video-dir media/new_strong_frames
```

预览：`media/stride_instep_strong_v0.1.15.mp4`，含正常速度与触球慢放。当前为固定球位、运动初态的单一加力档；实测、淘汰方案及范围见[加力射门记录](reports/2026-11-04-stronger-instep-shot.md)。

## v0.1.14：正脚背与脚外侧移动射门

新增正脚背、脚外侧两种完整流程：移动接近→真实触球进球→随摆与继续迈步→恢复站稳。两个最终14秒物理回合均通过，保留原脚内侧入口。预览为 `media/stride_instep_v0.1.14.mp4` 和 `media/stride_outside_v0.1.14.mp4`，均附触球近景慢放。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py --style instep \
  --output reports/new_instep.json --video-dir media/new_instep_frames
# 脚外侧：--style outside，并使用新的输出路径
```

当前验证范围为运动初态、固定球位的髋主导动作；方向/力度/高度联合标定及任意助跑距离尚未验证。恢复控制、碰撞坐标修正和完整实测见[新增踢法记录](reports/2026-11-04-more-foot-surfaces.md)。

## v0.1.13：跑着踢球

已完成固定场景的「连续奔跑→脚内侧射门→继续迈步→恢复站稳」。触球前3次交替腾空，1.575秒触球时仍以2.299m/s向前跑动，中间没有停步；球进门，末段连续站稳9.575秒。全程无摔倒或非脚部碰球。

Tang实际预览：`media/stride_shot_v0.1.13.mp4`，含正常速度和触球／落脚0.25倍慢放。这是运动初态、固定球位的髋主导脚内侧踢法；静止起跑、更多跑动踢法及方向／力度／高度联合控制仍待完成。详见[跑动射门记录](reports/2026-11-04-stride-shot.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_shot.py \
  --output reports/new_stride_shot.json --video-dir media/new_stride_frames
```

## v0.1.12：奔跑后真实触球射门并恢复

已把同一自由G1的「奔跑→减速调整→脚内侧触球进球→恢复站稳」接到一个物理回合。固定场景实测6次交替腾空，6.22秒触球，末段连续站稳7.185秒；全程无摔倒、无非脚部触球。

Tang实际预览：`media/run_to_shot_v0.1.12.mp4`。这版仍从运动初态开始，尚未验证静止起跑、其他踢法或联合方向／力度／高度控制。详见[真实射门衔接记录](reports/2026-11-04-running-to-real-shot.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_to_shot.py \
  --output reports/new_run_to_shot.json --video-dir media/new_run_to_shot_frames
```

## v0.1.11：连续腾空跑步与减速衔接

同一自由29DoF G1已在实际物理回放中完成6次交替腾空，并减速、按实测支撑状态交接控制；6.94秒内没有摔倒或非脚部碰撞。新增速度、真实脚底支撑和接触时序反馈，保留原执行器限制。

**完整任务仍未完成：**末段尚未达到站稳标准，零速度起跑在1.44秒失稳，奔跑与真实射门尚未衔接。两组方向／力度／目标高度小试均未通过；更多踢法高球、近竖直脚背与不同球位／距离也没有新的通过结果。详见[本轮结果、失败原因与复现](reports/2026-11-04-running-feedback-and-transitions.md)。

Tang实际回放：`media/run_to_handoff_follow_v0.1.11.mp4`。原有助跑射门、脚外侧和挑球默认配置保持一致，实验控制不替换已验证动作。

## v0.1.10：同一 G1 的跑步训练小试

这一版开始训练原 29DoF G1 的跑步策略，并修正了参考动作起步时骨盆与左髋的胶囊碰撞。原有助跑射门、脚外侧和挑球预览保留。

两轮各400次训练已完成，但持续腾空跑步仍未通过：最终固定起点回放前进约2.21m，在1.46s时倾斜失稳。这个版本提供可复现的训练／诊断工具，不把训练回报当作完整流程完成。

距完整目标仍缺持续跑步与射门衔接、方向／力度联合控制、更多踢法的高球及任意球位／距离适应。近竖直脚背也没有由已有进球结果证明。详见[目标差距与本轮实验](reports/2026-11-04-running-training-pilot.md)。

## v0.1.9：快速挑球试跑

脚内侧挑球已跑通，三组触球／向上随摆参数对应首次飞行峰高约31／35／44cm（球心高度），加强组连续离地约0.52s。它们是快速实测示例，尚未做精确力度或三维定点标定。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py --profile loft
# 另外两组：--profile low_arc / --profile medium_arc
```

详见[高球快速试跑记录](reports/2026-11-04-lofted-shooting.md)。沿用原自由机器人、真实球和执行器限制；没有给球施加额外冲量。


## v0.1.8：脚外侧新流程与原生力度定点

新增小腿主导脚外侧的完整助跑、触球、进球和恢复流程，独立扰动验收 **6/6通过**。三种已有动作在各自原生力度下，瞄准球门左／中／右三点（x=-0.3/0/+0.3m，y=-3.7m），联合验收 **18/18通过**，最大误差14.28cm。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style outside
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style instep --target-x 0.3
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style instep --target-x -0.3
```

**持续腾空跑步、轻／中／重与定点联合控制仍未通过验收。**当前版本保留真实失败数据，不提供未通过的三档参数表。外侧流程尚未标定目标／力度。详见[完整结果与限制](reports/2026-11-04-running-aim.md)及[冻结配置](configs/runup_shooting_v2.json)。旧版本记录及指纹请在对应版本分支复现。


## v0.1.7：连续短程助跑与两种挥腿方式

新增同一自由G1上的「短程助跑 → 触球射门 → 随摆 → 恢复站稳」。髋主导正脚背、髋主导脚内侧、膝主导正脚背在冻结的球位／初态扰动矩阵中 **18/18通过**，每回合完整验证10秒；158项CPU回归通过。详见 [实现与验收报告](reports/2026-11-04-runup-sequence.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style inside
```

冻结配置：[runup_shooting_v1.json](configs/runup_shooting_v1.json)。这是约1米的快步助跑，尚未验证持续腾空跑步、任意距离助跑、脚外侧新流程或新动作的定点／力度标定。髋／膝主导指接触前足端运动贡献，不是能量份额。历史默认预览继续保留。


GitHub: [YOUessi/KineticKick](https://github.com/YOUessi/KineticKick). Each version uses a separate `version/vX.Y.Z` branch; see [version control and local asset notes](VERSION_CONTROL.md).

**Aim:** learn coordination principles from human soccer movements, then generate novel, natural, physically feasible humanoid movements. Task success is a constraint, not the main scientific objective.

## v0.1.6：以出球效果和全过程平衡评价随摆

用户明确：随摆应服务于有效触球、力度／轨迹控制以及整个动作的平衡，摆得更远本身不是目标。新诊断 `tools/kick_contact_metrics.py` 按连续受力段记录接触时间、法向冲量和观察到的球速，同时报告过程中根部与接触力的变化。连续受力超过80ms不会自动被当作二次碰球；零力采样后的受力恢复单独记录，但仍不能仅凭力信号证明几何分离。

旧正脚背演示中，脚前摆由约15cm增至25cm，记录到的受力时长、法向冲量和首次零力采样处的球速却完全相同；脚外侧实验还暴露了恢复恶化。详细数据和测量边界见 [目标校正记录](reports/2026-11-04-contact-and-balance.md)。这是诊断与目标修正，**没有新增已验证的踢球控制器**；原默认动作不变。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/kick_contact_metrics.py \
  reports/v013_posture_demo_instep.json reports/v015_follow_final_demo_instep.json \
  --output reports/contact_effect_comparison.json
```

v0.1.5的旧阈值与结果保留供历史重放，不再代表当前优化目标。目标仍是有效且可控的出球、合法触球和全过程平衡，不能通过更换指标宣称旧失败动作已成功。

## v0.1.5：随摆试验（稳定性未通过）

本轮已做出球离开后继续摆腿的动作，但**尚未稳定实现随摆、指定触球、进球和站稳的完整组合**，普通预览仍默认使用 v0.1.3。最后一轮正脚背为3/6同时通过；脚外侧的膝关节过渡会把急收腿延迟到后面，单环境录像还出现原生终止，因此未采用这些参数。

分支保留随摆测量、可选后续关节控制接口，以及明确未验收的试验配置。44个诊断回合和4轮独立矩阵的结果（含失败）见 [随摆试验记录](reports/2026-11-04-follow-through.md)。Tang上的单次成功动作对比：`media/kick_follow_through_v0.1.5_experiment.mp4`；该录像不代表稳定验收。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --trials configs/follow_through_feasibility_trial.json --seed 945 \
  --output reports/follow_through_experiment_live.json
```

此命令显式运行实验；普通 `--style instep` / `--style outside` 继续使用旧稳定姿势。近竖直脚背触球、指定落点与力度控制也仍未解决。

本版扩展了控制器源码与证据记录，旧定点／力度标定会在启动仿真前因指纹不符而拒绝运行；没有把旧验收结果换上新指纹。复现旧定点实验请使用保留的 `version/v0.1.4` 或更早分支。

## v0.1.4：近竖直脚背可达性试验（射门未通过）

用户希望触球脚的脚尖朝下、脚背接近竖直。本轮确认摆腿时可以达到接近竖直的姿态，但 **30 个诊断回合中，尚无一例同时完成近竖直姿态、正脚背触球、进球和站稳**。已停止这组参数试验，该版本当时的普通预览保留 v0.1.3；本分支也继续保留 v0.1.3 默认配置。

本分支新增严格姿态／离地间隙审计和明确标记为未验收的可达性试验配置；没有发布新的稳定控制参数。诊断、复现命令及下一步问题见 [近竖直脚背试验记录](reports/2026-11-04-vertical-instep.md)。112 项代码测试通过不代表物理射门通过。

## v0.1.3：正脚背与脚外侧姿势修正

正脚背改为脚尖下压、脚背朝向球门；脚外侧减少身体侧转和右髋过度扭转。冻结参数后，在球位左右各偏移 2 cm、两组初始扰动的检查中，正脚背、脚外侧和保留的脚内侧均为 **6/6 完成接近、指定触球、进球和站稳，共 18/18，无摔倒**。姿势与触球由实际物理接触和关节数据复核。

普通预览继续使用 v0.1.3 稳定姿势：

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style outside
```

每次只启动一个仿真器，关闭窗口后再切换踢法。Tang 上的同视角前后对比含触球慢放：`media/kick_posture_v0.1.3_comparison.mp4`。参数保存在 `configs/free_shooting_posture_v1.json`，验收命令与限制见 [姿势修正报告](reports/2026-11-04-kick-posture.md)。接触仍偏脚前部；本次是近距离静止球姿势检查，**新姿势的定点和力度尚未重新标定**。带 `--target-x` / `--power` 的旧实验请在 `version/v0.1.4` 或更早分支复现；本版源码指纹已改变。

## v0.1.2：方向与力度控制实验（精度未达标）

新增固定球门内左／中／右目标和轻／中／重力度接口，支持三种触球方式。**目前仍不能稳定定点射门**：54 次独立扰动回合中，27 次同时满足落点误差 ≤15 cm、球速误差、指定触球面及站稳要求。脚内侧 11/18、正脚背 4/18、脚外侧 12/18；正脚背的力度档位尤其不稳定。完整失败记录、实测球速及复现命令见 [本版本报告](reports/2026-11-04-targeted-shooting.md)。

以下历史命令适用于 `version/v0.1.4` 或更早分支：

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --style inside --target-x -0.3 --power high --output reports/targeted_demo.json
```

`--target-x` 可选 `-0.3 / 0 / 0.3` 米；`--power` 可选 `low / medium / high`。三种踢法的请求速度不同，数值见报告。这些是实验控制，不保证实际命中；窗口显示实际误差。连续任意方向、任意力度和高球目标尚未验证。不加这些参数时，当前分支运行上方 v0.1.3 稳定姿势。

## v0.1.1：自由根部三种触球射门（2026-11-04）

已实现并完成有扰动的物理验收：走近球 → 正脚背 / 脚内侧 / 脚外侧射门 → 恢复站稳。30 次冻结条件验证中成功 28 次：正脚背 10/10、脚内侧 10/10、脚外侧 8/10。两次失败均为脚外侧射偏，30 次均恢复站稳。 三种方式的完整录像保存在 Tang 的 `media/free_shooting_v0.1.1_verified.mp4`（24.6 秒）。适用范围、失败条件和物理证据见 [本版本验证记录](reports/2026-11-04-free-shooting.md)。

在 Tang 上逐个运行（每次只开一个仿真器）：

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style inside --config configs/free_shooting_v1.json
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style instep --config configs/free_shooting_v1.json
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style outside --config configs/free_shooting_v1.json
```

关闭当前窗口后再运行下一个。`configs/free_shooting_v1.json` 保存三种控制参数与验收条件；使用球和球门状态输入的预训练策略，加上触球阶段的关节控制。模型与素材仍保存在 Tang 本地，环境准备见 [VERSION_CONTROL.md](VERSION_CONTROL.md)。当前验证的是近距离、静止球的仿真射门；不代表任意起点导航或实机能力。

以下保留早期研究记录与原始研究目标。

## 初始研究资源


- Verified Tang: NVIDIA 4090 Laptop (16 GB), Isaac Sim 4.5, Isaac Lab local checkout after 2.1.0; limited free disk (~61 GB at audit).
- PAiD reference motion dataset: 10 standard + 3 stylized G1-retargeted .npz clips (50 FPS); retrieved via sparse checkout at `third_party/HumanoidSoccer/motions`.
- `tools/motion_audit.py`: schema checks, joint smoothness/integrity descriptors and optional motion-tracking diagnostics. **This is an instrument, not a trained natural-motion generator.**
- Preliminary `reports/reference_inventory.csv` produced by auditing references, not robot rollouts.
- BeyondMimic tracking code (official archived): https://github.com/HybridRobotics/whole_body_tracking
- PAiD soccer code (Isaac Lab v2.1.1 requirement): https://github.com/TeleHuman/HumanoidSoccer

## Run the available audit (CPU only)

```bash
cd /home/you/projects/kinetickick
/home/you/anaconda3/bin/python -m unittest discover -s tests -v
/home/you/anaconda3/bin/python tools/motion_audit.py \
  --input third_party/HumanoidSoccer/motions \
  --output reports/reference_inventory.csv
```

## Reproducibility guards

1. Do not present *retargeted reference motions* as executed robot policy rollouts.
2. Do not label low joint-jerk or low MPJPE as human likeness or aesthetic quality. Such proxies can prefer slow, weak, unproductive movement.
3. Raw `.npz` stores no body/joint names, nor ball/contact labels. Built-in Isaac Lab G1 has 37 joints/44 bodies whereas PAiD motion files have 29/30: NEVER use the built-in G1 semantic ordering for PAiD. Obtain joint/body names from the *PAiD G1 URDF runtime*, validate the array ordering and synchronize contact events before reporting foot-slide, proximal-to-distal sequence or kick-specific timing. Optional `--joint-map` requires `_num_joints` and `_verified_against_source=true`.
4. Do not change the existing Isaac Lab installation. BeyondMimic is for the installed 2.1.0 line, PAiD has a 2.1.1 requirement; isolate versions.
5. Honor upstream data and model licenses. PAiD is CC BY-NC 4.0; do not publish included motions in a new public code repository without license review.
6. For future model comparisons: freeze train/test splits by source clip/player, same robot, same task success envelope, identical physics and training budgets.

## Next gate: genuine baseline replication

- [ ] Obtain clean BeyondMimic checkout + G1 assets in a separate environment (keep under disk budget).
- [ ] Verify Isaac Lab simulator launches and the G1 robot scene can step without GPU OOM.
- [ ] Confirm semantic joint/body mapping **in runtime** and export it to a versioned JSON.
- [ ] Generate one genuine robot rollout from pretrained checkpoint or short training.
- [ ] Export root/joint/body traces and timed contact labels from simulator.
- [ ] Compare real rollout to reference, report timing errors and failure cases.
- [ ] Create matched A/B videos (same robot and camera) for a blinded perceptual study.

## Planned primary research endpoint

Primary: independent human-rated soccer-motion naturalness in **matched, randomized, blinded pairwise video comparisons**, stratified by task and initial ball state. Use more than one rater per clip, measure inter-rater agreement and cluster uncertainty by motion/observer. Report task success and stability as guardrails.

Secondary:
- joint/segment phase relationship, prox-to-distal peak timing, support-foot contact and foot sliding (require semantic/ground-contact logs)
- torso–pelvis–swing-leg coupling and follow-through continuity
- motion diversity at matched functional success
- generalization to held-out ball positions and soccer movement types.

**Hypothesis H1:** an explicit phase-conditioned, morphology-aware coordination prior improves perceptual naturalness relative to pose tracking and generic style reward **without materially reducing** valid contact or stability. It is *not established* until experiments confirm it.

A high smoothness score alone cannot verify H1.

## Study pre-registration skeleton

- Baselines: same reference data, G1 body and training budget: (1) pose-tracking policy (BeyondMimic or replicable equivalent), (2) action style/discriminator with no explicit coordination, (3) proposed phase-aware coordination.
- Ablations: remove relative phase, remove cross-body couplings, remove contact-aware phase transitions, remove morphology conditioning.
- Outcomes: blinded paired preference, successful legal kick, falls, foot sliding, COM dynamics, coordination metrics, runtime.
- No contribution claims before pretrained policy rollout, controlled comparison, and seeded repetitions.

## Citation and upstream

BeyondMimic (2025) https://arxiv.org/abs/2508.08241
PAiD (2026) https://arxiv.org/abs/2602.05310
Biomechanics-inspired soccer kicks (2024) https://arxiv.org/abs/2407.14612


## Historical Tang demo (2026-10-21): **ORIGINAL G1 FOOT ONLY**

The user rejected extra football shoes; the active live scene now uses **only the built-in G1 foot collider** and shows a right-foot medial-side **forward** pass, not lateral sweeping. The old shoe prototype is historical code, not used by the live scene.

Launch on Tang's GUI session (do not duplicate an already running Isaac Sim):

    cd /home/you/projects/kinetickick
    DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_bare_forward_pass.py

Current physics-verified headless comparison at the same ball position (0.40,0.20,0.113)m:
- hip yaw -1.30 rad -> ball after first contact vx=1.951, vy=0.452 m/s
- hip yaw -1.60 rad -> vx=1.908, vy=0.268 m/s
- hip yaw -1.80 rad -> vx=1.631, vy=0.108 m/s (3.8deg horizontal deviation from +X).
- At -1.80, approx local nearest face: **medial side** of original cube (not a directly measured PhysX surface point), filtered force ~80.63 N; successfully crossed 2.2m goal in GUI simulation.
- Root-fixed, task-specific **scripted** controller, NOT human-quality or RL-trained motion; large hip yaw may be unnatural/unsafe for actual G1.
- "Knee only" is currently an uncalibrated exploratory option: same fixed ball position did not yield effective foot contact, so it must not be compared as a successful pass to hip+knee.
- UI goal location is provided by USD stage, not onboard RGB perception.
- All visual demonstrations stay on Tang, no Mac copies.

Evidence: reports/test_bare_forward_inner_foot.log; reports/test_bare_forward_yaw_sweep.log; reports/test_bare_knee_only.log; reports/show_g1_bare_forward_pass.log; reports/2026-10-21-bare-inner-forward.md.


## Historical GUI (2026-10-21): three strike surfaces + calibrated low/medium/high force

Use tools/show_g1_power_kicks_v2.py on Tang with the same Isaac Lab environment and X display.
This supersedes the earlier single-style show_g1_bare_forward_pass.py view.

- Original G1 bare foot, no added contact shoe. Three contact PRESETS: FRONT / TOE-LIKE, INSIDE / MEDIAL, OUTSIDE / LATERAL. Different football start positions are deliberately matched to foot orientation.
- Joint swing completion times: LOW=0.43 s, MED=0.27 s, HIGH=0.16 s. Physics-only impact/speed, no direct ball velocity injection. Contact-point labels are coarse nearest-cuboid-face estimates.
- Extra independently calibrated KNEE ONLY (EXPERIMENTAL) motor-chain option. Fixed thigh, active knee extension; in the calibrated test the fastest prescribed movement was paradoxically slower than medium. Do not claim monotonically controlled force for this mode.
- Default AUTO CYCLE 3 KICKS; CYCLE 3 POWERS holds the kick kind and cycles the three power levels. CLOSE FOOT / WHOLE FIELD views and SLOW / NORMAL display pacing are available.
- Experimental results: reports/calibrate_power.json, reports/calibrate_knee_power.json and reproducible regression checks in tests/test_measured_kicks.py.
- Live physics telemetry and shot history: reports/show_g1_power_kicks_v2.log, reports/live_power_kicks.csv.
- Tang Isaac GUI was visually verified and physically measured as working. This is STILL anchored-base scripted motion, not free-standing humanoid soccer skill learning or onboard RGB camera goal recognition.
- No files are copied to Mac.


## 2026-10-21 研究阶段：人体自然步态 Stage 2（Tang）

当前重点已经从“普通 PPO 可以走路”转为“真实人体动作参考 + 统一29DoF G1本体 + 动力学模仿”。

### 已实测的事情
- 官方37DoF平地走路权重保留。V1/V2各64并行环境、120迭代已训练并保存，但在0.25/0.5/0.8m/s × 2成对扰动场景的联合膝、速度、躯干倾斜与承重脚滑动守门检查中均未通过。因此不能自动替换官方PPO。
- 已从hf-mirror下载5段LAFAN1 G1 29DoF人体衍生retarget动作，30fps、36列、累计34438帧约19.13分钟；以及完整29DoF URDF和38份STL。URDF 59处网格引用均存在，29活动关节与CSV对应。
- 从5段源动作提取各8秒参考片段（跑步参考筛选真正的较快部分），验收关节限制/四元数/运动连续性。生成数据bank（5×240×29关节，含速度、足部运动学伪支撑标签），注意伪支撑不等于测力真值。
- Tang实时REFERENCE viewer使用实际29DoF G1完整渲染mesh，不创建任何额外PhysX刚体或collider。WALK-1/2/3/RUN-1和慢放按钮已验证正常工作。这个 viewer 只是运动学动作捕捉重定向回放，不能宣称训练出自然行走。
- Tang tests: 28 passed。

### 最常用入口
当前Tang观看人体自然动作参考：
  /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_human_reference_walk.py

物理37DoF三策略人工对比：
  /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_gait_policy_ab.py

上面的两个 Isaac Sim GUI 避免同时开启以节省4090 Laptop GPU 16GB显存。
完整研究记录：reports/2026-10-21-stage2-human-gait-reference.md
多速度物理评测：reports/gait_robustness_matrix_20261021.md
29DoF素材审查：reports/lafan1_g1_walk_motion_audit.json
29DoF参考bank元数据：reports/g1_29dof_reference_bank_manifest.json
素材存在data/目录，不上传GitHub、不复制Mac。

### 下一关卡
把29DoF URDF变成物理可控的Isaac Sim/Isaac Lab本体，检查质量惯量、关节轴、碰撞、脚底和PD配置；继而用BeyondMimic跟踪人类动作参考，训练能够维持自由根部平衡、脚底低滑移、自然躯干与臂腿协调的控制策略。当前系统没有找到标准URDF importer插件，本体物理转换还没有完成，不能把运动学回放作为物理成功。
