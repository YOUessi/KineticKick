# KineticKick — 具身足球论文研究执行记录（2026-10-23）

## 核心科学问题与验收
从优秀人类足球动作学习跨部位全身协调规律，使自由根部 Unitree G1 在不同球位、目标及力度条件下生成自然、连续、物理可执行的助跑→支撑落地→后摆→前摆→物理触球→随挥→站稳动作。动作自然度和跨情景协调关系是预期主要科研贡献；进球率和平衡是功能守门指标。下面报告全部为无球动作的 **Stage I 物理运动跟踪**，不得宣称已经踢球，也不得把人体动作参考的运动学回放说成训练后机器人表现。

## Gate A：上游模型、骨架顺序、物理一致性（已经通过）
- PAiD 上游源码：third_party/HumanoidSoccer，commit e72e470。原版 URDF：third_party/HumanoidSoccer/source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf；原版 meshes 通过 sparse checkout 加入。
- 仅替换 35 个 package:// mesh URI 以便 Isaac 运行；运行版：configs/paid_original/paid_main_resolved.urdf；hash 和改动审计：reports/paid_original_runtime_urdf_manifest.json。未修改 Isaac Lab 全局安装。
- PAiD 原版和现有 LAFAN1 29DoF 本体的 29 个活动关节、轴及运动学参数一致；但碰撞模型和附加固定连杆不同，训练需采用 PAiD 原版，避免不同脚底几何混用。
- unlabeled PAiD NPZ joint_pos 的列顺序是 Isaac articulation runtime order，不是 URDF XML 的关节出现顺序。采用前者在 7 帧的 30 个 body 上得到数值精度的 FK 一致性；错误顺序导致约 0.09248 m 匹配平均位置误差。
- PhysX 自由根部精确姿态写入、关闭 reset 扰动、单物理步一致性检查：7 帧的 PAiD 原版平均 body 位置误差约 0.00133 m、最大 0.01519 m。该结果不意味着高动态轨迹可无误执行，仍须强化学习。
- 历史最初约 0.1513 m 偏差主要受 reset 扰动/参考缓存或比较时机影响，不是整体骨架尺寸误配。
- 当前项目 40 项单元测试全部通过，包含 4 项新 Gate A 回归测试。见 tests/test_paid_original_gate_a.py 与 reports/paid_gate_a_full_tests.log。

## Gate B：Stage I 物理动作跟踪进度（尚未通过）
单条 PAiD standard-001_right 参考，260 帧 / 50 FPS / 5.20s；29 个 G1 关节，free-root PhysX，没有足球、球门命中目标或外加足球鞋。

| 训练配置 | 采样量 | 原版严格阈值评估：首次终止 | 完整 260 帧 |
| --- | ---: | ---: | ---: |
| PAiD 原版奖励的适配 PPO 64×160、seed74 | 245760 env-steps | 约 8 步 | 否 |
| 课程化 96×320、seed84，训练 EE_z=0.40m、加 alive/terminated、关闭早期随机扰动 | 737280 env-steps | **114 步**，2.28s | 否 |
| 相位强化① 96×240、seed84，80–112帧随机起点，35% 从0开始 | 552960 env-steps | **135 步**，2.70s | 否 |
| 相位强化② 96×200、seed84，108–148帧随机起点，40% 从0开始 | 460800 env-steps | **143 步**，2.86s | 否 |

所有 frozen evaluation 采用原 PAiD 0.25m EE_z 终止阈值，统一 MotionCommand 的帧0缓存，完整260控制步尝试。主要指标来自 **一个无扰动初始条件** 的 deterministic policy 轨迹；8 个同条件并行环境不算独立试次。以上训练方案预算和奖励不同，尚不满足严谨消融比较，只能作为探索性回归结果。此时任何盲评自然度、射门成功率、跨任务泛化都没有证据。

## 已定位的重要工程错误与真实失败机制
1. Isaac Lab 2.1 的环境 step() 先计算 termination，再执行 command_manager.compute()；新的 env.reset() 后 body_pos_relative_w 可能全部是零。早期独立评估因此错误地在 step1 终止（相对末端 z 最大错误可达约 1.112m）。已在 tools/eval_paid_stage1_original.py 中通过在首次 policy action 前设置 time_steps=-1、motion._update_command() 修复，确认 frame0 偏差仅约 0.00041m。
2. 初版课程化 strict eval 于 step114 同时触发 anchor_pos_z 和 ee_body_pos。参考在帧100左脚平面速度约 2.96m/s、帧120右脚约 5.22m/s。实际 torso z误差由 step30约0.044m → step100约0.107m → step113约0.247m，明显支撑转换失稳。
3. 相位强化① 135步和相位强化② 143步均仅触发 anchor_pos_z；而不是单纯手脚高度超阈值，故进一步放宽 EE_z 不会解决。V2帧120时 torso 参考0.861m、实际0.730m；帧142参考0.869m、实际0.621m。某些阶段参考右脚高抬但实际右脚仍接近地面，暗示摆腿未正确离地/支持腿承重不足。
4. Phase V2 strict eval: 存活期间14个选定body的平均位置误差约 0.0764m；30N足部受力门控的 ankle rigid-body 平面速度代理均值约 0.1006m/s。这不是精确脚底接触点滑动，也不是“优美程度”。Phase V1对应~0.0691m与~0.0738m/s；更久不代表各指标单调更好。

## 可复现工程文件
- tools/prepare_paid_original.py，tools/audit_paid_urdf_reference.py，tools/probe_paid_physics_parity_clean.py。
- tools/train_paid_stage1_original.py（独立初始基线），tools/train_paid_stage1_curriculum.py，tools/train_paid_stage1_phase_focus.py，tools/eval_paid_stage1_original.py。
- tools/paid_stage1_contact_rewards.py，tools/train_paid_stage1_contact_phase.py：额外骨盆/躯干高度、摆动脚离地和支撑接触力奖励，只用于独立候选研究分支。严格独立评估结果补充于文末。
- reports/paid_reference_urdf_fk_audit.json，reports/paid_physics_parity_clean_paid.json，reports/paid_physics_parity_clean_local.json。
- reports/paid_stage1_trace_phase114.csv，reports/paid_stage1_phase_focus240_strict_trace.csv，reports/paid_stage1_phase_focus_v2_strict_trace.csv。
- checkpoints: experiments/paid29_stage1_original/seed74_n64_iter160/last_model.pt，experiments/paid29_stage1_curriculum/seed84_n96_iter320_ee0p4/last_model.pt，experiments/paid29_stage1_phase_focus/seed84_n96_iter240_phase80_112_zero0p35/last_model.pt，experiments/paid29_stage1_phase_focus/seed84_n96_iter200_phase108_148_zero0p4/last_model.pt。

## 下一关卡与科研规范
Gate B仍未通过。应首先解决支撑转换、躯干塌陷与摆腿离地，完成260帧并多种初始化条件下的稳定性测试；之后进入 Gate C，在有草坪、单个真实物理足球、球门的场景下从稳定走向球到正脚背实际撞击再随挥站稳，不能外部给球速度、固定根部或把运动学动作当策略。Gate D才是接触事件相位+跨部位协调表征的论文创新，需要与PAiD/BeyondMimic等一致预算基线做消融和自然度盲评。

重要限制：参考的13段 PAiD motion 是机器人重定向动作，不能直接等同于新采集的优秀人类足球视频；后续仍需要真实人体视频与动作相位/接触标签。PAiD数据的 CC BY-NC 4.0 许可需要审查，不得随意推送原始 motion 到公开 GitHub。所有原始权重/数据仅留 Tang，不复制 Mac；此次没有破坏性升级 Isaac Lab，也没有进行无关磁盘清理。

## 接触相位奖励候选实验
（运行完成后在本节追加严格冻结测试结果。）
## 2026-10-23 02:29 — Stage I 单一固定场景完成与 Tang 实时 GUI

此前的 Gate B 未完成记录是先前阶段历史结果；最新 contact-phase PPO 已在严格单场景测试中完整运行 260/260 帧。

- 最新模型：`experiments/paid29_stage1_contact_phase/seed84_n96_iter200_phase108_148_contactv1/last_model.pt`。PAiD 原版 29DoF、自由根部 PhysX、PPO；增加躯干高度、摆动脚高度和支撑接触奖励。没有真实足球，也不是射门成功。
- 严格单场景测试：`reports/paid_stage1_contact_phase_v1_strict.json`；在冻结无扰动的一次测试中完成 260/260 步，参考时长 5.20 s；位置跟踪误差均值 0.069493 m，30N 受力门控脚踝水平速度代理均值 0.149223 m/s。单次成功不等于鲁棒。
- 真实 GUI：`tools/show_paid_stage1_physical_live.py`；在 Tang 的 DISPLAY=:0 打开 Isaac Sim 4.5，界面 `KineticKick | G1 PHYSICAL PPO LIVE`。支持 BEST/PHASE/BASELINE 切换、暂停、重启、慢放与相机视角。场地为不参与碰撞的渲染装饰，G1 是真正 PhysX PPO。
- 修复 GUI 重播问题：Isaac Lab 环境 step 使用 torch.inference_mode 导致 command manager 的缓存重置时报原地更新错误，改为 torch.no_grad；GUI 已可连续重播。
- 02:29 时 15 次可视化循环中有 10 次运行 260 步、5 次在高度偏差触发后提前重置。多次循环并非独立随机测试，反映当前物理稳定性还需进一步解决。实时日志：`reports/show_paid_stage1_physical_live.log`。
- 下一步：冻结多种子与随机接触/起始条件评估，再在物理环境加入单一足球，验证真实触球、出球、随挥与恢复。Gate C 尚未完成；不要把无球动作跟踪认作自然足球射门。

在 Tang 物理桌面启动可视化（勿重复启动 Isaac Sim）：

```bash
cd /home/you/projects/kinetickick
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority /home/you/conda_env/env_isaaclab/bin/python tools/show_paid_stage1_physical_live.py --checkpoint BEST
```
