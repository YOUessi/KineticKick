# KineticKick v0.1.48 — 首次真实训练触球前 29DoF 全身协调 PPO 与完整物理对照

**研究范围：** 原自由根部 G1 29DoF、原PD关节真实速度/力矩限、原足球/脚部PhysX接触、球门与球体判据不变，在触球前施加学得的全身动作，而射后仍由原v32已经训练的PPO处理。真实14秒独立验证依据原5ms物理采样，而不是看训练奖励或踢球演示视频。

## 1. 架构、训练与数据诚实边界

新Actor感知160维：154维原G1全身状态、脚部载荷/姿态和29关节实际速度，再附加3维球相对机器人位置与3维真实球体速度；动作输出29DoF。限制真实触球前1.02–1.54秒、正确支撑/摆腿脚负载、球距及原限速90%以下才可干预，单关节参考最大0.018rad，每20ms变化最多0.003rad。原v32恢复Actor冻结并独立控制触球后身体，不曾从已经评估的六组选择模型。
训练奖励包含原完整后接触稳定性、原5ms真实关节速度惩罚、球体实际出射方向/速度，以及从3DSP职业比赛训练片段获得的一条弱跨部位方向变化参考。该数据为广播视频估计的低频3D关键点，而非专业高帧率动作捕捉、真实脚球接触/支撑反力/足端能量链真值，不提供真正的专业抽射动作自然度监督。
事先固定 seed 4801、并行 G1 环境 64、最大 96 PPO 更新/ 420.0 秒预算；实际执行 96 次更新，真实训练用时 316.25 秒。
18个原本已用于开发的足球初始条件组成训练分布；六组与此前精确参数完全不重合的足球XY偏移/G1根部初速度组合在训练前锁定。它们仍在相同Isaac G1物理任务范围内，不是新足球比赛/机器人硬件环境。

## 2. 禁用新模块时的物理完整性

从原v32中心场景对比的新零干预回合，完整持续2800个5ms真实物理步，共有28个物理字段逐步一致。
原射门、接触、真实关节、球速、机器人恢复与原PPO动作未改变；完整两份JSON可能含不同只读诊断字段，因此不宣称完整文件字节相同。

## 3. 六组预注册新球位14秒真实物理配对

| 球位 | XY偏移(m)/G1初速差y(m/s) | 原v32严格 | 新v48严格 | 新真实全身动作步 | 旧/新出球(m/s) | 新最大实测qdot/原限制 | 旧/新超速样本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| preimpact_new01 | (+0.005,-0.028)/-0.067 | PASS | PASS | 6 | 4.6735/4.6770 | 0.9073 | 0/0 |
| preimpact_new02 | (-0.017,-0.005)/-0.018 | PASS | PASS | 6 | 4.6667/4.6646 | 0.9078 | 0/0 |
| preimpact_new03 | (+0.028,+0.006)/-0.041 | **FAIL** | **FAIL** | 6 | 3.2449/3.2474 | 0.9074 | 0/0 |
| preimpact_new04 | (+0.017,-0.015)/+0.041 | **FAIL** | **FAIL** | 6 | 3.4256/3.4308 | 0.9094 | 0/0 |
| preimpact_new05 | (-0.028,+0.026)/+0.017 | **FAIL** | **FAIL** | 6 | 4.8346/4.8342 | 1.0029 | 1/1 |
| preimpact_new06 | (-0.007,+0.015)/+0.068 | **FAIL** | **FAIL** | 6 | 3.4454/3.4605 | 0.9245 | 0/0 |

**固定原v32完整合法通过 2/6；新触球前全身PPO 完整通过 2/6。**
原失败转为新通过 0 个，原通过退化 0 个。真正非零动作改变的全新球位 6/6。
全任务硬判据一律同时要求完整14秒、原29关节的任意真实qdot样本不超限、合法右正脚背≥95%接触冲量、整球入门、出球≥3.5m/s、正常助跑/击球、无非法身体触球、最终连续站稳；绝不把碰球算进球、把进球算自然或者把训练回合数算成功率。

## 4. 真实失败条件和科研含义

- preimpact_new03: lawful_physical_right_instep_impact, original_ball_fully_inside_goal, original_task_success, real_ball_speed_xy_at_least_3_5_mps.
- preimpact_new04: real_ball_speed_xy_at_least_3_5_mps.
- preimpact_new05: actual_29joint_speed_within_original_limits, original_ball_fully_inside_goal, original_task_success.
- preimpact_new06: full_14s_original_freeroot_PhysX, lawful_physical_right_instep_impact, actual_recovered_original_G1_standing, original_runup, original_task_success, real_ball_speed_xy_at_least_3_5_mps.

新PPO第一次改变真实触球前29DoF动作，仍继承原射门专家作为底层动作模板。当前数据尚不能说明专业运动员式的近端到远端关节加速度序列、髋-膝-踝发力效率、右足/足球法向冲量和身体自然美感是否有任何提升。
六组条件每组每方法只有一个物理回合，样本量小，已有的后续训练不得将本组六组再次当作未知分布；不能给出可靠人口成功率或真实硬件安全承诺。若新模型偶然进球，但原qdot超限或摔倒，同样严格判失败并拒绝晋升。

## 5. 复现和归档

真实模型训练：tools/train_preimpact_wholebody_v048.py；G1全身学习控制：tools/preimpact_wholebody_v048.py；原14秒独立物理运行：tools/run_preimpact_wholebody_v048.py；批次调度：tools/run_preimpact_study_v048.py；真实物理审计：tools/audit_preimpact_wholebody_v048.py。
实验配置及源代码/数据SHA：configs/learned_preimpact_wholebody_study_v048.json、configs/state_conditioned_strike_v19_train.json、configs/state_conditioned_strike_v19_zero.json、configs/state_conditioned_strike_v19.json；测试tests/test_preimpact_wholebody_v048.py。
原始训练检查点、失败/训练日志、完整PhysX数据和第三方体育视频都保留在Tang的Git忽略目录；GitHub只提交代码、测试、冻结配置和真实研究报告。

**结论严格限制：本版是带真实物理反馈的预触球全身小脑探索，不是已经证明自然高爆发正脚背抽射的论文；专业数据、视觉自然度盲评、跨摩擦/球参数分布和多个独立训练种子仍是下一阶段必须完成的工作。**
