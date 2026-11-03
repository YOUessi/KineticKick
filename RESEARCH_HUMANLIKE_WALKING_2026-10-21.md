# KineticKick：人形机器人自然步态第一阶段研究记录（2026-10-21）

## 研究目标

先让 G1 学会自然步行：支撑期适度伸膝、摆动期自然屈膝、完整支撑—摆动时序、脚不打滑、骨盆/躯干协调和反向摆臂。随后才研究助跑、支撑脚落位和自然射门。稳定行走与动作自然性是两个必须分别验证的目标。

## Tang 已执行的内容

- NVIDIA 官方 G1 37DoF 平地 PPO 作为基线，不修改官方权重。
- 建立完整 gait 评估程序：tools/eval_humanlike_gait.py，可采集关节、左右脚接触力、站立/摆动相位、躯干倾角、速度和脚部滑动代理指标。
- 45N 接触力阈值之外，又采用5/15/30/75/120N敏感性检查，排除只在一种阈值下“迈步过快”的假象。
- 第一轮 V1：64环境、120 PPO 更新、184320 采样，添加支撑期直膝奖励和腾空惩罚（同时小幅调整原步态奖励）；训练完成，独立权重保存。
- 第二轮 V2：从相同官方起点重训相同预算，强化防滑与躯干姿态惩罚；训练完成，独立权重保存。
- 两轮候选和官方在相同0.5m/s速度命令、410控制步、评估种子42下进行 A/B 对比。
- 增加 Tang 实时观察界面：KineticKick | WALK NATURALNESS A/B LAB。已实际测试 OFFICIAL PPO / V1 KNEE / V2 SUPPORT 按钮切换成功，GUI 能持续往返走路超过4圈。头部绿蓝速度箭头关闭，足球场和单个黑白足球不变。

## 实测对比

| 指标 | 官方 | V1 | V2 |
| --- | ---: | ---: | ---: |
| 平均前进速度，m/s | 0.3880 | 0.4163 | 0.4157 |
| 跟踪0.5m/s指令的平均绝对误差，m/s | 0.1120 | 0.0837 | 0.0843 |
| 左膝支撑期绝对关节坐标中位数，rad | 1.0207 | 0.8197 | 0.8630 |
| 右膝支撑期绝对关节坐标中位数，rad | 1.0147 | 0.8985 | 0.9144 |
| 躯干倾斜角P95，度 | 3.662 | 7.536 | 6.425 |
| 左足承重时脚踝刚体平面速度P90，m/s | 0.0290 | 0.1250 | 0.1187 |
| 右足承重时脚踝刚体平面速度P90，m/s | 0.0434 | 0.0751 | 0.0668 |
| 无脚承重采样比例（45N），% | 13.51 | 5.19 | 3.12 |
| 跌倒/环境重置次数 | 0 | 0 | 0 |

## 结论（重要）

新的奖励能减少支撑膝关节屈曲，并在单场景提高速度跟踪，但两版都使躯干倾斜与足部滑动显著高于官方。因此二者都没有通过完整步态质量门槛，不能称之为更像人类走路或可以直接替换官方策略。步态自然性不是单一膝关节角/腿更直能代表的指标。

其中脚部水平速度是负载状态 ankle-roll 刚体速度代理，不是细分鞋底接触点的准确滑动，需后续更高精度接触建模。

## 下一步：人体动捕+模仿与强化学习结合

1. 引入 LAFAN1 人体步行、慢跑、起步/停步/转身片段；从原始人体动作中提取骨盆—髋—膝—踝、臂腿相位、支撑/摆动时间和脚跟到脚尖的过渡规律。
2. 统一机器人体：当前NVIDIA官方平地控制37DoF，而公开PAiD/BeyondMimic/LAFAN1 G1资源主要使用29DoF。不可直接复用关节索引或权重。优先通过官方Unitree RL Lab的29DoF模型和完整USD/机器人网格创建并验证统一的研究本体。
3. 复现 BeyondMimic 动作跟踪基线。训练物理可执行的人体动作条件策略，而非直接执行 kinematics-only 关节角视频。
4. 提出相位/形态条件的全身协调先验，通过脚部接触、躯干平衡和任务速度三类约束阻止奖励Hack。
5. 多速度和种子的稳定性验证后，用匿名配对视频由独立评审对动作自然度评分；向自然足球助跑/射门迁移。

## 工程阻塞与不做的事

- Tang目前到huggingface.co的网络访问超时；29DoF正式机器人完整 USD/mesh 和原始人体动捕素材还未完整获取。
- 训练的V1/V2是参考无关的生物力学奖励预实验，不是已经完成LAFAN1模仿训练。
- Tang UI中途切换权重不是严格A/B评测；只用同 seed、同起点的headless完整轨迹比较数据。
- 不会对现有G1安装环境作破坏性升级，不会向Mac复制素材，不会把原数据随意推GitHub。

## 复现入口

研究目录：/home/you/projects/kinetickick

- 训练奖励：tools/humanlike_gait_rewards.py
- 训练器：tools/train_gait_pilot.py
- 评测器：tools/eval_humanlike_gait.py
- 接触敏感性：tools/gait_contact_sensitivity.py
- 对比工具：tools/compare_humanlike_gaits.py
- 官方数据：reports/human_walk_baseline_v050.csv + json
- V1数据：reports/human_walk_pilot_v050.csv + json
- V2数据：reports/human_walk_v2_v050.csv + json
- V1权重：experiments/gait_naturalness_pilot_20261021/seed40_env64/last_model.pt
- V2权重：experiments/gait_naturalness_pilot_20261021/v2_seed40_env64/last_model.pt
- 交互GUI：tools/show_g1_gait_policy_ab.py
- 运行日志：reports/show_g1_gait_policy_ab.log

论文参考：
https://arxiv.org/abs/2505.20619
https://github.com/HybridRobotics/whole_body_tracking
https://github.com/unitreerobotics/unitree_rl_lab
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
