# KineticKick / Tang G1：不同脚面击球 + 力度分档（2026-10-21）

## 用户目标
1. 在实时 Isaac Sim 上看到**明显不同的脚法**，而不是不断重复一套髋关节角度变化的摆腿。
2. 能选 LOW / MED / HIGH 力度，并实际改变 G1 关节动作和物理足球出球速度。
3. 保留原始 G1 脚板和既有足球，**不加足球鞋、不改脚部碰撞体、不往足球直接注入速度**。

## 当前实装
实时脚本：tools/show_g1_power_kicks_v2.py
统一动作配置：configs/kick_modes_v1.json
数据：reports/calibrate_power.json (9次真实物理仿真试验)、reports/calibrate_lateral.json
实时追踪：reports/show_g1_power_kicks_v2.log、reports/live_power_kicks.csv
可视化：内侧 (INNER SIDE)、外侧 (OUTER SIDE)、正面脚板前缘 (FRONT/TOE)，默认 CYCLE 3 KICKS。CLOSE FOOT 近景、WHOLE FIELD 远景、SLOW/ NORMAL 播放速度。
只添加三个**无碰撞的彩色小球视觉提示**（位于目标触球区域）；这些不是鞋，也不参与碰撞。

## 出球力度实测，单位 m/s

| 预设与脚面（碰撞由脚部几何推断） | LOW | MED | HIGH |
| --- | ---: | ---: | ---: |
| FRONT / TOE-LIKE (不能冒充人体正脚背) | 2.199 | 3.800 | 4.908 |
| INNER / MEDIAL | 1.069 | 1.615 | 2.031 |
| OUTER / LATERAL | 1.319 | 1.964 | 2.682 |

测试中改变的是 G1 **髋关节/膝关节设定动作完成时间**，分别为 0.43、0.27、0.16 秒；足球线速度及力来自物理碰撞，不是直接写球速。需要注意：这三个动力档 **不保证在其他球位或机器人本体上仍严格单调**，且主语是固定躯干模拟机，不是自由平衡真人动作。

当前前向 front 实际接触区域在脚板前缘或下表面，不是正脚背曲面。外侧和内侧都是原脚板表面“近似最近面”估计，不是 PhysX 精确 contact point；且常接近侧面下缘/边角。不能将其称为真人高保真的正脚背/脚弓/外脚背已学会。

## 真实 GUI 回归验证
- SHOT_START medial medium -> PHYSICAL_BALL_HIT ... INNER / MEDIAL，峰值球速 1.615m/s。
- SHOT_START lateral medium -> PHYSICAL_BALL_HIT ... OUTER / LATERAL，峰值球速 1.964m/s。
- SHOT_START front medium -> PHYSICAL_BALL_HIT ... TOE / FRONT，峰值球速 3.800m/s。
- GUI 依次切换上述三个模式，日志、球速和接触面估计均改变。
- 11个自动测试通过（6个动作审计、4个脚法力度校准、1个膝关节单独摆动对比）。
- 球门位置仍由 USD stage 提供，非真正摄像头视觉识别。3种模式采用不同起始球位：不能直接将跨模式接触力作为公平方法优劣对比。

## 下一阶段研究，而非本次的虚假成果
1. 真正正脚背/脚内侧/外脚背的高保真碰撞位置和接触法向，需要有可靠的物理/机器人实测。
2. 切换固定基座为自由机器人，测试支撑脚、骨盆、摆腿、随挥等全身协调，而不是硬编码关节序列。
3. 对比大腿带动小腿 vs 仅膝关节伸展，要在均能产生有效触球的条件下重新调整各自动作工作空间，避免因为 knee-only 未击中而做出错误结论。
4. 完整科学结果的主评价仍然是运动自然度、正确触球区域、连续性、机器人稳定性和功能表现，不是单纯球速。

## Tang 图形桌面启动
cd /home/you/projects/kinetickick
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_power_kicks_v2.py

**只在 Tang 实时展示，没有也不会自动复制到 Mac。**


## 2026-10-21 最终实时 GUI（取代旧版）

当前 Tang GUI 脚本：tools/show_g1_power_kicks_v2.py，窗口 KineticKick V2 | KICK TYPES + FORCE，**仍然只有原生 G1 脚板碰撞体**。

具备两个独立观察维度：

1. CONTACT REGION（脚部不同接触区域）：FRONT (TOE)、INNER SIDE、OUTER SIDE。分别通过接触几何近似判断前缘、内侧与外侧。主界面默认 CYCLE 3 KICKS 自动依次播放、CLOSE FOOT 近距离观察。足部黄色/绿色/蓝色小球只是无碰撞的提示符，不是足球鞋。
2. MOTOR CHAIN（发力运动链）：THIGH -> SHANK（髋-膝联动的正面击球）与 KNEE ONLY (EXPERIMENTAL)（固定大腿、膝关节单独伸展）。膝单独伸展在相同足球初始点实现物理触球，但不同输入执行速度不保证出球速度单调。

### 膝单独摆动实测

在球位置 (0.38,-0.12,0.113)m 时：
- Low -> 球水平速度峰值 0.793 m/s，瞬时最大球力 40.0N。
- Medium -> 1.206 m/s，63.1N。
- High -> 0.776 m/s，41.5N。

由此可见电机动作更快未必让球运动更快——触球发生在不同摆腿相位和几何接触区域，必须联合优化发力、运动节奏和球位。该分支只作为 EXPERIMENT，不与已校准的三档脚法强弱混同。

已在真实 Tang Isaac GUI 人工点测：KNEE ONLY 按钮触发 SHOT_START knee_only medium -> PHYSICAL_BALL_HIT ... TOE / FRONT，球速峰值 1.206 m/s；CYCLE 3 POWERS 按钮触发同一击球动作 Low 2.199 -> Med 3.800 -> High 4.908 m/s 的真实球速变化；最后恢复 CYCLE 3 KICKS 自动演示。

运行日志：reports/show_g1_power_kicks_v2.log；每一次 GUI 射门的物理结果写入 reports/live_power_kicks.csv。代码配置 configs/kick_modes_v1.json；脚法物理数据 reports/calibrate_power.json；小腿实验 reports/calibrate_knee_power.json；回归测试 11/11 通过。

本研究距离“优美、自然、真实的人类足球动作”仍有距离：根部固定且动作是预设关节轨迹，并非经过动作模仿/自然度优化的 RL 策略，也没有真正的摄像头视觉识别。后续要引入动作跟踪、全身支撑、平衡以及关节相位学习。
