# KineticKick / G1 原始脚板正向脚内侧推球实验 (2026-10-21)

本实验按用户要求完全取消足球鞋和鞋形碰撞体。G1 原生脚板约 20.3 cm 长、6.55 cm 宽，脚内侧面本身可与足球发生刚体碰撞。右髋旋转带动脚板内侧朝球门，腿向前摆动，而不是侧向扫球。

## 实测物理结果

Isaac Lab 2.1.0、Isaac Sim 4.5，根部固定 G1 37DoF，球直径约 22 cm、质量 0.43 kg。相同球初始坐标 (0.40,0.20,0.113) 米，相同髋膝运动链：

右髋 yaw -1.30 rad：首次触球后球速度 vx=1.951m/s，vy=0.452m/s；水平偏转约 13°。
右髋 yaw -1.60 rad：vx=1.908m/s，vy=0.268m/s；偏转约 8°。
右髋 yaw -1.80 rad：vx=1.631m/s，vy=0.108m/s；偏转约 3.8°，脚—球碰撞力约 80.6N。

在 -1.80 rad 例子中，球心在足部刚体坐标系的位置约 (0.004,0.124,-0.096) 米。根据足部立方体最近表面估计，主要是内侧面而不是脚尖，但不等同于精确 PhysX 接触点（存在下缘角接触可能）。

实时三维场景使用完全没有附加鞋模型的原生 G1，名为 KineticKick | ORIGINAL G1 FOOT。默认以髋膝联动、右脚内侧推球。读取球门 USD 三维位置后依据之前的出球数据选择髋 yaw，摄像头自主识别尚未实现。球已真实穿越距离 2.20m 的研究用球门，记录 GOAL_CROSSING INSIDE。

## 文件

运行脚本：tools/show_g1_bare_forward_pass.py
两批物理试验：tools/test_bare_forward_inner_foot.py、tools/test_bare_forward_yaw_sweep.py
日志：reports/show_g1_bare_forward_pass.log
试验结果：reports/bare_forward_inner_foot_tests.json、reports/bare_forward_yaw_sweep.json
界面实拍：media/g1_bare_forward_live.jpg

## 局限

1. 这不是从人类视频学习的自然足球动作，根部被固定且髋外旋幅度非常大；当前只是在不加鞋的前提下验证“脚内侧面向正前方→足球向前运动”的物理可行性。
2. 需要联合身体/骨盆/支撑步姿态减少髋部极限旋转，使整条动作链像人而不是固定躯干转腿。
3. 必须验证更真实的人体支撑平衡与精确接触区域，不把 G1 矩形脚板误称为真实人体足弓。
4. 人类视频动作自然度是论文目标；球是否进入球门是功能约束而非最主要科研创新。
5. 当前机器人是 Isaac Lab 内置 37DoF G1，不能不校准就使用 PAiD 29DoF 动作。

按照用户明确要求：只在 Tang 实时展示，不复制到 Mac。
