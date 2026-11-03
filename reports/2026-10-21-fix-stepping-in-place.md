# Tang G1「原地踏步」修复核查（2026-10-21 20:06）

## 真正原因
Tang 原先运行的 show_g1_soccer_alignment_drill.py 在进入 lane_align 后，不断将球旁的Y目标在 +0.58 和 -0.58 米间切换，X坐标几乎不再前进。因此视觉上一直横向挪步、原地踏步。它并没有执行真正长距离接近/助跑。

## 新的正式实时演示
主脚本：tools/show_g1_runup_distance.py
日志：reports/show_g1_runup_distance.log
真实物理行走记录：reports/live_g1_actual_runup.csv
窗口名：KineticKick | REAL RUN-UP DISTANCE

新状态机：RUN_UP → FINAL_STEPS → READY（未学会射门）→ RETURN_TURN → WALK_BACK → FACE_GOAL → RUN_UP。
- 起点X=-1.60m；单个0.43kg黑白足球X=+2m；球门在X=5m；草坪仍为非碰撞渲染层。
- 一直使用自由根部、官方G1预训练行走PPO + 真实PhysX脚地接触。
- 往返靠真实机器人行走，没有回合重置/机器人瞬移。
- 面板显示根部实测位置/实测速度、命令速度、距离目标以及累计路径长度。

## 物理实测
第一轮RUNUP_LONG_APPROACH_DONE：step282，前进X 2.607m；
第一轮RUNUP_ARRIVED_PRE_SHOT_ZONE：step378，前进X 3.110m；
第一轮RUNUP_RETURNED_TO_START_PHYSICALLY：step978，X=-1.400m；
RUNUP_NEXT_LAP 1：step1158。
第二轮RUNUP_LONG_APPROACH_DONE：step1429，前进X 2.61m；
第二轮RUNUP_ARRIVED_PRE_SHOT_ZONE：step1525，前进X 3.113m；
第二轮RUNUP_RETURNED_TO_START_PHYSICALLY：step2131；
RUNUP_NEXT_LAP 2：step2310。
行走中实际约0.5m/s，接近时降速，机器人无固定躯干，没有球被人为赋速。

## 限制
本阶段证实的是长距离走向足球与位置准备，而非真正的足球助跑/射门。仍缺自由身体支撑脚落点控制、摆腿、自然动作模仿、碰撞出球以及随挥恢复，不能因为画面会动就宣称已经完成足球动作学习。球门信息来自USD仿真真值，不是摄像头识别。

只有Tang上的实时GUI和本地文件，未复制到Mac。
