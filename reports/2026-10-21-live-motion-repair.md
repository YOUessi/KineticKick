# Tang 实时仿真“机器人不动”问题与最终演示

2026-10-21

## 诊断

旧程序 show_free_g1_goal_approach.py 在第580步调用运行中 env.reset()，随后GUI日志停止，虽然进程还在运行但三维画面不更新。

使用RL环境自动重置解决了程序停顿，但实际检测发现从第二回合起机器人、足球高度发生约0.74m的非物理漂移，headless长回合对照也复现。直接在GUI中回写机器人和足球的根部状态进行回放仍可能停顿。因此当前方案绕开所有回合重置/传送动作。

## 已验证的实时演示

启动文件：/home/you/projects/kinetickick/tools/show_free_g1_continuous_walk.py

真正自由根部的G1采用官方预训练平地走路PPO，场景包含动态物理足球与球门，真实足底受力。动作以Goal-conditioned planner规划航向和站位，进入准备区后自主转身走回，再转身向足球走去。动作连续，未直接对机器人root做teleport或调用env.reset；不是训练完成的射门策略。

运行日志：/home/you/projects/kinetickick/reports/show_free_g1_continuous_walk.log
真实GUI截图：/home/you/projects/kinetickick/media/g1_continuous_walking_verified.jpg

可验证事件：
- LIVE_APPROACH_TARGET_REACHED_TURN_TO_RETURN：到达准备区，开始转身回走
- LIVE_RETURN_TARGET_REACHED_NEW_APPROACH：返回后再次接近足球
- 已至少重复两个完整往返；过程机器人根部高度约0.62~0.65m，没有出现上述漂移。

Tang GUI窗口：KineticKick | CONTINUOUS G1 WALKING
RUN APPROACH：取消手动停止；HOLD POSITION：临时发零速度命令；GOAL LEFT/CENTER/RIGHT：改变球门并从当前位置重新规划；RESTART WALK：从当前位置重新规划，并不瞬间传送回原点。

运行中脚步真实动，但本演示不是射门、不是助跑技能学习，更不是人体运动自然度结果。不要把GUI射门演示中的固定根部脚摆与这个自由行走演示混淆。

## 后续

核心科研任务依然是支撑脚落位稳定、髋-骨盆-躯干协调、自由平衡下射门动作学习，以及多脚面触球与目标方向控制。
所有输出只在Tang本地，不复制Mac。
