# 2026-10-21 Tang G1 头部箭头与自由根部射门排查

## 用户纠正，已找到准确原因

固定基座 G1 触球演示没有箭头；切换官方平地行走 PPO 后，出现绿蓝两箭头，看似长在脑袋上。并不是模型头部损坏或机器人换了脑袋。

真实来源：
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py 的 CommandsCfg.base_velocity.debug_vis=True；
IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py::_debug_vis_callback
在 robot.data.root_pos_w + (0,0,0.5m) 位置渲染 goal_vel_visualizer 和 current_vel_visualizer 两个速度方向箭头。

## 实际修复

- 从未修改的 tools/show_g1_runup_distance.py 复制干净 G1 程序：tools/show_g1_runup_no_arrows.py。
- 场景创建前设置 cfg.commands.base_velocity.debug_vis=False，不修改任何G1机器人模型/关节/头部USD。
- 运行时以 G1_HEAD_ORIGINAL_INTACT True velocity_debug_vis False goal_marker_exists False 验证 debug marker 没生成。
- Tang GUI 实际截屏确认 G1 头部没有绿蓝箭头（media/g1_no_velocity_arrows.jpg），仍能向前行走。
- 窗口名：KineticKick | ORIGINAL G1 / NO ARROWS。
- 当前live程序不再应用任何新的头壳修改。

## 为什么尚不能展示真正射门

当前通用 flat G1 PPO 仅学会站立、行走、转向，没有射门策略。之前固定根部 G1 射门是关节脚球物理原型，不能直接复用到自由根部。

进行了两次自由根部、真实0.43kg PhysX足球和脚地接触试验，并修复动作输出的关节顺序映射、推理张量的原地修改导致的卡顿。

第一组：球(0.23,-0.12,0.113)，尝试注入手工摆腿。
- 第0.20s发生脚球接触，力约8.58N，足球水平速度峰值1.314m/s；但这是正式摆腿之前的误触球，不能叫射门。
- 机器人随后失稳，最低根部高度约0.14m，1次场景重置。
- 日志：reports/free_root_kick_complete.log。

第二组：球(0.48,-0.12,0.113)，较低幅度手工摆腿。
- 足球接触力0N，球速度0m/s，完全没有触球。
- 机器人失稳、最低根部高度约0.115m，1次场景重置。
- 日志：reports/free_root_kick_ball048_low035.log。

结论：固定躯干射门动作硬叠加到自由行走 PPO 会导致误触球、踢空或摔倒；当前没有真实的、稳定的足球射门策略。

## 下一步研究门槛

1. 在通用平地行走底座上训练最后一步支撑脚目标落位和髋部/骨盆重心协调。
2. 验证支撑脚接触力、足底不滑、机器人姿态和触球位置，才能准许进入射门阶段。
3. 训练物理可执行的自由根部射门策略，包括摆腿、触球、随挥、重心恢复。
4. 融合真实人类足球视频的相位与全身协调先验，动作自然度作为论文主要研究目标。

所有东西保存在 Tang，不复制到 Mac。
