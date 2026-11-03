# 2026-10-21 Tang 足球画面与机器人行为核查

## 用户截图说明

截图中的黄色较大球才是足球，靠近机器人脚下的绿色和橙色小球分别是之前从脚球任务规划器生成的“助跑接近位置”和“支撑脚落脚位置”可视化标记，不是3个足球。
机器人显示 TURN TO RETURN，根本不是射门，只是Goal-conditioned locomotion返回下一个助跑起始点。机器人后仰、弯膝与转身生硬是当前通用G1 Flat PPO行走控制器和预设路径造成，不能用更真实的足球模型解决。

## 已执行的场景改造（真实 Isaac Sim GUI）

当前 Tang 实时进程入口：
tools/show_g1_soccer_training_ground.py

- 一颗足球：PhysX 0.43kg，半径0.11m的真实刚体球形碰撞体；没有为纹理另外添加刚体、球或碰撞体。
- 传统黑白图案：12个五边形、20个六边形。三维渲染面板与球体刚体同一USD父节点，跟着球运动；新增的两个三角网格仅渲染用途。实装于 tools/classic_soccer_visual.py。
- 消除了两颗易误认足球的绿色、橙色球形辅助标记；用不会参与物理接触的地面细叉号代替。
- 加宽训练草坪，施加交替绿色条纹，保留原PhysX地面；加入白色训练场线、球门网。实装于 tools/soccer_field_visual.py。
- 尝试中性石墨色头部外观（不修改机器人关节、动作或碰撞）。仍需根据正式G1外观审查头部材质的可见效果。
- 关掉悬浮的Simulation Settings面板，调整观察相机，让足球、机器人与球门更清楚。
- Isaac GUI脚本保留低层真实G1 walk PPO，但当前行为仅为Goal-conditioned approach + re-position loop，不是射门/助跑/足球控制策略；界面主动标记 Navigation only / not kicking。

## 验收检查

已通过2026-10-21的Tang runtime日志：
- FOOTBALL_PITCH_VISUAL_READY
- TRAINING_GOAL_NET_VISUAL_READY
- G1_HEAD_NEUTRAL_VISUAL_APPLIED
- SOCCER_VISUAL_READY: pentagons=12; hexagons=20; physics_colliders_added=0
- SOCCER_BALL_PHYSICS: rigid_objects=['football']; mass=0.43; radius=0.11
- 实时连续行走状态日志继续更新，没有动作回合重置漂移。
- 已从真实GUI截图检查黑白球、绿色训练场以及网门；
- 新增3个足球网格数值测试，总计20个单元测试通过。

## 仍然没解决、不能对用户掩盖的问题

1. 当前机器人在球前转身原路回走属于导航实验，不是人类足球射门，动作不自然、膝盖弯曲和后仰真实存在。
2. 球体虽然为真实刚体，现阶段尚未校准足球空气动力学（阻力、马格努斯力）、表面摩擦、滚动和自转参数；纯视觉黑白面板不等于高保真球体物理模型。
3. 机器人脚部仍是原37DoF G1 简化碰撞配置；并非真人正脚背高保真动作。
4. 只有实现带支撑脚落点/接触时相的人类动作先验、自由根部全身足球策略后，才有资格展示真正的自然助跑、射门和随挥。

## 后续真正推进

- 优先替换“到球前转身回走”的泛化行走演示，不要冒充射门。
- 最后支撑步精度与重心转移 → Free-root射门与左右目标角度控制 → 协调身体的Video→Motion / PAiD / BeyondMimic基线 → KineticKick创新、自然度盲评。
- 保留世界状态观测与USD目标作为oracle；真正摄像头目标识别要另独立验证。
- 只在Tang上实时展示，不向Mac复制文件。
