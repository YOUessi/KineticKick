# 2026-10-22 Tang：修复 LAFAN1 G1 参考回放“漫无目的乱走”

## 真实原因（修复前）
- 旧 Tang GUI 脚本 tools/show_g1_human_reference_walk.py 循环播放 walk1_subject1_selected8s.npz。
- 8秒素材累积移动约4.446m，起终点仅相距2.417m，直线效率只有54.4%；中途转向明显。
- 脚本用时间取模直接回到第一帧，每8秒会将参考网格从终点瞬间跳到起点。
- 该viewer只做29DoF人体运动学参考回放，不具备目标导航或自由根部平衡策略；不能称为机器人已经自主行走。

## 修复
- 新实时GUI：tools/show_g1_straight_walk_reference.py
- 原始动捕参考筛选器：tools/select_straight_gait_references.py
- 非碰撞标记：tools/gait_route_visual.py
- 参考文件：data/human_motion/straight_reference29/*.npz
- 元数据：reports/straight_gait_reference_manifest.json
- 运行日志：reports/show_g1_straight_walk_reference.log
- Tang本地截图：media/g1_straight_a_to_b_goal_reached.jpg

从原始30FPS LAFAN1转到G1 29DoF的运动CSV中选择连续、路径近似直线、朝向与运动方向一致的源数据，只做单次平面世界坐标旋转+平移；29个关节动作不做加工或目标条件训练。

| 模式 | 源文件起始帧(停止帧不含) | 时长s | 目标位移m | 路径长度m | 直线效率 |
| --- | --- | ---: | ---: | ---: | ---: |
| WALK SLOW | walk4_subject1 1200:1380 | 5.97 | 3.275 | 3.315 | 98.8% |
| WALK NORMAL | walk2_subject3 555:735 | 5.97 | 4.651 | 4.681 | 99.4% |
| WALK BRISK | walk1_subject1 2415:2595 | 5.97 | 5.637 | 5.679 | 99.3% |
| RUN STRAIGHT | run1_subject2 3930:4020 | 2.97 | 6.175 | 6.191 | 99.7% |

界面有起点 A、终点 B 的黄色横线，显示实际X位置、目标距离、横向偏差、时间。到达 B 即停止并保持最终姿态，不会自动绕路/回绕；点击 REPLAY A -> B 才重新播放。可切换慢走/普通走/快走/跑以及跟随近景/全路线镜头。

## Tang 实测
- GUI进程：show_g1_straight_walk_reference.py，窗口 KineticKick | STRAIGHT HUMAN WALK A -> B。
- WALK SLOW 从0.000到3.275m，正确标记终点并停下。
- 在真实 GUI 点击 NORMAL WALK，WALK NORMAL 从0.000到4.651m，到B时停止，最终横向偏差0m。
- 31项回归测试通过，旧的无限循环参考viewer已停止。
- 只在Tang，不复制到Mac。

## 未解决/不能夸大
此项目仍只有运动学参考回放，不是G1自主选择目标或模仿人类步态的RL控制；29DoF物理Articulation、动作追踪训练、足地接触和平衡控制尚待实现。B点冻结的末帧仍可能是原动捕中的行走姿态，不应冒称自然停止动作。后续应进一步选择包含真实减速/停步的人类数据并训练真实物理控制策略。
