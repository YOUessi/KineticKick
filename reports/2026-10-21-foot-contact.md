# Tang Isaac Lab：G1 三种脚法方向与足球物理接触测试（2026-10-21）

## 研究目标
机器人从人类足球视频中学习自然、优美的全身运动规律，能够运用正脚背、脚内侧（足弓）、外脚背触球。踢球成功率和稳定是约束，不是最主要的科学贡献。

## 当前实时演示（正在 Tang 显示）
程序：/home/you/projects/kinetickick/tools/show_g1_foot_contacts.py
Isaac Sim 4.5；Isaac Lab 2.1.0 系列；内置 G1 37 DoF；质量 0.43 kg、半径 0.11 m 的刚体足球。G1 根部固定，脚部按照预设轨迹运动；足球受到真实刚体接触冲量，未人为赋予踢出速度。Isaac UI 使用英文按键：Instep、Medial pose、Lateral pose、Auto cycle all three presets（中文字体显示异常）。

## 测试结果（GUI 首次脚—球接触，按模式分别测试）
- 正脚背方向：踝 pitch -0.48 rad，接触力约 206 N，足球接触时速度约 4.18 m/s。
- 内侧朝向：髋 yaw +0.48、踝 roll +0.12 rad，接触力约 234 N，足球接触时速度约 4.52 m/s。
- 外侧朝向：髋 yaw -0.48、踝 roll -0.12 rad，接触力约 233.5 N，足球接触时速度约 4.52 m/s。
接触力来自过滤到足球刚体的脚部 ContactSensor；速度是球的仿真刚体速度。三种模式足球初始坐标不同，不能由这些力/速度比较脚法优劣。报告来自 logs/show_g1_foot_contacts.log 及 headless 扫描 reports/contact_probe.json。

## 重要边界
G1 内置模型 ankle pitch 软限位 [-0.609,+0.659] rad（约 -35° 至 +38°）；ankle roll 软限位 ±0.236 rad（约 ±13.5°）。踝单独无法将脚竖直旋转 90°，但是可以通过髋、膝、踝联动使脚整体调整击球姿态。
现有 3 个模式是“脚部朝向预设”，虽确认物理触球，但没有识别碰撞点属于脚背、内侧或外侧；不能声称具备真人的精确三种脚法。机械刚性脚板也缺少真人足弓、柔性脚趾和立体脚背曲面。

## 后续科研任务
1. 获得 Foot-local 接触点/法向，识别命中脚背、脚内侧、外脚侧、脚尖的区域。先检查 G1 当前 USD collision mesh，必要时设计可制造的鞋形碰撞几何或多个局部接触 patch。
2. 在真实视频动作模仿中显式监督：目标脚面、接触点、触球时相、支撑脚、髋膝踝相位、随挥。训练新动作生成策略而非固定关节动画。
3. 取消根部固定，验证自由人形机器人在保持人体运动自然度的情况下仍可稳定触球。
4. 与 BeyondMimic / PAiD 相同初始球位、相同本体、相同物理设置对比，结合盲评自然度、有效接触、失败率、滚动/旋转结果进行完整科研消融。

## 运行与复现
- 脚部三模式实时展示：tools/show_g1_foot_contacts.py
- 物理接触网格搜索：tools/foot_ball_contact_probe.py
- 接触数据：reports/contact_probe.json
- 实时记录：reports/show_g1_foot_contacts.log
- 足关节软限位：reports/foot_sweep_probe.json
- 本项目实时在 Tang GUI 上观察，不再复制到 Mac。
- 说明：此演示是内置 G1 37 DoF 而非 PAiD 29 DoF，暂不能直接用于论文级人体自然运动对比。
