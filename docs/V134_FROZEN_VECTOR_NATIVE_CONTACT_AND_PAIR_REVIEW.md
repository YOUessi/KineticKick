# KineticKick v134 · 原生PhysX contact向量修复与配对审核预冻结协议

基线SHA 521861d6edb807c71968fc640f975614ea5a10e3（v133）；本协议**先独立提交**、后研发与检验。v131另有未提交全身共享隐变量脚本，不得触碰。原v54的三个固定回合全物理步数为2800、2800、1028。

## A. 真实PhysX ABI修复
根据NVIDIA 2026-07-23 Contact Reports官方文档，callback中contact_data[i].impulse为**三维向量**，不是v133 CPU mock使用的标量；事件头包含num_contact_data、stage_id、type、contact_data_offset。使用三维冲量向量及其模长，不冒称法向冲量；复制原始position/normal/separation和完整actor/collider path。CONTACT_LOST可为零manifold点；空报告不证明无碰撞。

拒绝：标量impulse、非有限数、重复或越界contact片段、假路径、stage/步时钟错配、unknown owner、未覆盖点、伪造正负控制或自称来自native。runtime路径解码只在Isaac Sim环境懒加载；CPU样例始终UNATTESTED。

## B. 原G1 200Hz耦合物理证据
SHA锁定并只读v54原3个物理回合，核查全29DoF、COM、左右足支撑代理、非脚力、最后真实物理帧及命名事件时差：5.135s峰值、5.140s终止前最后真实状态、5.145s命名事件不得填补虚构关节姿态。开发可未来接入真实PhysX的只读事件sink、状态时钟与OFF/ON全部共同物理字段不变性对照；CPU模拟只证明解析功能。

## C. 安全与资格门禁
保留正脚背冲量占比≥95%、球速≥3.5m/s、进球、原29关节速度/加速度/力矩、非脚碰撞、完整14s恢复。未知collider pair绝不授权motor。新增正负例包括vector/scalar、CONTACT_LOST、offset、owner map、时钟、OFF/ON不变、伪造数据来源。GPU忙时新增真实实验为0，不借CPU测试冒充训练或物理。

## 真实系统限制与公开API依据
官方依据：https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html 与 https://docs-prod.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html （后者明确使用impulse[0..2]并解码intToSdfPath）。文档不是本机已取得原生contact的证据，待正/负PhysX控制和OFF/ON真实验证。

Tang的v120全项目pytest PID242644占用RTX4090，绝不抢占、终止或启动第二套IsaacLab。磁盘剩余必须≥20GiB。原CMU/C3D/模型/PhysX/媒体全部只读，JSON/NPZ/PNG和训练权重按.gitignore留Tang本地。未见测试集不能回调参。
