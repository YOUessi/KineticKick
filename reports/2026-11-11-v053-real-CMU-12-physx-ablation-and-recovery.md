# KineticKick v0.1.53 — 真实CMU全身相位先验：12组预注册G1物理对照与射后失败机制

## 核心研究问题与结论

继续v0.1.52在Tang真实完成的 **2次候选+1次关闭控制器** 实验，不再把离线6维人体姿态余弦MSE减少视作正式自然度证据。v53先冻结4组精确的球位偏移/初始机器人横向速度，每组由同一个原有IsaacLab G1自由根部29DoF仿真分别执行三条**完全不改参数**的控制：原v32小脑、v52 CMU10_02人体协调、v52 CMU10_05人体协调。全部按照原脚球真实碰撞、原电机/关节限速、完整14秒正常终止、正确脚面/球门/球速/跑动/恢复的联合门槛计算，不允许单凭足球已入门判机器人成功。

本批真实物理共 **12回合**、4个相同初始条件，每组原始数据与配置均SHA校验。只有一个人类CMU动捕被试提供两条出球标记可用的片段；人体零点是球标记运动起点**并非**物理脚球接触。没有高水平职业球员动作自然度真值、没有人体GRF/正脚背冲量，也没有多种子或硬件复现。

## 1. 三策略并列：全部物理门禁

| 控制策略 | 严格通过（4组） | 生效非零关节参考的球位 | 非脚接触提前终止组 | 原始关节qdot超速组 | 冻结Wilson描述性区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| v32原冻结 | 0/4 | 0/4 | 1 | 0 | [0.000, 0.490] |
| 真实CMU 10_02 | 0/4 | 4/4 | 0 | 1 | [0.000, 0.490] |
| 真实CMU 10_05 | 0/4 | 4/4 | 1 | 0 | [0.000, 0.490] |

这里的区间仅是小样本描述，**不是置信度足够的真正论文泛化证据**；这些球位均在已开发的相同物理环境和有效控制幅度内。

## 2. 所有球位原始结果（无选择性排除）

| 原始球位与速度 | v32 | CMU10_02 | CMU10_05 |
| --- | --- | --- | --- |
| trial01 xy=[0.0118, -0.0512] v=+0.0878 | FAIL, 4.248m/s, 新参考0帧（legal_right_instep_contact） | FAIL, 4.267m/s, 新参考4帧（legal_right_instep_contact） | FAIL, 4.266m/s, 新参考4帧（legal_right_instep_contact） |
| trial02 xy=[0.0778, 0.0595] v=+0.1168 | FAIL, 2.321m/s, 新参考0帧（legal_right_instep_contact,entire_ball_goal） | FAIL, 2.326m/s, 新参考4帧（legal_right_instep_contact,entire_ball_goal） | FAIL, 2.325m/s, 新参考4帧（legal_right_instep_contact,entire_ball_goal） |
| trial03 xy=[0.0381, 0.1071] v=-0.1333 | FAIL, 2.042m/s, 新参考0帧（legal_right_instep_contact,original_task_outcome） | FAIL, 2.042m/s, 新参考5帧（actual_qdot_under_original_limits,legal_right_instep_contact） | FAIL, 2.042m/s, 新参考5帧（legal_right_instep_contact,original_task_outcome） |
| trial04 xy=[0.0864, 0.1131] v=-0.0820 | FAIL, 2.832m/s, 新参考0帧（real_physx_full_14_seconds,legal_right_instep_contact） | FAIL, 2.838m/s, 新参考5帧（legal_right_instep_contact,entire_ball_goal） | FAIL, 2.836m/s, 新参考5帧（real_physx_full_14_seconds,legal_right_instep_contact） |

相对原v32的配对结果（仅作探索，不做后验调参）：
- 真实CMU 10_02：原 FAIL → 新 PASS：0组；原 PASS → 新 FAIL：0组。
- 真实CMU 10_05：原 FAIL → 新 PASS：0组；原 PASS → 新 FAIL：0组。

## 3. v52中心球位事故：重新区分未恢复、非脚碰撞与真实跌倒

原v52曾将`cmu_10_05`完整射门失败写成“倒地”。新审计逐5ms重新对齐原始三个物理过程，确认原始失败的 termination 标记是 `physical_nonfoot`，它**不等于**已经有可靠的根部翻倒证据：
- v32原冻结：截止14.000000s；根部高度0.7814m，根倾1.58°；相对原v32最大根位移0.000m；termination=[[]]。
- 真实CMU 10_02：截止14.000000s；根部高度0.7861m，根倾5.64°；相对原v32最大根位移0.248m；termination=[[]]。
- 真实CMU 10_05：截止5.140001s；根部高度0.7860m，根倾3.07°；相对原v32最大根位移0.235m；termination=[['physical_nonfoot']]。

失败组在5.14s的根部高度仍超过原高度门限0.45m且倾角远低于60°：能确认的是**非脚部接触触发物理终止，因而未能完成合法14秒恢复**，但不能由该证据声称“机器人整个身体摔倒”。当时旧记录只有总非脚接触力，没有命名具体手臂/膝/髋的碰撞体。旧录制器把最终 +1μs 终止快照作为额外一行；严谨划分为1028个均匀5ms物理步 + 一个终止标记（共1029条保存的记录），不会伪造后续缺失物理轨迹。

另外，机器人姿态的峰值偏离和失稳的发生在足球已离开脚之后，证明“触球前几毫弧度修正 + 对应余弦代理MSE下降”无法作为射后未来安全的硬门槛。需要接触后0–1.25s甚至完整恢复期内实测29DoF/支撑状态和非脚碰撞体，而不是只研究球刚离脚的速度。

## 4. 全链路证据、隔离条件与复现

- 预注册SHA固定协议：`configs/cmu_g1_physics_matrix_v053.json` SHA256=f683e566c9c1190b187515af757233aacddb937b976a85efd528a358289f9ee5。
- 顺序批量真实仿真器：`tools/run_cmu_physics_matrix_v053.py`（GPU互斥、原始文件只写一次、进度断点严验SHA、防覆盖/篡改）。
- 全部12份原始球/本体/脚面/关节速度真实物理独立评审：`tools/audit_cmu_physics_matrix_v053.py`，真实物理明细和报告脚本不自称训练了新网络。
- 中心真实非足碰撞时序与姿态解释：`tools/inspect_cmu_recovery_mechanism_v053.py`。
- 图像保存在Tang本地media/v053_*.png，所有原PhysX完整JSON、CMU原始C3D/AMC、模型权重和参考NPZ依照.gitignore不进入GitHub。
- 单元/来源冻结/全部实验完整性回归：`tests/test_cmu_batch_v053.py`。

## 5. 下一整批研究门禁

严谨的下一批必须解决全身生物力学数据缺乏和触球后恢复控制之间的耦合：先为有明确触球/支撑证据的真实人体高频射门建立 phase/contact/stance 数据表，再研究独立任务安全监督的自然协调条件生成器；用与本批不重复的球位和不重叠的人体被试验证，禁止以自动化跑了多少实验取代研究方法创新。当前两条CMU先验还不是论文的高爆发抽射模型。
