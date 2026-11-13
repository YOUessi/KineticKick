# KineticMimic｜算法架构、实验合同与研究验收规范

> **身份：方法研究候选（Research Proposal）**。本文件规定将要实现和验证的系统，不意味着已实现 BFM、SuperMimic、真实专业人体冲量监督或新29DoF射门Actor。研发目标与模块优先级见 [科研总路线](RESEARCH_OBJECTIVE_AND_ROADMAP.md)。

## 1. 为什么必须改变早期 Mimic

截至 v0.1.158 的实验说明：

- PAiD提供现成G1足球重定向参考，最早「髋主导/膝主导、脚内侧/正脚背/脚外侧」来自原参考/原预训练控制器+人工关节阶段目标，不是从真人职业足球动作端到端学习。
- v39专业比赛视频估计3D的时序CVAE在独立动作上不及Ridge；v73真实CMU GRU跨被试测试劣于简单线性预测。
- v52真实CMU经可微FK/IK投到G1的700×29参考真实作用仅短时、极小修正；v104/v105十组对比显示旧参考在触球前0.150–0.225秒才出现可测全身变化，不能声称学得完整运动链。
- v108人体时序条件化完整20×29未来速度MAE约0.043854，**劣于**无人体时序的等容量模型约0.037690；新增443个关节软位置余量违例，静态投影虽消掉新违例却未获动力学执行资格；400ms未来触球F1约0.952不及事后训练数据时钟基线1.000。
- v156/v158的原URDF Pinocchio 35D动力学、关节努力约束与离线QP为重要基础，但缺真实浮动基6D、原生碰撞collider/manifold与足底CoP源级真值，不能当作真实高速抽射物理策略。

**新的研究核心不应再是逐帧优化 `||q - q_ref||²`，而是学到可被不同球位/人/运动阶段复用的全身协调行为表示，并通过接触驱动和全身物理控制执行。**

## 2. 模块接口与可替代技术

### 2.1 人体动作表示（Human Motion / Task-Space Encoder）

给定人体姿态、骨盆根位姿、左右脚/手任务空间轨迹：

```python
HumanMotion = {
  "player_id": ...,                 # 真实个体分组，防数据泄漏
  "clip_id": ...,                   # 源片段/时间/FPS/许可和质量SHA
  "pelvis_root_pose": ...,          # 真值/视频估计需分层
  "body_pose_or_markers": ...,      # 身体几何
  "support_phase": ...,             # observed/proxy/unknown，必须有证据等级
  "foot_ball_event": ...,           # verified/proxy/unknown，不允许默认verified
  "ball_position_and_velocity": ..., # 观测/假设不能混用
  "human_GRF_or_CoP": ...,          # 无同步测量时为None，而不是填零
  "source_qualification": ...
}
```

输出可解释的身体相对几何与阶段特征（骨盆、躯干、左右腿、两臂的**跨时间**关系）。数据只提供估计姿态时，不能把相位假设称作真实角动量、肌肉做功或法向冲量。必须能保留原人体运动从助跑到随挥全过程，不得硬选短窗口伪造全序列。

### 2.2 跨形态重定向（Human → Original G1 29DoF）

目标是生成**可被小脑追踪**的G1完整参考，包括：

```python
G1MotionReference = {
  "fps": 50,
  "root_pose": ...,             # 注意实际原G1浮动基36q/35v
  "joint_pos_29": ...,
  "joint_vel_29": ...,
  "foot_task_poses": ...,
  "pelvis_and_trunk_task": ...,
  "stance_events": ...,
  "contact_event_confidence": ...,
  "original_URDF_and_joint_order_SHA": ...
}
```

优化以足端/骨盆/躯干/双臂任务空间相对运动和连续时间运动学为基础，而非直接复制人体角度；必须验证G1原29关节顺序、可达性、关节限位/速度、身体碰撞距离及时间连续性。**源参考离线通过不能升级成真实机器人动作。**

### 2.3 Behavior Latent / Skill Generator（待训练的大脑）

给定完整人体示范 `m_{1:T}`、运动上下文 `c`、潜变量 `z`：

```text
z ~ q_phi(z | human_motion, contact_context)
z_t ~ p_theta(z | G1_observation, ball_state, support_state, shot_goal)
target_{t:t+H} = decoder_theta(z_t, phase_t, context_t)
```

初版优先做小CVAE动作隐空间作为**明确、可复现的Baseline**；研究性添加可解耦的全身动作相位、触球状态条件、运动学约束。比较 Forward–Backward BFM目标嵌入（不假设一定优于CVAE），同预算做无latent与逐帧Mimic对照。latent维度、先验、损失权重必须**仅凭训练和验证**选择，绝不在测试人/新球位上微调后重新叫盲测。

输出至少能区分骨盆/躯干、左右臂、支撑腿和摆腿的真实时序变化；只有输出向量、动作样本或视频，不等于G1 PhysX执行成功。

### 2.4 全身 Mimic Policy（小脑）

输入 `(s_t, target_{t:t+H}, support_feedback)`，输出原29个关节动作，必须在浮动基、GRF、关节限速和真实碰撞环境中稳定跟踪。首先用PAiD与LAFAN1参考建立标准运动追踪基线；SuperMimic、HoloMotion的技术/开源接口是**待评估和复现的候选**。

训练/推理日志必须区分：
- **reference q/dq**：期望轨迹；
- **measured q/dq**：原G1真正PhysX观测；
- **commanded action**：下发的真实29DoF控制动作；
- **physical contact force and ball state**：独立物理状态，不可由动作目标推断；
- **source and model hashes**：允许追溯到底哪个策略真正产生了运动。

### 2.5 接触条件化与真实物理约束（Contact / Whole-Body Controller）

完整策略可含以下损失（具体权重和形式待预注册、对照优化）：

[
mathcal L =
lambda_{mathrm{track}} mathcal L_{mathrm{motion}} +
lambda_{mathrm{relative}} mathcal L_{mathrm{coordination}} +
lambda_{mathrm{contact}} mathcal L_{mathrm{contact events}} +
lambda_{mathrm{impact}} mathcal L_{mathrm{robot impact}} +
lambda_{mathrm{prior}} mathcal L_{mathrm{behavior prior}}.
]

- `L_motion`：身体任务空间/关节参考与执行跟踪，不能等同自然性。
- `L_coordination`：髋—躯干—腿—双臂相对阶段、顺序和共享变化，需从数据估计，严禁将人工相位假设冒充真人实测动量功。
- `L_contact_events`：接触触发阶段变化，训练标签来自真实G1足地与足球接触，或标记为可信的人体接触数据。**绝对仿真时钟、预先固定撞击时刻不是物理感知。**
- `L_robot_impact`：只从真实机器人PhysX足球冲量/出球速度获得监督。球侧矢量冲量要由真实同时间步正确碰撞源测量，非脚净力或力模积分代理不可冒充矢量冲量/专业人体力数据。
- `L_behavior_prior`：可比较AMP判别式、CVAE latent或FB行为先验，不能未经复现宣称已有BFM性能。

**物理硬约束不能用上述软加权替代。** 需要同一实际时间戳的浮动基q/v、全部原29 q/dq/经过真实URDF和驱动认证的qdd/τ、足底法向/摩擦/CoP与接触多边形、真实碰撞collider接触双方/间隙、足球接触姿态、球速/方向、14s恢复。缺关键源时标记 `NOT_CERTIFIED`，不是将缺测值填0或「暂时默认安全」。

## 3. 同预算、同数据的强对照与消融

| ID | 算法 | 用来回答什么 |
|---|---|---|
| B0 | 固定 PAiD/原v32/现有G1策略 | 旧工程化射门能做到什么，失败来自哪里 |
| B1 | Standard Motion Mimic（无latent） | 直接参考轨迹追踪的可执行基线 |
| B2 | Standard Mimic + AMP/style prior | 仅风格分布是否足以改善自然动作 |
| B3 | Whole-Body Latent（无contact） | 行为隐变量是否带来多样性/泛化 |
| B4 | Contact-conditioned Motion Mimic（无latent） | 接触本身是否有效 |
| B5 | Full KineticMimic（latent+coordination+contact+real robot impact） | 新机制是否有独立收益 |
| N1 | 仅右踢腿/冻结双臂 | 是否真正保留全身协调 |
| N2 | 人体时间反转/跨人随机替换/相位标签交换 | 是否真的依赖人类时间规律 |
| N3 | 固定训练集撞击时钟（time-only） | 是否只是记忆每个脚球撞击发生时刻 |
| N4 | 随机/打乱behavior latent、相同latent重复采样 | 隐空间是否可控且有意义 |
| N5 | 无真实contact/无真实impact、只有几何MSE | 几何风格是否掩盖动力学负例 |

所有策略必须在同一原G1 URDF/USD、相同足球物理参数、同一训练预算/环境数、相同初始化条件与真正独立的测试集合上比较。额外HumanX/AMP-Kick/SoccerBot/MetaMotion等相关工作以**真实可复现代码、许可证和接口**为前提再纳入；不把论文展示视频当我们的独立测量数据。

**强反证规则：** 若无人体时序的等容量模型更好、time-only撞击F1更高、或者仅用旧v32时进球而新模型从未实际控制G1，本阶段判研究假设未通过。必须保留负结果。

## 4. 三个层级的实验许可与晋升门槛

### L0：DATA / OFFLINE ONLY

允许运动编码、重定向、轨迹插值、CVAE训练、latent插值/时间反转、静态URDF/QP检验。不得用离线关节MSE、Pinocchio/URDF的拟合结果或足端期望速度声称真实球速、接触力或安全控制。

### L1：CERTIFIED G1 PHYSX BASELINE

原v32/PAiD、已知安全策略在SHA冻结原G1 PhysX运行且通过真实完整帧、严格全部硬门禁。此级仅验证**原基线**和物理基础设施，不能声称新隐空间策略成功。

### L2：CERTIFIED NEW KineticMimic ACTUATION

必须能够给每个真实PhysX回合提供新模型输入、输出的29关节指令、实际原G1 q/dq、G1浮动基、真实命中接触面/球速/足地支撑/非脚碰撞、14秒末段恢复与确认。若原生碰撞证据不足或关节和动力学硬门禁拒绝，实验回到L0/L1；不能因部分球位进门而升级。

### L3：PAPER-LEVEL GENERALIZATION / NATURALNESS

必须有未见人、未见动作、未见球位和有效种子的独立组；生成动作要在合格任务样本上由盲评者比较自然、连续、协调与新颖性，报告评分者一致性、置信区间和失败案例；标准Mimic、AMP、BFM/latent和负控制共同比较。只有L2和L3同时成功，才可以提出自然高爆发全身足球技能方法贡献。

## 5. 核心指标和必须保留的物理门禁

| 类型 | 可报告的指标 | 不可混淆之处 |
|---|---|---|
| 自然性（主） | 盲评成对偏好、相位协调误差、全身动作多样性与新颖性 | 相似MSE/AMP评分只是代理，不能自动定义运动美感 |
| 全身执行（主） | 跑步→plant→coil→swing→impact→follow/recovery阶段覆盖、29DoF时序、根/末端动态 | 时间窗口内非零29关节不等于全身动态技能 |
| 正脚背合法（硬） | 真实机器人触球的脚背冲量份额≥0.95，接触部位/方向/时间 | 触球、球过门、任一非零足球力都不能替代 |
| 实际出球（硬+研究副） | 真实水平/三维球速，历史最低功能阈值3.5m/s、出球方向/进门 | 3.5m/s不是“暴力抽射”标准；比较同条件球速和冲量分布 |
| 安全（硬） | 原29关节q/dq/qdd/τ；真实足地接触/CoP/防滑；非脚接触<20N；完整14s站稳恢复 | 缺失τ或实际runtime collider时不能用0/假定值自动通过 |
| 统计分母 | 球员/动作/球位/实际改变物理随机性的种子数 | 时间重叠窗与只读ON/OFF重复轨迹不算独立样本 |
| 证据资格 | 原输入SHA、确切物理帧、right-censor、原生接触对、运行时URDF/USD | root xyz与tilt不能重建完整root pose/velocity |

## 6. 不该再重复的错误

1. 不把手工相位、关节平滑、静态QP或模型生成的NPZ冒充真实完整射门。
2. 不只挑容易进球的中心球位来报告；同一条件的ON/OFF被动观测是校准，不是增加独立样本。
3. 不用未认证的CMU ball-marker运动起点当人体精确首次脚球触球，不用净force算完整冲量。
4. 不以旧物理里v32成功归功于新CVAE/latent生成，必须有新模型实际命令及其真实影响。
5. 不在已见人/球位/种子上调参后再次宣称新盲测；必须新协议与新留出。
6. 不用原G1 URDF foot local collision-cylinder边界当作USD/PhysX真实接触斑块和CoP。
7. 不因离线指标、数字迭代或版本号增长而隐藏缺失的数据、碰撞资格、安全和真实动作评测。
8. 不将某论文名字、模型复现计划或工程结构误写成已安装、已接入、已验证。

## 7. 代码、设备、GitHub交付约束

- **GitHub 是代码与进度事实源**。每批先冻结实验代码、数据源SHA和协议，随后执行训练/真实物理/独立分析/专项及适用的回归测试，再完整push并用 `git ls-remote` 核对SHA；绝不覆盖旧branch、旧原始物理或更改原生物理门禁。
- 机器人GPU工作遵守全局互斥锁、IsaacLab单进程、设备/显存/磁盘/工作树干净与超时保护。若用户已有长任务，占用方需取证审查，**不抢占或擅自杀掉**。
- 原始受限的CMU/C3D/第三方视频、Git忽略的运动、接触/物理JSON、模型PT/ONNX、图与视频均留原Tang，合法许可证与引用完整记录；不向GitHub推大文件或他人有版权的数据。
- README/本规划是**研究方向合同**，具体经过事实核验的完成情况只能来自对应每个版本的独立报告和本地原始数据。新模型只有通过L2才能成为G1候选控制器，只有L3才能成为论文主结论。
