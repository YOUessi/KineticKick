# KineticKick v0.1.128｜原 v54 左腕—左髋净力事件：真实29DoF、几何接近与日志时序取证

**日期**：2026-11-13（Tang 本地，UTC+8）

**工作分支**：`version/v0.1.128`，从已验证 `v127` SHA `e435616866c7285d0e745f23cfaf6ebe62b9a811` 建立全新干净worktree。

**结果之前冻结并推送的协议SHA**：`8ce2aa190847ed60e480811e59a05fd05ab7430f`。

**限制**：严格只读审计历史原29DoF G1真实PhysX轨迹；本批**没有新增物理射门、没有训练或部署新控制器**。由于既有 `v120` 的全项目 `pytest` 进程 PID `242644` 占用 RTX4090 （260MiB、持续CPU负载），并且GPU互斥规则仍然生效，所以不抢占、不终止、不启动IsaacLab。

## 1. 解决了什么实际科研问题

在v54中，CMU10_05原G1于名义时间5.145s出现左髋 `left_hip_roll_link` 与左腕 `left_wrist_yaw_link` 的等值反向ContactSensor净力，单侧大小**25.8901532895N**。此前只能说与自碰撞吻合，不能证明直接接触双方。v127分析了*其他*29回合，却未重建这次特定失败。

v128现在**第一次针对发生该事件的相同三条v54真实原G1轨迹**，从SHA锁定URDF与200Hz实测全29DoF重新计算四组腕—髋原始圆柱的外包胶囊间距下界，并独立与原v52未附加传感器的全部共同物理记录逐字段比较。

### 原始输入资格与精确OFF/ON验证

- 3份原 `v054` 仪表开启PhysX结果全部重新SHA256认证，连同原v54独立诊断审计、等值反向净力分析、原PAID 29DoF G1 URDF，共核验5份实验源文件与1份URDF。
- 此外读取原 `v052` 仪表关闭的3份真实物理JSON，按v54冻结的原SHA再次核验，并在v128中**独立比较9个共同物理字段**：`traces`、`contacts`、`kinematics`、`records`、`sequence_records`、`running_evidence`、`stride_shot_evidence`、`stride_shot_success`、`termination_reasons`；**三次均完全一致**。原控制配置与训练权重SHA也一致。因此原只读传感器未改变这些已有物理记录，但不表示未来新观察器天然只读。
- 原始仪表日志共**6,629条**，其中v54失败尾部有1条 **5.140001s终止标记**，只是5.140000s原物理状态的复制（仅 `time_s` 与 `terminated` 不同）。这不是新的0.005s物理积分步，更不是5.145s碰撞后的29DoF实测状态。去重后共**6,628个实际200Hz物理状态**：2,800 + 2,800 + 1,028。
- 旧 `records[0].success` 报告成功2/3，但这不是新策略成功率、也不是本批重新认证的全部严格门禁成功率。两个正常场景运行满14s；失败场景在5.14s后终止。

## 2. 核心结果：首次把那次真实失败的腕髋几何风险与净力记录关联

### 全观察窗口的URDF原始左腕—左髋圆柱外包胶囊下界

| 原v54策略 | 实际200Hz物理状态 | 原结果 | 该策略全程最小几何间距下界 | 在5.140s的下界 |
|---|---:|---|---:|---:|
| frozen_v32 | 2,800 | 原报告PASS、14s完成 | 11.421mm（12.405s） | 141.270mm |
| CMU10_02 | 2,800 | 原报告PASS、14s完成 | 26.140mm（0.030s） | 83.450mm |
| **CMU10_05** | **1,028** | **physical_nonfoot失败** | **0.011591mm（5.135s）** | **0.013511mm** |

计算方法继承v127完整原29DoF正运动学与URDF `joint origin`、`collision origin`、轴线/圆柱半径及长度；外包胶囊间距下界是保守几何排除工具。**0.011591mm≈11.59微米**，距离零极近，足以说明旧方法可能处于碰撞风险边缘，但不能以近似为零推断两USD碰撞体已经实际接触。

v128额外在**3策略×3个真实采样时刻（5.130/5.135/5.140s）共9个G1状态**调用独立SciPy有界数值优化，重新求三维有限线段最短距离，和v127解析实现的最大差异 **2.78×10^-17m**。这是几何算法的独立数值交叉检验，**不是PhysX contact manifold验证**。

### 时间上的真实观测关联与偏移

- CMU10_05早在 **5.080s**，左腕—左髋原URDF几何下界就首次进入冻结的2cm警戒区，距离5.145s命名传感器事件 **65ms**。最后100ms的20个真实物理状态中，13帧落入该警戒区。这是历史数据上的回顾性预警分析，**不是独立未见场景验证**；v127冻结阈值时已知v54存在失败，不得声称无泄漏的新预测成功。
- 该失败的**trace非脚净力峰值25.8901538849N**在 **5.135s**，恰好与**URDF几何最小间距11.59微米**出现在同一trace物理状态；其前后两步下界约为0.565mm（5.130s）与0.0135mm（5.140s）。
- 与此相对，独立 `nonfoot_contact_diagnostics` 中两条命名净力记录的时间戳为 **5.144999885s**，比 `traces` 中净力峰值 **晚10ms**。这表明记录路径/时间对齐存在需要诊断的偏移，**不能把不同流中的同名物理时刻自动视为同步**。
- 原始 `traces` 最后有效关节姿态为 **5.140s**；5.140001s是被证实的非物理终止标记。因缺少 **5.145s** 的真实29DoF姿态，事件时刻仍有一个近似5ms不可观测区间，**不允许按末两帧线性外推制造新q真值**。
- 原v54命名净力事件还记录两G1 link origin位置相距 **0.113883m**、力方向完全相反。位置是link原点而不是直接的contact point。等值反向净力与极小几何下界共同强化了“同侧腕髋自碰撞”假设，但相互作用对象仍需原生PhysX `actor0/actor1`、`collider0/collider1`、接触点、法向、冲量及同步帧证据证明。

**应使用的科学表述**：在原v54失败回合中，原URDF左腕髋圆柱的外包胶囊下界于非脚净力峰值trace附近接近0，且两命名link具有等幅反向净力；该证据**强烈支持**直接自碰撞作为待检验机制，但**不能认证真正的PhysX接触双方**。原日志具有5.145s真实姿态缺失、约10ms净力时间戳不一致和可能USD/URDF collider差异等限制。Native pair状态仍为 **UNKNOWN**。

## 3. 29DoF/重心/双足与硬门禁没有被忽略

三个场景都使用原29个关节的完整 `joint_pos`/`joint_vel`、原始 `joint_velocity_limits`、COM三维轨迹、双足地面净力代理、非脚净力，计算触球前400ms、后1250ms及失败前100ms，且将触球与14s恢复分别判定。

原数据三个场景的全程29DoF最大速度限额比分别为 **0.9081、0.9081、0.9081**，因此这次v54的故障*不能*归因于一个已记录的原29DoF `qdot` 超限。三例脚背冲量份额均为1.0；CMU10_05尽管进入原记录中的球门，却未通过完整恢复/非脚接触安全门禁。它在5.135s的根高度约0.7856m，原根倾角约3.07°，不能仅凭提前终止或净力事件推断机器人整体跌倒。

双足净力只能作为 **ground support proxy**，不是合格的测量足底GRF/CoP或人类正脚背冲量真值。报告保留所有失败与代理限制，不因为进球或近似几何不接触而更改原始物理PASS/FAIL。

## 4. 实际新增源码与可复现接口

- `tools/v128_v54_event_geometry.py`：按冻结协议逐文件SHA认证、严格识别5.140001s非物理复制终止行、再次直接执行v52/v54原共同物理字段逐值验收、重建3例全29DoF原G1腕髋间距及COM/双足/关节速度阶段动力学，拒绝一切假Native证据，输出独立 `x` 模式审计文件。
- `tools/v128_independent_scipy_geometry.py`：9个真实原G1原始时刻通过独立SciPy约束优化交叉验证近接触的微米级几何计算。
- `tools/v128_plot_v54_event.py`：输出历史v54三控制器腕髋距离对照和失败回合COM/支持/29DoF/非脚净力耦合负例图；所有PNG保留Tang本地。
- `tests/test_v128_v54_event_geometry.py` 和 `tests/test_v128_independent_contact_negative.py`：针对伪造Native认证、错配SHA、数据腐坏、非物理终止行、非同步采样、净力不对称、29DoF安全、独立SciPy验证及真实3例完整重放执行负例测试。
- `configs/v128_frozen_v54_collision_event.json` 与 `docs/V128_FROZEN_V54_EVENT_COLLIDER_GEOMETRY.md`：先于分析推到GitHub的冻结协议。

复现（要求原始Tang大JSON及v116原URDF仍在、GPU资源无需占用）：

```bash
cd /home/you/projects/kinetickick-wt-v128
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/home/you/conda_env/env_isaaclab/bin/python tools/v128_v54_event_geometry.py --output reports/another_new_v128_audit.json
/home/you/conda_env/env_isaaclab/bin/python tools/v128_independent_scipy_geometry.py --output reports/another_new_v128_scipy.json
/home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_v128_v54_event_geometry.py tests/test_v128_independent_contact_negative.py
```

本轮**独立扩展CPU回归163 passed、0 failed**（包含v107/v117/v124/v125/v126/v127及本次全部相关测试）；没有宣称全项目GPU回归通过。

**本次真实审计结果SHA256**：
- `reports/v128_v54_original_event_audit_FINAL.json` → `15f3c0c7169161fac780479f094656d5e1eecbea8cc2bd72483482e251caf363`
- `reports/v128_scipy_independent_FINAL.json` → `727a1a0c5e6993251eab56642b48194c13d5c02ebef486eace7a32bc949285c6`

以上原始审计JSON、历史大原始数据及PNG仍在Tang本地，由原 `.gitignore` 排除；GitHub仅保存源码、协议、测试和论文式文字结论。

## 5. 论文假设与下一批依赖

**H128-A**：不同全身时序控制器会在正脚背触球后改变非踢腿支撑、髋躯干与摆臂—腕部空间位置，导致晚期恢复自碰撞安全区间压缩；仅触球前短弱CMU参考不足以保证完整自然动作的物理安全。v128对已知失败例提供较强的时序与几何支持，但没有因果干预、原生接触所有权及独立未见样本支持，不能断言已证明。

**下一步必须做**：等待原v120 GPU进程结束或由拥有者安全处理后，首先实现原生PhysX collider-pair/manifold正碰撞和分离负控制；记录步进顺序及同一物理步的q、两个碰撞体USD path、contact point/normal/impulse，并与原v54三场景同配置OFF/ON验证全部共同物理状态零干扰。碰撞安全认证后，冻结新种子/新球位、独立人体数据资格与29DoF全身潜在时序生成器，完整覆盖静止缓起、助跑、支撑、髋/躯干和摆臂加速、正脚背击球、随挥落地及14s恢复，不能再以1mrad局部参考或离线余弦/MSE作研究终点。所有严格门禁包括进球、脚背≥95%、球速、每一原始关节qdot、非脚真实碰撞与恢复。

**最终边界**：这一批通过了历史原PhysX完整数据审计、强负例与几何独立交叉验证，但本批**新增真实G1射门0次、新学习策略电机动作0次、新原生接触pair认证0次**；自然高速度职业式抽射仍未实现。
