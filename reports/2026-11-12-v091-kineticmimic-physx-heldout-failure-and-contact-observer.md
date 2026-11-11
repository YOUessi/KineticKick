# KineticKick v0.1.91 — KineticMimic 因果全身接触观察器 + 4次新原G1 PhysX零成功负例

2026-11-12（UTC+08）。独立研究分支 `version/v0.1.91`；来自已经发布的v88主事实提交 `c87672181f0a8224aeaece6e11dae2a06a6f5ab8`，未依赖正在编写的v89/v90未提交内容。**冻结协议及执行器在任何新G1物理前提交**：`b4220a0822076390fd88bd0c8f2e8975cd136533`；物理矩阵 SHA256 `ebfcb676a198552c7c06dbdeca88f9da8eab09c315d7977dd629c08d9024ed64`。

## 1. 实际执行与安全约束

Tang Ubuntu RTX4090在线。开始前无GPU计算进程，独占项目共享`kinetickick_physics_gpu.lock`，空闲磁盘约37GiB（硬门禁20GiB），原始CPU/PhysX/CMU及weights SHA只读验证。新干净工作树在 `/home/you/projects/kinetickick-wt-v091`；根据v84现存只读来源建立1903个**缺失文件**的符号链接，没有覆盖已存在路径。代码、冻结协议、检查脚本、测试与文字报告提交；30MB PhysX JSON、NPZ、训练权重、图及日志保留Tang本地，不进GitHub。

冻结两个之前未使用的球位 E(+0.12,+0.02)m / seed1319 / velocity delta +0.02m/s；F(-0.10,-0.04)m / seed1327 / delta -0.02m/s。各对照 `v32` 和未改动的 `CMU10_02` 控制器，采用自由根29DoF Unitree G1、200Hz CUDA PhysX。seed不同**不等于真正独立的根初态扰动**。所有4个子进程均按原source SHA和物理起止记录认证；无超时、无重做、无数据覆盖，原控制器没有被新观察器改变。

## 2. 4次新增真实G1 GPU PhysX：严格通过0/4

| 冻结条件 | 真正200Hz帧 | 触球后球速5–110ms代理 | 正脚背冲量份额 | 非脚净力峰值 | 独立门禁 |
|---|---:|---:|---:|---:|---|
| E_v32 | 736 + 1终止注释 | 3.2208m/s | **0%（ambiguous）** | 48.474N | **FAIL**：脚背、球速、非脚、14s恢复 |
| E_cmu02 | 2800 | 3.2955m/s | 100% | 0N | **FAIL**：球速低于3.5 |
| F_v32 | 320 + 1终止注释 | 5.1418m/s | 100% | 180.009N | **FAIL**：未进门、非脚、14s恢复 |
| F_cmu02 | 320 + 1终止注释 | 5.1432m/s | 100% | 192.983N | **FAIL**：未进门、非脚、14s恢复 |

总计 **4次真正原G1 CUDA PhysX执行、4176个真实5ms物理状态、3个+1us非物理终止注释、严格0/4通过**。独立脚本重做全部29关节实测速度（最大关节限速比约0.908，不超限）、脚背冲量分箱、球速、左右足支撑、COM、非脚强制终止、14s恢复的联合门禁，并与预注册子进程的严格判定4/4一致。**进球不能代替正脚背有效冲量；5.14m/s也不能代替进门、身体安全和随挥恢复。**

失效定位：
- E_v32：虽然球进入球门，但首次球接触表面分箱为`ambiguous`，真实正脚背份额0%。约3.675s产生**right_wrist_yaw_link ↔ right_hip_roll_link 两个刚体的反向净力**（峰48.474N）并非脚安全终止；由于原生contact manifold/collider路径未认证，只能叫强间接证据，绝不叫实测碰撞双方。
- E_cmu02：正脚背、进门、全关节限速、完整14秒和恢复均通过，但球速仅3.2955m/s，故独立硬门禁FAIL；不能把原来源的宽松`success=True`误当全门禁通过。
- F两条件：真实右脚正脚背触球，速度约5.14m/s但球未进门；约1.61s双双出现**pelvis与right_hip_roll_link 反向净力**（180.009/192.983N）后非脚终止，14秒恢复不存在。相等反向净力仍不是形状级pair/manifold真值。F两条**仅有触球后约35ms**真物理数据，不能假装有触球后1.25s随挥。

E两个实验具有触球前0.4s与后1.25s真实窗口；F的触球后窗口**不完整，所有后续恢复/随挥评价均作censored**。4张真实200Hz动态图均保留本地并含原始SHA。

## 3. 本批真正训练的KineticMimic第一阶段：causal contact-conditioned全身 observer，非控制器

新增 `tools/v091_coordination_observer.py`：独立读取实际原G1 PhysX、源SHA、29关节实测q/dq、真实双脚载荷/支撑模式、COM、球相对位置、足端速度，50Hz采样，预测真实+100ms全部29归一化关节速度、未来支撑模式和窗口首次脚球接触标签。输入严格只含当前可观测量，**没有未来接触时刻/真实冲量/人类GRF或测试标签泄漏**。

数据：原v84的C/D两个不同球位，每场景`v32`、`CMU10_02`；测试新冻结E/F（4场景-控制组合）。所有模型同用固定lambda=10和仅训练集拟合的标准化，5个独立冻结方法：
`velocity_persistence`、`clock_only_ridge`、`state_only_ridge`、`contact_conditioned_ridge`、`contact_feature_shuffled_negative`。先用v84 C->D做开发审计，再用所有C/D训练、E/F holdout，**没有依据E/F改动特征、lambda或物理门禁**。训练生成的权重`reports/v091_*coordination_observer.npz`仅存在Tang。

新holdout在真实first-impact窗口的归一化29DoF qdot MAE（越低越好）：

| 方法 | E两条完整窗口平均 | E/F四条含F截断窗口（诊断性） |
|---|---:|---:|
| 直接速度持续 | 0.077492 | 0.083170 |
| 仅时间clock | 0.052804 | 0.059700 |
| 当前机器人状态（无接触条件） | 0.031933 | 0.031791 |
| **当前支撑/contact-conditioned** | **0.031685** | **0.031337** |
| 支撑条件序列打乱负对照 | 0.031954 | 0.031790 |

仅E完整窗口，接触条件相较state-only改善约**0.78%**，然而支撑状态预测macro-F1反而下降（E平均0.7853→0.7728）；髋/腰/支撑/摆腿/对侧摆臂峰值相位时序误差均约**0.104s**，无改进。F因安全早停仅有删失窗口，不能纳入完整运动自然性/恢复结论。**因此没有足够证据声称Contact-conditioned模型已改善人体式全身自然动作**。此模型本质是“实际G1短时观测预测”，不是从优秀真人学到的29DoF动作生成器；从未向机器人电机输出新动作，新增学习策略真实射门数=0。

## 4. 真人人体数据资格研究与论文技术路线

新增 `docs/V091_HUMAN_DATA_QUALIFICATION.md` 与 `docs/V091_KINETICMIMIC_METHOD.md`。本地6段CMU C3D属于真实120Hz光学人体几何，**无可靠职业正脚背实时GRF/球冲量**。网页独立查证[GroundLink官方同步全身C3D+GRF/CoP](https://csr.bu.edu/groundlink/)（有正式真实支撑动力学数据）以及[足球头球落地/侧步数据](https://data.mendeley.com/datasets/hscscdhg49/1)（并非射门）。都不能直接作为真人专业正脚背冲量真值；**当前全资格的优秀真人射门样本仍为0**，未下载或虚构数据。

拟研究的真正新 Mimic 需联合事件条件的支撑脚—髋/骨盆—腰躯干—摆腿—对侧摆臂时序、允许跨形态差异的Motion、真实G1脚球Contact/Impact和14秒Recovery；已在方法文档写为**可证伪新假设**，不是已完成算法。原G1 GPU collider距离及真实manifold未经认证前，新策略fail-closed，只能比较原安全基线。

## 5. 软件实现、独立验收与可复现性

新增/冻结：
- `configs/v091_frozen_matrix.json`、`docs/V091_FROZEN_PROTOCOL.md`、`tools/v091_gpu_batch.py`：锁SHA的GPU互斥、源哈希验证、4个write-once子进程、严格停止、原全29关节门禁。
- `tools/v091_coordination_observer.py` + `tests/test_v091_coordination_observer.py`：真实G1数据加载、状态/接触条件模型、删失窗口、场景隔离、5种消融与数据泄漏负例。
- `tools/v091_independent_dynamics.py` + `tests/test_v091_independent_dynamics.py`：独立逐源SHA、终止非物理行审查、实测非脚/COM/左右足力与真实相位、严格门禁独立比对、4张原G1物理图。
- `tests/test_v091_protocol.py`：不泄漏新球位、冻结配比和强硬安全门禁。

完整项目CPU回归：**1011 passed / 2 skipped / 69 subtests passed / 0 failed**，日志SHA256 `e6af359a8a86c32c9e9d6b2a8b3fd5f39e157150a13ac9f11a7ac77016d55d1e`。针对性测试20/20通过。原项目其余所有分支、CMU/C3D、权重、30MB真实物理JSON未覆盖或删除。

Tang本地核心证据：
- 冻结G1 GPU批次：`reports/v091_g1_physics_final.audit.json` SHA256 `bae59ae4955e825aa0352862769e07e3574052664955fc89eda8596cb67cf560`
- 独立动力学审计：`reports/v091_independent_dynamics.audit.json` SHA256 `0132a62d148ece92cb52ba39b858c3d9bcb404931acd6a1c6b03532df9d1f370`
- 真实新场景 holdout：`reports/v091_holdout_coordination_observer.json` SHA256 `077da2370a927c3a26a94efd0caca3338a098dc8e5648eff85424b9ea37f07e3`
- 4张失败图：`reports/v091_physx_{E_v32,E_cmu02,F_v32,F_cmu02}.png`，每张SHA见独立audit；全部仅Tang本地。

## 6. 论文结论等级与下一批

结论：v91 **完成4个全新、原始29DoF真实物理对照，并取得0/4严谨负结果；完成一套全身因果contact observer和独立场景实验**。接触条件化未显著改善相位自然性，新策略还没有经过PhysX执行。最重要的新技术障碍不是缺少几个奖励项，而是**非法/非脚净力事件在新侧面（右腕—右髋、骨盆—右髋）再次复现，且原生报告系统依旧不能认证collision manifold**；同时真实优秀人类专业全身射门和冲量监督仍缺失。

下一批：优先修复/正控制校准原生actor/collider/contact-manifold + 与原G1只读一致性；查准E/F的真实右侧危险形状；资质可用的GroundLink GRF用于支撑规律预训练，独立搜集专业足球全身合法数据，**先真实学会人体协调再修改29DoF动作**；最后冻结新球位真实actuation消融并对自然性/高速度正脚背/支持/恢复一起验收。没有取得相应证据前，不能宣布“自然爆发抽射成功”。
