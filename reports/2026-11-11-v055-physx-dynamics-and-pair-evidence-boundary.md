# KineticKick v0.1.55｜真实 PhysX 15 回合、200Hz 全身失效动力学与接触双方证据边界

日期：2026-11-11。基准：`version/v0.1.54` 的 `3cac1736f72218cf7bb864059f3e0a21c08b7da6`。开发分支：`version/v0.1.55`。

## 本轮结论（不得略去负结果）

**15 次真实原 Isaac Lab/PhysX/G1 29DoF 自由根部闭环物理运行已完成**（3 次原中心球位精确复跑 + 2 种新球位/横向初速度 × 3 原策略 × 2 种预先指定 seed = 12 次）。不是离线重放、姿态 MSE 或脚部运动学演示；保存全部原始真实 5ms 物理轨迹 JSON，共约 285.5 MB，仅留 Tang / `reports/`，每条 SHA256 保存在本地审计 JSON。GPU 顺序互斥、未改动原 PD、质量、接触几何/摩擦、非脚终止、恢复网络或球体运动；未训练新策略，也不晋升模型。

**冻结 holdout 严格成功：原 v32 = 0/4，CMU 10_02 = 0/4，CMU 10_05 = 0/4。** 这里分母 4 表示物理执行的回合数，**不是 4 个独立随机条件**。经新追加的独立 seed 审计，1201 与 1509 在两类场景、三个策略的 6 对物理 `traces/contacts/records/sequence_records/termination/nonfoot` **全部逐值相同**；因此只有每策略两种不同球位与速度场景、每场景一个唯一轨迹，另有一次确认精确重复性的重跑。不能将原计算器输出的 n=4 Wilson CI 用来声称跨种子泛化或估计真实成功率。本轮 2 种独立场景均失败；样本极小。

### A｜直接 PhysX 刚体对与 collider path：**未证实**

从 v54 已确认 `left_wrist_yaw_link` 与 `left_hip_roll_link` 在原中心球位 CMU 10_05 终止时各自受到大小 25.890153N、方向恰好相反的**净传感器力**，连杆原点距离 0.113883m。此现象与两刚体相互接触高度一致，**不是 narrow-phase manifold/contact-pair/collider paths 真值**。本轮冻结排障包含 3 个未形成可评分物理输出的 native observer 试运行：首次在 reset 前构造新接触 view 长时间停滞；第二次 reset 后新 self-articulation pair view 使子进程在写物理结果前退出；第三次不创建新 view、只启用额外 observer 仍在首次物理输出前终止。日志保留 `reports/v055_diag_center_frozen_v32_*failure.log`；均未计为 PhysX 回合。**不得把 observer 接口写在源码里就宣称它运行成功。**

本轮用于可评分的 15 次物理运行**明确不开启新 pair observer**，只保留 v54 已验证的 named `ContactSensor.net_forces_w`；原 3 次中心球位与 v54 所有共有原物理字段逐值一致，但这不证明新增 pair 采样的因果无扰动性。原生刚体 pair、接触点、法线、实际碰撞 collider/USD 路径仍为 **UNVERIFIED**，不能据此更改碰撞 filter 或调低非脚门槛。

CPU-only 新关联诊断在 v55 全 15 个原报告中发现：中心失败 10_05 的**净力对称候选**仍是左腕/左髋；holdoutB 六个对应失败回合均为 `pelvis`—`right_hip_roll_link` 反向净力候选（约 234.32–252.56N 峰值），在接触后极早的 1.64s 左右触发原 `physical_nonfoot` 终止。候选来自原 body-level 净力矢量相加约为零；**并未证明真实两碰撞形状的接触，也不能排除多对隐藏同时接触**。见 ignored `reports/v055_nonfoot_net_force_pair_candidates.json`。

NVIDIA 当前 Isaac Lab Contact Sensor 文档说明 `net_forces_w` 是传感器刚体的聚合力，特定配对必须使用单个 sensor body 对相应 filter partner 的 `force_matrix_w`（并遵循后端、过滤和 contact data capacity 限制）。下一版应先在**隔离小场景**核对本机 Isaac Sim/PhysX 与有效 filtered pair，再回原 G1 做三回合 pair-on/off 严格 parity；形状级路径需要单独的 native contact report 或 raw contact 证据。不能在生产评测里盲建 self-articulation view。

参考：https://docs.nvidia.com/isaaclab/concepts/sensors/contact-sensor

### B｜200Hz 原 G1 全身、支撑与后恢复

原固定 29DoF `joint_pos/joint_vel/joint_target` 与全部原 qdot 限、实际全身质心位置/速度、ground-filtered 双脚法向负载和命名非脚净力均可对每个真实 5ms 样本追溯。由 v54 中心物理读入的 `tools/analyze_existing_200hz_v055.py` 已提供 SHA 锁定旧证据分析；新 `tools/audit_pair_dynamics_v055.py` 对全部新 15 回合做每个 5ms 原 qdot 限、左右臂/腿/腰峰值时刻、COM 支撑与完整恢复窗审计。触球 1.57s，完整 0.4s 前窗 + 1.25s 后窗应约 330–331 个样本；正常回合包含至 14s 的 2800 次原物理步。

- 中心原 v32/CMU10_02：`2800` 帧，14s 完整、正脚背比例 1.0、进门、≥3.5m/s、恢复，严格 PASS；CMU10_05 在 `5.140001s` 提前终止，`1029` 条采样含 1µs 终止附加快照，虽触球进门且球速达标但**无14s恢复**，严格 FAIL。v54 三份源重分析 29DoF 速度比原限均未超过1（中心最大约0.908）。
- holdoutA 两个 seed 各三策略全部完整14s、正脚背、出球速达标、实际关节限速与射后恢复均通过，**6/6 全未进球**，故严格成功 `0/6`。
- holdoutB 两个 seed 各三策略在 `1.640001s` 左右 `physical_nonfoot` 提前终止，记录约329条原5ms+终止快照，**6/6 未进球且未恢复**。原 1.57s 脚背触球以后仅约70ms可观察恢复，**不能编造未发生的后1.25s/14s数据**。6条回合原非脚传感器峰值约234.32–252.56N；对应命名净力相反刚体为 pelvis/right_hip_roll_link 候选。
- **holdout 12/12** 原实际所有29DoF关节速度未超限，正脚背触球比例达到门槛且球速≥3.5m/s；**0/12进球**；后 1.25s 完整可观察仅 holdoutA 6/12。球速与脚面合法**不是**整套射门完成的替代门禁。

双脚数据只是已过滤场景接触法向力，不是人体 GRF、完整地面反作用力矢量/摩擦锥，也不能在没有精确脚底 polygon 的情况下认证动态 ZMP/CoP/支撑多边形。没有碰撞网格最小 signed distance；腕髋连杆**原点**距离只能当风险辅助指标。不可称已证明“自碰撞导致跌倒”。

### C｜新的全身协调控制接口（仅可执行 CPU 设计，**未上机、未学习**）

`tools/wholebody_safety_interface_v055.py` 定义真实原 G1 **按顺序明确命名的29个关节**、相位驱动的多关节耦合目标刺激，以及将期望弧度残差映射至原 normalized action 的 `adapt`。关节包络、每20ms关节指令变化、实际原 qdot 预算、承重脚/质心、左腕髋距离和已发生真实 pair 接触均在同一安全判定中，任何失败返回**未经改动的原动作**。关键修正：新接口必须显式提供经过认证的原 G1 关节名称顺序和实时 PhysX 刚体对测量可用性；如果两者任一缺失，**fail-closed**，从而保证 v55 尚未获得 manifold 的情况下绝不能悄悄启用“安全”协调增量。`phase_coupled_intent` 非零覆盖>20关节、峰值≤0.075rad，但它是**合成接口测试输入**，没有经过专业人体资料拟合，不能说已学会全身发力，也不是能保证安全的 CBF/QP。

下一阶段可验证研究假设（尚不是当前结果）：通过 **human foot-contact-verified phase encoder → stance-aware 29DoF whole-body temporal policy → collision/contact-aware predictive whole-body controller → physically scored 14s recovery** 四段解耦学习和约束，能在原 G1 物理里改善助跑-支撑-髋/躯干旋转-摆臂-摆腿-脚背碰撞-随挥落地的协调，而不仅是 1mrad 的局部轻微修正。应先获取可溯源高频正脚背接触真值和支撑标签；缺失时把人类相位当**代理弱标注**，不得把先验当真值。

必须 **先冻结训练/验证/测试主体与动作划分、控制输入输出和 mesh/支撑门禁，再训练**；训练 reward 应结合动态支撑安全、对抗性碰撞对、脚背法向冲量、时序全身表征、真实足端速度/出球速度与14s恢复，需无作弊物理运动量注入及 no-contact/no-foot/mismatched-phase 负例。整个冻结 holdoutA/B 已被看过，今后不得拿它们调参后仍当未见测试。

### D｜严格 gate / 数值解释

原 `tools/audit_pair_dynamics_v055.py` 逐回合调用 v53 严格门禁（原 14s PhysX、自由根、全部原实际 qdot、脚背≥95%冲量、球完整进门、≥3.5m/s、真实助跑/恢复、原接触力）。无新控制器上线，因此此次**评估的是旧 v32 与两个 v52 CMU 弱参考，不是新的全身策略成功率**。三策略在未见本轮球位均为 0/2 个*不同物理条件*；6 次 seed replay 是相同时间序列的复现，不能提供随机方差信息。统计效应与多主体运动自然度一律未验证。

### E｜冻结与完整可复现性

1. SHA 协议：`configs/physx_pair_dynamics_v055.json`，父 SHA `3cac1736...`，15 固定 case、原三策略原配置 SHA、旧物理源 SHA、原模拟器/控制关键脚本 SHA；CPU-only `tools/run_pair_dynamics_batch_v055.py::freeze` 前后执行。GPU lock：`experiments/kinetickick_physics_gpu.lock` flock 非阻塞，启动前检查 nvidia-smi compute PID 与磁盘至少20GiB；已有数据不覆盖，断点只核 hash。
2. 真实执行：`/home/you/conda_env/env_isaaclab/bin/python -u tools/run_pair_dynamics_batch_v055.py`。现有 15/15 证据，不应为了“多跑”覆盖；全部本地原JSON/日志及失败pilot日志由 `.gitignore` 保留。
3. 独立评估：`python3 tools/audit_pair_dynamics_v055.py`（首次生成严格报告）；`python3 tools/audit_seed_replay_v055.py`（首次生成种子证据）；`python3 tools/scan_nonfoot_force_pairs_v055.py`（首次生成净力对称候选）。这些原数据报告输出均为 **不允许覆盖**；重检前应读取现存结果并校验原 SHA，不允许改写历史证据。
4. CPU 测试：`python3 -m pytest -q tests/`；真实历史报告在 `reports/v055_full_physics_dynamics_audit.json`、`reports/v055_pair_batch_progress.json`、`reports/v055_seed_replay_audit.json`、`reports/v055_nonfoot_net_force_pair_candidates.json` 与实际图片 `media/v055_existing_physx_wrist_hip_failure_dynamics.png`，仍**仅本地保留**。
5. 数据资格：CMU 仅同一被试两段 120Hz 足球动作，以球标记推断事件而非真实脚背 collision manifold；无精确人体接触时刻/力、GRF、跨被试独立数据，不能称职业级自然爆发力已经习得。

## 下个可验证决策

**先解决 A 的测量可信度、不要直接在 holdout 上训练。** 用与本机 Isaac 版本匹配的单刚体→单刚体已知接触隔离实验验证 GPU filtered pair/报告能力，随后谨慎对原三个中心回合进行物理状态逐字段 pair on/off parity；取得确凿 contact manifold/path 后才能做安全约束学习。并预注册新的不同足球轨迹/多人动作独立保留测试组。当前底层控制器保持冻结，所有正负证据继续保留。
