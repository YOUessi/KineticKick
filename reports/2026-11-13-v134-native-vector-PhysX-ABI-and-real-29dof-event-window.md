# KineticKick v0.1.134 · 原生PhysX接触冲量ABI修复、全29DoF时窗取证与负例

日期：2026-11-13（Tang本地）。基线：v133 / 521861d6edb807c71968fc640f975614ea5a10e3。冻结先行提交SHA：dbe76c45527dc4595dd90ebaa3754f81f051ffba。

## 1. 已实际执行的安全核查

- Tang / Ubuntu / RTX4090 在线。原目录 /home/you/projects/kinetickick 是v64，原始未跟踪文件 =100===，未修改；v131有共享隐变量候选未提交文件，未修改。新建此前不存在的版本 version/v0.1.134 和独立目录 kinetickick-wt-v134，从远端确认的干净v133建立。
- 已核查远端全部 version/v0.1.X 数值版本，创建v134之前最大是v133；核查磁盘剩余约30GiB，满足20GiB保护线。
- v120全项目pytest PID242644仍持续运行并占用RTX4090约260MiB，累计超过11小时、原日志0字节；未杀、未暂停、未启动并行IsaacLab。本轮新增真实PhysX 0次，训练 0次。
- 按前述v128 SHA冻结来源独立读取三份原v54全部真实物理原始JSON；不覆盖、删除原C3D/CMU、权重、NPZ、JSON或媒体。所有新审计JSON/四张PNG以及pytest日志按.gitignore留Tang本地，GitHub只存代码/协议/测试/本报告。

## 2. A：发现并修复v133接触适配器与真实NVIDIA接口不一致的关键问题

NVIDIA原生Contact Reports文档（2026-07-23）说明ContactData.impulse是三维向量；官方示例使用 impulse[0], impulse[1], impulse[2] 求模。v133的CPU mock采用1.5标量，v133有限浮点校验也只接受标量，真实NVIDIA Float3回调会被拒绝。因此v133上游的“CPU通过”不能证明该代码能消费真实PhysX contact数据。

新模块 tools/v134_native_contact_vector_gate.py 实际改用：
1. 从借用的callback headers/contact_data立即复制stage_id、event type、actor0/1、collider0/1完整路径及offset/count；每个点复制world位置、法向、三维冲量、冲量向量模长、沿报告法向的有符号投影、separation、face indices。它不把向量模长误叫正脚背法向冲量。
2. 接受CONTACT_LOST零manifold点，**不允许**将零点当作从未碰撞的证明；拒绝scalar impulse、非有限数据、越界/重复contact数据片段、未覆盖数据、stage错配、重复actor/collider、坏USD路径。
3. 懒加载真实runtime的PhysicsSchemaTools.intToSdfPath，支持v133的只读USD最近刚体归属映射，并对callback actor与USD owner精确匹配；未知owner或owner不符全部拒绝。
4. CopyOnlyContactRecorder只注册只读contact callback，记录独立physical-step提示，不自动把callback次数当作物理步；原29DoF同物理步校验、缓存源、真/负PhysX控制、OFF/ON状态不变性全部必须在真实Isaac runtime另行证明。即使CPU虚构完全一致，也始终 native_pair_certified=false、motor_execution_authorized=false。

对应NVIDIA文档：
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html
- https://docs-prod.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html

**重要边界**：本轮没有运行Isaac Sim，此模块还没有原生callback正负物理验证。当前左腕—左髋的PhysX collider pair/manifold仍为UNKNOWN，不能把历史25.890154N等幅反向力直接认证为腕髋自碰撞。

## 3. B：完整200Hz物理证据与0.4s/1.25s的因果时窗

tools/v134_real_29dof_coupled_window.py按原始v128协议逐文件SHA256核对并读取3个原v54回合共6,628步，每步包含29 q、29 dq、机器人COM、双足净力支撑代理及非脚净力。以失败组命名sensor时间5.145s为**时钟锚**，固定窗口4.745–5.140s（80真实帧）及5.145–6.395s（251个包含端点的帧）。成功控制器的对应锚仅表示相同时钟，不意味着那里有同一种接触事件。

| 原控制器 | 真实200Hz全程步数 | 前0.4s真实帧 | 后1.25s真实帧 | 缺失后帧 | 前0.4s qdot/原限额最大比 | 前0.4s非脚力峰(N) |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v32 | 2800 | 80 | 251 | 0 | 0.258950 | 0 |
| CMU10_02 | 2800 | 80 | 251 | 0 | 0.235701 | 0 |
| CMU10_05（失败） | 1028 | 80 | **0** | **251** | 0.307105 | **25.890154** |

失败组5.135s发生非脚力峰、5.140s最后真实关节状态、5.145s才有命名sensor事件。不能在5.145s补一个虚假的机器人姿态，更不能凭5.145s之后不存在的251帧判断机器人安全恢复。

新量化的分组全身动作证据（仅是原G1的**观测关节角位移**，非人类发力）：

| 原控制器 | 左臂肩肘腕最大角位移(rad) | 右臂肩肘腕最大角位移(rad) | 腰/躯干最大角位移(rad) | COM水平路径(m) |
|---|---:|---:|---:|---:|
| frozen_v32 | 0.130447 | 0.265651 | 0.063878 | 0.137968 |
| CMU10_02 | 0.219901 | 0.235982 | 0.076006 | 0.105548 |
| CMU10_05 | **0.763156** | **0.515181** | **0.220252** | 0.101124 |

上表是预定义事件时窗的关节相对第一帧峰值，表明CMU10_05前0.4s中左臂/躯干角度显著更大，结合以前的URDF碰撞几何接近可作为**后续假设**，但不能据此证明导致非脚接触、更不能依据已见失败回合调参并声称盲测效果。

新增独立来源复核：tools/v134_real_29dof_coupled_window.py还重新读取原v52观察器OFF以及v54 ON的各3份真实PhysX JSON，双侧分别验证SHA256，再逐值比对9类共同物理字段：traces、contacts、kinematics、records、sequence_records、running_evidence、stride_shot_evidence、stride_shot_success、termination_reasons。**3/3原回合、9/9共同字段全部精确一致**。这是重新核验既有v52/v54的历史观察器不干扰证据，不是v134新观察器的PhysX OFF/ON实测，不能认证native pair。审计已存Tang本地reports/v134_real_coupled_v54_original_v2.audit.json。

三回合完整时域的原29DoF qdot限额最大比均为0.908129，故本次v54失败不能由该指标的已记录超速单独解释。支撑数值只是已记录足底净力代理的时窗积分，不是实测人体GRF、CoP或可靠零力矩点。

另外，四张真实失败图分别绘制原29DoF qdot最大限额比、非脚力与20N门限、左足支撑代理及COM水平路径；失败组曲线严格截止5.140s，不补绘数据。

## 4. C/D：控制许可、负控制及仍未完成的工作

本批实现了CPU级完整状态记录的OFF/ON字段逐值比较器：q、dq、根部姿态/速度、COM、左右足支撑、非脚力、球位姿和速度、motor torque，丢一个字段、记录长度不一致或任何值不一致都拒绝；这套比较器本身不证明源于两次真实同种子rollout。

保持现有严格门禁：正脚背冲量≥95%、球速≥3.5m/s、进球、所有原29关节qdot/加速度/扭矩、非脚碰撞及14s恢复。CPU合成的全PASS数据、损坏的路径、缺少runtime时钟、或者空callback都不能授权motor。

研发假设（待新真实物理及人体数据资格认证后预注册正式训练）：
- H1：带相位连续共享隐变量的支撑—骨盆—躯干—对侧臂—摆腿协调生成，若用真实人体时序而非伪接触标签训练，能改善完整助跑与随挥自然性；
- H2：运行时collider-pair和双足支撑约束会降低非脚碰撞及恢复失败，而非仅优化触球瞬间；
- H3：以球速、正脚背冲量和进球等硬门禁审查，协调性改善与射门表现必须同时成立，不能以离线角度MSE作替代。

当前人类C3D只有2名被试且无可信真实正脚背冲量/同步GRF，不能宣布已学会专业球员发力；v131共享潜变量独立候选仍是未经PhysX执行的工程候选，本批未挪动其未提交文件。新增策略实时物理rollout 0、成功率不适用。

## 5. 实际测试、可复现命令及风险界定

本批代码：tools/v134_native_contact_vector_gate.py、tools/v134_real_29dof_coupled_window.py、tools/v134_plot_real_29dof_event_window.py；专项负例：tests/test_v134_native_vector_contact.py、test_v134_physics_invariance_and_falsification.py、test_v134_real_coupled_event_window.py。

初次测试发现合成缺帧负例的时间起点仍覆盖完整前0.4s，因此预期异常未出现，修正测试数据起点后通过；随后提高OFF/ON字段集合要求，新增29DoF维度与根姿态拒绝测试、真实原v52/v54 9字段逐值一致重验、单字段篡改拒绝测试。保留首次失败记录，不把第一次失败隐藏为成功。v134专项最终89 passed；扩大到v124、v127–v130、v132–v134的CPU回归最终345 passed、1 skipped、0 failed（CUDA_VISIBLE_DEVICES为空；未执行GPU全项目测试），日志reports/v134_expanded_cpu_pytest_FINAL_v3.log留Tang本地。

复现（仅CPU，不调用GPU）：
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/v134_real_29dof_coupled_window.py --output reports/v134_new_unique.audit.json
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/v134_plot_real_29dof_event_window.py --output-prefix reports/v134_new_unique_plot
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_v124_* tests/test_v127_* tests/test_v128_* tests/test_v129_* tests/test_v130_* tests/test_v132_* tests/test_v133_* tests/test_v134_*

输出采用exclusive-create不覆盖本地旧图或JSON；第二次运行须换新文件名。原始JSON的SHA由冻结v128配置保证。PNG/log/审计JSON未纳入GitHub。

下一轮必须先完成PhysX隔离球体正/负控制，校准官方向量冲量及stage/physical-step clock，再使用原G1三控制器同seed观察器OFF/ON三组配对真实PhysX、原29DoF+球+力等全部共同物理字段逐值不变性，才能认证左腕—左髋真实collider pair；在GPU独占并确认数据资格后推进可执行全身协调控制和未见球位、多种子硬门禁评测。
