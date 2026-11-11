# KineticKick v0.1.60｜原29DoF G1原生接触订阅与200Hz全身观测器二因素剖析

**日期**：2026-11-11。**父SHA**：`4107bebb4ee4604d2316acf3a2bfad86a49ac339`（已推送v0.1.59）。**本次分支**：`version/v0.1.60`。冻结协议 `configs/g1_observer_factorial_v060.json`，SHA256 `638fee983e0884a4d1622e52b96426b9c52d79bcb9429006bc2bf0afca247b5c`；运行前写入 `reports/v060_protocol_lock.json`，报告和原始轨迹全部write-once。本批是**观测能力及失效定位诊断**，不是新控制策略对抗实验，不计独立未见测试集，不能据此声称人体自然协调性能改善。

## 结论摘要

本次冻结并真实启动 **10 个原始G1／PhysX案例**（CUDA 6、CPU 4），其中 **7/10有完整可审计原G1 29DoF、200Hz物理JSON**（CUDA 3/6、CPU 4/4），**3/10在CUDA显示READY后无任何G1轨迹JSON**。不可评分3例不能计入严格成功/失败分母。7个可评分案例严格全任务成功 **2/7**：均为相同基线v32的CUDA无观察器或仅原生接触订阅；CPU全4例FAIL，另一CUDA CMU10_05 FAIL。**因这两个PASS是同一物理轨迹的观测器复跑，不是独立成功样本；本批不存在新训练策略。**

本批核心诊断：

1. **CUDA原生contact subscriber单独启用可完成物理且逐字段无扰动**：相同CUDA、相同seed=1201、相同v32和球位，关/仅native各2800个真实物理行、25类字段完全相同（**0处差异、最大数值差0**）。CMU10_05的CUDA仅native版本同样产出1029行，在5.14s触发nonfoot终止，不能因为球进门且脚背份额1.0算PASS。
2. **CUDA启用v055全身pair-dynamics observer即发生无G1数据的失效模式**：v32仅dynamics、v32 both、CMU10_05仅dynamics三例均READY后无JSON，虽退出码0。对应去掉v055、仅保留v056 native subscriber时可运行。因此故障与`--pair-dynamics-v055`的CUDA路径强相关；但尚未定位具体PyTorch CUDA/PhysX张量操作、初始化时序、宿主进程退出原因，**不能用相关性冒充单一API因果证明**。
3. **CPU两个只读观察器均未影响既有被测物理**：CPU v32无观察器/仅native/仅dynamics的333条轨迹（332实际5ms + 1终止状态标记）、25类字段逐值完全相同。原生订阅确实提供真实G1骨盆—右髋碰撞**双方actor/collider路径及3个接触点**，而不仅是净力推断；CMU10_05 CPU仅native也获3点。
4. **GPU native=0头不是“没有接触”证据**：CUDA v32、CMU10_05有完整的200Hz输出，原生回调却0头、0真实接触点；GPU左腕—左髋候选至今没有direct pair/manifold证明，CPU骨盆—右髋真值不能移植到GPU自碰撞结论。

## 冻结物理矩阵：不修改原控制器

| Case | 设备 | 观测器 | 有效原G1 | 真实时长 / 200Hz行 | 严格结果 | 直接接触证据 |
|---|---|---|---|---|---|---|
| cuda_v32_none | CUDA | 无 | 是 | 14.0s / 2800 | PASS | 未激活 |
| cuda_v32_native_only | CUDA | native | 是 | 14.0s / 2800 | PASS | 0 headers；不能反证不存在 |
| cuda_v32_dynamics_only | CUDA | v055 | **否** | READY后无JSON | **不可评分** | 不可评分 |
| cuda_v32_full | CUDA | native+v055 | **否** | READY后无JSON | **不可评分** | 不可评分 |
| cpu_v32_none | CPU | 无 | 是 | 1.66s / 332+标记 | FAIL | 未激活 |
| cpu_v32_native_only | CPU | native | 是 | 1.66s / 332+标记 | FAIL | 3个骨盆—右髋真实point |
| cpu_v32_dynamics_only | CPU | v055 | 是 | 1.66s / 332+标记 | FAIL | 未请求native |
| cuda_cmu05_native_only | CUDA | native | 是 | 5.14s / 1029 | FAIL | native=0，不能反证 |
| cuda_cmu05_dynamics_only | CUDA | v055 | **否** | READY后无JSON | **不可评分** | 不可评分 |
| cpu_cmu05_native_only | CPU | native | 是 | 1.66s / 332+标记 | FAIL | 3个骨盆—右髋真实point |

这里严格结果遵循原controller定义：球门、正脚背冲量≥95%、球速≥3.5m/s、**所有29关节原始qdot限制**、非法非脚碰撞和终态恢复必须同时满足。可评分7例中全部峰值原qdot / 原限制 <1（CUDA约0.90813、CPU约0.90983），但其他门禁失败依旧FAIL，不得用单项良好掩盖整体失败。CPU种子1201在约1.58s触球、1.66s非脚提前终止，**之后≥1.25s的恢复轨迹不存在**。CUDA CMU10_05虽然进球、脚背份额1.0、峰值qdot合格，但5.14s出现physical_nonfoot，因此绝非全任务成功。

## 真实接触point与身体状态的独立证据

本次CPU原生`omni.physx`接触报告给出`/World/envs/env_0/Robot/right_hip_roll_link`与`/World/envs/env_0/Robot/pelvis`，以及对应`/collisions`后缀的两侧**实际collider路径**、接触点世界坐标/法向/冲量与separation；例如CPU v32的第一个点时间为**1.65999996s**，位置约`[-0.228445,-3.909860,0.712943]`m，单位法向约`[-0.967871,-0.249814,-0.028605]`，separation约`4.04e-5`m。本次不是把两个相反的净力当作pair真值。约1.66s的CPU原生碰撞与非脚提前终止时间一致；其他时间、其他后端不能无条件归因为同一pair。

CPU v32的非脚终止截断了原本应有的1.25s后观察窗口；CUDA v32保留14秒、支持完整接触前0.4s以及接触后≥1.25s至14s。CPU和CUDA使用不同物理执行后端，**不能把它们轨迹差异作为只读观测器扰动**；同后端同seed的直接对照才支持无扰动。全身原29DoF原始qdot、CoM、双足支撑力、触球时刻、非脚力记录有可追溯的本地200Hz数据；不把单纯的双足net force写成专业人体GRF。

## 原因定位与下一个可证伪实验

v0.1.59只能知道“CUDA full无JSON而off可运行”；本版通过冻结单因素拆开，排除了“只要开启native subscriber就导致GPU无法运行”这个宽泛解释，缩窄为**v055 pair-dynamics路径与无输出强相关**。保留的日志在`FREE_SHOOTING_READY`后无`FREE_SHOOTING_RESULT`，不能凭子进程exit=0声称成功，也不能假装获得缺失的200Hz物理。

下批应在**新版本新SHA协议**独立定位v055在CUDA的具体首个非法操作：分别启用仅COM/get_masses、仅body-com tensor读取、仅contact_forces+body_names、仅路径report轮询、逐帧tensor到CPU转换等最小字段子集，以可审计`last_successful_stage`和原G1完整JSON作为双重判定。每个候选必须有同设备/同seed observer-off校验和安全退出，避免把GPU compute死锁引入长期训练；找到最小可复现故障前不得在CUDA训练实时碰撞约束控制器。CPU native已认证的骨盆—右髋接触点可用于collision-risk监督信号，但**GPU左腕—左髋以及人体接触/真实GRF依然未证实**。

真正研究方法应在可复现观测器稳定后独立冻结：真实人类正脚背助跑—支撑换足—髋/躯干旋转—腿部鞭打—双臂反向角动量—随挥落脚的**相位条件全身生成器**，带29DoF动力学、足地接触可行域、非脚自碰撞余量和后恢复约束。数据分为专业人体数据资格集、训练、验证及未见测试，禁止用本次已看过的seed1201来调参后称泛化；需先取得高质量足球强射人体动捕/足底GRF/球接触真值，否则只允许使用CMU非强射相位代理并明确上界。不能以1mrad residual、离线余弦MSE或RoboNaldo式碰球当研究终点。

## 执行证据、复现与不可覆盖门禁

本批复验GPU无其他compute进程，磁盘剩余约44GiB（≥20GiB），全局互斥锁`experiments/kinetickick_physics_gpu.lock`、write-once`reports/v060_*.[json|log]`、源文件/原policy哈希及协议SHA均已验证。重跑`tools/run_backend_batch_v060.py`只读取并确认已有10条SHA，不覆盖原报告。**需要重新执行真实物理时必须另开版本和独立新路径**。数据、CMU/C3D、模型权重及3000万字节原始PhysX JSON只留Tang；GitHub只保留预注册协议、脚本、tests和本报告。

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/run_backend_batch_v060.py   # SHA安全复核，不重新覆盖
python3 tools/audit_v060_observer_factorial.py --output reports/NEW_UNIQUE_v060_independent.json
python3 tools/plot_v060_factorial_dynamics.py --audit reports/v060_independent_audit_complete_20261111.json
PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking \
 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
```

独立审计器对**每帧25类物理字段必须实际存在**作明确验证，防止off/on均丢同一字段被错误宣布“parity”；合计5组同后端或既往CUDA baseline的物理25类逐值对照。本机已生成`media/v060_observer_factorial_validity.png`和`media/v060_measured_200hz_nonfoot_and_support.png`，只留本地，真实失败与被截断轨迹均保留，未用零填充缺失物理。
