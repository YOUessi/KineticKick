# KineticKick v0.1.59｜原G1 CPU观测器无扰动实证、CUDA原始物理恢复与失效定位

**2026-11-11**，父提交：v0.1.58 `3eaa84e45b7eb7bc64a51f7e8a289e6602833f12`。从清洁v58创建唯一新分支 `version/v0.1.59`，启动任何实验之前新建协议 `configs/g1_observer_parity_v059.json`，**协议SHA256 `f0ffcfd60e423ba648c0c7df2e58d0ee806137ff86b80a4b5769256808788b0d`**。全部原29DoF Unitree G1、原v32/CMU冻结控制器、未训练也未修改原物理配置。严禁将这批诊断样本作为未见测试。

## 一、真实PhysX实验矩阵：10次启动、9次有效、1次严格PASS

| 冻结case | 后端 | 只读原生订阅+全身动力学 | 真正原G1 JSON/200Hz | 完整任务 | 关键事实 |
| --- | --- | --- | --- | --- | --- |
| cpu_v32_off_s1201 | CPU | 关 | **有效333行**（332物理+1终止状态） | FAIL | 1.66s非脚早停 |
| cpu_v32_full_s1201 | CPU | 开 | **有效333行** | FAIL | 3个原G1骨盆/右髋原生接触点 |
| cpu_cmu02_full_s1201 | CPU | 开 | **有效333行** | FAIL | 3个原G1骨盆/右髋原生接触点 |
| cpu_cmu05_off_s1201 | CPU | 关 | **有效333行** | FAIL | 1.66s非脚早停 |
| cpu_cmu05_full_s1201 | CPU | 开 | **有效333行** | FAIL | 3个原G1骨盆/右髋原生接触点 |
| cuda_v32_off_s1201 | CUDA:0 | 关 | **有效2800行，14.0s** | **PASS** | 原G1完整14s、正脚背冲量份额1.0、球进门、恢复 |
| cuda_v32_full_s1201 | CUDA:0 | 开 | **无JSON** | **不可评分** | 启动/READY后无原200Hz结果，退出码0 |
| cuda_cmu05_off_s1201 | CUDA:0 | 关 | **有效1029行，5.14s** | FAIL | 正脚背份额1.0、进门，但非脚早停/未恢复 |
| cpu_v32_full_s2333_perturb | CPU | 开+真实seed扰动 | **有效2800行，14.0s** | FAIL | 进球且恢复但正脚背份额仅 **0.560997** |
| cpu_cmu05_full_s2333_perturb | CPU | 开+真实seed扰动 | **有效2800行，14.0s** | FAIL | 进球且恢复但正脚背份额仅 **0.529152** |

**严谨分母：9次有真实原G1物理结果，严格成功1/9；CPU 0/7、GPU 1/2，另有1次未出物理数据的CUDA尝试，不算0或1。** 有效种子2333同时改变了真实首帧根位置与关节状态，不是仅修改标签。两CPU新seed虽完成14秒并进门，但因脚背冲量**未达95%**，仍一律FAIL。GPU v32的合法进球不能提升运动自然性论文结论；CPU与CUDA不具有物理等价性。

所有有输出的9例按原 `29DoF` 关节速度和原限制审计，峰值上限比约0.908–0.910，**此批无qdot超速**，但不能豁免脚背接触、非法非脚接触、≥3.5m/s出球以及恢复等全部门禁。对于5个CPU未扰动早停样本，接触后的至少1.25秒不可观测，不补造后恢复。2个CPU扰动后完整14秒样本已经观察到恢复，但冲量面归属不合格。具体停机与非脚峰值应查各自本地锁定原始JSON。

## 二、关键突破：同一CPU后端真实证明只读观测不改变可比物理字段

冻结相同CPU后端、相同 `seed=1201`、相同原模型、球/球门、原控制器，无观测 vs 只读观测，独立程序 `tools/audit_v059_observer_parity.py` 对每个5ms采样的25类共同物理字段（200Hz时钟、根和球状态、29关节位置/速度/目标/力矩估计、双足支撑、非脚力、各接触及序列相位等）逐字段对比，结果为：

- **v32 CPU off/on：333行，包括终止标记；0处不一致；最大数值绝对差0.0。**
- **CMU 10_05 CPU off/on：333行；0处不一致；最大差0.0。**
- **v58已锁定CPU full与v59 CPU full再次复现：333行；0处不一致；最大差0.0。**

注意：333行中只有332个200Hz物理行，终止后1微秒是标记而非真实substep。这证明**当前CPU回合此配置下的observer parity**，不是CPU/GPU等价，也不是所有随机种子、所有后端和所有场景的普遍无扰动证明。

原生 `omni.physx` contact reporter 在CPU观测条件下：v32、CMU10_02、CMU10_05的未扰动回合**各发现3个真实原G1骨盆—右髋接触point及两侧真实actor/collider path**（约1.660、1.665、1.670秒），包含法向、位置、冲量、分离距离；这不是简单的两个连杆净力反向，也不是隔离双盒子玩具实验。不同控制器在1.40秒起已发生实测物理差异，但仍相继早停。扰动seed2333的两个完整CPU样本对应此pair **0个已认证point**，不能从零事件反证其他自碰撞。

**原v54的GPU左腕—左髋候选仍没有直接 native manifold 真值**，请不要把CPU骨盆—右髋物理接触偷换为它的认证。

## 三、CUDA故障定位：不是修复argv后控制器完全不工作

v57/v58的CUDA完整观测版本出现`READY`但无G1 JSON。v59冻结矩阵控制了后端、原控制器和资产，在**关掉v055全身pair observer与v056 native subscriber之后，CUDA v32完整运行14.0s并严格PASS，CUDA CMU10_05也完成到5.14s真实失败**。同一基线CUDA同时开启两个只读观察器又无结果，进程尽管返回0仍不能认证物理。

因此可以支持一个比“GPU无法运行”更精确的结论：**额外观测器的启用路径（单个native subscriber / v055 observer / 二者耦合或初始化顺序）与CUDA原始物理无结果强相关**。在尚未单独切分两个观察器前，不能唯一归因某个 API；更不能因CPU观测无扰动就把CUDA缺失看作正常或把GPU力对伪造为真值。

## 四、身体运动/安全门禁与新方法定位

- CPU seed1201：`t_contact=1.580s`，在`t≈1.660s`非脚停止；球面归属ambiguous，正脚背冲量0，球门未通过。左/右脚支撑、CoM、身体连杆位置、原29qdot和原生接触三维point都存于本地；后续1.25秒和14秒**缺失**。
- GPU v32 seed1201：原冻结控制器可完成真实14s全任务；这证实既有基线有效，但**不是新的人体协调生成成果**。
- GPU CMU10_05 seed1201：虽然实际触球脚背份额100%且进球，`physical_nonfoot`在5.14秒切断恢复，严格FAIL。
- CPU seed2333：两条原G1完整14s都进门和恢复，但脚背冲量仅56.1%/52.9%，不能放宽阈值来宣布成功。
- 原生pair证据可用于研究**骨盆—腿摆动空间碰撞安全**，但未取得人体专业正脚背GRF/触球真值，也不允许在v59这些已看过的失败案例上暗调大幅动作控制器。

本版未训练新模型，因此不声称自然全身发力/职业式大力抽射。研发贡献是：恢复原G1真实GPU物理，实验确认CPU观测等价，以及取得原G1 CPU真实接触双方和Manifold；为接下来可训练、可验证的**相位条件全身动作协调 + 支撑/自碰撞安全投影**打基础。数据资格依据 `docs/V058_HUMAN_KICK_DATA_QUALIFICATION.md`：真实人类足球强射足球接触/GRF目前仍未取得，已有120Hz CMU只是非真值相位代理。

## 五、执行和复现（源码GitHub、原始数据留Tang）

源码与实验预注册：`configs/g1_observer_parity_v059.json`、`tools/run_backend_batch_v059.py`。实验保留独一无二的本地 `reports/v059_*.json` / `*.log`、`reports/v059_batch_progress.json` / `reports/v059_protocol_lock.json`，含源SHA、协议SHA、GPU文件锁、20GiB空间门禁、无其他CUDA作业、每例write-once。独立报告 `tools/audit_v059_observer_parity.py`，CPU纯逻辑测试 `tests/test_v059_observer_parity.py`，失败图生成器 `tools/plot_v059_physics_parity.py`，图片仅留Tang `media/v059_cpu_observer_qdot_parity.png` / `media/v059_cpu_g1_native_hip_pair_nonfoot.png`。

全部锁定原实验 JSON/NPZ/媒体/CMU/C3D/权重不覆盖、不删除、不上传。任何重新运行必须使用**新的唯一输出路径**，不可复用当前`v059`报告。数据独立审计入口：

```bash
cd /home/you/projects/kinetickick
python3 tools/audit_v059_observer_parity.py --output reports/NEW_UNIQUE_v059_independent_audit.json
PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking \
  /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
```

下一批先新分支冻结**GPU子系统二因素诊断**（none、仅v055、仅v056、both及最小只读callback），同一个原G1 seed与控制器独立对照，定位导致CUDA READY后无输出的确切组件；确证GPU native path/manifold与无扰动物理一致才能进入碰撞风险标签。然后建立从足部真实触球时序到29DoF全身动作的约束控制接口，在新随机种子与保留独立未见场景中冻结训练、验证和测试分割。禁止把 `RoboNaldo` 式碰球、离线余弦MSE、1mrad残差包装成终点。
