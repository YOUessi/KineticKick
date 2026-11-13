# KineticKick v0.1.142 — 原生PhysX接触取证链 + 原G1全状态只读准入 + CMU实物数据资格审查

2026-11-13。**父版本：** `version/v0.1.141` `4edf26fd332b3df03e4e85007d64f5056796ca38`。
**实现前冻结协议SHA：** `e9cc197bd1e26615c9596f2456caa22312595366`，具体内容见 `docs/V142_FROZEN_NATIVE_ADMISSION_AND_HUMAN_DATA.md`。

## 1. 资源保护和实验范围

经过授权Remote Desktop Commander在Tang检查，本轮主目录 `/home/you/projects/kinetickick` 位于dirty `version/v0.1.64`（原 `?? =100===`）；没有切换/覆盖它，未触碰v131及其他工作树。从干净v141 SHA创建单独 `/home/you/projects/kinetickick-wt-v142`，先冻结协议再研发。根磁盘还剩约30GiB（97%占用），高于最低20GiB。RTX4090已有v120 Python pytest PID **242644**，约260MiB显存、CPU持续运行，执行已约20小时。本次没有抢占GPU、结束该进程，也没有重复启动IsaacLab。

冻结了**三种原G1控制策略 × OFF/ON = 6次真实配对试验**，以及孤立已知接触ON/OFF与隔离非接触控制三种原生校准；这是未来待执行的计划表，**不等于已做了6+3次真实实验**。本批新增真正PhysX回合 **0**、新策略训练 **0**、新严格物理成功率 **不可计算**。

## 2. A/B — 接触证据链研发（真实GPU取证仍被阻塞）

代码 `tools/v142_native_pair_ledger.py` 与 `tools/v142_runtime_bridge.py` 将原v134 3D vector-impulse callback扩展成可组合的 **同真实物理步写时观测接口**：

- PhysX回调仅立即复制原生actor path、collider path、contact manifold、3维矢量冲量、法向/位置/间距。回调数与物理步数分别记录，不误把callback event当作5ms物理帧。
- 来自原G1 runner的每个真实200Hz物理步必须提供12种完整字段，包括全部29DoF q/dq/电机扭矩、浮动基位置/朝向/速度、COM、双足与非脚力、球体位姿与速度；物理步必须连续且从1开始，callback与poststep记录必须同号。
- 使用实际USD只读祖先owner map，要求collider path→rigid-body owner与原生actor paths精确对应；必须是 `left_wrist_yaw_link` 和 `left_hip_roll_link` 两个不同刚体、同事件有非零矢量冲量的有效manifold，才形成待认证候选。禁止模糊名字匹配、等大反向力直接证明自碰撞。
- 引入全字段OFF/ON逐步严格对比、SHA256来源、write-once输出与不可覆写保护。真实回调仍需已在运行时经过正负控制、真实来源认证和独立物理不变性；CPU测试fixture或自声明的native=true**永远不能完成物理认证**。
- 即使未来callback中一无所获，也不能据此证明没有碰撞；若ContactReportAPI在USD中缺失，先冻结并报告传感器覆盖不足，不擅自修改机器人或控制器。

**本次没有实际挂载v142 sidecar到新的Isaac Sim G1物理回合**。新接口的运行时适配仍需在GPU可用时与原控制器只读对接后完成真实认证。对历史腕髋事件，原生PhysX actor/collider pair/manifold结论仍为 **UNKNOWN**。

### 历史真实物理SHA验证

本轮真实执行v134 SHA锁定历史原PhysX证据审查，不是重新仿真：

| 原控制器 | 原真实200Hz状态 | 覆盖时长 | 结论 |
|---|---:|---:|---|
| frozen_v32 | 2,800 | 14.000s | 有完整状态，**非**严格射门PASS |
| cmu_10_02 | 2,800 | 14.000s | 有完整状态，**非**严格射门PASS |
| cmu_10_05 | 1,028 | 最后真实q=5.140s | 提前终止；**无**后接触恢复真值 |
| 合计 | **6,628** | 历史3回合 | **新增0回合** |

失败组真实trace非脚力峰值发生在5.135s；命名传感器5.145s存在事件，但该时刻缺少同物理步完整29DoF状态，不能填造5.145s的根姿态或任何后续1.25s/14s恢复结果。历史传感器OFF/ON的逐值一致不等于**本次新sidecar**的非侵入性已认证。

## 3. C — 两段真实120Hz CMU C3D实物数据资格检查

本次不是仅从文件名推断：使用当前环境的`ezc3d`读取真正C3D帧数组和analog通道，审计对应v50真实数据资格记录。全部数据留Tang本地，GitHub只提交代码及聚合指标。

| 原CMU Subject 10 | 原始C3D帧 | 实际采样率 | analog通道数 | 能证明的内容 |
|---|---:|---:|---:|---|
| `10_02.c3d` | **590** | **120Hz** | **0** | 真3D人体marker序列、非真实正脚背接触力 |
| `10_05.c3d` | **436** | **120Hz** | **0** | 真3D人体marker序列、非真实正脚背接触力 |

原v50总数据审计覆盖**6段足球类CMU片段、两个被试、3,358帧**，但现有6段中**经过准确标注的人体触球时刻=0、正脚背冲量=0、职业级人体生物力学真值=0**。原v50有部分快球marker运动证据，不能当作真实接触点或力量。所选两段**同一个Subject10**，因此这两段不能做被试不重叠的train/test。C3D analog通道为零不代表人体不存在真实GRF，只表明**这些文件没有提供同步力板通道真值**。

本地来源清单声明科研用途许可，但原始TLS证书链验证和上游签名校验均非完整认证；需后续明确授权范围和可信度。此阶段可用作不合格运动先验探索，**不能**作为优秀职业球员真实正脚背力量传递、撞击冲量或同步GRF学习的真值。

## 4. D/E — 实际CPU负例、图与回归

新增源代码：
- `tools/v142_native_pair_ledger.py`：源SHA锁、真实历史复核、严格12字段状态校验、同物理步矢量manifold配对、拒绝伪造认证、write-once；
- `tools/v142_runtime_bridge.py`：原控制器不变的订阅/物理步对齐接口、即时复制、状态深拷贝、callback数量与物理tick分离；
- `tools/v142_human_data_qualification.py`：实读CMU C3D markers/analog、v50审计与来源manifest验证、明确缺失接触力/GRF/职业标签；
- `tools/v142_batch_readiness.py`：真实GPU进程/磁盘/锁/工作树审查、冻结3×2物理对照清单、默认只读拒绝物理执行；
- `tools/v142_plot_source_gap.py`：由SHA验证结果绘制历史物理截断图、人体真值缺口图。

本批新增两个专项测试文件：**75 passed**，覆盖时钟偏移、异常manifold、非法标量冲量、真实状态缺字段、错乱/重复step、缺失USD actor所有权、空callback非证、错误GPU资源状态、输出文件覆盖保护、布尔值冒充步号、CMU实物数据资格及副作用隔离。其中一次失败测试发现`True==1`使mock物理步布尔号被错误接受，修正后全部通过；保存的第一轮失败日志未覆盖。

扩展CPU回归范围v124–v142：**748 passed, 2 skipped, 2条既有SciPy边界裁剪警告, 0 failed**。这是明确选取的CPU扩展回归，**不是全项目GPU/PhysX回归**。

本地真实实验审计与图（所有输出通过`open('x')`首次创建，不覆盖既有证据）：
- `reports/v142_initial_real_data_and_resource_review.json`：SHA256 `943a8157efd122d25232b6d4aab12d38330a05b34c95f8e5e67eb3603130c261`；
- `reports/v142_independent_historical_sha_recheck.json`：SHA256 `3903aa44614569b87ffcaa26cf76bc9c080e721b71d068a70498e9c51fee9b67`；
- `reports/v142_source_gap_20261113_real_physx_trace_coverage.png`（历史真实5ms状态覆盖与缺失）；
- `reports/v142_source_gap_20261113_human_truth_eligibility.png`（实物人体先验/接触/GRF/职业真实性区分）；
- `reports/v142_unit_phase1.log`、`v142_unit_phase2.log`（故障负例，保留）、`v142_unit_phase2_fixed.log`、`v142_selected_cpu_regression_20261113.log`。

## 5. 原生取证与论文主线依赖

现阶段v142提升的是**可证伪的真实物理取证链**，而不是通过重复C2/QP运动学筛查冒充全身协调生成已完成。实际优先级：

1. GPU安全释放且磁盘>=20GiB后，以真正原生隔离接触、非接触/ON/OFF校准回调actor/collider路径与step时钟；保存原始manifold，确认USD只读映射可用。之后在原v54三策略下做**3×OFF/ON=6**个同种子原G1物理回合，严格整段全字段逐值比对；不改变原控制器，不手工添加ContactReportAPI后假装read-only。
2. 实际接入浮动基、足底GRF/CoP、摩擦、扭矩、全29DoF关节限制及非脚碰撞门禁。v139–v141数学轨迹包络不等于物理可执行，也不解决身体动力学。
3. 获取真正优秀人类正脚背发力链的可核查数据与合法许可，冻结被试不重叠划分，区别先验与真实碰撞/GRF真值；训练/优化前冻结未见球位和种子。
4. 以**脚背冲量占比>=95%、合法入门、球速>=3.5m/s、全29DoF真实qdot合规、无非脚碰撞、14s恢复**的联合门禁报告真实任务率；任何缺测按未知/不通过，不用“碰到球”算成功。

**本批实际状态：新PhysX=0、真人专业发力学成=否、真实腕髋原生pair=UNKNOWN、CPU 748 passed。保留负结果和设备互斥，不掩盖研究尚未完成。**

## 6. 复现（新的输出文件名，不覆盖原证据）

```bash
cd /home/you/projects/kinetickick-wt-v142
PY=/home/you/conda_env/env_isaaclab/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/v142_batch_readiness.py --output reports/v142_recheck_new.json
$PY tools/v142_native_pair_ledger.py --output reports/v142_original_v54_recheck_new.json
$PY tools/v142_plot_source_gap.py --input reports/v142_recheck_new.json --stem reports/v142_recheck_new
$PY -m pytest -q tests/test_v12[4-9]*.py tests/test_v13[0-9]*.py tests/test_v140_nonlinear_dense_reaudit.py tests/test_v141_continuous_envelope.py tests/test_v142_native_contact_and_human_qualification.py tests/test_v142_runtime_bridge.py
```
