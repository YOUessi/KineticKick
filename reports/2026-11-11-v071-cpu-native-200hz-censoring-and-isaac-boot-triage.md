# KineticKick v0.1.71｜GPU 忙碌条件下的完整 CPU-only 取证批次：200Hz 29DoF 跨事件审计、提前删失、失败可解释性与 Isaac 启动故障分层

**日期：2026-11-11，独立分支 `version/v0.1.71`。科学状态：对已见过的 v67 真实原 G1 物理记录进行严格只读、事后取证（postregistered forensic），不是新的冻结盲测，也不是新的训练结果。**

### 1. 完整执行前安全审查和冻结

- Tang 在线；现有主目录 `version/v0.1.64`、v65–v70 独立工作树，v65/v66/v68/v70 含原已有未提交数据／工作，不碰、不覆盖、不合并；v67 和 v69 工作树干净。GitHub 已有最新远端 `v0.1.69`，v70 新协议 8493a6c 在另一工作树正执行，未干预。
- 盘符 `/` 剩余约 **39GiB**、已用 96%，高于保守20GiB门槛，但禁止复制 30–45MB 物理 JSON/CMU/C3D/权重/媒体和其他多余存储。新分支由远端 clean `v0.1.69` `0ae40fb` fork 得来，**协议提前独立提交为 `89a3a7581dd557699172e08265571f31335c20dc`**；冻结协议 config SHA256 **`b1e54c1b0880e0e80c08481c13f127397bf569a20300b4e65294bdea32672e94`**。
- 当时 Tang 上 `/home/you/conda_env/env_isaaclab/bin/python` 正在执行 `v070` 串行 PhysX batch，GPU 有真实进程，故 **v71 禁止 GPU / IsaacLab**，运行分析和 pytest 时设置 `CUDA_VISIBLE_DEVICES=''`。没有插队、杀进程或修改 v70 原控制器/种子/文件。**v71 新物理回合：0**；无法提供 v71 新物理成功率。
- 冻结 v67 **9个**原始 source JSON SHA 与批次索引 SHA，全部读取原工作树路径而非重新模拟；含其 4 条 **提前终止、没有完整 wrapper/backend tag** 的记录。另用早期冻结 progress 内的原日志 SHA、真实 `V058_NATIVE_START/RETURN cuda:0` 和 `Environment device: cuda:0` 标志单独认证其 CUDA 已运行，但仍不把早停标成 `VALID_G1`。所有 raw/report/模型来源字节保持不变。

### 2. 完整的真实原 G1 200Hz 多事件证据：9条历史物理回合，无补帧

新 `tools/audit_v071_native_event_ledger.py` 逐条校验源 SHA、9个原始 CUDA 日志 SHA、29个真实 q/dq 与原速限 batched(1,29)、逐步 5ms 时间、真实全身CoM / 原 root / 双脚 ground support 力 / nonfoot 刚体聚合净力。对具有同帧 `safe_body_v062` 的完整回合，还严格校验观察者 time/index 与所有29关节对齐；对提前终止无 observer 的回合明确 NULL，不制造距离。真实 G1 原接触时刻前0.4s 至后1.25s 为唯一窗口，若无接触则缺失，后14s 不补帧。

| 已冻结 v67 场景 / 策略 | 真实 5ms 行 | 最后真实秒数 | `−0.4,+1.25` 窗口 | 最大原关节限速比 | nonfoot 最大聚合净力 N | 合法全部门禁 |
|:--|--:|--:|:--:|--:|--:|:--:|
| a / 原v32 | 2800 | 14.000 | 全 | 0.9074 | 0 | PASS |
| a / 合成时序顺序 | 2800 | 14.000 | 全 | 0.9074 | 0 | PASS |
| a / 合成时序反序 | 2800 | 14.000 | 全 | 0.9074 | 0 | PASS |
| b / 原v32 | 741 | 3.700001 | 全 | 0.9078 | 37.709 | FAIL：非脚+14秒恢复 |
| b / 合成时序顺序 | 2800 | 14.000 | 全 | 0.9078 | 0 | **FAIL：脚背份额72.983%** |
| b / 合成时序反序 | 2800 | 14.000 | 全 | 0.9078 | 0 | PASS |
| c / 原v32 | 329 | 1.640001 | **无真实接触** | 0.9136 | 2272.839 | FAIL：非脚早停 |
| c / 合成时序顺序 | 329 | 1.640001 | **无真实接触** | 0.9136 | 2225.377 | FAIL：非脚早停 |
| c / 合成时序反序 | 329 | 1.640001 | **无真实接触** | 0.9136 | 2287.163 | FAIL：非脚早停 |

原G1速度证据按所有29个关节**真实物理输出**重新评价，没有任何关节速限豁免。已见9组历史物理中 **4/9 全门禁合法**（3个独立球/重置条件，而非9个独立种子、不是新数据的成功率），5/9失败/不完全：其中 b / 顺序虽然14秒、射门进球、球速代理约4.390m/s，旧wrapper宣称PASS，但实测只有 **0.729834707982299** 正脚背冲量，必须FAIL。b / v32在接触1.575s后又真实活动至3.700001s，故后1.25s窗口有数据，不等于具备14s恢复。c 三个均 **没有真正球接触时间**，不能从CMU光学标记代填。根高度、倾角和 COM 高度也不能充当整体站立认证。

b 回合触球前0.4秒同物理支撑力：基线左约236.25 N/右107.26 N、顺序约236.43/107.26 N、反序约236.49/107.26 N；是 PhysX 原实测足底**法向净力统计**，不代表完整接触多边形/GRF 矢量，也不证明人体支撑迁移。a三策略接触后球速代理约5.198/5.231/5.171m/s，扰动改变幅度很小，不能提出合成方向优势。b中不同控制的左腕—左髋 **link-origin** 最小距离约0.138/0.129m（另有b原v32因提前终止缺失 observer），不能视作 mesh collider 距离；强非脚净力本身不是 native pair。

本批额外生成 Tang 本地真实图：`media/v071_original_200hz_nonfoot_censored.png`（3个真实场景×3个时序策略，早停后**不画延长轨迹**）及 `media/v071_context_b_measured_29dof_com_support.png`（真实接触锚定的29关节平均|qdot|、CoM高度、左右足力差；**不是人类发力学习证据**）。图与30MB+原数据仅保留 Tang，无上传。

### 3. A：GPU PhysX 双刚体 contact/manifold 真实路径证据依然不成立

加入 `has_direct_cuda_wrist_hip_pair()` 硬门检查：候选必须至少具备 GPU 原G1声明、原 PhysX 两个 collider 路径分别含 `left_wrist_yaw_link` 和 `left_hip_roll_link`，可用的三维接触点/法向和正法向冲量。即使字段齐全，这也**仅是 schema 前提**，不能替代受信的 PhysX 原生生产者、instrumented/uninstrumented 全部25物理字段比对、相同冻结种子及原始 native callback 证明。任何同名链接的等反力、`25.890153N` 净力、CPU 的另一个碰撞对、仅有link-origin间距，不能冒充 GPU 真 manifold。实际9条旧 GPU 源中 **直接证明的左腕—左髋 pair = 0**。在 GPU 已被 v70 使用时，禁止开启额外 Isaac 来伪造或自称完成该目标。

### 4. v70 额外只读取证：重复 TIMEOUT 并不是射门物理失败

`tools/audit_v071_v070_launch_stage.py` 只读取 **v070 已写入进度索引的完结子进程**，按该索引独立核对 log SHA，对每条按命令进入 AppLauncher → Kit 配置载入 → native scene / Environment CUDA → V058 native return → 原JSON分级，禁止读当前仍在写的日志、禁止占用GPU。得到的冻结时点快照见 Tang `reports/v071_independent_v070_launch_snapshot.json`，**仅代表当时进度，不是 v70 最终结论**。

观察到的已完结 v70 子进程连续 `return_code=124`、超时约240.4秒，日志确实进入 Isaac Sim AppLauncher / Kit 配置载入，但在 `Environment device: cuda:0` 原场景打印前停住，且没有 `V058_NATIVE_RETURN` 或原G1 raw JSON。正确事件分类是 **ISAAC_KIT_BOOT_OR_SCENE_INIT_NO_G1_EPISODE**，不允许把这些 `NO_G1_OUTPUT` 记作射门失手或模仿学习负例。日志出现 GLFW 初始化和 DriverShaderCache 警告，但**不足以认定具体根因**；后续应在GPU释放后，另开新版本进行有界单进程 native 控制器启动/场景创建测量，不可以无预检地继续发12个240秒的无证据 job。

### 5. 可复现性、科学假设与下一 GPU 研发门槛

新的主要代码为独立 CPU 的 `audit_v071_native_event_ledger.py`、`audit_v071_v070_launch_stage.py`、`plot_v071_native_failure_dynamics.py`，配 `tests/test_v071_cpu_physics_evidence.py` 中时钟乱序、原qdot(1,29)、无原CoM、95%脚背反例、nonfoot净力不能变成pair、缺后1.25s、14s删失、v070无raw绝不记物理失败、HASH篡改拒绝、历史9例真PhysX交叉检查等**15项**。

科研假设依然是学习**人体支撑足plant → 近端髋/骨盆躯干旋转 → 踢腿膝踝速度耦合 → 对侧摆臂 → 鞋面触球 → 随挥/着地恢复**，从原始120Hz光学人体点轨迹构建相位/相对时序，再用独立的真实机器人CoM、支撑力、qdot、足球接触和碰撞层安全约束做 joint policy，而不是放大合成幅度至1mrad足残差或仅拟合离线余弦。**当前CMU/6段光学人体球标记没有真实足球法向冲量、同步足底GRF或可靠精英运动员资质，因此任何将其称为职业抽射力学GT的训练均禁止。**未来必须获得并合格标注，按被试隔离训练/验证/测试，不能在 v67/v70 已看过的失败样本上调参后再报告留出集增益。

下一次大批依赖：先让现有 v70 自主结束且释放GPU；查清 Isaac Boot/Kit卡住的环境先验（driver/graphics context/init阶段），仅有界新版本修复，**不得改原冻结工作树/日志**。成功后以3次原物理、原控制器全25字段无扰动核对资格拿到左腕—左髋真实 collider path/manifold；随后可重新冻结具有可测支撑、mesh-clearance和29DoF安全可行域的新的全身协调 control interface，再批量 GPU 运行留出 seed/球位和反序、无摆臂等真实负控制，全部原强制门禁一致；旧v67数据只能作为历史压力测试，不是证明新策略优越的盲测。不能声称当前已经实现专业级自然暴力正脚背抽射。

**运行命令**：`python3 tools/audit_v071_native_event_ledger.py --source-dir /home/you/projects/kinetickick-wt-v067/reports`、`python3 tools/audit_v071_v070_launch_stage.py`、`MPLBACKEND=Agg python3 tools/plot_v071_native_failure_dynamics.py`、`CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/`。所有审计输出文件用排他创建 `open('x')`，禁止覆盖历史读入文件；纯解析不含 sim load。

### 6. 实际测试与输入保护（本批最终结果）

- v071 自身新增测试：**15 passed**，覆盖本地伪造数据的严格否定测试和原始 v67 9例 SHA-lock 历史交叉验收。
- 在 `CUDA_VISIBLE_DEVICES=''` 严格不运行GPU物理的条件下，全项目 `tests/` 最终为 **755 passed + 51 subtests passed，69 skipped，0 failed**。69项因非 CPU 环境/其他本地环境所需条件跳过，不当作通过；**755不是真实G1物理回合数**。
- 新工作树的旧回归测试初始缺失 gitignored 资产而失败；根据报错仅在 v071 独立目录建立 53 个 original source SHA 锁定的只读用途资产引用/2份 40KB URDF 本地副本，既不修改主工作树，也不上传任何模型/实验 JSON/原始动捕。完整测试后逐份重算并验证 **53/53 原始输入SHA未变**。测试日志保留 `reports/v071_full_pytest_cpu_v5.log`，这些原始数据按 `.gitignore` 继续留在 Tang。
- 可复现实验的分母：**本批真实 PhysX 新回合 0**（GPU被另一已存在v70占用），**历史原 PhysX 原始输入9**、历史严格成功 **4/9**、GPU原生左腕—左髋 manifold **0项认证**。不能把历史回放计数并入新训练/新物理成功次数。
