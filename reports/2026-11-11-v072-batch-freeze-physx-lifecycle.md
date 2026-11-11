# KineticKick v0.1.72 — ISAAC 物理退出阶段定位 + 原始 29DoF 支撑耦合审计 + 原 CMU 光学数据资格

**科学状态：独立冻结、有界诊断 + 已见数据的事后取证；没有新的完整 G1 物理射门。日期：2026-11-11。**

## 1. 来源、机器、不可变协议与源数据

从 v71 GitHub 确认的 `ca2e3f51a7d30467bb6416b9abc0e7468cac2155` 建立独立干净 `version/v0.1.72` worktree，先提交冻结协议 `b3f7b7225ae235afc64b060e18eaa9ddbe0ed8c9`。协议 `configs/v072_runtime_triage_and_native_pair.json` SHA256 `a05cc82092bd45944f364361e05df0661f1608e56058f5004080ce322e298cc2`，包含7个原关键源SHA、GPU互斥与20GiB磁盘阈值、65s单次smoke门禁，以及只有无误退出才允许的原 v54 3次G1配对实验计划。

Tang（Ubuntu RTX4090，Driver 580.178.04）在线，起始没有其他 CUDA compute PID；磁盘约39GiB可用。v70原独立批次12次均结束。v65/v66/v68/v70既存未提交文件保持原状；不覆盖任何旧工作树、CMU/C3D、原模型或原始PhysX JSON。未强推、未删除文件。实验脚本与论文报告上 GitHub，logs/原始JSON/NPZ/媒体仍只在Tang。

## 2. 实际新 GPU 实验：一次 bounded smoke，G1回合0

按冻结单进程 `tools/v072_isolated_runtime_probe.py`，使用原封不动的 `tools/isaac_smoke.py`，真实GPU Isaac Sim/SimulationContext.reset() 后8次 step **确实执行**，约第3.3秒已打印 `KINETICKICK_ISAAC_SMOKE_OK`。但该成功标记位于 `finally: app.close()` *之前*，进程65s超时 returncode=124，没有正常退出。独立 `tools/v072_stage_qualification.py` 据此纠正初始笼统的 KIT_BOOT_TIMEOUT 标签：**SMOKE_STEPPED_EIGHT_TIMES_THEN_EXIT_OR_APP_CLOSE_TIMEOUT**。现有证据没有堆栈，不能进一步断言卡在 app.close 本身还是 Kit/Python finalization。原日志/结果均保留，不重写。唯一新GPU进程为此次smoke；**原29DoF真实 G1 踢球=0；新策略成功率没有分母**。

历史 v70 的12份源SHA锁定日志均以约240s超时、0完整G1原始结果，属于无法评分的 `NO_G1_OUTPUT`；不能记作12次踢球失败。原 v67 **成功**的 G1日志同样有 GLFW 警告，因此不能把“GLFW failed”文本直接定为故障原因。因为预冻结的 gate0 退出失败，**不启动原 v54 的3个对照、不冒险连续产生无效GPU作业**。原 G1 左腕—左髋直接contact-pair/manifold仍 **0个认证**；旧5.145s双方25.89N等大反向净力仍仅是强烈嫌疑，非真实 collider path 认证。所有原始本地证据 `reports/v072_kit_empty_headless.log`、`v072_kit_empty_headless_result.json`、`v072_independent_boot_lifecycle.json`。

## 3. 真正200Hz原 G1 全身动力学：九份已见历史资料，禁止训练后留出集声称

新增 `tools/v072_measured_support_coupling.py` 读取v71冻结9份原 v67 SHA；**15,728条原记录（含4条早终止+1微秒末尾记录，非独立5ms步）、5/9个14s完成、6/9真实触球且具备[-0.4s,+1.25s]完整窗口**，c球位3组在1.640001s真实非脚早停、无任何真实触球窗口；b基线3.700001s早停。以原29个 `joint_vel`、原不同关节 `qdot` 限制、whole-body COM、双足 `support_foot_force_N`、腕髋link-origin距离和各腿/腰/臂 RMS实测窗统计、±100ms时序相关输出独立表；**相关不代表因果传力，足底净法向力不是CoP或人体GRF，origin距离不是shape clearance**。

a球位触球前0.4s左足支持法向份额≈0.444（3策略）；b球位≈0.457（3策略）。v71严格全部门禁原历史结果为4/9；v72另行审查的部分门禁也为4/9，但不能作为替代全部门禁的独立成绩。b合成顺序虽进球且完成14s，却只有约72.98%正脚背冲量 → 必须FAIL。真实终止/支撑图仅Tang：`media/v072_original_g1_early_termination.png` 和 `media/v072_original_support_and_waist_coupling.png`，不补画早终止后的曲线。

## 4. 六段真实120Hz CMU球-足点云资格：5组同被试/1组另一被试，0个GRF真值

新增 `tools/v072_cmu_geometry_qualification.py` 全只读检查6个原光学 C3D (10_01/02/03/05/06、11_01)，均为原120Hz毫米点阵，并识别3个足球WEP标记、左右足6个标记与各原始缺帧掩码。六段都有光学近球点候选，但 **无可信同步力台GRF、无鞋面真触球冲量、无职业运动员资格认证**。10_03虽观察到RTOE离球标记约0.00517m，仍不能宣称真实冲击。样本被试组10包含5段、组11仅1段；训练/验证/测试不得随机以clip混分同一被试。不能用这组数据宣称“学会职业式暴力抽射”。SHA逐段验收，结果与候选时间留本地 `reports/v072_six_cmu_optical_suitability.json`。

## 5. 下一版本严格依赖与可检验论文假设

**H1（尚未验证）**：学习支撑脚plant→髋/骨盆与腰扭转→大腿/膝踝速度耦合→对侧摆臂→鞋面撞击→随挥落脚的时序生成表征，在原29DoF PD/PhysX、真实双足支持、29关节速限、同向冲量与接触安全硬约束下，较仅脚部残差/固定相位生成能在未见球位和跨被试条件下生成更自然、更安全且具有足够球速的动作。**H0**：纯光学相位和合成信号不改善任何严格的真实物理指标。不得把离线余弦、单次进球、微小1mrad动作或碰球视为论文终点。

阶段依赖：**先在新冻结版本对AppLauncher→Context.reset→step→app.close entry/exit分段打点及堆栈取证，消除退出超时并实现正常关闭**；再用原v54三次原控制器 off/on 只读 contact 观察和全部共享物理字段严格比对。只有native collider0/1路径、接触点、法线、法向冲量与准确CUDA路径足够时确认真正的左腕—左髋 pair；空callback仍不等于无接触。再冻结新全身支撑/安全控制，训练/优化与留出球位、种子、逆时序/去摆臂/去腰时序负例，对每次脚背≥95%、球门、球速、全部原qdot、非脚碰撞、14s恢复独立严格验收。原G1安全接口 `wholebody_safety_interface_v055.py` 对缺native形状证据仍是fail-closed，不能装作已经达到硬件安全。

## 6. 全部测试、数据保护、复现

新增 `tools/v072_isolated_runtime_probe.py`、`v072_stage_qualification.py`、`v072_measured_support_coupling.py`、`v072_cmu_geometry_qualification.py`、`v072_plot_physical_failure.py` 和 `tests/test_v072_safety_and_source_qualification.py`。27个新专项测试均PASS；**全项目CPU-only：782 passed、69 skipped、51 subtests passed、0 failed（20.01s）**。项目完整测试使用53份原本在Tang的gitignored fixture的独立只读引用/两份40KB URDF复制；验收后 **53/53 原始依赖 SHA保持，9/9历史G1原始SHA保持，6/6 C3D原始SHA保持**。72版原始独立产物全部排他写入 `open('x')`，禁止第二次同文件覆盖；GitHub只提交此文本/脚本/配置/测试。

原地只读测试：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/`。相关分析器一律write-once，已有同名结果应拒绝覆盖，复现必须独立新目录配置新目标名并先SHA核对。GPU已有非本项目进程或磁盘<20GiB则不得启动GPU物理。
