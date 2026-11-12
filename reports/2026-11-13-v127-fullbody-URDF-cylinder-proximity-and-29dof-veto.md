# KineticKick v0.1.127 — 29回合全29DoF URDF 碰撞几何重建与有限时域安全接口

**日期**：2026-11-13
**Git 分支**：version/v0.1.127
**方法冻结 SHA（运行结果之前已推远端）**：8ce91b872f8f7702dec9020281c42dd22e03cf88
**输入源**：历史版本v116/v117/v118的原始PAID G1自由根200Hz PhysX轨迹；v125独立SHA认证清单；原始PAID resolved 29DoF G1 URDF（v116 SHA锁定）。
**声明**：本批没有执行任何新的 IsaacLab/PhysX 射门、没有训练新的真人策略、没有新的电机动作。所有动力学与成功门禁来自已执行且通过独立原始SHA核验的真实物理记录，几何推断是**离线的URDF重建**，绝非PhysX原生contact manifold测量。

## 1. 宿主安全、证据与实验来源

首先核查Tang在线、原Git远端分支最高为v0.1.126，v126工作树干净。在未使用的version/v0.1.127建立隔离 worktree 并在分析前推送研究协议。根目录已有v0.1.64工作树包含未跟踪文件，未触碰；旧worktree、CMU/C3D、模型、30MB原始物理JSON均没有修改。磁盘可用约30GiB，超过20GiB门槛。

GPU上依然有已有的v120 pytest进程PID 242644：检查时已经运行超过5小时、占用RTX4090约260MiB与1个以上CPU核心；按GPU互斥原则不终止、不抢占、不启动新的IsaacLab。CPU计算在独立目录以低优先级运行；CPU测试禁用CUDA_VISIBLE_DEVICES，避免额外GPU工作。

原始数据来源于v116（10次）、v117（11次）、v118（8次）历史真实PhysX，总计**29个回合×2,800个200Hz样本=81,200物理帧**。其中24个独立「8球位×3控制器」组合，五次是同条件ContactSensor观察器OFF/ON重复，不能当作独立成功。历史24个条件的严格结果仍为**13 PASS / 11 FAIL**，没有改变或重算硬门禁定义。所有原始物理JSON、批次master、控制器种子、Git提交和v125独立审计SHA均需要一致才参与。

## 2. 新开发：完整URDF前向运动学与外包碰撞体几何下界

新工具tools/v127_collider_proximity.py为原始29DoF关节顺序重建39条URDF关节连接（40个link），每个200Hz物理帧使用实测q进行完整正运动学：显式采用每条joint的origin xyz、rpy、axis、固定/转动类型，使用每个collision的origin xyz、rpy以及圆柱半径/长度。

针对四对预冻结配对：左腕yaw/左髋roll、右腕yaw/右髋roll、左腕yaw/右髋roll、右腕yaw/左髋roll。每条有限圆柱轴线段按其半径外包成胶囊体，并求三维**有限线段到线段的最小距离**，减去双方半径，得到一个保守表面间距下界。

**严格逻辑**：圆柱包含于外包胶囊，所以若该下界>0，可以排除对应URDF圆柱在该姿态下碰撞；若下界≤0，只能说不能排除，不代表发生接触。此结论只适用于原始URDF模型和重建q，不具备确认运行时USD collider是否相同、PhysX自碰撞过滤、actor/path/contact point/normal/impulse的效力。

使用**独立SciPy Rotation逐链矩阵组合**校验了5组29DoF随机状态下4个目标link的旋转和平移，与主实现逐元素一致；使用SciPy有界连续优化对8组随机3D线段距离作独立参考，两者在测试公差内相符。还测试了平行、相交、错位线段、NaN、失效URDF和错误关节维数等负例。

## 3. 真实数据分析（严格区分几何、接触、动力学）

在全部24个独立物理条件的14s内，四对的**每一帧**外包胶囊间距下界都严格为正：

| 预冻结URDF圆柱配对 | 24条件合并14s最小下界 | 14s内下界≤0的条件数 | 能排除什么 |
|---|---:|---:|---|
| 左腕yaw / 左髋roll | **+0.010014m** | 0/24 | 这些历史q下，原URDF所列两圆柱没有接触 |
| 右腕yaw / 右髋roll | **+0.013138m** | 0/24 | 同上 |
| 左腕yaw / 右髋roll | **+0.210290m** | 0/24 | 同上 |
| 右腕yaw / 左髋roll | **+0.243519m** | 0/24 | 同上 |

注意，这是**四对指定圆柱的几何排除**，不是整个G1机器人无自碰撞认证，也不是证明运行时PhysX没有发生腕髋碰撞。

预冻结时间窗统计（均来自实测q、真实时间戳、独立SHA、触球时间）：

| 窗口 | 左腕/左髋最小下界 | 右腕/右髋最小下界 | 四配对≤2cm提醒 |
|---|---:|---:|---|
| 触球前400ms | +0.089871m | +0.090429m | 0/24个独立条件 |
| 触球后1,250ms | +0.139585m | +0.062678m | 0/24个独立条件 |
| 其余至14s恢复 | **+0.010014m** | **+0.013138m** | 左侧13/24条件出现提醒；右侧1/24 |

因此，原URDF腕髋圆柱间的**临近风险集中在晚期恢复**，而不是触球前0.4s或触球后前1.25s。这一结果支持未来控制将随挥、落地和更晚恢复纳入完整安全时域；但“2cm提醒”只是设计警戒线（不是测出的PhysX接触冲量），也不能作为自然全身协调被学习的证据。

本批24个独立场景和五个观察器重复的四配对几何数列逐帧审计：五组OFF/ON的2,800帧×4配对**完全一致**；原物理状态一致性另由既有v125报告提供，不能用仅几何数列代替状态全字段对照。

这里一个关键解释边界：早期v54在**另外一组**CMU10_05物理轨迹的5.145秒观察到左腕和左髋大小25.890153N、方向相反的净力，那不是本批v116–v118的24个独立场景。本批无法据另一批的URDF间距排除那次事件；那次 **native actor/collider/manifold仍为UNKNOWN**。本批历史29回合的nonfoot净力峰值没有构成v54事件的直接复现，不能混淆实验批次。

## 4. 全29DoF有限时域前置安全否决接口

新增tools/v127_lookahead_safety.py，把实测URDF的4配对2cm几何提醒引入已有v117七阶段TemporalPlan / v107_motor_admission的**联合离线审查**：保留29个原q、qdot、qddot、τ、joint position、支撑力代理、支撑滑移、时间戳、真人来源资格及原生碰撞witness等拒绝理由。

即使在单测中人为构造全部“真”的witness、声称来源合格，接口仍然固定返回**motor_execution_authorized=False，physical_motor_command_issued=False**。它只产生离线否决或风险说明，不导出命令、不改旧控制器、更不将URDF几何的分离或合成证明升级为可下发电机许可。将来必须用真实PhysX正负控制和全程物理试验独立认证。

此实现不是旧1mrad参考、不是脚步小残差、不是“已学习的人体动力学”，而是允许下一代整条29DoF运动生成器在引入原生碰撞真值之前就明确拒绝不安全候选的可执行接口边界。真人运动资格仍不具备专业射门冲量、GRF、CoP及跨三组被试独立划分。

## 5. 负例、修复、测试与出图

- 初次CPU审计揭示原始29DoF velocity limits存放为(1,29)而非(29,)，修正按29个值展开；**未修改真实数据**。
- 第二次揭示v125既有“触球前/后”时间切片不能代表完整14秒qdot最大值；修正为与已认证原hard gate中的完整qdot比值逐值核对。这同样避免把“早期阶段左删失”错误变成安全结论。
- 完成**38项v127几何及全时域接口测试**，并用原env_isaaclab Python但CUDA隐藏、pytest插件自动加载禁用执行跨v107/v117/v124/v125/v126/v127的**149 passed / 0 failed** CPU相关回归。未报告全项目GPU测试通过。
- 真实审计结果：reports/v127_original_g1_capsule_geometry.audit.json，SHA256为5f81a667b22328071dab839000e843214b6c4347256a531b9c26e741dc381a13。
- 真实数据失败/几何图：reports/v127_physics_full14s_geometry_lower_bound.png 和 reports/v127_physics_phase_proximity_alerts.png，源码tools/v127_plot_geometry.py可重复出图。大文件、图像和审计JSON按.gitignore留在Tang，不上传GitHub。

复现命令（在v127工作树，需原已认证source worktrees与URDF仍可用）：

    CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 python3 tools/v127_collider_proximity.py --output reports/v127_repeat_new_unique_filename.audit.json
    CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 python3 tools/v127_plot_geometry.py --audit reports/v127_original_g1_capsule_geometry.audit.json --outdir reports
    CUDA_VISIBLE_DEVICES=-1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_v127_collider_proximity.py tests/test_v127_lookahead_safety.py

独立重复必须采用新文件名，CLI以x模式拒绝覆盖审计证据。已有原始数据只读，未重新跑物理；初次因程序验证错误失败的尝试保留在当前批次日志。

## 6. 论文假设、未解决部分与下一依赖

**H127**：在机器人全身恢复阶段，姿态和关节时序改变会让腕/髋相对几何距离快速缩短，因此真正的29DoF动作生成必须具备**触球后≥1.25秒直至完整14秒的动态接触安全约束**。本批确认原始URDF两个同侧圆柱在不少历史恢复片段出现2cm内接近，但全部24独立条件仍处于严格正的外包胶囊间距；还没有证据表明它们发生实际碰撞。

**下一工作必须**：GPU释放后，采用真正USD碰撞原生actor/collider正控制与分离负控制；确认**运行时**USD collider是否仍是这些圆柱、碰撞API绑定位置、contact report step时序和contact pair数据结构，并在原始v54失败轨迹的同一3个回合OFF/ON逐字段复跑。只有获得真实碰撞双方真值才可标记腕髋自碰撞。之后才是合格高水平真人足球全身时序数据、真实静止起步和支持/防碰撞硬约束的学习生成策略，预先冻结全新球位/种子后才允许GPU训练与独立物理评价。

**最终状态**：本批新物理射门**0次**，本批新真实物理成功率不适用；历史严格结果13/24未更改。未证明自然或职业式高速正脚背抽射，原生碰撞双方门禁仍未通过。所有负结果保留。
