# KineticKick v0.1.158｜原G1 URDF 29电机约束的全身35D假定支撑QP：冻结对照、负结果与来源审计

**日期：** 2026-11-14
**基线：** version/v0.1.156（\`998c17b93619637443f883784e9daa2539348423\`）；v157存在独立未提交文件，未修改
**预冻结协议 SHA：** \`aa9deaeba02024f175af1d7b9918d1764d5cd01a\`
**证据等级：** 原始G1 URDF来源参数 + Pinocchio 2.7.0 **离线CPU动力学** + 原v54历史PhysX来源校验；不是本轮新PhysX实验或从高水平人类足球动作学得的控制器。

## 0. 安全与来源核查

Tang当前在线，根目录 \`/home/you/projects/kinetickick\` 仍为v64（d807850），v157独立工作树处于未提交开发中。v158从干净v156创建新worktree，不删除、不覆盖任何历史版本。GitHub版本分支已核查，v158此前未使用。磁盘约30GiB可用（总使用率97%，高于20GiB保护底线）；原v120 pytest PID242644持续占用RTX4090计算上下文约260MiB，严格不启动新IsaacLab/PhysX/GPU作业、不杀进程。

原CMU/C3D、权重、URDF、原PhysX大JSON和生成的审计/图像继续存于Tang本地；其他测试需要的受忽略旧资产以**指向原有路径的符号链接**挂接，不复制大型原始数据或修改目标文件。

## 1. 本轮真正要检验的两个机制假说

此前v154、v155只是合成M/h、假定扭矩界与假定足底模型；v156才完成原G1 URDF来源35维质量矩阵与雅可比的建立。v158将**同一原URDF源状态**接入受全身动力学、6D假定足底支撑、原29个执行关节位置/速度/URDF额定努力约束的QP，完成严格配对、无训练的努力上界消融。

动力学条件：

\[
M(q)\ddot q+h(q,\dot q)=S^\top\tau+\sum_{\text{hypothesized feet}}J_f^\top w_f,\qquad
J_f\ddot q+\dot J_f\dot q=0.
\]

其中全广义速度35维（6浮动基+29执行关节），一个或两个脚的每只支撑力矩为6维。每步5ms，对29个关节使用URDF原\`lower/upper/velocity/effort\`。还保留法向非负和假定摩擦 \(\mu=0.6\)。脚底力矩边界由真实URDF碰撞体**世界方向旋转后的外包围盒**得到，**仅是可能接触范围的宽松必要条件，绝非实测CoP/GRF、真实USD鞋底接触多边形或真实无滑移证据**。碰撞面、腕髋manifold仍未知。

6阶段、每阶段29关节协同加速度参考（人工确定性函数，包括髋、腰、手臂和摆腿）不再是1mrad姿态残差；但其强度和方向是**研究用人工激励**，绝非优秀足球运动员动捕/发力真值，不能被认为完成了人类技能学习。

## 2. 先冻结、后实验：96组同源全身QP

预冻结分母：6动作阶段（approach / plant / coil / swing / followthrough / recover）×2假定支撑（左脚 / 双脚）×4种子＝48个原URDF模型人工姿态；每态配对比较**URDF原额定努力上界**与**100倍努力反事实**，共96个QP，两个条件除努力上界外完全相同。额外预声明8类错误来源×2种子＝16组必须拒绝；预先固定12组独立RNEA/ABA身份检查。

| 冻结结果 | 真实来源URDF努力上界 | 100倍上界反事实 |
|---|---:|---:|
| 数值求解状态 \`solved\` | **48/48** | **48/48** |
| 29关节加速度跟踪RMSE均值 | 1.929222838 rad/s² | 1.929222722 rad/s² |
| 最大原URDF努力利用率（对原限值归一） | 0.505596 | 0.505596 |
| 是否证明真实物理射门成功 | **否** | **否** |

两组共96/96符合求解器数值状态及独立动力学/支撑/约束残差审计。48个配对的平均RMSE变化（100倍−原额定）约 **−1.19×10⁻⁷ rad/s²**；最大单配对绝对变化约 **5.49×10⁻⁶ rad/s²**。原努力上界在本批人工低动态/假定支撑工况中没有激活：最大原URDF努力使用率仅50.56%。

尤其是人工\`swing\`目标，原努力条件的跟踪RMSE均值约 **6.3860 rad/s²**；扩大努力上界也几乎不变。因此**不能把这些工况的误差直接归因于额定关节力矩饱和**。它可能来自假定6D支撑、目标冲突、目标正规化权重等，需后续冻结消融再区分。也不能据此推导高速射门中不存在力矩瓶颈；这些不是专业人体踢球或真实接触/冲量工况。

本批12组原URDF模型独立RNEA/ABA身份重检中，最大RNEA一致性误差约 **1.71×10⁻¹³**。这仅认证原URDF数值计算的一致性；并不认证运行时PhysX的惯性、地面反力或接触器件。

## 3. 来源篡改、负控制与可复现性负结果

冻结16/16项错误来源注入成功拒绝，包括错误29关节名顺序、错URDF SHA、非法根四元数、缺少支撑、冒充运行时接触、冒充合格人类技能学习、非法努力上界、非有限动力学偏置。专项测试进一步加入6种源码矩阵/参考篡改：伪造M、h、J、Jdotv、脚几何外包络及冻结全身参考，**均被重新读取的URDF/Pin源身份拒绝**。输入来源校验实际逐次重新计算 \`M/h/J/Jdotv\` 和足部几何，不信任单纯声明的SHA字段。

本轮源码加固之前和之后各执行完整96组CPU求解，保存为不同**write-once本地证据**：
- 首次：\`reports/v158_original_urdf_actuation_cpu_20261114.json\`，SHA256 \`08f68e86fb94c36bb36e10761c65a2c7ecd5bb115edeeeafe84d1d23e9025759\`
- 来源加固：\`reports/v158_original_urdf_actuation_cpu_sourceverified_20261114.json\`，SHA256 \`48c92e1d17d0954d6c74e34d68e618408a83fdacfb727ca523ca4ff1e056e24f\`

**必须保留的负结果：两次JSON并非字节完全一致**；96例求解状态和来源SHA相同，95/96行数值完全相同，1/96行出现数值差，最大差约 **1.33606×10⁻⁴**（包括力/加速度解的浮点差异）。没有在观察后调整OSQP设置/收敛阈值使其看起来一致，也没有以“同种子”为由声称严格位级复现。独立比较见 \`reports/v158_repeat_difference_20261114.json\`，SHA256 \`1b085af7495ea781953dc666677c6c38c9ce9a27d3da71cf9acc2a6ab2622d37\`。求解数值的字节级可复现性是后续需要单独解决的工程问题。

两张图（原URDF相位RMSE和29关节努力上界余量）已在Tang实际生成；只作为离线模型数值图，不作为真实动作或合法射门图。两张最终图SHA256分别为 \`88c7da882173b48b9ada1a737f87d60d036f5546afce74fa43791c467ed91733\` 与 \`436cb05ba5173272c6fe3a56afe5edfdfeeeb1e9b7c4996001815bc04e8fd648\`。

## 4. 独立复核旧真实物理证据（没有新PhysX）

历史v54三回合重新通过已有source SHA：**2,800 + 2,800 + 1,028 = 6,628个原200Hz状态**。名义接触窗口合计993帧，CMU10_05提前终止前最后0.4秒只有81帧完整记录，不制造5.145秒关节状态或14秒恢复。原真实历史轨迹仍缺根四元数、根部6D速度、同物理步真实足底GRF/CoP和原生collider/manifold，因此无法合法重建真实35D动态控制输入；并且不能将腕髋25.890153N等大反向净接触力认定为直接腕髋碰撞。

旧v53四球位×三控制器**12场**严格成功仍为各0/4，不因原位重新审计而产生新物理实验。合法正脚背冲量≥95%、球速≥3.5m/s、进球、全部29关节qdot、非脚撞击和随挥恢复仍需未来同一次实际PhysX中逐门禁验证。

本次新增真实PhysX回合：**0**；当前新控制器严格物理成功率：**未知/不可计算**；左腕—左髋原生collider pair/manifold：**UNKNOWN**；优秀人体真实足球训练：**0**。

## 5. 实现、测试与复现

提交源码：
- \`tools/v158_source_actuation_wrench.py\`：原29执行器顺序与原URDF限值映射、真实源M/h/J6、6D必要支撑力矩边界、35D-QP、原源复核与独立残差审计。
- \`tools/v158_frozen_actuation_campaign.py\`：冻结96组配对及16项负控制、12项独立RNEA/ABA和历史6,628状态来源复核，**本地write-once**证据。
- \`tools/v158_independent_reproduction_audit.py\`：不设事后通过阈值，保存两次离线运行的逐项非位级差异。
- \`tools/v158_plot_actuation_ablation.py\`：生成两张离线实证数值图，不提交PNG。
- \`tests/test_v158_original_urdf_actuator_source.py\`：140项专项CPU测试，含篡改与安全拒绝。
- \`configs/v158_frozen_original_actuator_urdf_wrench.json\`、\`docs/V158_FROZEN_SOURCE_ACTUATION_VS_UNBOUNDED_EFFORT.md\`：原始SHA冻结研究协议。

最终专项回归 **140 passed**；扩展v124–v158 CPU测试 **2,021 passed、2 skipped、6条既有SciPy SLSQP裁剪警告、0 failed**。第一次扩展回归曾因新worktree缺少与原资产的只读符号链接出现4项失败；恢复原v156既有\`experiments\`/\`data\`/\`third_party\`/\`media\`/checkpoint等原路径符号链接，19项相关测试重检全部通过，最终完整回归再次全部通过。失败日志未删除。

复现（仅Tang CPU，写新文件名，绝不覆盖旧证据）：

\`\`\`bash
cd /home/you/projects/kinetickick-wt-v158
export PYTHONNOUSERSITE=1 PYTHONPATH=tools
/home/you/conda_env/env_isaaclab/bin/python tools/v158_frozen_actuation_campaign.py \
  --output reports/v158_REPLAY_NEW_FILENAME.json
/home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_v158_original_urdf_actuator_source.py
\`\`\`

原G1 URDF SHA256：\`dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b\`；Python环境NumPy 1.26.4、SciPy 1.15.3、Pinocchio 2.7.0、OSQP 0.6.7.post3。实际离线代码不能发出机器人控制命令。

## 6. 下一批研究依赖与论文假说

1. **GPU竞争解除后优先原生接触证据**：在真实运行时USD中检查是否为实际腕髋collider启用报告；独立校准PhysX回调时钟和两种正负接触控制；原三控制OFF/ON六回合逐200Hz状态对照，要求配对collider paths、同物理步接触manifold、完整浮动基状态及read-only仪器不改变物理。禁止把无回调等同于无碰撞。
2. **真实脚部接触/支撑资格**：读取实际场景（可能与URDF不同）的脚底碰撞几何，记录ground-only GRF/CoP、6D足底Jacobian、真实净力及非脚事件，同时保留全部原29 qdot、14秒恢复。现有URDF外包络不是这项实测。
3. **论文主贡献而非反复QP玩具**：获取跨被试、精确触球、真实3D高水平正脚背动作和同步支撑约束；只有资格通过后，冻结训练/测试集再训练真正的全身动作协调潜变量、阶段时序和足底约束的闭环策略。严格对照无人体先验/弱1mrad旧基线/全身学习策略，在未见球位和多种子原G1 PhysX中测高速正脚背、球门、触球合法性、非脚撞击、关节/扭矩和恢复。

**边界：** v158是一个完成了预冻结、来源认证、96组真实URDF参数离线QP、负控制、独立复核、测试和可重复Git源码交付的研发批次；没有真实PhysX新射门、没有训练合格优秀人体足球数据，也不能将固定6D支撑的数值最优解宣传为运动自然性或职业式暴力抽射已实现。
