# KineticKick


## v0.1.63：8次原29DoF G1 PhysX，真实合成相位全身动作负结果（不冒充人体学习）

承接已验证 v0.1.62 CUDA 200Hz全身只读观察，本版在干净分支、预注册SHA锁定源与种子后，**串行执行8次原G1/PhysX物理**：4组球位/独立种子×原v32/新合成多关节相位参考。**8/8有效**、原v32 **4/4 PASS**、合成参考**3/4 PASS**，合并7/8。hold_b合成参考虽入门、全部原qdot和14秒恢复均通过，但实测触球分类ambiguous、合法脚背冲量占比仅**59.22%**（门槛95%），所以严格FAIL且不升级默认控制器。每一对0–0.5s共25类原物理字段逐值相同；中心基线与v0.1.62的2800行全部原字段逐值相同。新控制真实幅度最大约**0.01738rad**、7步非零控制，处于保守支撑/速度/腕髋间距代理限制下；**不是专业运动员人体时序学得的自然暴力抽射**。

本版还逐行对齐此前回调快约10ms的时钟标签到原200Hz测得状态，保持原值以供审计；认证完整29DoF位置、速度和所有原始速度上限。GPU原生真实左腕—左髋contact pair/manifold仍缺，不能用净力或连杆原点距离冒充真值。人体6段CMU未提供精确脚球撞击或人体GRF。代码含相位候选、纯安全门测试、原PhysX冻结8回合runner、后续独立审计与实测绘图。**769 passed + 69 subtests passed**（本版最终记录）。原物理大JSON及图仍只留Tang。

[严格8回合实验表和失败归因](reports/2026-11-11-v063-real-29dof-synthetic-phase-ablation.md) · [专业人体数据资格门禁](docs/V063_HUMAN_DATA_QUALIFICATION.md) · 协议 configs/g1_phase_batch_v063.json。


## v0.1.54：三次原G1碰撞传感器复跑，定位10_05失败时左腕/左髋等大反向接触力

接续上一批12回合原PhysX比较，本轮另开branch，冻结14项原版本源码/3组真实物理结果SHA。只在原已知球位的v32-zero、CMU10_02、CMU10_05三种原控制配置上使用现有`--contact-diagnostics`开启**只读**命名非足刚体合力记录，一共实际运行三组独立原G1/Isaac物理回合。三组的真实200Hz追踪、完整接触、50Hz29DoF关节/目标、射门和全部终止结果与原v52历史源**在所有共同物理字段逐值相同**，新增诊断无策略/本体/球状态扰动；源数据保存在Tang忽略目录，没有人工修改或挑选。

失败组CMU10_05再次在5.140001s`physical_nonfoot`终止。独立碰撞传感器在约**5.145s**同时记录左手腕`left_wrist_yaw_link`和左髋`left_hip_roll_link`各25.890153N、方向**完全相反**的原始合接触力，两个连杆原点相距0.1139m；两个原成功对照都没有≥5N的命名非脚碰撞。证据强烈符合**左腕—左髋自碰撞**，而不是声称“整个机器人已经跌倒”；但net contact force仍不提供PhysX碰撞对/contact manifold，因此**可能的自碰撞尚不是严格验证的碰撞对真值**。下一批应验证碰撞对而非放宽安全终止门槛；保留v32/v5默认小脑不变。

审计与来源：`configs/cmu_named_nonfoot_diagnostics_v054.json`、`tools/run_named_nonfoot_diagnostics_v054.py`、`tools/audit_named_nonfoot_diagnostics_v054.py`、`tools/infer_nonfoot_pair_v054.py`、`tools/write_named_nonfoot_report_v054.py`；[完整真实PhysX失败碰撞体研究报告](reports/2026-11-11-v054-original-PhysX-named-nonfoot-contact.md)，新增测试tests/test_named_nonfoot_v054.py。**655 passed +69 subtests passed**。仍不是专业人类自然暴力抽射学习成功；暂时没有新训练的小脑/脑网络，也不升默认策略。

## v0.1.53：四球位×三冻结控制器=12次真实PhysX，人体时序先验未带来严格射门成功

本轮按照一个完整研发批次（冻结协议、12个原始G1 PhysX回合、200Hz失败归因、独立审计、全项目测试、GitHub发布）执行，而非每一步等待“继续”。在训练/重定向任何新策略前，冻结4组球位xy偏移/机器人初速度，比较原v32、小幅真实CMU10_02先验、小幅CMU10_05先验；全部12个控制/球体结果有SHA证据、原脚球碰撞与原执行器限制保持不变。**严格合法正脚背、≥3.5m/s出球、全14s、跑动/接触/射后恢复及实际qdot安全：三个方案均0/4成功**；新CMU参考在两组各4/4真实生效，但没有改善严格过关，部分场景出现新的真实关节超速。trial01球速约4.25m/s但正脚背冲量分数不足95%，trial02/03约2.0–2.33m/s且脚面不符，trial04原v32和CMU10_05都出现非脚物理接触提前终止。用户要求的专业类人暴力抽射仍未实现，默认控制器绝不升级。

重新对v52失败案例做精确时间戳归因：5.140001s终止源共1029行记录，其中1028个标准5ms物理步+1个1µs终止快照，根高度约0.786m和倾角约3.07°并不证明已经“整体倒地”，只能确认原ContactSensor的非脚部接触被认为不合法，阻止14秒恢复。真实人体骨骼数据虽然是120Hz，但源动作出球起点不是碰撞真值、两段来自同一人，离线方向余弦MSE也不是自然度。完整报告[真实CMU三策略12次物理审计](reports/2026-11-11-v053-real-CMU-12-physx-ablation-and-recovery.md)；[整批研究路线和完成门槛](docs/RESEARCH_BATCH_SERIES_2026-11-11.md)；配置configs/cmu_g1_physics_matrix_v053.json，工具run_cmu_physics_matrix_v053.py、audit_cmu_physics_matrix_v053.py、inspect_cmu_recovery_mechanism_v053.py及全测试；Tang原数据不进入GitHub。**651 passed +69 subtests passed**。下一连续批次做命名非脚碰撞体诊断及后续动作稳定的物理归因，后续才有资格考虑真正全身生成人体动作而不是1mrad参考修正。

## v0.1.52：真实CMU足球动作相位→可微29DoF G1协调生成→实际物理正负验证

从已核实 v0.1.51 的 120Hz CMU 踢球动捕和原29DoF G1 URDF几何对照出发，对单人两段有连续球标记运动的真实人体片段 **10_02/10_05** 分别固定96次优化，建立带梯度的6D骨段相位到G1全身29关节参考的有界离线投影，产生两条 **700×29** 的50Hz参考。离线相位目标误差分别下降26.63%/23.44%，但这只是人体/机器人并不完全解剖等价的几何余弦代理分数，且人体t0为*推测的球标记出球起点*而非真实脚球撞击，不能代表专业爆发力或自然度。

两组均通过原 G1 独立 style 物理门控实际施加5步全身关节参考，**顺序完成两次真实Isaac PhysX/29DoF脚球物理射门和一次 ZERO 禁用对照**。10_02在完整14s/2800个5ms物理步中实现合法右脚正脚背触球、4.6341m/s进球、最终8.715s站立恢复，原实测关节限速无超限；10_05虽然也以4.6391m/s正脚背进球，却在**5.14s触发physical_nonfoot并倒地**（仅1029步，最终恢复失败），所以严格的两个物理候选结果是 **1 PASS / 1 FAIL**，不因球已进门、两条离线相位代理误差改善而掩盖实际失败。关闭新参考的ZERO和原v32在全部共有物理记录字段逐值一致，旧档案额外的7个只读字段没有伪造。新参考对实测关节姿态最大改动约0.0012–0.0013rad，离职业人类真正大力抽射很远，默认v5/v32控制器**不升级**。

实际源码：tools/build_cmu_g1_candidate_v052.py（可微生成）、tools/audit_real_cmu_physx_v052.py（包括提前终止失败样本的真实物理审计）、tools/write_cmu_real_physics_report_v052.py（只读结果出报告）；完整[独立研究报告](reports/2026-11-11-v052-real-CMU-phase-controlled-G1-PhysX.md)、[复现指南](docs/CMU_REAL_PHASE_TO_G1_V052.md)、19份源SHA冻结协议configs/cmu_g1_phase_projection_v052.json、两个新增测试模块。Tang本地原CMU/PhysX/模型/NPZ受.gitignore保护，不上传原始文件。全项目**647 passed，69 subtests passed**。后续优先研究准确真实人体击球时刻和落脚/随挥的物理稳定性，再研究有监督的全身协调生成、样本外球位/被试和自然度盲评。

## v0.1.51：真正CMU 120Hz人体踢球与G1原29DoF的6D跨形态时序接口（不伪装成人类动力学）

直接复用v0.1.50已SHA审计的6段真实CMU动捕，选其中**仅2段球标记有连续高速运动证据**的CMU10_02/10_05，和v0.1.48已冻结、原PhysX真实脚球撞击的G1 v32/v48两条29DoF轨迹，建立**六个同定义但非解剖学等价**的3D段落余弦（双膝、双肘、双大腿与躯干）相对时序接口。人体无对应G1足趾，排除原人体8特征中的两个踝—足尖通道。每条各自用运动开始或真实碰撞前0.40s至0.02s的20个50Hz采样，合计**4对20×6的真实人体/G1比较**。G1 Torch原URDF FK六通道与独立NumPy/SciPy FK差异<1e-12，全部源文件SHA冻结。

与CMU10_02/10_05的人体动作比较，原v32身体相对自身第一帧的6D余弦变化MSE分别0.06536/0.08735，新v48为0.06538/0.08738；G1身体相对姿态变化幅度仅为该两段人类代理幅度约0.698/0.585倍，**这不是自然美感、爆发力或训练效果指标**：人体时间零点是推断球簇运动开始，机器人为实际PhysX足球力接触，身体骨架也不等价。新v48旧PPO几乎没有改善这种相对时序差异。此版不重新训练G1或伪称已完成跨形态重定向。

实际交付完整[人体–G1跨形态时序报告](reports/2026-11-10-v051-real-human-G1-coordination-baseline.md)、冻结配置configs/cmu_g1_cross_morph_v051.json、源码/测试、两张Tang物理数据图和可**一条命令**从CMU源审计到G1对照验证的运行器 tools/run_cmu_g1_research_batch_v051.py。复现指南[CMU足球到G1整个批次](docs/CMU_SOCCER_TO_G1_BATCH.md)，依赖pin requirements/cmu-mocap-v050.txt；原始CMU/G1大文件保持Tang本地。完成**636项主测试+69项子测试**，一键流程本身再验证11项测试全过。默认机器人和球控制策略不晋升，仍需真正高帧率专业正脚背接触/发力真值，才能将此对照升级为安全的G1人体式暴力抽射动作生成器。

## v0.1.50：真实CMU 120Hz人体足球动捕全面接入与球轨迹缺失审计

在Tang真实取得官方CMU两名被试的6段足球踢球数据（14个原始文件：6 C3D、6 AMC、2 ASF，共约20.7MB），全部原文件本地SHA256校验，C3D/AMC帧数一一对应，合计**3358帧原始120Hz三维人体动作**。新数据不同于旧3DSP广播视频AI估计的3D，已解析为19个人体全身标记/骨架点、骨盆归一化坐标、8维真实三维段落余弦、足球三WEP标记簇与缺失掩码、脚—球簇距离和原AMC根动作。来源 https://mocap.cs.cmu.edu/ ，依据官方条款用于研究，原始数据不上传GitHub；Tang上CMU证书链不全，下载脚本明确使用TLS非验证、仅以本地SHA追踪文件连续性，不伪称上游签名。

**数据资格审计：6/6原始人体全身姿态基本可用；仅2/6(10_02、10_05)观察到≥8帧连续高速足球标记运动，粗略球标记簇速度约14.05/13.76m/s；10_03只有两帧高速数据其后丢失，其余3段没有可靠出球轨迹。合格的精确真实脚球接触时刻/正脚背冲量/支撑反力/职业球员生物力学真值均为0/6。** 三WEP簇质心不是经过校准的球心，不补齐NaN或将首次运动判作精确击球时刻；只有2被试，按被试划分的5/1样本不能支持稳健泛化。已实际绘制六段球追踪质量图和两条人体髋膝踝/手臂时序图，保存在Tang media/v050_*，GitHub仅代码、协议、报告和测试。

详见[120Hz真实人体足球动捕资格、球轨迹和运动数据报告](reports/2026-11-10-v050-real-120hz-soccer-mocap.md)。完整项目**631项测试+69项子测试**通过。默认G1、PPO和旧球物理不改。下一批以这批更可靠的真实人体3D为基础构建**跨人形本体可比较的真实全身姿态时序先验**，但在接触真值缺失时不能冒充职业式暴力抽射已学成。

## v0.1.49：真实触球发力链审计（12个原PhysX回合，200Hz足端/全身速度分解）

在已通过的v0.1.48原G1 29DoF自由根部实测物理上，以真实第一次右脚球接触时刻为零，逐份SHA核验12个既有物理回合、提取每次接触前0.30秒至后0.35秒的131帧×5ms信号。对原物理记录给出的右脚速度严格还原为根部+右髋+右膝+右踝四个世界速度矢量分解，并和原29关节实际qdot/裁剪PD估计力矩、支撑负载、真实球线速度/旋转/0.43kg足球平动能对照。**这是运动学可观测性而不是人类肌肉能量传递或动作自然度评分。**

六组相同球位原v32触球时的足端朝门速度平均 **4.773 m/s**、新v48为**4.775 m/s**，配对差异平均绝对值只有 **0.007 m/s**。原六组髋部在球门方向贡献6/6为正、膝/踝6/6为负；新v48符号关系完全相同，简单右髋pitch—右膝—右踝pitch绝对角速度峰值不下降时序判据原/新均0/6（**该判据不代表人类正确动作**）。这表明v48新PPO虽有真实动作输出，并未改变已有全身足端速度构成，无法据此宣称具有职业式爆发性。

完整按原真实PhysX和原始试验SHA生成的表格、数据来源、人体和机器人解剖学不可等同的科研限制见[触球发力链观测与物理分解研究报告](reports/2026-11-10-v049-true-kinetic-chain-observability.md)。两张原物理绘图存Tang media/v049_*；源/NPZ大体积数据不上GitHub；原默认控制器和29DoF物理限制不更改。**625项主回归测试与69项子测试通过**。后续必须补足真实专业人类高速动作/脚球接触标注，不能仅用弱3DSP视频估计来定义发力顺序。

## v0.1.48：首次触球前全身29DoF PPO真实训练与原PhysX完整对照（已确认负结果）

从v0.1.47/50f8e50在独立分支完成真正的触球前29DoF全身PPO。原G1自由根部、PhysX球/脚真实接触、原PD电机和29关节速度/扭矩限、原v32学得的射后恢复小脑保持冻结。新160维输入包括原154维机器人身体、关节和支撑状态、球相对位置与球速；在1.02–1.54s的原右脚触球前添加最大0.018rad、每20ms变化≤0.003rad的29关节残差。目标含真实球出射方向/速度、原始5ms实际qdot风险、来自训练集3DSP视频估计的**弱**人体姿态协调参考。它不是职业球员高速正脚背真实动捕或爆发力真值。

固定seed4801、64环境96次PPO更新，Tang RTX4090实际训练316.25s，新模型SHA256 `16c6dccae019c92a7326c7cdf7e9cb9595e865832fe345712e309f4b29fdb628`，未从测试集挑选模型。先验证关闭新Actor时14s原PhysX真实字段/原动作完全等价，再在训练前冻结的6组全新精确球位/机器人初速上跑旧v32、新v48各一次共12次：**原v32严格通过2/6，新PPO仍2/6；0组失败被修复，0组原通过退化；新PPO在6组均实际产生6个非零全身控制步**。新球速比旧版本仅几毫米每秒差异，第05组存在实际关节超速、第06组非正脚背且非脚接触提前终止。严格正脚背、全球进门、原5ms真实qdot/恢复/≥3.5m/s门槛不改。

详见[真实触球前全身PPO及12次原物理对照报告](reports/2026-11-10-v048-learned-preimpact-ppo-physx.md)，相关真实曲线在Tang `media/v048_learned_preimpact_true_physx_comparison.png`。**620项回归测试、69项子测试通过；默认策略不升级，不能声称已生成专业人类自然暴力抽射。** 下一研究应先量化真正近端→远端发力时序、全身协调和脚球冲量，而不是用很小的残差再调奖励。

## v0.1.47：真实29DoF G1恢复策略PPO微调（72次更新、独立6组PhysX：严格通过无提升）

在v0.1.46/a0eb01e基础上，从冻结v32 29关节恢复PPO权重warm-start，不改原IsaacLab G1自由根部、PD真实速度/扭矩上限、球/脚碰撞与完整正脚背验收。保留原接触后恢复奖励，新增对实测5ms关节速度超过原限70%时的平方裕度惩罚，按训练前固定协议在Tang RTX4090上用64环境执行72次PPO更新（约202s），新Actor权重SHA已冻结；此奖励不是CBF安全认证，也不是职业球员人类爆发抽射动力学。

随后在**6个训练前冻结的新球位+G1初速度精确三元组**执行旧v32与新微调PPO各一次14秒真实PhysX，共12次，新旧全部关节限速合法、射门判据和恢复按原物理评估：**原v32严格任务4/6，新PPO仍4/6，0组修复、0组退化**。两者所有触球前动作与出球速度逐值一致；两失败组因球未合法入门、非目标脚背及球速不足而失败，接触后恢复PPO无法改变已经确定的接触球路。站立时间有时改善有时下降，不能宣传统一提升。原v32/默认v5没有升级。

完整代码/原始SHA锁定训练协议与14秒配对审计见[真实G1恢复PPO强化学习报告](reports/2026-11-09-v047-real-ppo-contact-recovery.md)。Tang保留原模型checkpoint、日志与12次原PhysX JSON；模型/第三方运动视频不上GitHub。下一步必须真正触及触球前全身运动生成与脚面/球路/力学耦合，而非继续只优化触球后的关节稳定。

## v0.1.46：真实触球后、落脚前的全身过渡控制（负结果与明确安全退化）

从v0.1.45/01e6441隔离实现真正作用于原G1 29DoF关节的接触阶段全身落脚过渡参考，在真实脚球接触后0.06–0.62s，基于原脚部接触胶囊最低位置/下降速度、左右脚实测支撑力、29关节实测qdot调整左右腿、腰部和双臂；原PPO、人体视频估计弱参考、G1自由根部、原PD电机速度/力矩限、足球真实碰撞/球门完全不改。单关节附加目标限制≤0.016rad、每20ms增量≤0.002rad，**这些参考界不提供真正的物理安全证明**。

按冻结15个原PhysX真实回合：零控制2800个5ms物理步与原v41共有28项物理字段及所有原控制动作严格一致；2个已知旧开发场景中，原超速trial01在19步真实新增全身控制后变为无超速且合法进球，原本成功trial03也保持成功。但**6个精确新球位/初速组合**原v41严格成功4/6，新v46严格成功**3/6**，新控制全部6个球位真实生效（18–19步），0组从失败变成功、**1组从原成功变失败**。新球位04接触后0.685s右膝真实qdot达30.049rad/s，超原20rad/s限值（原v41同一时刻仅5.549rad/s），同时右脚瞬时支撑约4359N。其他失败包括脚面不合法、球速过低/射偏；均按原判据保留。由此明确**不推广v46为默认控制器，也不能宣称职业式类人暴力抽射已学会**。

完整原始对照与冻结SHA见[触球后全身落脚过渡物理研究报告](reports/2026-11-09-v046-anticipatory-landing-physx.md)。14次配对/开发主动回合均按预注册运行且无事后调参；协议调度器键名错误在首次主动实验前修复并记录原字节与变更，原零动作结果不变。图media/v046_known_trial01_real_landing_qdot.png和media/v046_six_ball_cases_physical_qdot.png保留Tang本地，不上传第三方素材与大体积原始物理记录。

## v0.1.45：真实触球后29关节恢复调节器，15次PhysX实测负结果

从v0.1.44隔离冻结原G1自由根部29DoF、原球/碰撞、原v32 PPO和v41人体估计姿态参考，新建在**真实脚球首次接触、分离确认后0.1–1.25s**基于实测qdot/原始限值给关节位置目标施加有限反向修正的控制器，不调用v44误报安全的未来速度MLP。已按固定协议跑完**15次完整或提前终止的真实PhysX场景**（1个严格零动作、2个旧开发例、6组新球位各两种控制器），逐5ms检查原速度/正脚背/球速/站稳。

严格零模块动作/28个共有物理字段完全等价；在所有实际启用场景中**附加恢复调节器真实非零步数均为0**，因为接触后合格窗口里50Hz控制时刻实测qdot未超过0.68报警限，危险关节超速反而可能在5ms物理子步中晚发生。六组新参数旧v41严格通过**0/6**，新v45**0/6**；这是负结果，不能把未触发的调节器报告为安全优化。全项目**596项测试、69项子测试通过**。完整失败与全部球位记录见[触球后恢复调节器15次真实物理报告](reports/2026-11-09-v045-contact-recovery-governor.md)。默认控制器和原硬限制未改变。

## v0.1.44：计划动作条件全身接触后1.25秒qdot预测（真实训练，但安全判断负结果）

从v0.1.43的12条真实G1/PhysX事件数据提取按原球位/初速度分组的144个触球前物理观测（67维状态+事先选定的协调参考控制标志），以触球后1.25s内**29关节实际最大速度/原限值**为训练目标。4个物理条件组训练、1组验证、1组离线测试（原回合以前已检查，绝不是前瞻性新盲测）。RTX4090上按冻结配置训练单个三层MLP1000步，验证选第50步。首次训练后仅TorchVersion元数据反序列化失败，保存并修复格式，完全相同种子与超参复现的**全部网络权重逐元素差为0**，没有以测试效果重新选模型。

测试MAE：当前速度持续预测0.18295，训练均值**0.04149**，Ridge0.04513，MLP0.04256。模型略优于Ridge但不如训练均值；**测试中真实未来超速的12个同回合观测窗口，MLP全部预测为安全**。因此模型不具备危险状态识别可靠性，不能替换G1安全小脑。完整训练/元数据修正/模型对照与限制见[未来全身qdot预测研究](reports/2026-11-09-contact-future-model-study.md)，Tang本地图media/v044_delayed_joint_risk_forecaster_baselines.png。**未训练新的足球执行策略、未改变原物理，也没有取得职业人体抽射动力学真值。**

## v0.1.43：真实触球事件全身动力学数据与延迟超速风险审计（只读）

接续v0.1.42的六组新球位/初速度配对物理测试，在不新增训练或仿真、也不修改任何G1限速/足球碰撞/旧控制器的情况下，以**真实第一次脚球接触**为t=0，将12个实际14秒原PhysX回合对齐为各251步×5ms（接触前0.25s、后1.0s），提取29关节真实qdot、PD估计力矩、支撑脚负载、质心、右脚速度和球速，所有源结果SHA冻结。数据与图保存在Tang，源码/协议/实验报告入Git。

完整14秒中5个回合存在真实关节速度超限。只用接触前250ms关节速度是否超过限值90%预警，会有**2个漏报/1个误报**；v41第一组在触球前速度峰值仅占上限71%，触球后0.46s左踝却超速，原v32另一组在**触球后1.04s**才发生左膝超速。因此下一阶段全身控制需要显式研究至少1.25秒的触球后随挥/支撑/落脚风险，而不是只优化球离脚时的瞬间速度。风险研究详细记录见[触球事件安全动力学报告](reports/2026-11-09-contact-event-delayed-safety.md)。**这是回顾性真实物理机制证据，不是安全控制器、世界模型泛化或职业人类发力规律已完成。**

## v0.1.42：六组新球位/初始速度配对物理测试——显示正面收益与安全退化

在已推送v0.1.41之上，事前随机种子4201冻结六组三维初始状态（足球x/y偏移与G1初始y速度），每组以完全相同环境各运行原v32与v41全身人体姿态弱先验控制一次，总共12次14秒真实PhysX，没有在观察结果后重训或调参。严格原门槛包含合法正脚背、完整足球越门线、实际关节限速、14秒无异常终止、球速及射后站稳。

**原v32严格通过2/6，新v41严格通过3/6，2组原失败得到修复但1组原成功反而失效。** 六组全部有真实非零新增关节参考，12组都把球送进门，但有的因非法/混合脚面触球或真实关节超速被判FAIL。第一组新v41左踝在触球后0.46秒测得37.047rad/s，超过原37rad/s限制；绝不以进球掩盖。新旧球速均在约4–5.24m/s之间，距离类职业球员爆发式正脚背抽射仍有很大差距。

这是六个此前未命名的连续参数组合，但仍在同一任务/物理范围内，每组一次，样本极少，不能宣称统计显著泛化、真实机器人安全或自然度提高。完整原始数据SHA、所有正负条件和动力学归因见[新球位配对物理研究报告](reports/2026-11-09-v042-novel-ball-generalization.md)。**暂不将v41升级默认策略，继续冻结原v32/v5基线；下一研究要加入全身触球后动态稳定和专业击球生物力学数据。**

## v0.1.41：人体姿态启发的全身动作参考真正进入G1，三个已知开发球位物理任务通过

将 v0.1.40 人体协调参考与旧足球球位误差门控分离，在保留原v32自由根部29DoF G1小脑、真实PD/执行器限速、原球、触球与全程14s恢复硬门槛下，按预注册固定次序执行零动作中心、开启中心、两个历史射偏条件。**中心零动作与原PhysX逐值完全一致；3个开启的开发条件均实际产生5–6次全关节参考修正、完成完整合法正脚背进球、原限速零超限和回合末恢复**。冻结v32原本在这3个条件1/3进球，v41为3/3；新球速依次4.631332、4.743554、4.313940m/s。机器人端的11维人体代理协调MSE实际改善仅0.66%–0.83%，非常小，不能据此宣称职业人体自然美感或真实爆发式正脚背发力链条。

v41第三方人类数据仍是未校准的3DSP比赛视频估计3D，且三球位均为此前开发条件、**不是未知球位泛化**。首次B审计把只适用于中心“旧误差门控零值”的诊断错用于偏球位，完整保留初始审计后将不适用指标标为null，其他真实物理/限速/进球门槛不改。完整实验、冻结来源与局限见[全身协调参考独立物理验收](reports/2026-11-09-coordination-style-physx.md)，图存Tang media/v041_three_known_case_physical_shots.png。默认v5未升级。

## v0.1.40：人体跨部位协调参考→G1全身29DoF投影（已接入物理环境，门控未激活，拒绝提升）

实现原G1全身URDF的Torch可微正运动学（与独立NumPy FK误差约1e-15）、专业比赛3DSP训练片段相对协调变化的20帧×29关节优化、支撑/摆腿几何约束和700控制步全关节轨迹生成。离线代理目标误差下降50.9%，参考峰值0.00504rad、每步变化上限0.0012rad；但这不是人体真实发力或机器人自然度。冻结配置与source SHA保留，人体仅使用既定训练集片段00001。

真实PhysX零修正测试完整14秒，2800物理步与原v32机器人/球/接触/动作逐值一致。打开候选后虽然原射门仍合法、以4.631m/s进球且站稳，但**新增全身动作实际为0**：原小脑误差修正门控在中心球位始终为零，错误抑制了运动协调参考。这不是新协调方法成功，后续开发球位/未见测试全部停止，默认v5不变。详见[完整可微协调投影及物理门控报告](reports/2026-11-09-whole-body-coordination-projection.md)。下一实验需独立分开动作先验激活与误差补偿门控，再冻结新的PhysX对照。

## v0.1.39：职业比赛估计3D射门姿态→跨部位协调时序学习与G1正运动学接口（严格负对照）

实采并完整CRC/SHA核验3DSP/AutoSoccerPose职业足球射门数据：200段、4000帧、17个广播视频估计的3D关节点（并非高速光学动捕或已标注正脚背接触真值）。构建11组人体躯干/肩/腿/臂方向余弦，按足球比赛隔离划分108训练、47验证、45独立测试（12/4/4场）。固定1200次梯度更新的监督条件VAE时序训练，以前8帧预测未来12帧，真实独立测试MSE为0.170381，优于最简单静止预测0.235215但**不如训练集Ridge基线0.141046，四场独立测试比赛皆劣于Ridge**。模型只预测协调特征，不是可控制G1的生成策略；没有用测试集进行调参或后验挑选。

另从两组已保存的自由根部G1真实PhysX回合重建原29DoF URDF全身FK，提取同名11维机器人协调关系（40帧），URDF双脚link原点与原仿真测量最大差7.62×10⁻⁷m。这建立的是数据/机器人特征接口，不是人体关节/触球/功率的等价或真实人机迁移。已有12项针对性测试通过；原第三方视频、处理数据、训练权重、图像留Tang不上传。完整实测数据、失败对照与科研边界见[职业射门协调时序实验](reports/2026-11-08-professional-shot-coordination-pilot.md)。**P2高质量职业正脚背动力学数据和P3真实类人爆发抽射仍未完成**。

## v0.1.38：真实脚背扫动速度方向反馈（物理负结果，不升级控制）

在冻结v32恢复/射门控制基础上，新增基于实测右脚材料点线速度、足球速度及球门方向的雅可比投影反馈，限制附加关节修正≤0.022rad/每20ms变化≤0.004rad。最初实现遗漏了`enabled=false`禁用分支，导致错误零实验提前终止，已保留失败数据、补写禁用路径测试并修正。修正后的14秒零附加反馈与原v32全部2800个物理步、700控制步、接触和结果逐值一致（原偏门依然偏门）。

按预注册规则只运行一次固定候选旧射偏B：原始正脚背仍触球，但球速4.729m/s未进球；在2.645秒左膝真实角速度21.439rad/s超过原20rad/s上限，4.52秒触发非脚部接触终止，无法完成14秒或恢复站立。**实验候选被拒，后续球位和未见条件不再执行；无新训练、无控制器升级**。21项定向回归测试通过，完整失败证据见[扫腿速度方向反馈负结果报告](reports/2026-11-08-foot-sweep-direction-negative-result.md)。

## v0.1.37：真实脚球法向／切向冲量与抽射方向机制

沿用冻结 v0.1.36 学习策略和 v0.1.32 原控制，对两组历史射偏情境进行只读接触对照：新增3次14秒 PhysX 重放，加上 v0.1.33 原先已验证的 signed-force 数据，比较四个控制／球位组合的完整脚球冲量。所有新回放的真实物理执行和相应原轨迹一致，逐5ms动量平衡误差约10⁻⁶ N·s。已修复旧射偏A的合成接触方向由−1.57°变为+6.30°；未修复的旧射偏B则由+1.60°变为+0.82°，向错误方向变化约0.79°。**该版本是测量／失效机制研究，未修改抽射控制、训练模型或默认v5，未声称人体自然协调已学会。**

事前冻结的55个接触采样窗口、法向／摩擦分离、控制来源例外和旧物理流逐值等价验证均有记录；27项针对性CPU测试通过。研究记录见[真实脚球冲量对照](reports/2026-11-08-strike-contact-mechanics.md)。Tang图表：media/v037_measured_contact_directions.png。原始JSON、训练权重、媒体继续留Tang。

## v0.1.36：v0.1.35 触球前学习策略的严格物理复验（未通过总门槛）

已按预先冻结的研究协议，在 Tang 单独运行 v0.1.35 最终 29 关节模型的两组历史射偏场景。第一组从偏门变为合法正脚背进球，水平出球4.313m/s、末段站稳9.51s；第二组合法正脚背出球4.758m/s、站稳9.69s，但整个足球越过门线时横向偏左3.75cm，**完整任务仍未通过**。两组均完成14秒真实PhysX、没有实际关节速度超限，前600ms原动作逐值一致。新增21项针对性CPU检查通过。

按事先声明的双旧射偏必须都通过的硬门槛，**停止中心/强力/新留出条件评估，不加训、不调参、不替换默认v5**。这两个场景均出现于v35训练集，不属于真正未见泛化。新的v12配置仅用于隔离评估，训练权重与原始JSON只在Tang。完整记录见[触球前模型独立验收](reports/2026-11-07-prestrike-independent-acceptance.md)。

## v0.1.35：全身触球前反馈训练已保存，按用户要求暂停

新增独立29关节触球前策略，保留冻结v32射门与恢复。18项定向CPU测试通过；零修正2800物理样本、700控制样本与v32逐值一致。唯一一次600.027秒训练完成64次更新，最终权重已保存在Tang。**训练后射门验收尚未进行，效果待验证；工作已按用户要求暂停，默认v5不升级**。详见[保存进度与待验收项](reports/2026-11-07-prestrike-training-paused.md)。

## v0.1.34：横向摆脚反馈的固定候选未通过

基于实测脚位/脚速新增共享横向路径反馈，保留冻结v32恢复。零修正完整逐值复现；同一套参数的两个旧偏门仍未命中。第一例出球4.296→4.452m/s、站稳9.145s；第二例左支撑脚踝37.056rad/s超过原37rad/s，按协议停止剩余条件。**候选被拒，默认v5不升级，P0未完成**；15项定向CPU测试通过，无训练或新扰动消耗。见[实际结果与停止原因](reports/2026-11-07-early-path-feedback-negative-result.md)。另完成[控制基线接口与预算核查](docs/research/2026-11-07-controller-baseline-compatibility.md)及[三个新人体数据来源排除记录](docs/research/2026-11-07-new-fullbody-kick-sources.md)，未冒充P2/P4完成。

## v0.1.33：补齐球侧切向冲量，原控制不变

一例旧偏门的只读回放与v32全部2800物理样本逐值一致。显式重读原始signed contact数据后，55个5ms采样的球动量最大残差1.39×10⁻⁶N·s；向左切向冲量超过向右法向冲量，合成方向由+2.053°变为−1.521°。脚面朝向接近目标，但扫脚速度方向仍偏约19.5°。**旧偏门未修复，没有新训练或控制更改**；结果支持下一步联合处理身体/摆脚路径和触球曲面。原采集非负假设错误及严格拒绝保留，见[冲量机制与边界](reports/2026-11-07-ball-impulse-direction.md)。20项定向CPU检查通过，原资产与旧版本均保留。

## v0.1.32：学习触球后恢复，保留新扰动失败

共享29关节反馈在球离脚后介入，旧18条件全部完成14秒且未观察到原关节速度限制超限，首击/出球保持3.923–5.525m/s；完整进球任务仍16/18。此前晚期右膝超速条件通过，六个匹配对照有三例恢复更慢。第一项新扰动因触球面不符并偏门失败，按预定规则停止剩余两项：**19/21已执行，完整P0未通过，默认v5不升级**。

一次600秒PPO的最终checkpoint用于全部候选，不是人体协调生成器。MUVY职业球员多视角素材已取得，精确同步/标定/3D与正脚背资格仍未完成。Tang实际预览`media/learned_recovery_v0.1.32.mp4`包含原速与四分之一慢放；渲染回合独立通过，出球4.075m/s、末段站稳9.995s。方法、退化、负例、31项定向检查及来源见[本版研究记录](reports/2026-11-07-learned-post-shot-recovery.md)。权重与原始证据留Tang，冻结基线与旧分支保留。

## v0.1.31：支撑参考对齐的独立负结果

单例预定空间对齐试验保持正脚背首击和进球，第二次落地力峰3878→2532N，但随后侧移并在3.60秒触发非足接触终止，未恢复站稳。**完整任务未通过，候选不升级默认控制**。本轮停止空间参数试验，转向实测支撑切换及全身恢复；18项定向CPU检查通过，原19冻结资产不变。见[独立结果与证据](reports/2026-11-07-support-reference-negative-result.md)。

## v0.1.30：触球姿态退出、晚期落脚诊断与职业人体数据

实验控制消除了已测慢初速回合的严重落脚超速，中心4.631m/s和较强组合5.525m/s的完整物理轨迹保持不变。**矩阵尚未通过**：6个已测条件原射门任务均成功，但1个在后续第二次落脚仍超速，剩余12旧条件和3个新扰动未继续执行。v6是实验候选，默认v5与冻结基线保留。

新增原地面作用点/摩擦锚点只读诊断，以及SkillCorner职业比赛、KTH多视角人体数据流程。数据精度、真实正脚背标签仍有缺口，P2尚未完成，没有训练新的协调生成器。结果、负例、64项定向检查和下一步机制试验见[本版研究记录](reports/2026-11-07-state-conditioned-landing.md)。

## v0.1.29：抽射动力学测量与真实人体协调数据

保留原控制器，补齐脚底法向力／摩擦力、骨盆／胸廓／上臂运动与事件时序分析。三次完整14秒观测回放的8400个原物理样本与v28逐值一致，中心与较强组合出球仍为4.631／5.525m/s。测量发现慢初速场景落脚时膝关节38.05rad/s、超过原20rad/s限制，并伴随明显冲击；**原任务通过不代表动力学质量已通过，P0仍需修正**。

已取得5段CMU原始人体足球光学动作并生成2761帧协调表示，缺失／上游处理／新插值标记分开保存。球员水平、正脚背类型和真实击球时刻尚未确认，不能据此宣称P2或人体协调迁移已完成。没有训练或替换现有模型。

实测、复现和边界见[测量版本记录](reports/2026-11-07-research-measurement.md)、[动力学基线](reports/2026-11-07-explosive-strike-baseline.md)、[P0独立审计](reports/2026-11-07-p0-acceptance-audit.md)、[人体协调数据](docs/research/human-kick-coordination.md)。

## v0.1.28：提前步态反馈，修复组合条件踢空与失衡

新增基于实测身体横向位置/速度的早期步态适配器，以及原URDF双小腿路径预测避碰。原v27的15个球位/初速条件从 **14/15提高到15/15** 完成“连续跑动→合法正脚背进球→恢复站稳”；另留的3个新条件均触球并站稳，但完整进球仅 **1/3**，总计 **16/18**。剩余失败是出球方向偏出；不能宣称任意组合泛化。关闭学习修正后，修复组合再次踢空。

默认配置`configs/state_conditioned_strike_v5.json`，冻结v15/v27和原执行器/接触/验收条件保留。Tang实际预览`media/state_feedback_instep_v0.1.28.mp4`含原速与明确标注的慢放；录像出球4.23m/s、末段站稳6.10秒。方法、失败候选、独立测试和完整对照见[步态与摆腿路径研究报告](reports/2026-11-07-shank-path-adaptation.md)。

## v0.1.27：组合扰动下的可达摆脚与恢复

修复触球后骨盆/大腿自碰撞，以及不可达脚姿态请求使摆脚修正归零的问题。最终同一控制在15个离散球位/初速条件中完成14/15个“跑动→真实正脚背进球→恢复站稳”：九个已知条件9/9、修正中已观察的四个组合4/4、另行预留的两个确认组合1/2。旧版九个同条件为8/9；冻结v0.1.15保留。局部测试不代表任意球位或静止起跑泛化。

默认配置为`configs/state_conditioned_strike_v4.json`；旧配置保留。Tang实际预览`media/combined_instep_v0.1.27.mp4`包含原速及明确标记的慢放，录像回合水平出球4.313m/s、末段站稳10.01s。失败、消融、全部条件和复现见[组合抽射研究报告](reports/2026-11-06-combined-strike-adaptation.md)。

## v0.1.26：实测状态驱动的跑动射门适应

同一控制器在原始条件、球横向±5cm/远近±10cm、初速±0.20m/s的七个开发条件中，完整流程由冻结基线 **1/7提高到7/7**：连续跑动→真实正脚背触球进门→恢复站稳。较快初速实测触球时仍以2.32m/s前进，末段站稳9.25秒。模型根据实际球/身体状态修正落脚和触球相位，新增前1秒有界起步适配，原冻结基线保留。

两组未参与本轮参数辨识的球位＋速度组合通过 **1/2**，仍是局部标定结果，不能视为任意组合泛化或静止起跑能力。最终原速/慢放预览在Tang：`media/adaptive_instep_v0.1.26.mp4`。具体对照、消融、失败组合与复现见[状态驱动抽射报告](reports/2026-11-05-closed-loop-strike-learning.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_placement_strike.py \
  --config configs/state_conditioned_strike_v1.json \
  --velocity-delta=-0.20 --output reports/new_adaptive_fast.json \
  --video-dir media/new_adaptive_fast_frames
```

## v0.1.25：起跑—真实触球—恢复联合残差学习，候选未通过

新增共享训练／评估的全身残差与时序策略、成功物理回合的动作先验，以及独立初速扰动比较。零修正回放复现冻结成功动作；一次修复后的 533.6 秒联合训练仍未保住原始场景，原始及初速 ±0.20 m/s 三条件为 **0/3**，同条件冻结基线为 **1/3**。四组球位条件未继续运行，**不替换冻结基线，不宣称适应问题已解决**。

实测失败时刻、已修复的目标裁剪问题、训练成本与后续研究判断见[联合射门学习记录](reports/2026-11-05-joint-strike-learning.md)。原始动作先验、权重和物理日志按约定保留在 Tang 本地。

## v0.1.24：状态反馈抽射实验，未通过鲁棒性验收

保留 v0.1.23 冻结基线，新增一次初速扰动 PPO 微调、实测球位／身体状态驱动的落脚与时序反馈、非脚部碰撞诊断。原七条件完整通过仍为 **1/7**，补充 ±0.10 m/s 两条件候选与冻结控制均 **0/2**；部分场景回退，**不替换可靠基线，不宣称已解决球位／初速适应**。

代码和失败数据用于下一轮包含触球及恢复的联合训练。结果、消融、文件和复现见[状态反馈抽射负结果记录](reports/2026-11-05-adaptive-strike-negative-result.md)。本地权重和标定数据不随 GitHub 分发。

## v0.1.23：冻结加力正脚背的科研基线评估

研究主线转回全身抽射。冻结v0.1.15控制、模型和物理参数，完成7个预定初始条件的局部敏感性评估：仅原始条件完整通过。球位变化暴露偏门、触球部位和恢复问题，初始速度±0.20m/s均在触球前触发物理终止；这不是随机成功率估计。

新增接触冲量、足端参考点速度、估计关节功率/功、骨盆转动与接触相位时序分析，未训练或调参。入口 `tools/run_strike_baseline_matrix.py` 和 `tools/analyze_strike_baseline.py`，固定协议 `configs/strike_baseline_v1.json`。具体结果、证据边界与科学图见[冻结抽射基线报告](reports/2026-11-05-frozen-strike-baseline.md)。

## v0.1.22：停球后调整落脚并定向传球

已接通同一物理回合中的“动态接球 → 确认球停稳 → 按实测球位小步调整 → 脚内侧定向传球 → 恢复站稳”。统一控制通过地面0°/±12°和高空停球后0°的四个场景；根据身体重心和实际支撑调节两脚承重，落脚朝向与空中击球朝向分开。

入口 `tools/run_receive_pass.py`，配置 `configs/receive_pass_v1.json`，支持方向角或目标坐标转方向。预览 `media/receive_step_pass_v0.1.22.mp4`。当前是右脚短距离调整、左脚支撑，不是任意绕球走位或精确停到传球目标点。实测与复现见[停球后小步调整与传球记录](reports/2026-11-05-receive-step-pass.md)。

## v0.1.21：观测来球后抬脚与低余速停球

同一个双脚落地接球站姿，根据实测球高度、方向和速度预测接触，再完成脚弓卸力、安全收脚与恢复站稳。六个地面/高空来球组合均有通过的实际录像，末速低于0.02m/s、总角速度低于0.3rad/s；仍是近静止而非绝对零速度。

入口 `tools/run_reactive_receive.py`，共享配置 `configs/reactive_receive_v1.json`，预览 `media/reactive_receive_v0.1.21.mp4`。范围为地面0.9–1.5m/s、高空水平3.6–4.3m/s与±8°，使用仿真状态观测，未验证任意来球或视觉识别。实测、标定范围、失败与录像来源见[动态脚弓接球记录](reports/2026-11-05-reactive-receiving.md)。

## v0.1.20：脚弓停球、安全收脚与恢复站稳

地面和高空两个固定场景的完整物理录像均通过：脚弓主动卸力，收脚后不再碰球，并分别在末段连续站稳2.80s与2.76s。最终球速为0.034/0.039m/s，达到近静止门槛，仍有少量余速。

入口为`tools/run_balanced_receive.py`，配置为`configs/balanced_receive_ground_v1.json`与`configs/balanced_receive_aerial_v1.json`。预览：`media/balanced_inside_receive_v0.1.20.mp4`，包含正常速度和触球/收脚慢放。当前从准备姿态接固定来球，任意来球与走动接球未验证；实测、局限与复现见[脚弓停球恢复记录](reports/2026-11-05-balanced-inside-receiving.md)。

## v0.1.19：脚弓停地面球与高空球实验

两种来球都已实现真实脚内侧接触和触球前回撤卸力。地面球曾降到0.016m/s，但恢复站立时再次被推走；高空球在落地前由5.665降到1.432m/s，恢复阶段仍失稳。**两种完整停球均未通过**，这是固定准备姿态下的实验入口。

控制入口为`tools/run_inside_receive.py`，分别使用`configs/inside_receive_ground_experimental_v1.json`与`configs/inside_receive_aerial_experimental_v1.json`。预览：`media/inside_receive_experimental_v0.1.19.mp4`，含真实正常速度与慢放，明确保留失败过程。实测、失败诊断和复现见[脚弓接球实验记录](reports/2026-11-05-inside-foot-receiving.md)。

## v0.1.18：主动下撤实验

接球控制加入来球预测、脚部位置/速度反馈和支撑腿协同。固定场景中，脚在首次触球前已向下运动0.35m/s；但随后仍有小腿碰球，**完整卸力停球尚未通过**。实验入口为`tools/run_aerial_cushion.py`，参数为`configs/aerial_cushion_experimental_v1.json`；不要把这份实验配置视为可用停球模式。

完整实测、淘汰记录与复现见[主动卸力实验记录](reports/2026-11-05-active-cushioning.md)。旧接球原型、射门和挑球入口保持不变。

## v0.1.17：接高空球实验原型

已记录从1.05m高处来球、脚背实际触球并恢复站稳的完整回合：落地前球速由4.09降至0.53m/s，末段连续站稳3.335s。**完整停球仍未完成**：主动下撤偏晚，球最终以0.39m/s滚走；连续颠球尚未实现。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_aerial_control.py \
  --plan configs/aerial_receive_experimental_v1.json \
  --output reports/new_aerial_receive.json --video-dir media/new_aerial_receive_frames
```

这是准备姿态、固定来球的原型。预览：`media/aerial_receive_experimental_v0.1.17.mp4`；参考重建、未通过项和实测依据见[接高空球记录](reports/2026-11-05-aerial-receive-prototype.md)。旧射门和挑球版本保持不变。

## v0.1.16：短助跑挑球模式

已重新录制脚内侧短助跑挑球：球心最高42cm，球底离地31cm，持续腾空0.50s，进球后末段连续站稳3.92s。复用已有可用控制，原普通射门和加力档保留。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py \
  --config configs/chip_shot_v1.json \
  --output reports/new_chip.json --video-dir media/new_chip_frames
```

预览：`media/chip_short_run_v0.1.16.mp4`。**连续跑动中不停步挑球仍未稳定**；本模式为固定场景的短助跑小挑球。实测与本轮尝试见[挑球记录](reports/2026-11-04-chip-mode-preview.md)。

## v0.1.15：正脚背加力射门

新增正脚背加力档：同球位、同球门，水平出球速度由3.77提高到4.63m/s（+22.8%），球进门，14秒回合末段连续站稳8.19秒。沿用原关节限制，不给球添加外部冲量。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py \
  --plan configs/stride_instep_strong_v1.json \
  --output reports/new_strong.json --video-dir media/new_strong_frames
```

预览：`media/stride_instep_strong_v0.1.15.mp4`，含正常速度与触球慢放。当前为固定球位、运动初态的单一加力档；实测、淘汰方案及范围见[加力射门记录](reports/2026-11-04-stronger-instep-shot.md)。

## v0.1.14：正脚背与脚外侧移动射门

新增正脚背、脚外侧两种完整流程：移动接近→真实触球进球→随摆与继续迈步→恢复站稳。两个最终14秒物理回合均通过，保留原脚内侧入口。预览为 `media/stride_instep_v0.1.14.mp4` 和 `media/stride_outside_v0.1.14.mp4`，均附触球近景慢放。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py --style instep \
  --output reports/new_instep.json --video-dir media/new_instep_frames
# 脚外侧：--style outside，并使用新的输出路径
```

当前验证范围为运动初态、固定球位的髋主导动作；方向/力度/高度联合标定及任意助跑距离尚未验证。恢复控制、碰撞坐标修正和完整实测见[新增踢法记录](reports/2026-11-04-more-foot-surfaces.md)。

## v0.1.13：跑着踢球

已完成固定场景的「连续奔跑→脚内侧射门→继续迈步→恢复站稳」。触球前3次交替腾空，1.575秒触球时仍以2.299m/s向前跑动，中间没有停步；球进门，末段连续站稳9.575秒。全程无摔倒或非脚部碰球。

Tang实际预览：`media/stride_shot_v0.1.13.mp4`，含正常速度和触球／落脚0.25倍慢放。这是运动初态、固定球位的髋主导脚内侧踢法；静止起跑、更多跑动踢法及方向／力度／高度联合控制仍待完成。详见[跑动射门记录](reports/2026-11-04-stride-shot.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_shot.py \
  --output reports/new_stride_shot.json --video-dir media/new_stride_frames
```

## v0.1.12：奔跑后真实触球射门并恢复

已把同一自由G1的「奔跑→减速调整→脚内侧触球进球→恢复站稳」接到一个物理回合。固定场景实测6次交替腾空，6.22秒触球，末段连续站稳7.185秒；全程无摔倒、无非脚部触球。

Tang实际预览：`media/run_to_shot_v0.1.12.mp4`。这版仍从运动初态开始，尚未验证静止起跑、其他踢法或联合方向／力度／高度控制。详见[真实射门衔接记录](reports/2026-11-04-running-to-real-shot.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_to_shot.py \
  --output reports/new_run_to_shot.json --video-dir media/new_run_to_shot_frames
```

## v0.1.11：连续腾空跑步与减速衔接

同一自由29DoF G1已在实际物理回放中完成6次交替腾空，并减速、按实测支撑状态交接控制；6.94秒内没有摔倒或非脚部碰撞。新增速度、真实脚底支撑和接触时序反馈，保留原执行器限制。

**完整任务仍未完成：**末段尚未达到站稳标准，零速度起跑在1.44秒失稳，奔跑与真实射门尚未衔接。两组方向／力度／目标高度小试均未通过；更多踢法高球、近竖直脚背与不同球位／距离也没有新的通过结果。详见[本轮结果、失败原因与复现](reports/2026-11-04-running-feedback-and-transitions.md)。

Tang实际回放：`media/run_to_handoff_follow_v0.1.11.mp4`。原有助跑射门、脚外侧和挑球默认配置保持一致，实验控制不替换已验证动作。

## v0.1.10：同一 G1 的跑步训练小试

这一版开始训练原 29DoF G1 的跑步策略，并修正了参考动作起步时骨盆与左髋的胶囊碰撞。原有助跑射门、脚外侧和挑球预览保留。

两轮各400次训练已完成，但持续腾空跑步仍未通过：最终固定起点回放前进约2.21m，在1.46s时倾斜失稳。这个版本提供可复现的训练／诊断工具，不把训练回报当作完整流程完成。

距完整目标仍缺持续跑步与射门衔接、方向／力度联合控制、更多踢法的高球及任意球位／距离适应。近竖直脚背也没有由已有进球结果证明。详见[目标差距与本轮实验](reports/2026-11-04-running-training-pilot.md)。

## v0.1.9：快速挑球试跑

脚内侧挑球已跑通，三组触球／向上随摆参数对应首次飞行峰高约31／35／44cm（球心高度），加强组连续离地约0.52s。它们是快速实测示例，尚未做精确力度或三维定点标定。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/lofted_shooting.py --profile loft
# 另外两组：--profile low_arc / --profile medium_arc
```

详见[高球快速试跑记录](reports/2026-11-04-lofted-shooting.md)。沿用原自由机器人、真实球和执行器限制；没有给球施加额外冲量。


## v0.1.8：脚外侧新流程与原生力度定点

新增小腿主导脚外侧的完整助跑、触球、进球和恢复流程，独立扰动验收 **6/6通过**。三种已有动作在各自原生力度下，瞄准球门左／中／右三点（x=-0.3/0/+0.3m，y=-3.7m），联合验收 **18/18通过**，最大误差14.28cm。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style outside
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style instep --target-x 0.3
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style instep --target-x -0.3
```

**持续腾空跑步、轻／中／重与定点联合控制仍未通过验收。**当前版本保留真实失败数据，不提供未通过的三档参数表。外侧流程尚未标定目标／力度。详见[完整结果与限制](reports/2026-11-04-running-aim.md)及[冻结配置](configs/runup_shooting_v2.json)。旧版本记录及指纹请在对应版本分支复现。


## v0.1.7：连续短程助跑与两种挥腿方式

新增同一自由G1上的「短程助跑 → 触球射门 → 随摆 → 恢复站稳」。髋主导正脚背、髋主导脚内侧、膝主导正脚背在冻结的球位／初态扰动矩阵中 **18/18通过**，每回合完整验证10秒；158项CPU回归通过。详见 [实现与验收报告](reports/2026-11-04-runup-sequence.md)。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode knee --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_runup_shooting.py --mode hip --style inside
```

冻结配置：[runup_shooting_v1.json](configs/runup_shooting_v1.json)。这是约1米的快步助跑，尚未验证持续腾空跑步、任意距离助跑、脚外侧新流程或新动作的定点／力度标定。髋／膝主导指接触前足端运动贡献，不是能量份额。历史默认预览继续保留。


GitHub: [YOUessi/KineticKick](https://github.com/YOUessi/KineticKick). Each version uses a separate `version/vX.Y.Z` branch; see [version control and local asset notes](VERSION_CONTROL.md).

**Aim:** learn coordination principles from human soccer movements, then generate novel, natural, physically feasible humanoid movements. Task success is a constraint, not the main scientific objective.

## v0.1.6：以出球效果和全过程平衡评价随摆

用户明确：随摆应服务于有效触球、力度／轨迹控制以及整个动作的平衡，摆得更远本身不是目标。新诊断 `tools/kick_contact_metrics.py` 按连续受力段记录接触时间、法向冲量和观察到的球速，同时报告过程中根部与接触力的变化。连续受力超过80ms不会自动被当作二次碰球；零力采样后的受力恢复单独记录，但仍不能仅凭力信号证明几何分离。

旧正脚背演示中，脚前摆由约15cm增至25cm，记录到的受力时长、法向冲量和首次零力采样处的球速却完全相同；脚外侧实验还暴露了恢复恶化。详细数据和测量边界见 [目标校正记录](reports/2026-11-04-contact-and-balance.md)。这是诊断与目标修正，**没有新增已验证的踢球控制器**；原默认动作不变。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/kick_contact_metrics.py \
  reports/v013_posture_demo_instep.json reports/v015_follow_final_demo_instep.json \
  --output reports/contact_effect_comparison.json
```

v0.1.5的旧阈值与结果保留供历史重放，不再代表当前优化目标。目标仍是有效且可控的出球、合法触球和全过程平衡，不能通过更换指标宣称旧失败动作已成功。

## v0.1.5：随摆试验（稳定性未通过）

本轮已做出球离开后继续摆腿的动作，但**尚未稳定实现随摆、指定触球、进球和站稳的完整组合**，普通预览仍默认使用 v0.1.3。最后一轮正脚背为3/6同时通过；脚外侧的膝关节过渡会把急收腿延迟到后面，单环境录像还出现原生终止，因此未采用这些参数。

分支保留随摆测量、可选后续关节控制接口，以及明确未验收的试验配置。44个诊断回合和4轮独立矩阵的结果（含失败）见 [随摆试验记录](reports/2026-11-04-follow-through.md)。Tang上的单次成功动作对比：`media/kick_follow_through_v0.1.5_experiment.mp4`；该录像不代表稳定验收。

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --trials configs/follow_through_feasibility_trial.json --seed 945 \
  --output reports/follow_through_experiment_live.json
```

此命令显式运行实验；普通 `--style instep` / `--style outside` 继续使用旧稳定姿势。近竖直脚背触球、指定落点与力度控制也仍未解决。

本版扩展了控制器源码与证据记录，旧定点／力度标定会在启动仿真前因指纹不符而拒绝运行；没有把旧验收结果换上新指纹。复现旧定点实验请使用保留的 `version/v0.1.4` 或更早分支。

## v0.1.4：近竖直脚背可达性试验（射门未通过）

用户希望触球脚的脚尖朝下、脚背接近竖直。本轮确认摆腿时可以达到接近竖直的姿态，但 **30 个诊断回合中，尚无一例同时完成近竖直姿态、正脚背触球、进球和站稳**。已停止这组参数试验，该版本当时的普通预览保留 v0.1.3；本分支也继续保留 v0.1.3 默认配置。

本分支新增严格姿态／离地间隙审计和明确标记为未验收的可达性试验配置；没有发布新的稳定控制参数。诊断、复现命令及下一步问题见 [近竖直脚背试验记录](reports/2026-11-04-vertical-instep.md)。112 项代码测试通过不代表物理射门通过。

## v0.1.3：正脚背与脚外侧姿势修正

正脚背改为脚尖下压、脚背朝向球门；脚外侧减少身体侧转和右髋过度扭转。冻结参数后，在球位左右各偏移 2 cm、两组初始扰动的检查中，正脚背、脚外侧和保留的脚内侧均为 **6/6 完成接近、指定触球、进球和站稳，共 18/18，无摔倒**。姿势与触球由实际物理接触和关节数据复核。

普通预览继续使用 v0.1.3 稳定姿势：

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style instep
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style outside
```

每次只启动一个仿真器，关闭窗口后再切换踢法。Tang 上的同视角前后对比含触球慢放：`media/kick_posture_v0.1.3_comparison.mp4`。参数保存在 `configs/free_shooting_posture_v1.json`，验收命令与限制见 [姿势修正报告](reports/2026-11-04-kick-posture.md)。接触仍偏脚前部；本次是近距离静止球姿势检查，**新姿势的定点和力度尚未重新标定**。带 `--target-x` / `--power` 的旧实验请在 `version/v0.1.4` 或更早分支复现；本版源码指纹已改变。

## v0.1.2：方向与力度控制实验（精度未达标）

新增固定球门内左／中／右目标和轻／中／重力度接口，支持三种触球方式。**目前仍不能稳定定点射门**：54 次独立扰动回合中，27 次同时满足落点误差 ≤15 cm、球速误差、指定触球面及站稳要求。脚内侧 11/18、正脚背 4/18、脚外侧 12/18；正脚背的力度档位尤其不稳定。完整失败记录、实测球速及复现命令见 [本版本报告](reports/2026-11-04-targeted-shooting.md)。

以下历史命令适用于 `version/v0.1.4` 或更早分支：

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --style inside --target-x -0.3 --power high --output reports/targeted_demo.json
```

`--target-x` 可选 `-0.3 / 0 / 0.3` 米；`--power` 可选 `low / medium / high`。三种踢法的请求速度不同，数值见报告。这些是实验控制，不保证实际命中；窗口显示实际误差。连续任意方向、任意力度和高球目标尚未验证。不加这些参数时，当前分支运行上方 v0.1.3 稳定姿势。

## v0.1.1：自由根部三种触球射门（2026-11-04）

已实现并完成有扰动的物理验收：走近球 → 正脚背 / 脚内侧 / 脚外侧射门 → 恢复站稳。30 次冻结条件验证中成功 28 次：正脚背 10/10、脚内侧 10/10、脚外侧 8/10。两次失败均为脚外侧射偏，30 次均恢复站稳。 三种方式的完整录像保存在 Tang 的 `media/free_shooting_v0.1.1_verified.mp4`（24.6 秒）。适用范围、失败条件和物理证据见 [本版本验证记录](reports/2026-11-04-free-shooting.md)。

在 Tang 上逐个运行（每次只开一个仿真器）：

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style inside --config configs/free_shooting_v1.json
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style instep --config configs/free_shooting_v1.json
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py --style outside --config configs/free_shooting_v1.json
```

关闭当前窗口后再运行下一个。`configs/free_shooting_v1.json` 保存三种控制参数与验收条件；使用球和球门状态输入的预训练策略，加上触球阶段的关节控制。模型与素材仍保存在 Tang 本地，环境准备见 [VERSION_CONTROL.md](VERSION_CONTROL.md)。当前验证的是近距离、静止球的仿真射门；不代表任意起点导航或实机能力。

以下保留早期研究记录与原始研究目标。

## 初始研究资源


- Verified Tang: NVIDIA 4090 Laptop (16 GB), Isaac Sim 4.5, Isaac Lab local checkout after 2.1.0; limited free disk (~61 GB at audit).
- PAiD reference motion dataset: 10 standard + 3 stylized G1-retargeted .npz clips (50 FPS); retrieved via sparse checkout at `third_party/HumanoidSoccer/motions`.
- `tools/motion_audit.py`: schema checks, joint smoothness/integrity descriptors and optional motion-tracking diagnostics. **This is an instrument, not a trained natural-motion generator.**
- Preliminary `reports/reference_inventory.csv` produced by auditing references, not robot rollouts.
- BeyondMimic tracking code (official archived): https://github.com/HybridRobotics/whole_body_tracking
- PAiD soccer code (Isaac Lab v2.1.1 requirement): https://github.com/TeleHuman/HumanoidSoccer

## Run the available audit (CPU only)

```bash
cd /home/you/projects/kinetickick
/home/you/anaconda3/bin/python -m unittest discover -s tests -v
/home/you/anaconda3/bin/python tools/motion_audit.py \
  --input third_party/HumanoidSoccer/motions \
  --output reports/reference_inventory.csv
```

## Reproducibility guards

1. Do not present *retargeted reference motions* as executed robot policy rollouts.
2. Do not label low joint-jerk or low MPJPE as human likeness or aesthetic quality. Such proxies can prefer slow, weak, unproductive movement.
3. Raw `.npz` stores no body/joint names, nor ball/contact labels. Built-in Isaac Lab G1 has 37 joints/44 bodies whereas PAiD motion files have 29/30: NEVER use the built-in G1 semantic ordering for PAiD. Obtain joint/body names from the *PAiD G1 URDF runtime*, validate the array ordering and synchronize contact events before reporting foot-slide, proximal-to-distal sequence or kick-specific timing. Optional `--joint-map` requires `_num_joints` and `_verified_against_source=true`.
4. Do not change the existing Isaac Lab installation. BeyondMimic is for the installed 2.1.0 line, PAiD has a 2.1.1 requirement; isolate versions.
5. Honor upstream data and model licenses. PAiD is CC BY-NC 4.0; do not publish included motions in a new public code repository without license review.
6. For future model comparisons: freeze train/test splits by source clip/player, same robot, same task success envelope, identical physics and training budgets.

## Next gate: genuine baseline replication

- [ ] Obtain clean BeyondMimic checkout + G1 assets in a separate environment (keep under disk budget).
- [ ] Verify Isaac Lab simulator launches and the G1 robot scene can step without GPU OOM.
- [ ] Confirm semantic joint/body mapping **in runtime** and export it to a versioned JSON.
- [ ] Generate one genuine robot rollout from pretrained checkpoint or short training.
- [ ] Export root/joint/body traces and timed contact labels from simulator.
- [ ] Compare real rollout to reference, report timing errors and failure cases.
- [ ] Create matched A/B videos (same robot and camera) for a blinded perceptual study.

## Planned primary research endpoint

Primary: independent human-rated soccer-motion naturalness in **matched, randomized, blinded pairwise video comparisons**, stratified by task and initial ball state. Use more than one rater per clip, measure inter-rater agreement and cluster uncertainty by motion/observer. Report task success and stability as guardrails.

Secondary:
- joint/segment phase relationship, prox-to-distal peak timing, support-foot contact and foot sliding (require semantic/ground-contact logs)
- torso–pelvis–swing-leg coupling and follow-through continuity
- motion diversity at matched functional success
- generalization to held-out ball positions and soccer movement types.

**Hypothesis H1:** an explicit phase-conditioned, morphology-aware coordination prior improves perceptual naturalness relative to pose tracking and generic style reward **without materially reducing** valid contact or stability. It is *not established* until experiments confirm it.

A high smoothness score alone cannot verify H1.

## Study pre-registration skeleton

- Baselines: same reference data, G1 body and training budget: (1) pose-tracking policy (BeyondMimic or replicable equivalent), (2) action style/discriminator with no explicit coordination, (3) proposed phase-aware coordination.
- Ablations: remove relative phase, remove cross-body couplings, remove contact-aware phase transitions, remove morphology conditioning.
- Outcomes: blinded paired preference, successful legal kick, falls, foot sliding, COM dynamics, coordination metrics, runtime.
- No contribution claims before pretrained policy rollout, controlled comparison, and seeded repetitions.

## Citation and upstream

BeyondMimic (2025) https://arxiv.org/abs/2508.08241
PAiD (2026) https://arxiv.org/abs/2602.05310
Biomechanics-inspired soccer kicks (2024) https://arxiv.org/abs/2407.14612


## Historical Tang demo (2026-10-21): **ORIGINAL G1 FOOT ONLY**

The user rejected extra football shoes; the active live scene now uses **only the built-in G1 foot collider** and shows a right-foot medial-side **forward** pass, not lateral sweeping. The old shoe prototype is historical code, not used by the live scene.

Launch on Tang's GUI session (do not duplicate an already running Isaac Sim):

    cd /home/you/projects/kinetickick
    DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_bare_forward_pass.py

Current physics-verified headless comparison at the same ball position (0.40,0.20,0.113)m:
- hip yaw -1.30 rad -> ball after first contact vx=1.951, vy=0.452 m/s
- hip yaw -1.60 rad -> vx=1.908, vy=0.268 m/s
- hip yaw -1.80 rad -> vx=1.631, vy=0.108 m/s (3.8deg horizontal deviation from +X).
- At -1.80, approx local nearest face: **medial side** of original cube (not a directly measured PhysX surface point), filtered force ~80.63 N; successfully crossed 2.2m goal in GUI simulation.
- Root-fixed, task-specific **scripted** controller, NOT human-quality or RL-trained motion; large hip yaw may be unnatural/unsafe for actual G1.
- "Knee only" is currently an uncalibrated exploratory option: same fixed ball position did not yield effective foot contact, so it must not be compared as a successful pass to hip+knee.
- UI goal location is provided by USD stage, not onboard RGB perception.
- All visual demonstrations stay on Tang, no Mac copies.

Evidence: reports/test_bare_forward_inner_foot.log; reports/test_bare_forward_yaw_sweep.log; reports/test_bare_knee_only.log; reports/show_g1_bare_forward_pass.log; reports/2026-10-21-bare-inner-forward.md.


## Historical GUI (2026-10-21): three strike surfaces + calibrated low/medium/high force

Use tools/show_g1_power_kicks_v2.py on Tang with the same Isaac Lab environment and X display.
This supersedes the earlier single-style show_g1_bare_forward_pass.py view.

- Original G1 bare foot, no added contact shoe. Three contact PRESETS: FRONT / TOE-LIKE, INSIDE / MEDIAL, OUTSIDE / LATERAL. Different football start positions are deliberately matched to foot orientation.
- Joint swing completion times: LOW=0.43 s, MED=0.27 s, HIGH=0.16 s. Physics-only impact/speed, no direct ball velocity injection. Contact-point labels are coarse nearest-cuboid-face estimates.
- Extra independently calibrated KNEE ONLY (EXPERIMENTAL) motor-chain option. Fixed thigh, active knee extension; in the calibrated test the fastest prescribed movement was paradoxically slower than medium. Do not claim monotonically controlled force for this mode.
- Default AUTO CYCLE 3 KICKS; CYCLE 3 POWERS holds the kick kind and cycles the three power levels. CLOSE FOOT / WHOLE FIELD views and SLOW / NORMAL display pacing are available.
- Experimental results: reports/calibrate_power.json, reports/calibrate_knee_power.json and reproducible regression checks in tests/test_measured_kicks.py.
- Live physics telemetry and shot history: reports/show_g1_power_kicks_v2.log, reports/live_power_kicks.csv.
- Tang Isaac GUI was visually verified and physically measured as working. This is STILL anchored-base scripted motion, not free-standing humanoid soccer skill learning or onboard RGB camera goal recognition.
- No files are copied to Mac.


## 2026-10-21 研究阶段：人体自然步态 Stage 2（Tang）

当前重点已经从“普通 PPO 可以走路”转为“真实人体动作参考 + 统一29DoF G1本体 + 动力学模仿”。

### 已实测的事情
- 官方37DoF平地走路权重保留。V1/V2各64并行环境、120迭代已训练并保存，但在0.25/0.5/0.8m/s × 2成对扰动场景的联合膝、速度、躯干倾斜与承重脚滑动守门检查中均未通过。因此不能自动替换官方PPO。
- 已从hf-mirror下载5段LAFAN1 G1 29DoF人体衍生retarget动作，30fps、36列、累计34438帧约19.13分钟；以及完整29DoF URDF和38份STL。URDF 59处网格引用均存在，29活动关节与CSV对应。
- 从5段源动作提取各8秒参考片段（跑步参考筛选真正的较快部分），验收关节限制/四元数/运动连续性。生成数据bank（5×240×29关节，含速度、足部运动学伪支撑标签），注意伪支撑不等于测力真值。
- Tang实时REFERENCE viewer使用实际29DoF G1完整渲染mesh，不创建任何额外PhysX刚体或collider。WALK-1/2/3/RUN-1和慢放按钮已验证正常工作。这个 viewer 只是运动学动作捕捉重定向回放，不能宣称训练出自然行走。
- Tang tests: 28 passed。

### 最常用入口
当前Tang观看人体自然动作参考：
  /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_human_reference_walk.py

物理37DoF三策略人工对比：
  /home/you/conda_env/env_isaaclab/bin/python -u tools/show_g1_gait_policy_ab.py

上面的两个 Isaac Sim GUI 避免同时开启以节省4090 Laptop GPU 16GB显存。
完整研究记录：reports/2026-10-21-stage2-human-gait-reference.md
多速度物理评测：reports/gait_robustness_matrix_20261021.md
29DoF素材审查：reports/lafan1_g1_walk_motion_audit.json
29DoF参考bank元数据：reports/g1_29dof_reference_bank_manifest.json
素材存在data/目录，不上传GitHub、不复制Mac。

### 下一关卡
把29DoF URDF变成物理可控的Isaac Sim/Isaac Lab本体，检查质量惯量、关节轴、碰撞、脚底和PD配置；继而用BeyondMimic跟踪人类动作参考，训练能够维持自由根部平衡、脚底低滑移、自然躯干与臂腿协调的控制策略。当前系统没有找到标准URDF importer插件，本体物理转换还没有完成，不能把运动学回放作为物理成功。

## 2026-11-11 最新研究审计 v0.1.55

原 G1 / PhysX **15 次新真实物理回合**（中心3个、两球位×三冻结控制×两 seed 共12个）和原 200Hz 29DoF、COM/双足支撑、非脚净力、全安全门禁独立核查已完成：全部 holdout 控制严格 `0/4`，其中球位 A 进球 `0/6`、球位 B `6/6` 因非脚碰撞在约1.64s提前终止。**6对不同seed原物理轨迹逐字段完全一致，因此不可声称4个独立随机试验。** 原生PhysX contact pair/manifold/collider paths仍无真值；左腕-左髋和 pelvis-右髋都只能称相反净力候选。新增fail-closed 29DoF全身协调动作接口仍未在PhysX启用，也未训练出专业射门；全测试676 passed +69 subtests passed。详情见 `reports/2026-11-11-v055-physx-dynamics-and-pair-evidence-boundary.md`、`docs/REPRO_V055_PAIR_DYNAMICS.md`。

## 2026-11-11 研究进展 v0.1.56

在冻结原控制器/原G1自由根部/原PhysX后，完成 **12次独立初始随机种子探索物理回合+3次中心只读parity**。新探索v32/CMU10_02/CMU10_05各自严格任务**1/4**，通过的是同一工况，不表明人类运动先验改善。6对两seed初始状态与200Hz实际trace均不同；三组中心原共同物理字段逐值完全一致。原接触回调在G1/GPU上仍0个，左腕—左髋接触manifold**没有被证明**。另以独立盒子隔离实验找到本机GPU接触filter有626N真实力却无法获取原生report，而CPU toy同样接触可得到239个原生报告并包含actor、collider与原生接触点；这些不是G1真值。29DoF时序/双足支持/截尾审计与691项回归+69 subtests已完成；没有训练成功高爆发自然正脚背全身射门。研究全文：`reports/2026-11-11-v056-g1-physx-seeded-backend-evidence.md`。
