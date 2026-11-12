# KineticKick v0.1.107 — 原G1 N/O双种子六回合PhysX、只读接触一致性与全身机械功取证（2026-11-12）

## 1. 安全、代码和预注册

Tang在线，Ubuntu RTX4090、GPU启动时没有其他计算进程，现有全局flock独占；剩余磁盘约35GiB，高于20GiB保护线。原始CMU/C3D、URDF、预训练权重、此前全部PhysX证据和v106带未提交状态的工作树均未覆盖/修改。新建干净worktree version/v0.1.107，父提交ae2117a4cf1719286d1dcd2f943628562677f21e；冻结源码和实验前的独立提交f6bad0695108cc86d1a92040a577d29e12dd2474，冻结矩阵SHA256 b45bfb1841fb273de04b7c11fe176af01c1d7ef8ae4182f2d8f108f52803cc32。六次物理执行先于后续分析开发、测试和报告。全部原始30MB级JSON与图片仅在Tang reports、依.gitignore保留。

预先确定 N(seed1703,球偏移[-.065,+.065],初速增量+.012)：原v32 OFF/ON、CMU10_02 OFF/ON；O(seed1709,球偏移[+.055,-.050],初速增量+.020)：原v32 OFF/ON。使用原G1自由根29DoF，未改变原射门控制器、策略权重和接触安全规则；物理200Hz，原Tensor只读观察器ON/OFF同参数配对。采用每子实验140秒watchdog、失败立即停止协议，没有GPU并发和测试集调参。

## 2. 新增六次真正可评分G1物理实验

| 冻结实验 | 真实物理 | 正脚背冲量份额 | 触球后5–110ms球速峰值代理 | 实测29关节qdot上限比 | 严格结果 |
|---|---:|---:|---:|---:|---|
| N/v32 OFF | 14s/2800步 | 100% | 5.175106m/s | 0.937052 | FAIL：goal |
| N/v32 ON | 14s/2800步 | 100% | 5.175106m/s | 0.937052 | FAIL：goal |
| N/CMU02 OFF | 14s/2800步 | 100% | 5.175733m/s | 0.937052 | FAIL：goal |
| N/CMU02 ON | 14s/2800步 | 100% | 5.175733m/s | 0.937052 | FAIL：goal |
| O/v32 OFF | 14s/2800步 | 100% | 4.520668m/s | 0.908675 | PASS |
| O/v32 ON | 14s/2800步 | 100% | 4.520668m/s | 0.908675 | PASS |

六次实际独立执行、全部认证原29DoF CUDA 200Hz物理，共16,800个真实时间步，六次都有完整触球前0.4秒、后1.25秒和14秒轨迹，无超时/补造/删失。全部非脚净接触峰值0N、源29DoF速度合法、恢复成功，N 4次唯一硬门禁失败是未进球。O两次满足原全部硬门禁，但4.52m/s只是最低3.5m/s门槛以上，不是已证明人类职业暴力射门。

**按真实执行次数2/6严格PASS；由于ON/OFF是相同物理轨迹的无扰动重复，按独立控制×种子场景只有1/3严格PASS，严禁按2/6夸大统计独立性。** 在N球位，CMU02相对v32的速度代理仅约+0.000627m/s，不能声称CMU带来实质改进，更不能称真人时序协调已学成。

## 3. 接触双方与只读无扰动认证

三组原v32-N、CMU02-N和v32-O完整OFF/ON配对，对全部共同200Hz原始轨迹、records、contacts、termination_reasons、kinematics、sequence_records、stride_shot_evidence、torque_evidence、ground_collision_paths进行递归逐数值零容差比较：**3/3 PASS，全部对照差异0**。源物理事件及全部硬门禁相同，不是只对root/qdot检查。

ON对左腕 yaw link→左髋 roll link及右腕 yaw link→右髋 roll link配置filtered tensor view。两侧各只报告单源单伙伴，sensor_count=1、filter_count=1、状态READING_BUT_NOT_NATIVE_MANIFOLD，本批六条目标filtered force均0。这不能证明没有直接腕髋碰撞：G1过滤通道缺少针对该真实pair的正控制，未获取原生actor0/actor1 collider shape路径、真实manifold与物理接触冲量。当前直接腕髋自碰撞**仍未认证**；历史反向净力也不能被替换成真值。

## 4. 全身动力学、力矩和时序，为什么不能当作真人学习结论

新增带原始文件SHA校验的独立200Hz审计：各29实测关节速度、真实COM与COM速度、双足接触法向力代理、非脚净力、根高度、触球前/后完整观测窗口、删失与原门禁。增加估算电机功率τ_est × qdot、不同身体组正负电机功、功率时间重心、触球时base/hip/knee/ankle对末端速度的分项贡献及实际两脚力时间积分。这些量可复核机器人运动机制，但**估算电机功不是人体全身生物力学功，足部力代理不是真人GRF/CoP**。

N/v32触球前0.4s双足法向力代理积分：左98.4293Ns、右43.0239Ns，右腿估算正电机功67.997J、左臂4.709J。N/CMU02分别约98.4531Ns、43.0239Ns、67.994J、4.716J。O/v32约左93.6451Ns、右40.4653Ns，右腿69.647J、左臂4.529J。这些是新真实机器人样本的测量，不能证明真人式发力、人体角动量传递或自然性改善。真实200Hz原轨迹图已在Tang本地，未传GitHub大图。

## 5. 实现与安全门禁

新批处理tools/v107_original_g1_batch.py、configs/v107_frozen_matrix.json和预冻结协议一次跑6次且write-once；独立的tools/v107_200hz_coupled_audit.py、tools/v107_mechanics_evidence.py与tools/v107_plot_failure.py完成重审、因果证据和可视化。tools/v107_motor_admission.py接收完整29个位置/速度/扭矩目标，并做全部29关节q/qdot/qdd/torque、积分一致性、已认证碰撞几何间距、支撑力、滑移、时效和SHA准入。无几何、支撑或native contact认证会fail-closed。**它尚未接入电机控制，也没有真人运动生成模型训练；合成单测的admitted不等于原G1安全。**

人体数据与研究假设见docs/V107_HUMAN_TO_G1_RESEARCH_GATE.md。未找到经过认证的职业式完整正脚背动捕+同步GRF/CoP+球冲量；当前CMU相位代理与robot-only训练不能充当这个监督。

## 6. 代码验证与负结果

新v107专项47 passed；全项目初测1,222 passed/1 failed/2 skipped/69 subtests，唯一失败是继承自v106的版本专属单测把当前分支名称硬编码为version/v0.1.106；在**新v107工作树**隔离该单测的git branch依赖，未修改v106原分支及任何冻结控制脚本。最终全量测试 **1,223 passed，2 skipped，69 subtests passed，0 failed**，日志均在Tang本地。旧证据只创建2018个只读符号链接用于回归，不复制不覆盖旧JSON。

## 7. 下一批必须解决的核心问题

先在隔离PhysX正/负控制中认证真正的actor/collider path和接触point/normal/impulse，区分API无回调、过滤路径层级、实际未发生碰撞；再回原G1做真正独立source-partner pair认证和OFF/ON完整无扰动复核。随后取得合格真人正脚背全身运动资格，构造支撑图/髋躯干驱动/反向摆臂/踢腿/随挥时相生成，和全29DoF碰撞、支撑、力矩及恢复安全接口闭环，在新的冻结盲测场景做脚部/摆臂/时序/安全消融。论文主结论必须是自然全身协调生成，而不是仅进球/低MSE。

**结论：v107真实完成6次原G1物理和3/3仪表无扰动，发现N以5.175m/s合法触球和站立但没进门，O旧控制器严格PASS；并未解决native collision manifold，未实现专业真人高速协调学习。**
