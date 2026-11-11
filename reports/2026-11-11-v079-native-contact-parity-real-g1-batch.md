# KineticKick v0.1.79｜原29DoF G1的真实PhysX碰撞观察器三条件配对实验

2026-11-11。**新GPU真实原G1物理执行5次、4条完整14秒、1条5.14秒真实非脚碰撞提前终止；源锁定下物理硬门禁4/5 PASS。两对完整的native off/on源物理字段逐值一致，但原生contact pair回调未返回任何header，因此左腕—左髋collider pair/manifold仍未被认证。**

## 冻结协议与设备

- Tang Ubuntu/RTX4090在线，运行前无其他CUDA PID，原GPU互斥锁全批次独占；磁盘可用约39GiB，大于20GiB硬限；原第三方源与CMU、原权重、30MB PhysX JSON仅经60个指向原主树的本地符号链接读取。
- 独立工作树 version/v0.1.79，基于v78冻结协议提交 96840df；本批配置SHA256为 8ca3de101c8df85bbd505ae8ee6f5a2ba75d78d5814619fac10d18c6088fe793；实验前protocol commit 3a5579a27c53d1e7070abbdfb7ddf0c99ef2973a，运行代码冻结提交 6b01882e8fc836e0eb7f597a44f9dcf9eaeefd89。
- 原v54三份GPU物理及v78成功14s输出已逐SHA检查；原v76控制器全部上游源码和运动/模型均按完整provenance确认。不覆盖、修改旧物理、模型和其他分支。原始输出/日志全部write-once、独立SHA。
- 预注册原v32 → CMU10_02 → CMU10_05，组内off→on，固定seed1201、真实29DoF自由根、200Hz、14s、CUDA PhysX；计划6条。**冻结规则首条未完成14s立即停止**，所以第6条CMU10_05 on未执行。

## 原始5次真实GPU物理结果

| 原控制/组别 | read-only原生report | 原始积分帧 | 实际观测终点 | 严格物理门禁 | 诊断结论 |
|---|---|---:|---:|---|---|
| v32 | off | 2800 | 14.000s | PASS | 脚背/球门/球速/全部29 qdot/恢复/非脚安全 |
| v32 | on | 2800 | 14.000s | PASS | off/on物理字段逐值完全相等；原生header 0 |
| CMU10_02 | off | 2800 | 14.000s | PASS | 脚背/球门/球速/全部29 qdot/恢复/非脚安全 |
| CMU10_02 | on | 2800 | 14.000s | PASS | off/on物理字段逐值完全相等；原生header 0 |
| CMU10_05 | off | 1028+1微秒终止证据行 | 5.140s物理帧；5.145s观察到非脚事件 | FAIL | physical_nonfoot，未达到14s |
| CMU10_05 | on | — | 未执行 | 不计成功率 | 按预注册停机，不能冒充失败或PASS |

本批物理硬门禁通过4/5（80%），是本批5条物理轨迹的描述统计，**不是未见球位/新种子/训练所得模型的泛化成功率**。四条完整组别原脚背冲量占比均1.0，原29 qdot峰值/限值约0.908129；v32峰值出球速度（接触后5至110ms proxy）为4.765318m/s，不能据此宣布职业式爆发强射。v32与CMU10_02轨迹前280/2800帧逐值相同，而后开始改变，说明CMU参考生效，但不能证明生成动作更自然、更快。此版wrapper认证依托源锁定v76原G1控制器+全上游SHA+CUDA/29DoF 200Hz真实记录，不能误等同v67采用的另一套V058标记证据。

## 原生contact pair认证：两对零扰动合同PASS，碰撞manifold仍无真值

两条完整off/on配对对全部冻结物理字段逐值完全相同（traces、contacts、records、sequence_records、29关节状态与限值、初始化、终止等），所以**2/2严格物理parity PASS**。原生观察者只注册既有PhysX callback，未Apply USD schema、未更改控制器/物理步长。两组native-on均显示左腕、左髋等目标原生prim具有PhysxContactReportAPI，但callback_calls=0、native_headers=0，没有actor/collider路径、接触点、法线或冲量数据。无回调是UNKNOWN，不是没有碰撞。

CMU10_05第5.145秒实际命名净力：left_hip_roll_link 为 [-21.220844, -14.821829, -0.537769] N，left_wrist_yaw_link 为完全相反向量；每端模长25.890153N。符合自碰撞假设，但净力不能单独证明一对直接接触的形状、碰撞manifold或碰撞primitive path。由于第6条native-on按停机规则未执行，不制造证明。官方NVIDIA接触回调文档显示在native回调中有actor0/actor1/collider0/collider1和contacts，但是否派发需实际验证；当前未获阳性结果。下一独立冻结实验重点是对这条失败回合进行read-only native get_contact_report/poll或更底层contact stream取证，并复核订阅、threshold、时间窗和物理parity。

## 真实200Hz全身时序与失败图

新独立工具读取5条原GPU JSON及log SHA并验29关节、实际qdot、真实COM、双足传感净力、非脚事件、髋-膝-躯干-对侧摆臂带符号时间滞后，明确最后1微秒终止证据不是200Hz积分帧。5条均有触球前0.4s到触球后1.25s的实测窗口。四条完整14s；CMU10_05触球后虽有窗口，却在5.14s中止，不能补造射后14秒的站立与恢复。净足力非测力板GRF或CoP，时序相关非能量因果证明。审计结果：Tang本地 reports/v079_independent_200hz_dynamics.audit.json；5张全时序COM/双足/非脚接触/qdot实测图保留本地media/v079_original_g1_native_pair_*_dynamics.png；每个原始JSON约30MB只在Tang，Git不提交。

## 下一论文创新假设与可执行接口

假设H1：带有真实运动员认证身份、跨被试完整120Hz以上助跑—支撑脚落地—髋躯干储能/转体—摆腿/对侧摆臂—脚背触球—随挥—双足落地的时间序列及同步GRF、球脚冲量，可学习29DoF全身协同的时序隐变量。它的主科学指标应是动作自然性、全身协同性、跨运动员时序泛化，而不是离线余弦MSE、脚部微残差或一次进球；需要运动员/动作风格隔离的训练验证测试和盲测，另外报告球速、脚背95%、非脚安全、关节qdot、14s恢复全部硬门禁。现有v73真实CMU的跨被试GRU相对等速外推表现更差、6段C3D缺同步优秀球员GRF和正脚背冲量，尚不能升级为真人职业动作强力先验。

假设H2：实际控制接口应采用20ms周期的29DoF相位条件全身动作增量，联合可验证的原G1关节限位/qdot、双足支撑多边形、重心及**同帧真实碰撞几何间隙**进行投影与拒绝，随后在原Isaac PhysX做真实触球与恢复反馈。仓库已有 tools/stance_safe_temporal_policy_v067.py 在没有真实碰撞几何witness时fail-closed，它只是待训练策略的安全入口，不是已认证的控制屏障。下一独立方法批次应在新运动员/新场景/新seed预冻结后比较真正全身时序模型 vs 无对侧臂 vs 打乱相位 vs 仅脚部增量 vs 原v32，不能在已经见过的失败样本上挑超参然后把它称为holdout。

## 测试、可复现性与保留失败

新增可复现GPU执行/原生pair检查 tools/v079_native_pair_batch.py，独立200Hz全身/截尾图工具 tools/v079_independent_200hz_audit.py，两组13个专项测试。系统Python首次全套因rsl_rl不在系统解释器site-packages而collection报15项依赖错误，保留Tang日志；改用原项目env_isaaclab无需修改依赖后全套**837 passed，67 skipped，51 subtests passed，0 failed**。首次绘图因新独立worktree没有media目录而失败，保留首次日志，创建本版media后retry2成功。没有覆盖任何原始物理文件或旧日志。本批最重要负例是原生header仍为零与CMU10_05非脚终止；请勿将4/5合法诊断回合包装成已经成功学习真实人类协调强射的论文结论。

