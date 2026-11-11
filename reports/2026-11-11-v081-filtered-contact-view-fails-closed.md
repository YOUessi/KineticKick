# KineticKick v0.1.81 — 原29DoF G1自碰撞filtered tensor观测负例、1次完整物理与严谨停止

2026-11-11。**本批不是新的人类足球学习成功**。虽然冻结设计为3种原G1控制条件 × 只读观测off/tensor = 6次真实物理，实际按照安全门禁仅启动了**2次**子进程。**1次原v32 off成功完成14s原G1 CUDA PhysX，严格射门PASS；另1次原v32 tensor 180s watchdog结束，未产生任何有效G1物理JSON，不能计为射门FAIL或零扰动物理PASS**。其余4个预注册条件因为观测器阻塞而依据冻结fail-closed规则未运行，不能宣称6次完成。

## 1. 分支、原数据及源锁定

- 独立干净原v80 d5e1bd944206dc7c10262dacaee39392fe94827d → version/v0.1.81。
- 协议冻结commit 377943cccda8bc2976cae32bdabfdd6a4443f2f3，冻结JSON SHA256 20fb3893988022e40ccacc5056a3b2a2c995750b1ed80aa838bb278b817ff38f；运行代码锁定commit 97d4933f9e0fa177a21e6cab4126dfd5fad218e2。
- Tang在线，RTX4090最初无CUDA占用，磁盘38GiB可用>20GiB；全球非阻塞GPU flock执行串行子进程。只保留本次新增的JSON/日志/图在独立忽略目录，没有改、删、重写任何历史CMU/C3D、模型、v54/v79/v80物理证据或其它dirty分支，未强推。
- 新工作树初始不含被gitignore排除的原运行资产，首次GPU批脚本在预检中因缺少 experiments/running_v013/stride_reference.npz 停止；该失败日志保留。随后从原v80工作树**仅复制13个指向Tang原文件的本地符号链接和空文件夹结构**，重新核SHA预检成功，无任何数据字节复制、改写或覆盖。第三方代码亦保持外部原路径。

## 2. 两次GPU尝试，严格隔离真实可认证物理和超时

| 实验 | 条件 | 结果 | 物理样本 | 严格安全与射门判定 |
|---|---|---|---:|---|
| v081 original_v32 off | 原原29DoF、seed1201、0.005s、控制器未改 | CUDA PhysX 完成14.000s | 2800 | PASS，基线重复 |
| v081 original_v32 tensor | 相同控制器并申请三条过滤RigidContactView（两个自碰撞方向+脚球正控制） | 180秒后watchdog终止本批拥有的子进程，无G1原始JSON | **0** | UNCERTIFIED，绝非真实射门失败 |

第一条物理的原始JSON SHA256 e2f5a89cf932e4af506285b1bad8c52e0995eba1df509f9e4a348641e5535d7f；第二条watchdog日志 SHA256 051138ec87b3b6ac6eaea2872aa78775186d3a4cb600c2b1c1064693c757753c。日志停止在初始policy网络打印之后，没有后续采样或完整step报告；**不能从日志推出究竟卡在view构造、物理初始化、还是后续执行**，更不能把“观测失败”换算为零接触。GPU子进程仅由本批自己启动并由自己watchdog终止；无其他计算进程被触及。后续因冻结UNSCORED停止规则取消cmu_10_02和cmu_10_05两组四次试验。

## 3. 真正的200Hz全身动力学证据

从第一条2800个真实5ms状态独立核查29关节实测qdot/上限、全身COM高度、双足支撑传感器净力、非脚接触、实际触球前0.4s及后1.25s和14s恢复。实际触球1.57s，两个事件窗口均被完整观察；29DoF最大实测qdot比例0.908129，非脚峰0N，COM高度范围0.639677–0.743509m，双足支持载荷>10N的帧比例0.65536。进球且正脚背冲量≥95%、球速proxy≥3.5m/s和原安全门禁通过，严格基线1/1。没有带观测器的第二条轨迹，**无法计算off/on物理一致性**，也没有左腕—左髋刚体对/网格collider/native manifold正面真值。

新增 tools/v081_independent_dynamics.py，对真实JSON及子进程日志逐SHA认证，并明确区分1次真实物理、1次未认证GPU尝试、4次**未执行**；等大反向非脚净力仅标注为待查接触候选，不当作直接碰撞双方。真实测量的1张全身动态图表留Tang本地media，原JSON30MB不上传GitHub。

## 4. 这次失败提供了什么科学信息

v80三组read-only get_contact_report 410个step均为0 native headers，仍有原cmu_10_05约25.890153N反向左腕—左髋命名净力。为避免用“没有native报告”推论“没有自碰撞”，本批尝试从同一PhysX张量后端创建左腕→左髋与左髋→左腕过滤视图，连同足背→球正控制，核验真实的contact data点、法向、标量冲量。v81代码拒绝缺字段/零冲量/非数等伪证据，也不把过滤view的命名路径当原生collider path。

本批观测器一旦附加便无法完成原物理记录，在给定180秒上限内**未验证无扰动物理**，这是明确负结果；不能声称“PhysX不支持自碰撞filtered view”，因为没有单独阶段打点/调用堆栈，更没有复现出错误归属。下一批应先把SDK/Kit启动、observer创建、reset及每次200Hz第一个step分层取证，区别view构造时机与内部PhysicsTensor调用；如果探针依然阻塞，改为USD原body碰撞schema只读盘点和原生报告正控制，原controller不动。不能盲目重复整组六次相同失败。

## 5. 人类学习方向与不可升级结论

v73真人120Hz几何GRU跨被试泛化比固定速度外推差约98%，v80六段原CMU中0段具同步GRF或合法正脚背冲量，职业身份无认证。当前只允许研究几何/相位代理，不能宣称职业自然爆发力“学会了”。未来真正论文方向应是支撑约束条件化的人体全身时序生成，耦合支撑脚→髋→躯干→摆腿→对侧摆臂→足背接触→随挥站稳，输出原29DoF动作及其速度参考并进行碰撞间隙、实际支持载荷、所有原实测qdot、球速/球门/正脚背≥95%、非脚20N及14s恢复独立GPU门禁。不可仅用1mrad弱参考或离线MSE取代整身物理。

## 6. 复现、代码、完整测试

- configs/v081_frozen_rigid_pair_tensor.json，docs/V081_FROZEN_TENSOR_PAIR_PROTOCOL.md。
- tools/v081_tensor_pair.py：双方向同体contact张量过滤、足球正控制、collider inventory、严格有限数/强度/证据检查。
- tools/v081_run_tensor_strike.py：仅在原v80控制器包裹下挂接观察器，不改策略、USD碰撞与物理门禁。
- tools/v081_tensor_batch.py：三冻结策略、六个计划条件、SHA/GPU互斥、watchdog/write-once/独立硬评测；预注册止损保护生效。
- tools/v081_independent_dynamics.py：对已真正完成的200Hz轨迹进行逐SHA、29DoF、支持、COM、失败截尾和绘图独立审计。
- tests/test_v081_tensor_pair.py新增11项否定控制与证据校验，均通过。
- 完整CPU测试初次因本新工作树没有旧gitignored测试fixture导致20个FileNotFoundError；仅创建Tang原JSON/日志符号链接，保留原文件及全部首次失败记录；后续一次运行仅因原v062日志没链接失败；最终链接旧原日志后全部回归绿色。不存在修改被测历史物理文件的“修测试”行为。最终pytest结果以独立日志为准。
- 原证据留在Tang本地，GitHub仅提交可复现代码、协议、测试、文字报告；本轮没有通过physx pair/mesh安全证明，也未实现训练出的自然爆发抽射。
