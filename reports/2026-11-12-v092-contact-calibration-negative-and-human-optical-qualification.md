# v0.1.92 — 真实碰撞取证仪表负例、六段人类光学几何资格与四条G1原始负例交叉审计

**日期：2026-11-12 Asia/Shanghai**。事实源 parent v91 `4ea0b5d740976428abeec3acb740e19d5a6e4bb0`。冻结物理协议提交 **`b11f53d059969eb415595d5e5d89b3135a7139e2`**；矩阵 SHA256 **`3d5772f66f93979be196175e28857138cba0572fd5dbbf29ef7eb10965f0ae38`**。原始JSON、C3D、weights、图留在Tang本地，未改任何旧分支。

## A. 物理正控新实验实际结果与安全停机

Tang在线，RTX4090无他人计算进程，磁盘约36–37GiB，独占项目全局GPU互斥。冻结的5个独立正/负控中实际启动 **1次 Isaac Sim CPU 物理尝试**，原G1射门新增 **0次**；其余4个试验因预注册运行证据失败门禁停止，不能记作已运行或射门失败。

- `v092_cpu_colliding_off` 子进程约4.23秒，OS返回码0，无看门狗超时；日志成功进入PhysX `SimulationContext.reset`，但 **未出现第一条200Hz step标记，也未写出真实物理JSON**，不能认证物理接触正例。
- 日志在 `V092_RESET_OK` 后直接出现 `V092_CLEANUP_ENTER`；`V092_APP_CLOSE_ENTER` 后进程退出却无 `V092_APP_CLOSED`。由于原程序没有在 `finally` 前明确打印异常栈，本轮不能在无堆栈情况下把原因精确指向view创建或Kit关闭。
- 日志明确显示 **`PxRigidDynamic::setLinearVelocity/setAngularVelocity: Body must be non-kinematic`**；这是本轮对刚体实验设置的有效负面诊断：不能向作为静态碰撞靶的kinematic box写刚体速度。
- 原计划 `report_mode=off` 的USD清单显示 ball/box `PhysxContactReportAPI=True`，因为资产构造配置包含 `activate_contact_sensors=True`；这使“off”根本不是无报告API对照。**因此本轮OFF/ON条件无效，即使后续获取轨迹也不能声称只读零扰动物理配对。** 没有修改冻结源码后偷偷重试；需要在下一个全新version改正并重冻。
- 根prim显示CollisionAPI=False不等于实际几何子prim没有collision，须在下一批遍历所有子形状并检查；不能臆断缺失。
- 原生actor/collider pair、contact manifold、左腕-左髋和右腕-右髋真实碰撞对全部仍未认证。

冻结批次事实审计本地：`reports/v092_frozen_contact_batch.audit.json`；单例日志SHA256 `539561859cb72f8757d1bfe7a17768b45035e291e17f2cf01c038b850150ca59`。

## B. 真实人类120Hz CMU六段有界资格 + 29DoF机器人交叉审计

新增 `tools/v092_motion_qualification.py`，直接读六份原始CMU C3D并计算SHA，按C3D标记单位把右脚趾与球体三个**表面标记的均值**作为**几何近似**，仅计算局部[-0.4,0]s躯干/骨盆/双臂/双腿相对标记运动速度峰值。真实触球时刻、真实左右GRF、真人职业运动员身份、足背球冲量**完全没有推断、人工补造或标为已证实**。

六份数据源来自被试组目录：subject10五份、subject11一份；根本不能支持可靠的subject-disjoint泛化验证。六段可提取一个几何最近接近候选，候选右脚趾到球三表面点均值距离分别约 **0.1715、0.16754、0.11599、1.69134、0.03635、0.17648m**。其中10_05候选甚至距离1.69m，**不能称实际踢球时刻**。整个片段内该右脚趾—球代理同时有效的帧比仅0.43–0.78；经首次探索性审计后，retry1采用更保守的>90%稳定观测资格规则，得到**满足该探索性资格规则0/6、职业正脚背真实冲量资格0/6**。这个后设资格规则不能冒充实验前预注册的标准，也不能将局部几何候选直接认定为真实接触。前一临时审计由于只看局部速度峰值，把6段误标为可学相位，已保留第一份结果作为负例；改为更严格几何/可见度规则，保留独立 retry1（SHA256 `f03c95e4c448755077c27671fa590f6b7a6e32c9af8c746`）。这不是改动已经冻结的物理协议，原始数据和历史试验从未修改。

重新加载v91真实GPU物理原始四场景E/F，逐份验证独立audit中的SHA、实测全29关节qdot和200Hz连续时间步、实际两足支撑力和非脚净力、第一脚球接触与恢复。**历史4个原29DoF G1射门严格通过0/4（本批新增0）**：E_v32 736帧+1非物理终止注释、E_cmu02 2800帧、F_v32/F_cmu02分别320帧+1非物理终止注释。F两段缺少触球后1.25秒完整窗口，不能填补后恢复。v91四个真实实验都是真实物理失败，不得和v92独立标定尝试混合为成功率分母。

独立两张真实来源图保留Tang：
`reports/v092_human_geometric_candidates.png` SHA `da452c61a573ebac6a54f32f9cbb587c3513b96163e7514f3a70f49c1f88f605`；
`reports/v092_g1_heldout_censoring.png` SHA `d4e5f6e80349ca843ac444775edb1f5e008255e9b83fda04c56942176332c386`。

## C/D. 安全可执行的研究界面与下一轮预注册修复

本批没有训练新的人类动作生成策略，也没有修改原G1控制器，因此模型真实试射为**0次，不能计算新模型成功率**。已有contact observer只预测机器人qdot，不能当作真人全身coordination mimic；六段几何近似不具备真实冲量/GRF标签。维持fail-closed：原生碰撞对校准和足够人体运动资格证据之前，禁止部署新的强关节动作。

**下一独立分支应先修正、冻结、再跑完整5条件：**（1）关闭资产自动接触报告，真实对照off/on PhysxContactReportAPI；（2）不对kinematic box写速度；（3）try/except在app.close之前保存完整原异常和阶段戳；（4）按碰撞形状真实USD路径匹配接触tensor过滤，并检查球静置/碰撞前后全部轨迹；（5）仅真实标定后在原G1进行只读碰撞对、OFF/ON逐值一致性验证，随后才能制定可驱动全身策略及新球位消融。

本轮冻结的失败不是“新足球策略失败”；所有关节、球、恢复的硬门禁继续不变。新增源码、测试、说明与此报告可在GitHub复现，原媒体和数据仅本地。
\n## E. 完整回归和版本校验\n\n完整项目CPU回归：**1028 passed, 2 skipped, 69 subtests passed, 0 failed**（reports/v092_full_suite.log，Tang本地），17项新增定向测试通过。失败的CPU物理probe原始日志及两份保留的人体审计没有覆盖。
