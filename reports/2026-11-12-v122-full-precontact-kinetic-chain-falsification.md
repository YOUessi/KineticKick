# KineticKick v0.1.122｜完整触球前29DoF动力学链：从v121左删失到可测反证

2026-11-12。独立CPU研发批次，分支 `version/v0.1.122` 从已验证干净 v121 `9362c0df6e624c76168e3704f4a80e4a7ef42907` 建立。冻结协议在首次本批分析前独立提交 `ae76315ed436543834e847523b6c8db547a3a935`，并已推送远端。脚本随后只读历史真实机器人物理轨迹；本报告不包含任何新物理射门。

## 一、执行安全与来源真实性

Tang 在线，RTX4090 已由另一个独立 v120 `python -m pytest -q` (PID 242644，nvidia-smi可见260MiB GPU内存)占用。本批未启动Isaac/PhysX/CUDA、未争抢GPU、未结束该进程。磁盘剩余31GiB，使用率97%，未执行清理和大文件复制。主仓库 v0.1.64 工作树有孤立未跟踪文件 `=100===`；v120 未完成的工作树也有未提交材料。均保持不动。新建的v122工作树起初干净，独立于v120。

| 来源（已冻结HEAD） | 历史200Hz原G1 PhysX回合 | 独立场景×控制器 | 严格PASS |
|---|---:|---:|---:|
| v116 `402b87f` | 10 | 9 | 4/9 |
| v117 `24857ba` | 11 | 9 | 6/9 |
| v118 `b2ff2fc` | 8 | 6 | 3/6 |
| 合计 | **29 / 81,200物理帧** | **24** | **13/24** |

按原始master逐项读取29份JSON并重算SHA256，所有原文件SHA和对应HEAD、29个实际关节、实际5ms格点、真实触球与全回合删失条件通过。5次只读OFF/ON属于重复，严格独立分母仍为24；16个同场景控制器配对。源master SHA256 与**此前v121本地已生成的**独立审计逐字一致：
- v116 `9080ae466ece92603c24b85e00f1424f211693580d2b0ec8fe9a6833ad3a88bc`
- v117 `76e5782e62df6cd47c571c100b720a56484024fa02398d6946548bc1838081da`
- v118 `800fb091cdf44b61be8bc1674e2930d4138c03be7d1118e2392c7cd7368716d5`

本批新增原G1实际物理执行**0回合**、新增已认证新策略电机驱动**0回合**。历史29回合是实测数据再审计，**不是本批新实验**。

## 二、关键科学反证：完整历史物理窗口推翻了0.4秒的伪时序

v121在触球前[-0.4,0)s观察到24/24的腰/右髋/右膝于第一帧已经活跃，完整功率链无法辨认。本批不调参数，按预注册同一原29关节与估算机器人电机功率定义，扩展到所有真实触球前200Hz样本[t=0.005s,t触球)，对比原[-0.4,0)窗口。连续3帧、0.5W下限、当组峰值20%的主阈值和预注册40%的敏感阈值均未随结果调整。

- **短窗口**：24/24至少有一个必要身体组左删失，无法由窗口起点判断全身功率顺序。
- **全触球前窗口**：24/24都能观察四个组按这一阈值定义的功率起点，0/24符合“腰→右髋→右膝→右踝”的观察排序。注意：这仅是否定该具体的*功率起点排序假设*；不代表反证所有可能的人体发力机制。
- **40%固定敏感性**：同样24/24可识别、0/24符合这一固定排序。不是为了优化结果而改变阈值。
- **高度依赖初始控制序列**：20%阈值下24个场景控制器条件全部得到同一序列：右踝0.020s、右髋0.025s、腰部与右膝0.045s。这说明所观察到的早期功率信号高度重复，不能把24组当成相互独立的专业动作发力样本。
- 左足>50N连续三帧代理起点24/24均在0.060s，晚于上述右髋起点0.025s，因而“左足支撑阈值已在髋功率起点前建立”在24/24均为**False under this specific threshold**。这并不能证明当时机器人失去地面支撑：右足、低于50N的左足力、起始姿态和接触解算都会影响解释，更不能将机器人足力代理当成人体GRF/CoP。

| 固定组 | 20%阈值起点（24个条件一致） | 40%阈值中位起点 | 完整触球前正估算电机功中发生于末400ms之前的中位比例 |
|---|---:|---:|---:|
| 腰 | 0.045s | 0.415s | 73.62% |
| 右髋 | 0.025s | 0.125s | 81.22% |
| 右膝 | 0.045s | 0.205s | 60.10% |
| 右踝 | 0.020s | 0.025s | 75.85% |

20%/40%阈值导致起点绝对时间相差很大，说明将“超阈起点”直接包装为真实运动组织启动时刻有明显语义风险。即使观察到四组顺序，也只能作为估算电机功率的描述指标，不能建立生物学因果传力，更不能认证出优秀真人的自然助跑与高速正脚背抽射。早于0.05秒的高功率尤其应警惕仿真初始化和预置控制器共同模式。

## 三、复现程序、负例和失败记录

本批交付：
- `tools/v122_full_precontact_chain.py`：重新核验每条原G1物理SHA、预触球完整窗口与末400ms、全29DoF估算电机功、左足支持、两个固定阈值、启动边界可辨性、配对、原始qdot审计以及独立实验分母。
- `tools/v122_plot_full_precontact.py`：生成实际机器人功率起点（20%/40%）、支撑代理时间线和触球前早期工作量比例图。PNG保留在Tang本地 `reports/v122_measured_full_precontact_onsets.png`、`reports/v122_robot_early_motor_work_fraction.png`，不上传Git。
- `tests/test_v122_full_precontact_chain.py`：13项独立CPU合成单测，专测短窗假左删失、完整起点仍左删失、次序反例、阈值稳健性、支撑阈值、29列、格点、SHA、OFF/ON排除和写一次结果。
- 原生接触报告下一批的安全正/负标定步骤（见下文）。

**不掩盖两个执行期间的失败**：首次证据序列化因numpy.bool_不能直接被默认JSON编码器处理，产生一份不完整且保持未覆盖的本地 `reports/v122_complete_precontact.audit.json`。随后仅改进序列化前验证（先完整编码再采用禁止覆盖的独占创建），新增写一次防覆盖测试，使用**新的** `reports/v122_complete_precontact_verified.audit.json` 成功复验。初次宽泛CPU回归50 passed/3 failed，仅因为clean工作树尚无Git忽略的v119原生接触原始证据引用；建立13条指向未修改v119原文件的本地只读符号链接之后，完全相同的专项CPU集合 **53 passed、0 failed**，保留首次失败记录。此前单独v122+v121回归25 passed。

无需GPU复现：
```bash
cd /home/you/projects/kinetickick-wt-v122
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/v122_full_precontact_chain.py --output reports/v122_reproduction_NEW_FILENAME.audit.json
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg python tools/v122_plot_full_precontact.py \
  --audit reports/v122_complete_precontact_verified.audit.json
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_v122_full_precontact_chain.py tests/test_v121_29dof_power_chain.py
```
输出采用open('x')写一次，重现时必须给新名字，严禁覆盖原审计。大JSON和媒体依.gitignore保持Tang本地；本批报告Git提交文字、协议、代码、测试。

**未执行整个项目全量回归**，因为另一进程v120持续占用GPU并在运行自己的全项目pytest，不能同时发起可能触及Isaac的测试。上述53 passed仅为所列CPU相关专项测试。

## 四、PhysX原生contact pair尚未认证

此前正碰撞Tensor显著有力、部分腕髋净力方向相反、原生header却仍为空。并不能推出真实左腕—左髋collider必然接触或不接触。本批没有启动接触探针。NVIDIA官方Omni Physics文档明确要求在真正rigid body/articulation prim上施加 `PhysxContactReportAPI`，正确订阅 `subscribe_contact_report_events`，确保阈值及第一步前订阅；native header给actor0/actor1/collider0/collider1、stage和contact_data_offset/count，每点包括position、normal、impulse。SDK数据只对当前物理步有效。

参考： https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html

参考： https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physx/docs/api/python.html

下次GPU空闲后必须先用隔离真实正碰撞、负分离与同一步读buffer取得双方path和冲量点正证书，再移入原G1目标腕/髋场景；OFF/ON重跑要重新比较全物理状态，不得把零header解释为否定。接触reporter只读、不改控制器/碰撞形状，且需要锁定Kit/PhysX具体API版本。

## 五、论文假设与下一阶段准入

本轮严格否定的是“原v32/CMU体系可以由预触球400ms的腰→髋→膝→踝假功率起点排序作为人体自然运动学习证据”。完整真实机器人轨迹虽然解除统计窗口左删失，结果反而明确不支持这一预注册功率起点顺序（0/24），且起点早期共同模式强烈。**不能把这与真正人体技能学成、技术创新通过或完全不可能存在其他协调策略混为一谈。**

下一步研究要重新采集自然助跑开始前的真实身体、地面力/CoP和足球冲量可溯源数据；为可能的29DoF全身学习策略提供七阶段高频条件化动作、足底支持、自碰撞/关节速度位置/力矩、触球后1.25s乃至14s恢复约束，先冻结未见球位/seed，实际取得硬门禁物理安全证据后才准入G1电机。保留每次关节超速、射偏、脚背冲量<95%、非脚碰撞的全部负例。

结论：本轮是**真实历史物理证据的CPU只读复验和明确理论反证**；没有新增PhysX试验、没有人体GRF真值、没有原生接触双方证明，也没有任何新的“自然高速抽射”能力认证。
