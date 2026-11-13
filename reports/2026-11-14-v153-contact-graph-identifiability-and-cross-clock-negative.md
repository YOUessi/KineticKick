# KineticKick v0.1.153：真实G1碰撞观测不可辨识性与跨物理步时间错位

**研究日期**：2026-11-14（Tang 本地）。独立 Git 分支 **version/v0.1.153**，来自干净的 v151 SHA **cda3b6e62a9d99c05842d201574c83585b1f2584**；v152 未提交的独立开发完全未改。**预冻结协议 SHA**：08774eae66c2bddf3796da47e6df31088d9a33b6。

## 结论和必须保留的负结果

原始 CMU10_05 的历史真实 PhysX 中，左腕、左髋命名刚体诊断的三维合净接触力严格等大反向，模长 **25.8901532895 N**。但这些合力本身**不足以证明或否认左腕与左髋之间的真实碰撞体对**。本轮在不更改原物理状态的条件下，构造四种符号接触图，它们在腕/髋两刚体的测量合力上完全等价；其中三种图**没有腕髋直接接触边**。独立关联矩阵重算 288/288 图，残差 **0 N**。

这一结论**只否证了从净力反演唯一碰撞图的逻辑**，并不证明其他具体碰撞发生，也不能把符号接触模型当成实际 G1 PhysX 模拟或几何可行解。真正的原生 collider paths、manifold 和同原生物理步对齐的数据仍为 **UNKNOWN**。

更关键的真实数据时间错位：原200Hz非脚净接触峰在 **5.135s**；最后有完整原29DoF状态的时间是 **5.140s**；命名腕髋传感器诊断记录在 **5.144999885s**。后者比峰值晚 **9.999885ms**、比最后完整状态晚 **4.999885ms**。本轮拒绝把这些事件解释成同一个真实物理步，拒绝为5.145s制造缺失的29DoF、COM、GRF或恢复状态。

## 1. 设备安全与原始来源

Tang 在线，文件系统剩余约 **30GiB**，使用率约97%。原v120测试 **PID 242644** 仍占用 RTX4090 约260MiB，并持续运行超过27小时；没有终止进程、没有竞争GPU、没有启动IsaacLab。独立v153工作树创建前验证v151干净且远端v153未占用；根目录原v0.1.64中的未跟踪文件不触碰；独立v152未跟踪的WIP也不触碰。

原v54协议SHA256：**add8693911cdd2489f3fb944adb77698238ddca50f14867a2a388b31eaeeeefa**。三份原始真实 PhysX JSON 按原 SHA 检查：

| 策略 | 历史JSON SHA256 | 原200Hz完整状态 | 完成14秒 |
|---|---|---:|---|
| frozen_v32 | f7a130707fdb761d7f7185d6805f4887d4ca929fec8965bfd35ae13d6666e01c | 2800 | 是 |
| CMU10_02 | e197cf6807709e077fa5cfd2b2582bf988dd1437f249bd61addf7835a7d2b255 | 2800 | 是 |
| CMU10_05 | d46a48f84dbaa7a367eac3e5dc399d4378007485c369c2b57c1c67775ccbfac1 | 1028 | 否，5.140s终止 |

**总计历史真实PhysX的6,628个5ms状态，且排除1条终止注释。新增真实PhysX为0次。** 历史v53原4球位×3控制器共12回合，三控制器各0/4严格通过的负结果保持不变，不能把中心球位记录当成成功新评测。

## 2. 真实腕髋证据与29DoF全身状态

| 观测 | 来源数值 | 有关限制 |
|---|---|---|
| 腕传感器 net force (x,y,z), N | (21.2208442688, 14.8218288422, 0.5377691388) | 非单独接触点冲量 |
| 髋传感器 net force (x,y,z), N | (-21.2208442688, -14.8218288422, -0.5377691388) | 不能唯一匹配腕刚体 |
| 双身体命名传感器净力模长 | 25.8901532895 N | 不是原生contact pair证书 |
| 两**刚体原点**距离 | 0.1138834365 m | 不等于碰撞形状最近距离 |
| 原5.135s非脚净力峰 | 25.8901538849 N | 与5.145s命名传感器时刻不一致 |
| 最后一条完整200Hz状态 | 5.140s | 随后的真实姿态/支撑缺失 |
| 命名净力事件 | 5.144999885s | 无同一时刻的完整29DoF读数 |

本批从原真实6,628帧再审计完整关节位置/关节速度/模拟关节扭矩、COM速度、左右足合净接触、非脚合净接触。CMU10_05最后0.4秒含81条5ms状态：最低合足净支撑71.168304N，最大非脚合净接触25.890154N，29关节中的最大瞬时关节速率9.197486rad/s，最大COM速度0.489837m/s。上述**不是孤立地面反作用力、CoP或真实电机力矩**，根倾角与高度单独也不能证实跌倒。

| 历史控制器 | 最高非脚净接触 | 原29关节最大qdot/限值比 | 模拟扭矩最大/限值比 |
|---|---:|---:|---:|
| frozen_v32 | 0N | 0.908129 | 1.000000 |
| CMU10_02 | 0N | 0.908129 | 1.000000 |
| CMU10_05 | 25.890154N | 0.908129 | 1.000000 |

仅适用于这三回合原中心球位；不推翻v53四球位失败、不代表不存在其他运动学安全问题。

## 3. 冻结288图的识别反例

设腕收到三维合力 **f**，髋收到 **−f**。对图每条接触边施加成对等大反向力，聚合后比较腕与髋的可观测合力：

| 图名 | 结构 | 腕/髋合力 | 直接腕–髋边 |
|---|---|---|---|
| direct | 腕–髋接触 | f / −f | 有 |
| two_to_third | 腕–第三体及髋–第三体两个接触 | f / −f | 无 |
| two_to_world | 腕、髋分别与外部环境接触 | f / −f | 无 |
| split_paths | 腕、髋各经第三体与环境分流 | f / −f | 无 |

上述第三体/环境图纯属**未认证几何可行性的符号图**，仅说明合净力无法唯一决定碰撞拓扑。冻结矩阵为 **24个方向×3个正幅度倍率×4图=288图**，每种图72例。独立于生成器的关联矩阵复核：288/288，72个观测分组内四类图的腕/髋合力逐分量一致，最大残差 **0N**。不宣称288次真实模拟，更不声称职业式人体发力链已训练成功。

## 4. 完整研发交付

- **tools/v153_historical_collision_clock_audit.py**：真实原PhysX SHA核对、原29DoF及5ms状态完整性审计、最后81帧全身和双脚接触数据、独立命名诊断钟差、拒绝补造数据。
- **tools/v153_contact_graph_identifiability.py**：冻结288图逐节点力平衡计算、三类非直接腕–髋反例、严格检查图输入与观测残差。
- **tools/v153_independent_graph_audit.py**：不调用图生成器，独立构建关联矩阵，重新检查288图，单独SHA与时序核对原真实历史源。
- **tools/v153_plot_collision_clock.py**：以真实v54样本绘制左右足与非脚净接触时间序列、5.135峰、5.140缺失边界与5.145命名诊断。
- **tests/test_v153_collision_graphs.py** 和 **tests/test_v153_historical_clock_provenance.py**：288组全矩阵验证及伪造拓扑、坏力、伪造认证、非法时间、改动源SHA、非有限值和终止注释对抗测试。

复现操作（需要原始Tang本地已认证的历史文件）：

~~~bash
cd /home/you/projects/kinetickick-wt-v153
PYTHONPATH=tools python3 tools/v153_historical_collision_clock_audit.py --output reports/v153_historical_collision_clock.json
PYTHONPATH=tools python3 tools/v153_contact_graph_identifiability.py --source-audit reports/v153_historical_collision_clock.json --output reports/v153_graph_matrix_288.json
PYTHONPATH=tools python3 tools/v153_independent_graph_audit.py --source-audit reports/v153_historical_collision_clock.json --matrix reports/v153_graph_matrix_288.json --output reports/v153_independent_matrix_audit.json
PYTHONPATH=tools python3 tools/v153_plot_collision_clock.py --output reports/v153_real_clock.png
PYTHONPATH=tools python3 -m pytest -q tests/test_v153_collision_graphs.py tests/test_v153_historical_clock_provenance.py
~~~

原数据、审计JSON、PNG及测试日志**只留Tang本地忽略路径，输出只写一次**；历史审计SHA256 **b49c0d917b6c989d7137f2a5d850cafdd5c065e2eb699fd485afd1e0c7e7b8b4**；288图SHA256 **dc42358690c5161885978521810fb8c5ac0ff66044002f4aa5ea1506f00abfb2**；独立审计SHA256 **bc15d69705048353a4e2627f7dde22730b1ccbb5f12c8d2937be2f4e38ca92c7**。

## 5. 测试结果与科研下一步

专项 **323 passed，0 failed**，覆盖全288图和35项其他正负校验。扩大到v124–v153的 **43个CPU测试模块**：**1478 passed、2 skipped、2项既有SLSQP运行警告、0 failed**；耗时52.52秒。

下一批先用独立真实PhysX正负控制标定是否在原场景已有接触报告资格、回调时间与collider owner。不得对原实验私自Apply新报告schema或改变控制器；再做同原控制器OFF/ON三策略共6个串行真实物理回合，比较全部原29DoF、COM、足底支持、球状态及原生接触。严格门禁包括正脚背≥95%冲量、球速≥3.5m/s、球门、所有原关节速度限值、非脚碰撞零容忍、自然助跑和14秒稳定恢复。必须在GPU互斥结束且数据合格后再训练闭环全身协调控制器，并独立使用未见球位与种子评测。

**真实新PhysX：0；真实新策略成功率不可计算；native左腕–左髋collider pair/manifold：UNKNOWN；专业人体正脚背与同步GRF真值：尚未认证。** 不把任何离线图、净力相等或旧1mrad参考包装成已完成的人体式高速正脚背抽射。
