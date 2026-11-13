# KineticKick v0.1.151 — 原G1 PhysX原生接触可观测性、六回合OFF/ON门禁及历史严格负例复审

**研究日期：2026-11-13（Asia/Singapore）**
**前置已验证事实源：** `version/v0.1.149`，`9d9575662db2b6434f30b35c93a457a889425f58`。
**本批预冻结协议SHA：** `2d835f72658c712dbc045a30cd97e13510db2234`。
**版本隔离：** v150已有协议commit和未提交的独立研发文件。本次没有使用、修改、提交或清理v150；v151从干净的v149单独分叉，不覆盖原v64脏主工作树或其他分支。

## 研究判断和实验边界

本轮优先解决 **G1左腕—左髋原生PhysX contact pair为何长期无法直接取证** 的前提条件，而不是把合成29DoF轨迹当成真实合法抽射。原v54传感器在5.145s输出两刚体等大反向净接触力（25.890153 N），只能作为怀疑自碰撞的间接证据。**没有目标碰撞体的原生报告API、真实时钟校准与同物理步的双方manifold，就不得宣称自碰撞证实或排除。**

特别注意：`tools/soccer_world.py` 含 `PhysxContactReportAPI.Apply`，用于另一个原型右足鞋面扩展；**这不证明原始G1左腕或左髋具有报告资格**。v149原控制器sidecar运行器本身不Apply此API，而是检查现有schema。因此真正的判断取决于**运行时USD实例**，离线文本检索的状态仍是UNKNOWN。

## 安全检查

Tang在线；原根目录 `/home/you/projects/kinetickick` 当前是带有既有未跟踪文件的 `version/v0.1.64`，未修改。磁盘可用约29–30 GiB，满足冻结20 GiB下限，但总用量已约97%。RTX4090仍由 **PID 242644** 的原v120 `python -m pytest -q` 占用约260 MiB，运行超过一天；不得kill、抢占或同时启动IsaacLab。原全局mutex仍存在。

**安全结果：新真实PhysX 0回合、新射门0次、新物理成功率N/A。** v151仅执行CPU物理来源复审、冻结批次生成、源资格检查及对抗测试。GPU预检返回 `V151_FOREIGN_GPU_CONTEXT_NO_PHYSICS`，没有自动越过门禁。

## 本批确实实现的代码

| 文件 | 作用和安全边界 |
|---|---|
| `tools/v151_native_campaign.py` | 对原v54策略配置和历史协议逐文件SHA核验，生成3原控制器×OFF/ON固定6回合命令、write-once独立日志/sidecar/原始结果；支持显式GPU执行，但默认只规划，实际启动还要每回合检查磁盘/GPU及旧v149的非阻塞互斥锁 |
| `tools/v151_off_on_exact_audit.py` | 对两份原29DoF 200Hz sidecar逐样本逐分量严格一致性比较，包括29q/29qdot/29模拟扭矩、浮动基位置四元数与6D速度、COM、双足/非足净力、足球位姿和速度；保持首个分歧步、每变量最大偏差和所有失配统计 |
| `tools/v151_native_observability_ledger.py` | 对现有USD reporting API是否可用、回调到达相位、真实碰撞体归属与三维冲量作fail-closed判定。缺失回调/缺失Report API/错位时钟均不等于“未碰撞” |
| `tools/v151_independent_historical_reaudit.py` | 独立重新SHA认证旧v53 12场与v54 3场真实PhysX证据，重新计算严格门禁布尔合取；不调用GPU、不改原数据 |
| `tests/test_v151_*.py` | 六条件清单、SHA防漂移、原数据完整性、独立parity、48个正负回调CPU矩阵、坏冲量/假时钟/伪来源/缺失200Hz帧/调乱qdot/错误API资格等对抗测试 |

所有真实原CMU/C3D、模型权重、旧PhysX 30MB JSON、媒体、实验NPZ保留在Tang。生成的新CPU计划、审计JSON和日志位于v151 worktree的 `reports/`，属于gitignored的原始研究证据；GitHub只提交源代码、配置、测试和Markdown报告。

## 已执行的历史真实物理负例复审

v54 三回合原SHA锁定的完整200Hz状态：`frozen_v32=2800`、`CMU10_02=2800`、`CMU10_05=1028`，共 **6628个真实PhysX状态**。失败回合还有1条终止注释，不等同于额外5ms真实状态。真正新运行次数为0。

v53 四球位×三原控制器共12回合的严格门禁逐项重算：

| 冻结原控制器 | 严格成功 | 不合格脚背 | 球速低于3.5m/s | 进球几何失败 | 全29DoF速度违规 | 14s未完成 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v32 | 0/4 | 4/4 | 3/4 | 2/4 | 0/4 | 1/4 |
| CMU10_02 | 0/4 | 4/4 | 3/4 | 2/4 | 1/4 | 0/4 |
| CMU10_05 | 0/4 | 4/4 | 3/4 | 2/4 | 0/4 | 1/4 |

以上失败门禁相互重叠，不可累加当成不同失败样本数量。没有用进球、完成14秒、或CMU小于约1mrad的局部修正覆盖脚背合法性、出球速度、全部29DoF原关节速度、碰撞和恢复约束。历史CMU两段来自同一位被试，没有验证真实职业足球人体接触/GRF，不构成人体爆发发力真值。

## 可复现操作和待解物理依赖

在v151分支的CPU环境：

```bash
PYTHONPATH=tools python3 tools/v151_native_campaign.py --output reports/v151_cpu_manifest.json
PYTHONPATH=tools python3 tools/v151_native_observability_ledger.py --output reports/v151_cpu_schema_inventory.json
PYTHONPATH=tools python3 tools/v151_independent_historical_reaudit.py --output reports/v151_cpu_original_physics_independent_audit.json
```

命令第二次执行应由于write-once文件已存在而拒绝；绝不覆盖证据。实际物理执行需RTX4090空闲、磁盘>=20 GiB、原v149非阻塞mutex保护以及所有输出路径尚不存在时，才可**显式**使用 `tools/v151_native_campaign.py --execute-physics`。它不会自动认证物理成功；还必须独立审计OFF/ON每个完整200Hz物理状态。首先需在安全、隔离的原生正/负控制中确定已存在的report API是否覆盖真正目标碰撞体，不得在原G1场景上擅自添加schema以强迫得到报告。

## 独立失败归因与后续科研路线

缺少回调的解释不止一种：现有G1 USD目标刚体未启用contact reporting、回调订阅资格不足、接触事件到达时间不属于`scene.update`包围区间、碰撞过滤阻止了真实接触、或真实场景确无该碰撞。必须在真实PhysX中校准正例和负例，获得真实collider path、manifold与时间归属；仅凭等大反向净力不能区分。

接下来依次完成：**GPU安全释放→原生正负校准→原控制器六回合同条件OFF/ON逐帧认证→足底独立GRF/CoP和浮动基力矩闭环→具备资格的高速人体正脚背动捕监督→真实全29DoF支持-摆腿-腰躯-双臂协同训练→未见球位多种子严格物理射门及14秒恢复**。

本批**没有实现新型职业式高速正脚背抽射，也未取得左腕—左髋接触的原生manifold真值**。这些仍为研究前置门禁，而不是可被CPU合成回调替代的结果。

## 测试与最终提交

最终专项测试 **110 passed**；扩展42个左右CPU测试文件的回归为 **1196 passed、2 skipped、6条既有SciPy SLSQP裁剪警告、0 failed、0 errors**（48.63秒）。完整最终日志为Tang本地 `reports/v151_expanded_cpu_final.log`。两次初始测试错误均保留并归因，不会以最终通过覆盖历史负结果。GitHub final SHA及远端HEAD在提交后核验。

### 本批可核验数字及受保护证据SHA

- 本轮冻结协议 commit：`2d835f72658c712dbc045a30cd97e13510db2234`；原v149父SHA：`9d9575662db2b6434f30b35c93a457a889425f58`。
- 新原G1物理实验：**0**；真实PhysX正负控制：**0**；原生左腕—左髋manifold：**UNKNOWN**；未来原物理OFF/ON六回合全部**未执行**。绝不把48组CPU模拟回调、固定六行dry-run清单当作真实试验。
- 实际独立重读/重新SHA认证**历史**v53 **12次**真实原G1回合、v54 **3次**回合 **6,628个200Hz状态**。历史三策略在4个球位上严格各为**0/4**；全部有4/4不合法脚背，3/4不满足3.5m/s出球速度，至少两次不满足球门条件。
- 冻结CPU六回合计划本地JSON：`reports/v151_cpu_manifest.json`，SHA256 `682ef294f3c6cf79d6928c3caa884995c04535c2758592f6a663c6656ca87cf1`。
- 只读源级Report API资格清单：`reports/v151_cpu_schema_inventory.json`，SHA256 `9dbbc6ade5106e3b2a40dc4f8ca3e8ee3b8e12c7b778123cdea67dbdfcbe7514`；绝不宣称检查过运行时USD。
- 历史物理独立严格门禁审计：`reports/v151_cpu_original_physics_independent_audit.json`，SHA256 `65874dc2fc82860cc96557ef71bb0f8a49258a86ad76f10fb33a51a6866c3249`。
- **真实历史负例图**（仅Tang本地gitignored）：`reports/v151_real_v53_12_physics_failure_gate_matrix.png`，SHA256 `d1af647a0c4b4923969a17049f635b803596df15abf12bb921272d287e8bdd41`；`reports/v151_real_v54_failed_nonfoot_5ms_timeline.png`，SHA256 `8bad17aa28b702a91379432225defdf5e9ac941952980330a362695ab02bd495`。后者在5.140s截止，不插值伪造终止后的完整恢复或5.145s同物理步manifold。
- 专项CPU测试 **110 passed**（`reports/v151_specialized_final.log`）；扩展CPU测试最终数字见`reports/v151_expanded_cpu_final.log`。此前专项测试第一次1项模拟NaN测试fixture自身在计算摘要时失败，已修正为正确验证非有限值拒绝；原日志`v151_new_test_first_run.log`保留。扩展测试首次12 failed/23 setup error，原因是新worktree缺少Git忽略的原历史研究图片/JSON与URDF资源软链接；已在v151建立37条**仅只读引用、经路径与原v148 SHA认证**的忽略链接，不复制、不改原数据。首次日志`v151_expanded_cpu_first.log`保留。

对未来论文提出的可证伪假设：**可靠同物理步接触事件能够区分左腕—左髋相互接触与并发环境碰撞；只读接触采集本身在全部29DoF与完整状态上不会改变原物理轨迹。** 两项假设均需真实PhysX校准及六回合配对，当前只有基于原始状态数据的CPU审计，没有证明或否定该假设。人体高级爆发技术仍缺失跨被试和同步GRF/精确脚背接触真值，尚不能启动自称专业球员风格的训练。
