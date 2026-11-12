# KineticKick v0.1.133：原生PhysX接触快照与同物理步证据链（CPU完成，GPU待释放）

## 1. 运行条件与冻结

- Tang在线；根目录`/home/you/projects/kinetickick`停留在旧v64且含用户既有未跟踪文件`=100===`，**完全未修改**。
- 从干净`version/v0.1.132` SHA `f6fab3682f9da3c3bfc4bb7d1e283656c5dbbf07`单独创建`version/v0.1.133`工作树。当前本地及GitHub尚无v133时才创建，无覆盖/force push。
- 先提交并推送预注册协议SHA `6cd4e597cc7ab4683fc097558d0df2fad9cb0d74`；后实施工具和测试。当前GPU被v120长期pytest PID 242644占用260MiB，日志原位置零字节，剩余磁盘约30GiB。**没有**启动GPU/IsaacLab、终止或抢占已有进程，也没有修改任何原PhysX、CMU/C3D、权重或v131未提交的共享隐变量生成器文件。
- 预冻结关键门禁：原29DoF/200Hz；脚背冲量占比≥95%、球速≥3.5m/s、全部原qdot、非脚力<20N、14秒恢复、进球；所有native事件均需callback正/负控制、同一步状态时钟认证、刚体和碰撞体完整USD路径及OFF/ON全部共同物理字段不变；未知时**拒绝电机授权**。

## 2. 本批真实实现的四个可复现模块

1. `tools/v133_native_contact_frame.py`：从`get_physx_simulation_interface().subscribe_contact_report_events`的borrowed callback buffer中**当场深拷贝纯数值**，保存PhysX header actor0/1、collider0/1、stage、slice offset/count、contact点/normal/impulse/separation、步时钟提示；拒绝slice越界/重叠、非有限值、伪路径、重复actor、非原200Hz时钟；所有输出始终`native_pair_certified=false`，即使合成测试给出全部正控制也不允许自动认证。
2. `tools/v133_runtime_stage_contract.py`：在真正Isaac Sim runtime可选加载的**只读**USD遍历，寻找每个CollisionAPI碰撞体最近RigidBodyAPI祖先，审计`PhysxContactReportAPI`是否已附着；完整保留一link多个collision形状，缺报告API/无owner的情况标为未认证。没有施加schema、创建prim或改变阈值。此工具**尚未在GPU上的Isaac Sim运行验证**，所以不能视为已取得真正USD/collider path。
3. `tools/v133_physical_event_ledger.py`：重新SHA256验证v128协议冻结的原v54三份真实200Hz原G1日志，重新检查原29个关节、末端marker剔除、qdot、COM、左右足净力代理及5.080–5.140s 13个采样步的真实数据；只输出有来源的历史数据和安全不确定性。
4. `tools/v133_plot_real_event_window.py`：绘出真实200Hz 3控制器的全29关节速度限额比、非脚接触净力代理、左足支撑净力代理三张失败分析图；5.145s只画命名传感器时间线，**不插值/补造末端q**。PNG、30MB级JSON等按gitignore在Tang本地，不上传。

## 3. 原始真实G1实验批量审计

| 历史控制器 | 真实步数 | 终点 | 原记录success/recovered | 5.080–5.140s非脚净力峰(N) | 全程29DoF最大qdot限额比 | 近事件左足/右足净力积分代理(Ns) |
|---|---:|---:|---|---:|---:|---|
| frozen_v32 | 2800 | 14.000s | true/true | 0 | 0.908129 | 24.2242 / 5.1417 |
| CMU10_02 | 2800 | 14.000s | true/true | 0 | 0.908129 | 10.9627 / 11.1682 |
| CMU10_05 | 1028 | 5.140s | false/false | 25.890154 | 0.908129 | 0.4353 / 19.6290 |

- **本批新增真实G1 PhysX回合0次，历史重审3次、真实200Hz物理步6,628。** 上表success为原报告字段，**不是重新独立执行全严格门禁后的新策略成功率**。
- 失败组真实非脚力峰位于5.135s；原命名Sensor时刻5.145s、最后实际关节状态5.140s，历史的5.140001s拷贝终止标记被剔除。缺少5.145s对应29DoF同物理步样本，任何collider-pair归因仍保持`UNKNOWN`。
- 近事件0.065s窗口三组原29DoF qdot最大限额比分别0.25895、0.13336、0.29508。仅看这一窗口无法判定整个14秒安全；失败组虽然最大qdot比不超1，仍因非脚力和恢复失败。
- 脚底净力是支撑**代理**，不能叫真值GRF/CoP；这里的纯数值审计既没有建立人体脚球真实接触，也未训练职业式高爆发足球协调策略。

## 4. 专项负例和真实测试

- `tests/test_v133_native_contact_frame_and_ledger.py`和`tests/test_v133_runtime_stage_contract.py`新增**46 passed**。测试覆盖borrowed buffer修改不影响快照、错误stage、token路径、重叠contact slices、离散时钟错位、错误原29关节顺序、伪造all-positive认证、空事件不能证明无碰撞、collider多形状、丢失报告API、无刚体owner以及源文件SHA等。
- 扩展CPU相关回归（v124、v127、v128、v129、v130、v132、v133）**240 passed、1 skipped、0 failed**；GPU全项目回归未执行。没有伪称原生PhysX callback已成功输出，因为本批没有新physx runtime。
- 实际原始审计生成`reports/v133_original_G1_event_ledger_v1.audit.json`，3张PNG，审计日志与完整pytest日志均保留Tang工作树本地，由原版本原始文件路径和冻结SHA再现。

## 5. 复现、边界及下一批物理依赖

在`version/v0.1.133`工作树运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python tools/v133_physical_event_ledger.py --output reports/v133_original_G1_event_ledger.audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python tools/v133_plot_real_event_window.py --output-prefix reports/v133_real_G1_200hz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/test_v133_native_contact_frame_and_ledger.py tests/test_v133_runtime_stage_contract.py
```

输出使用exclusive create，不覆盖已有物理记录/图片；重复运行另取输出名。

NVIDIA 2026-07-23 Contact Reports指南描述了`PhysxContactReportAPI`、`subscribe_contact_report_events`和`actor/collider`完整路径及contact_data offset/count/point/normal/impulse接口（https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html）。它是本实现的接口依据，但运行环境版本差异必须通过真实正负控制复验。

**GPU释放后的先后顺序**：①用隔离的两球体/刚体正碰撞和分离负例校准上面订阅与PhysX事件步时钟；②原G1只读USD所有碰撞体owner/报告阈值/运行时接触offset，获取左腕—左髋、左肩—躯干真正actor/collider/point/normal/impulse；③严格按同原控制器、球位、seed原3回合OFF/ON独立真实物理逐状态对比；④在这一前置门禁已验证且GPU独占条件下，推进全身动作共享协调生成 + 支撑与碰撞安全投影 + 未见球位/多种子PhysX严格门禁。当前不能以旧人类弱参考、离线MSE或单次进球冒充主论文创新。

**结论**：本批将之前仅有的URDF几何和命名净力关联，推进为可挂载的native callback快照、只读USD collider-owner映射、拒绝假同帧认证的证据合同及6,628个原始物理步的完整来源审查。A的真实contact pair仍未得到，C全身可执行生成和D多组新增真实物理仍因GPU阻塞和数据资格限制而未完成。
