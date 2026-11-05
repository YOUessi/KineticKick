# v0.1.24：状态反馈抽射候选的负结果记录

**结论：尚未解决用户要求的“球位／初速变化 → 调整落脚和触球时机 → 恢复站稳”。不提升为可靠控制版本，不替换冻结基线。**

原七个预定条件，冻结控制与本候选均为 **1/7 完整通过**；补充 ±0.10 m/s 的两个条件，两者均 **0/2**。每个条件一个确定性回合，均为开发／诊断数据，不是随机成功率或独立泛化评估。所有结果是 Tang 上原 G1 29DoF 的 PhysX 仿真，不是实物。

## 本轮做了什么

- 从 `version/v0.1.23` 的 `31e477b` 建立 `version/v0.1.24`，19 个冻结文件 SHA256 均未改变；原入口、旧模型和旧版本保留。
- 跑步 PPO 从冻结 checkpoint 热启动，在每次训练 reset 对世界 x/y 速度加入独立均匀 ±0.25 m/s 扰动。一次限时训练，实际 600.524 秒、记录迭代 472；不将训练完成当作物理验收。
- 新实验入口按实际球位、根部位置／速度计算有界落脚参考与相位修正。跑步与摆腿参考、原强力残差、支撑门控的速度脉冲共用调整后的时序。
- 足端平移只通过髋／膝四个近端关节求解，保留当时两踝关节的目标角，经过原关节范围和 ±0.25 rad 修正限制。正脚背姿态并不因此得到物理保证，仍以真实接触分类验收。
- 保留自由根、原执行器与碰撞参数、原终止和进球／恢复标准；初始化后没有机器人或球的位置／速度写入，没有外力托举。
- 规划器使用仿真真值状态和冻结原始场景的根部轨迹标定。它是显式反馈控制，**不是已学会球位条件化抽射或人类发力链的生成模型**。本轮只训练了跑步平衡策略。

## 固定候选 C 的最终结果

|条件|冻结完整通过|候选完整通过|候选水平出球速度 m/s|末段站稳 s|实际结局|
|---|---|---|---:|---:|---|
|原始球位／初速|是|是|4.727|9.795|完整通过|
|球 x −5 cm|否|否|5.159|9.840|正脚背、站稳，但偏门|
|球 x +5 cm|否|否|—|0.000|未触球，双小腿部位出现接触|
|球近 10 cm|否|否|2.538|0.000|左脚提前碰球；随后骨盆／髋部接触|
|球远 10 cm|否|否|—|0.000|未触球，双小腿部位出现接触|
|前进初速减 0.20 m/s|否|否|3.417|0.000|正脚背触球；恢复阶段小腿非脚部接触|
|前进初速增 0.20 m/s|否|否|—|0.000|小腿先碰球，随后正脚背接触|
|前进初速减 0.10 m/s（补充）|否|否|5.037|0.000|正脚背触球后物理终止|
|前进初速增 0.10 m/s（补充）|否|否|1.588|0.000|触球部位不符并物理终止|

“—”表示无接触或接触尚未结束即终止，不能把中途球速充作完整出球速度。完整通过包含跑动衔接、指定部位触球、进球、全身接触约束与恢复；站稳一项通过不等于整套任务通过。前进方向是世界 −Y，前进初速减 0.20 对应世界 vy 加 0.20。

## 有效改动与被否定的假设

新增状态反馈确实改变了动作时机，原始、减速和加速的首次正脚背接触分别为 1.510、1.595、1.480 s；后两个冻结条件都没有脚触球。初速扰动引起的触球前横向偏移有所减少，但这些局部改进没有转化为完整任务成功。

较快初速案例在 1.465 s 已出现 `right_knee_link` 对球约 837 N 接触，早于 1.480 s 的脚接触；这个链接包含小腿碰撞体，不能把后来的正脚背接触掩盖为合法射门。较慢初速在 2.455 s 出现约 212 N 的该链接非脚部接触，当时根部约 0.787 m、倾斜约 25°，球接触矩阵为空；没有单独过滤所有身体碰撞对，因此不武断断言这是膝盖触地。

原始候选末段站稳 9.795 s，但球 x +5 cm、球远 10 cm 相比冻结控制出现更早终止。新跑步策略改变了与旧射门绑定的步幅／相位；单独对脚位、触球时机和恢复交接加反馈，无法保证全身在分布外状态保持兼容。

## 有界尝试记录

|尝试|三个原始／减速／加速探针|决策|
|---|---|---|
|A：新跑步权重＋只调射门时序|0/3，原始场景左脚提前碰球|同步整段步态相位|
|B：整段相位＋标定接近路径|1/3，变速均能正脚背接触但碰撞／恢复失败|检查足端几何与恢复路径|
|C：保留踝姿态＋完整根部标定路径|1/3；完整七条件仍 1/7|固定作负结果研究对象|
|D：触球后平滑换回冻结跑步策略＋放宽相位上限|仍 1/3，未修复两个速度失败|拒绝新增复杂性，保留本地快照|

总计一次 600 秒 PPO 训练、20 个新增有界物理回合（A/B/C/D 各 3、另外 4 个球位、较小初速候选／冻结对照各 2）。原七个冻结结果复用，不重新训练或重跑基线。达到不收敛诊断点后停止继续堆叠手工参数；没有宣称完成适应能力。

## 下一项科研实验

本轮否定的是“只训练跑步鲁棒性，再靠原射门控制的局部反馈就足以获得端到端适应”的假设。下一轮应在同一个物理训练回合内包含真实球位分布、接近落脚、触球冲击和恢复，让策略直接接收球／支撑状态并学习全流程；加入保持原始成功轨迹的约束，避免以新跑步风格破坏原射门。先用小范围扰动和原始场景联合验收，再扩域，保留本轮反馈方案和冻结 v0.1.15 作消融对照。该联合训练本轮尚未实现或运行。

## 复现与文件

实验入口 `tools/run_adaptive_stride.py`，候选配置 `configs/adaptive_strike_v1.json`（明确 experimental_not_accepted）；串行对照 `tools/run_adaptive_strike_matrix.py`；离线审计和科学图 `tools/analyze_adaptive_strike.py`。`--disable-feedback` 用于消融，不是完整冻结基线；冻结对照仍使用 `tools/run_strike_baseline_case.py`。

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/build_strike_feedback_template.py
/home/you/conda_env/env_isaaclab/bin/python tools/run_adaptive_strike_matrix.py --cases center initial_slower_02 initial_faster_02 --output-prefix reports/v024_new_reproduction
/home/you/conda_env/env_isaaclab/bin/python tools/analyze_adaptive_strike.py --output reports/v024_comparison_recheck.json --figure media/v024_comparison_recheck.png
```

限时训练命令：
```bash
/home/you/conda_env/env_isaaclab/bin/python tools/train_running_balance.py --motion experiments/running_v011/run_stop_reference.npz --output experiments/running_v024/new_velocity_pilot --num-envs 1024 --iters 500 --max-seconds 600 --seed 2401 --init-checkpoint experiments/running_v011/run_stop_n1024_i500/last_model.pt --strict-contact --full-phase-sampling --initial-velocity-jitter .25
```

Tang 本地保留：`experiments/running_v024/velocity_jitter_pilot/` 内的权重、manifest、各失败方案源码／配置快照和补充速度清单；`reports/v024_*.json` 原始遥测与汇总；`media/adaptive_strike_v0.1.24_comparison_reviewed.png` 对照图。七个 C 结果来自 `v024_ankle_preserved_*` 的原始／±0.20 初速及 `v024_ball_positions_*` 的四个球位。补充速度用 `v024_velocity_candidate_*`／`v024_velocity_baseline_*`。

新权重 SHA256：`053007e300fee959fb8451d5e06eb885e41a68e6cde6b663be7e1674efe01428`。标定轨迹 `configs/strike_nominal_trajectory_v1.json` 由冻结中心回合生成，按资产约定留在 ignored 本地目录；生成入口核对配置中的指纹。新克隆依旧需要 Tang 的模型、原始物理报告、运动参考与模拟器。

离线审计已核对九对初始机器人姿态／关节、实际初速、球门／球位、模型与执行源码指纹、关节力矩／速度限制及完整 5ms 记录或真实终止。最终定向测试 **38 项通过，另有 45 个参数子检查通过**（禁用无关 ROS pytest 插件自动加载）；未运行全套测试、独立留出集或实机测试。

独立只读审阅发现候选审计未拒绝关闭反馈的消融模式／运行配置变化，以及空反馈记录可漏过。均已加入失败→通过的回归检查；复核九个真实报告结果不变。修订后的审计文件为 `reports/v024_adaptive_comparison_reviewed.json`。审核只修补证据校验，没有改变已测试控制器或新增 GPU 回合。
