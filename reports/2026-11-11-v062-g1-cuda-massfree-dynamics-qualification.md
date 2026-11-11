# KineticKick v0.1.62 | 8次预冻结原G1 PhysX：恢复CUDA全身200Hz安全观察，定位危险的质量加权路径

**日期** 2026-11-11；父源码 SHA `1b9a69063b68a04d41ea64223c30d198f68b08f6`（v0.1.61），独立冻结协议 commit `3493ce2`。配置 `configs/g1_massfree_probe_v062.json`、SHA256 `5ad61909b77e68f50c55a3f225e9f31985fd4346a5fb1c89111127c2b7a44134`。写入本地 `reports/v062_protocol_lock.json` 后运行全部8个实物理回合，实验输出write-once，无覆盖。原CMU/C3D、模型、JSON、媒体全部留在Tang，本报告只提供可复现协议/代码/数值，**没有训练新控制器**。

## A/B：物理测量恢复与失败机制核验

预注册 `safe_full / mass_no_mul / mass_clone / mass_after` 仅改变**只读**观测计算，不变动原G1 29DoF/球/PD/ContactSensor/碰撞过滤/非脚停止门槛。CPU与CUDA独立核验，不跨backend做parity归因。8次均真实启动原G1 Isaac PhysX；5次产出有效5ms原物理JSON、3次CUDA读取联合质量信息时无G1 JSON，严格视为**不可评分**。全量独立审计 `tools/audit_massfree_v062.py` 重新核验源协议/日志/原始物理文件SHA，非零无效结果不假设恢复轨迹；结果保存在Tang `reports/v062_independent_audit_20261111.json`。

| 预注册回合 | 后端 | 观测模式 | 有效200Hz帧 | 严格结果 |
|---|---|---|---:|---|
| cuda_v32_safe_full | CUDA | 29DoF+命名刚体位置/速度/非脚力，不读质量 | 2800 | PASS，14s |
| cpu_v32_safe_full | CPU | 同上 | 332（另1微秒终止标记） | FAIL，约1.66s非脚终止 |
| cuda_cmu05_safe_full | CUDA | 同上 | 1028（另1终止标记） | FAIL，约5.14s非脚终止 |
| cpu_cmu05_safe_full | CPU | 同上 | 332（另1终止标记） | FAIL，约1.66s非脚终止 |
| cuda_v32_mass_no_mul | CUDA | 质量读取，但不用质量×COM | 2800 | PASS，14s |
| cuda_v32_mass_clone | CUDA | 克隆质量→归一化→加权COM | 无G1 JSON | 不可评分；首帧`weighted_com`入口 |
| cuda_v32_mass_after | CUDA | 先COM，再质量→加权COM | 无G1 JSON | 不可评分；首帧`weighted_com`入口 |
| cuda_cmu05_mass_clone | CUDA | CMU10_05 克隆质量→加权COM | 无G1 JSON | 不可评分；首帧`weighted_com`入口 |

**2/5严格PASS不代表泛化成功**：两个PASS是相同v32、同球位、同seed1201策略的观察器复跑；其他3个有效案例非脚提前终止。3例无JSON既不是控制器FAIL，也不是PASS，不得混入严评。两种政策没有改动，只有诊断差异。

5组独立历史比较（同CPU/CUDA、同策略、同seed，跨原v060不改控制器）均核对**完整每帧25项原始物理字段逐值相同，最大差0，差异数0**，证明这些单独/combined只读测量对所测轨迹没有影响。相比v061，`safe_full`组合在CUDA**真实运行至14s/2800帧**，同时完整读取原29关节、10个命名刚体位置及速度、接触净力；避免了以前的CUDA质量×COM首帧路径。现有原物理 `whole_body_com_xyz` 可作为原已记录的CoM字段，**不能**把10个link的无权平均叫作机器人CoM，也不能把足部传感器合力冒充专业人体GRF。

**新增重要时间戳资格问题（不是新的物理变化）**：`safe_full`在同一索引 i 采集的原29个关节位置/速度，与原始物理 `traces[0][i]` **逐关节相等（最大差0）**，但 `env.sim.current_time` 标签比原始帧标签**领先约0.010s**，在这一批5ms帧中偏差基本固定。此不是10ms的机器人动作延迟真值。审计显式报警，绘图依据状态索引和原物理time_s对齐，不能按原观察器时间戳直接做接触前后窗口。后续需修订观察器记录state-frame time，但不能修改本批已冻结源码/已保存证据。

## 动力学与非脚事件：真实200Hz、无人工补帧

本批新`safe_full`观察器提供每个真实5ms物理步原29DoF、双足支持代理、10命名刚体世界位置/速度、左腕与左髋的link原点距离与命名非脚力；旧原物理trace同时保留whole_body_com、非脚总力和原控制器的合法脚球事件。独立检查得到：

- CUDA v32：14s整段、真实接触1.57s、触球前0.4s和后1.25s均完整；0N全段非脚net力；左腕/左髋link原点最短距离约**0.12469m**（5.80s），完整恢复严格PASS。
- CUDA CMU10_05：触球约1.57s，确实进球，但**5.14s提前非脚终止**，全14s恢复缺失；观测到原全身非脚净力峰值约**25.89015N**，左腕/左髋link原点最短距离约**0.11318m**（5.14s）。二者强烈支持碰撞风险，仍**不是**两侧真实collider/manifold的GPU直接证据。
- CPU v32、CPU CMU10_05：约1.58s触球，在1.66s均因`physical_nonfoot`提前终止；全身非脚净力峰值分别约**893.64N / 882.70N**，真实后0.08s可观察，**不能**当作有1.25s恢复轨迹。CPU与CUDA不是物理同构实例，不能直接用CPU解释CUDA自碰撞。
- `mass_no_mul`完整成功，且原物理与v060 CUDA基线严格逐值一致；`mass_clone`和`mass_after`首帧在`weighted_com`后没有OK标记、没有原物理JSON。由于该阶段同时包含CUDA Tensor乘法/求和/CPU转移，目前**只确认故障局限在此联合阶段**，不能武断定位单个CUDA算子，也不能将exit 0误判为成功。

新绘图工具 `tools/plot_massfree_v062.py` 基于本轮原始SHA对应的四组真实运动，分别在Tang的 `media/v062_real_*_failure.png` 绘制link距离、非脚净力、全部29原qdot限速比例、双足支撑代理。提早终止曲线在实际终止时截断；无JSON回合没有伪图。图像、原始G1物理JSON/NPZ及媒体不进入GitHub。

## 独立严格门禁、可证伪性与界限

每次物理严格综合：右脚正脚背冲量占比≥95%、进门、出球≥3.5m/s、原全部29个qdot限速、非脚碰撞停止和14s真实恢复。即使已进球也不能掩盖非脚终止。5个有效回合峰值实际qdot/原限均未超1，但失败仍失败。独立audit基于原 `stride_shot_success`、原 `termination_reasons`，不以可视化/根高度/姿态代理重新放宽判据。

**本批为观测仪表资格测试，并非独立未见策略泛化测试**：原seen seed1201、旧v32和CMU10_05，严格PASS重复不表示样本分布改善；没有生成人类式全身踢球策略，没有人体真实脚球冲量、职业球员触球与足底GRF真值。已有CMU只提供弱相位代理。GPU原生contact reporter仍0接触回调时不得以0头判断不存在碰撞；**GPU左腕—左髋真实碰撞双方/collider path/contact manifold仍未认证**，不许将CPU骨盆—右髋pair移植为GPU真值。

## 下一版研发接口：不再把诊断当作论文贡献

下版先将“状态索引准确配对、原物理COM字段、不再在GPU逐帧加权求COM”作为**有条件可用的安全观察路径**，并给自碰撞信号设置可信度等级（link原点距离=预警，不是碰撞形状间隙；无native pair=未知而非0）。随后才能接入可执行的**整段相位条件29DoF协调生成**：支持脚提前承重、踢腿近端/远端时序、躯干与摆臂角动量协调、随挥落脚；显式门控支撑、29关节速度/扭矩、非脚碰撞及14s恢复。冻结人类数据资格/训练-验证-未见划分和多随机扰动球位再作物理比较，不能在seen seed1201上秘密选超参。全项目自然动作评测还需多人体样本和盲评。

## 复现（不会覆盖write-once物理证据）

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/run_massfree_batch_v062.py
PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/audit_massfree_v062.py --output reports/NEW_UNIQUE_v062_audit.json
PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/plot_massfree_v062.py --output-dir media/NEW_UNIQUE_v062_figures
PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
```

本轮全部回归 **762 passed，69 subtests passed**；未改变任何历史CMU/模型权重/原PhysX结果。raw物理保留本地、GitHub只提交冻结配置/脚本/测试/科研文字。
