# KineticKick v0.1.57 — CPU/GPU G1 接触后端资格实验：失败审计与再现约束

2026-11-11 · 冻结父提交 `4714393ed447ef569bc1ebe757f6665929e24a32` · 仅原29DoF Unitree G1

## 执行状态与结果

Tang RTX 4090 Laptop 16GB，根盘余量约45 GiB，原有 CMU/C3D/模型/PhysX v52–v56 证据保持原地，无改写。执行前以 `version/v0.1.57` 从 v56 clean SHA 建分支；写出冻结配置、SHA 锁定和 write-once 本地报告，串行执行10个 preregistered case (6 G1 + 4独立双盒子 PhysX 控制)。本报告完成后源码可以提交，原报告 JSON、截图和日志仍在 Tang，并由本地进度 JSON 的 SHA256 验证。

**严禁将 v57 作为新的G1射门成功率结果**：

| 实验分组 | 原定 | 有可认证的 G1 原始200Hz JSON | 能认证的全任务成功 |
|---|---:|---:|---|
| 原 v32 / CMU 10_02 / CMU 10_05 × CPU | 3 | **0** | 不可估计 |
| 原 v32 / CMU 10_02 / CMU 10_05 × CUDA | 3 | **0** | 不可估计 |
| 双盒子正/负接触 × CPU/CUDA | 4 | 非G1 | 不可用于G1 |

**6次 G1 尝试均以子进程退出码0结束，但缺少最终 JSON**；日志停于 `FREE_SHOOTING_READY` / Isaac Kit 提示，不能声称真实29DoF轨迹已完成，更不能将结果误标为0/6失败或6次有效试验。尤其是源代码级复核确定：`tools/run_placement_strike_v057.py` 解析了外层 `--sim-device`，但向 `eval_free_shooting_backend_v057.py` 重建 `sys.argv` 时**漏传了该参数**，内层使用默认 `cuda:0`；3个原本标为CPU的G1日志确实输出了 `Environment device: cuda:0`。这使所谓CPU-GPU G1平行对照从原理上失效。缺失原始结果的另一直接成因尚未定位，不能推断为单独由参数漏传造成；保留失败日志供之后定位。

**4次隔离两盒子实验写出了真实有限时长物理报告，但进程收尾均触发180秒终止**，因此应标记 `WRITTEN_BUT_CHILD_EXIT_ABNORMAL` 而非干净成功。实际证据：CPU正接触过滤峰值627.058N、239个原生回调；CPU负对照0N、0回调；CUDA正接触626.339N、0回调；CUDA负对照0N、0回调。两盒子碰撞和 G1 自碰撞不是一个真值目标。当前 CPU/CUDA 原生回调可观测性差异与 v56 一致，但 **G1左腕—左髋 manifold、actor path、collider path 仍然未知**。零回调不能作为不存在碰撞的反证；反向净力仅构成候选归因。

## 重现实验与失败安全

`configs/physx_g1_backend_pair_v057.json` 是本批首次物理运行前冻结协议。驱动器 `tools/run_backend_batch_v057.py` 保证原输入 SHA256、源 SHA256、完整顺序、约20 GiB余量门禁、GPU文件锁、write-once数据以及逐次原始结果校验；所有CPU/GPU实验都必须以独立 backend 视为不同物理条件，绝不能假设轨迹等价。实验中实际出现的错配与无输出应按事实标注，不得更改冻结协议或篡改原本失败的日志。

`tools/audit_v057_backend_qualification.py` 是**事后只读**独立审计：校验协议/progress SHA，校验全部10份日志和所有已存在报告的SHA，交叉验证日志识别设备，严格要求G1输出存在、进程退出0且确实使用了冻结后端，才能计入有效物理样本；隔离两盒子永不算G1接触真值。审计结果 `reports/v057_independent_batch_audit.json` 为忽略上传的本地副本。相关单测验证缺失JSON不可计入分母、日志篡改检测、进度乱序和原始数据失配拒绝。

真实控制研究核心仍未完成：没有新模型学习真实专业正脚背接触、跑—支撑—摆腿—反摆臂时序；没有G1自碰撞真值；不能以过往球入门替代95%脚背冲量、≥3.5m/s出球、原29关节qdot、完整14s恢复与非脚碰撞门禁。v56 已有15个完成物理回合及严格结果，**不能与 v57 的6个无输出尝试混合统计**。v57 的工作贡献主要是发现 frozen-case 工作流丢失物理记录与CPU后端参数传递缺陷，防止错误统计进入论文。

## 下一版本 v0.1.58 可证伪实验协议（提案，不作为已执行）

1. 从清洁 v57 SHA 新建 v58；增加 argv backend 传播函数（纯CPU单测必须证明 `cpu` 能抵达内层模拟器），环境运行时设备回读和显式完整输出认证。首先各一组不含 G1 callback 的 CPU/GPU 原控制器短时定位实验，以实际JSON存在为放行条件；禁止在无输出异常下直接扩大批次。
2. 在正确后端上用原冻结控制器对照 read-only sensor/callback，独立真实物理正/负配对；必须有原生 path 和 manifold 才能称“G1自碰撞验证”；如 GPU 无回调只报告“不可观测”，不要给模型伪标签。
3. 全身29DoF 200Hz pre/post impact 耦合测量按真实终止时刻删失，无人类GRF、未验证接触点不可冒充真值。训练前另开身份/时间分离人体动作资格清单，冻结训练/验证/测试并隔离过去的C/D调参组。
4. GPU唯一持有、磁盘下限、旧数据不可覆盖、回归测试、push SHA核验仍为硬性门禁。

复现（需Tang原始资产、IsaacLab环境）：

```bash
cd /home/you/projects/kinetickick
PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking \
  /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests
python3 tools/audit_v057_backend_qualification.py \
  --output reports/v057_new_unique_audit_file.json
```

因 write-once 保护，切勿重用已存在的任何 `reports/v057_*.json` 输出路径。任何后续 v58 修复只能建立独立版本，不能更改 v57 冻结 source hash 或“修正”旧 v57 CPU 标签。
