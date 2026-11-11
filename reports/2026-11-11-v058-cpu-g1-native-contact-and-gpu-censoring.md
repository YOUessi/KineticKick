# KineticKick v0.1.58 | 原G1 CPU native manifold 真值与 CUDA 无输出负结果

日期：2026-11-11。基础版本：`version/v0.1.57`，父SHA `f7c3c2f696aea7e1a6cf955b6dc3632982958f50`。严格锁定文件：`configs/g1_backend_recovery_v058.json`，协议SHA256：`690fa214c40e9697dba8026294045b182edd58660eb7c6e1d3a43822b7f1f1c0`。原29DoF Unitree G1，原控制器无更改。

## 1. 本批实际执行及完整计数

在 Tang RTX4090 Laptop / IsaacLab 既有环境，已真实**启动7次原G1 PhysX作业**，完全按锁定次序和唯一目的文件记录；1次有完整可认证物理轨迹，6次 CUDA 运行虽然进程退出码0、日志出现 `FREE_SHOOTING_READY`，**没有最终 G1 原始JSON，不计有效试验或失败率分母**。

| 条件 | 原冻结控制器 | 是否真实执行 | 有效200Hz G1 | 严格结果 |
| --- | --- | --- | --- | --- |
| v056 CUDA | frozen v32 | 启动 | 无 | 不可评分 |
| v056 CUDA | CMU 10_02 | 启动 | 无 | 不可评分 |
| v056 CUDA | CMU 10_05 | 启动 | 无 | 不可评分 |
| v058 CUDA（已修正backend转发） | frozen v32 | 启动 | 无 | 不可评分 |
| v058 CUDA（已修正backend转发） | CMU 10_02 | 启动 | 无 | 不可评分 |
| v058 CUDA（已修正backend转发） | CMU 10_05 | 启动 | 无 | 不可评分 |
| v058 CPU（已修正backend转发） | frozen v32 | 启动 | **332个真实200Hz帧**，另有1个终止状态标记 | **FAIL** |

这是 **1/1 有效回合失败**，不是“0/7成功率”。CPU与GPU是不同物理后端，不能把CPU原生碰撞当作CUDA G1碰撞真值，不能将v52-v56 GPU的成功率合并到这里。

与v57相比，`tools/run_placement_strike_v058.py` 将外层 `--sim-device` 显式转发内层 `eval_free_shooting_backend_v057.py`。CPU日志确认为 `Environment device : cpu`、`V058_BACKEND_ARG_FORWARDED cpu`、`V058_NATIVE_RETURN cpu`、`FREE_SHOOTING_RESULT`，证明转发修复生效。六次CUDA日志确认 `cuda:0`，但未打印 `V058_NATIVE_RETURN`，根因仍未知。**不能推断是GPU传参缺陷导致无输出：它在修复后仍复现。**

## 2. 首次取得原G1的CPU直接native contact pair/manifold

唯一有效的原G1 CPU回合在球-脚第一次接触 `t=1.580 s` 后，原生 `omni.physx subscribe_contact_report_events` 返回了G1内部实际接触：
- actor A：`/World/envs/env_0/Robot/right_hip_roll_link`；
- actor B：`/World/envs/env_0/Robot/pelvis`；
- collider A/B：分别为上述actor路径的 `/collisions` 子路径；
- 物理接触点 `t=1.660,1.665,1.670 s` **3个原生point样本**，包含三维位置、法向、冲量向量和分离距离。两个后续样本的分离距离约 `-1.06e-5 m` 与 `-2.81e-5 m`；首样本为 `+4.04e-5 m`，可能属于接触容差，不能描述为该时刻的可见穿透。
- `native_headers=2362`、`callback_calls=332`、`errors=[]`、`dropped_events=0`、`report_capture_complete=true`。
- `pelvis-right_hip` 有5次header候选，其中3次有直接点样本；**只有这3点符合原生manifold证据门槛**。另外2次零点header不计真值。
- CPU本回合 **0个左腕—左髋直接point样本**，不等于其在其他回合或GPU从未碰撞。

这是此前“骨盆/右髋净力相反”归因的一项**真实原G1 CPU接触对证据**，但不是原v54 GPU左腕—左髋候选的确认，也不是整个CPU/GPU动力学等价的证据。

## 3. 200Hz全身过程、严格门禁和删失

独立CPU审计脚本 `tools/summarize_v058_censored_g1.py` 在只读5,499,979字节原报告上运行。原 `traces[0]` 共333行，其中最后一行 `t=1.660001s`、`terminated=true`，相对前帧只有1微秒，是**终止状态标记，不是200Hz物理帧**；合格物理帧为332行，从0.005到1.660s。独立v055动力学观察器为332帧；以此避免把终止标记错误算入200Hz样本。

- 真实触球 `t=1.580s`，前0.4s窗口覆盖，前窗口含79条原始trace采样；触球后仅0.080s、17条物理trace采样，**未达到1.25s，不能推断后恢复，更不能补成14s**。
- CPU本次在约1.66s因 `physical_nonfoot` 终止；非脚净力峰值 **893.6396N**。它是含多个刚体的**净力**，不能把全部893.6N归给单一骨盆—右髋原生接触对。
- 前0.4s左足支持法向均值 **244.98N**、右足 **96.08N**；触球后左足平均 **249.90N**、右足 **0N**（仅到终止）。这是模拟器的ground-filtered足部力，**不是人体受试者GRF真值**。
- 原G1全部29个关节的实测 `max |qdot| / 原上限 = 0.909826`，峰值在 `left_knee_joint`、`t=0.030s`；本次qdot没有超速，但其余门禁失败。
- 前窗口模拟CoM高度范围 `[0.67066,0.71516]m`，触球后到终止 `[0.71701,0.74171]m`；躯干抬高和低倾角**不能被当作完成站立恢复**。
- 右脚的球体接触面被分类为`ambiguous`、**正脚背冲量份额0.0**，未进门、未完成14s、未恢复，故严格FAIL。即使曾有球体接触，也不是合法高速度正脚背抽射。
- 早停后的接触窗口与所有后续动作均按缺失/删失处理。

本地真实失败图：`media/v058_cpu_g1_support_nonfoot.png`、`media/v058_cpu_g1_original_qdot_ratio.png`。大体量的原JSON、日志、图以及lock/progress SHA副本留在Tang，依据`.gitignore`不上传；GitHub提交复现代码、冻结配置、单测和这份文字报告。

## 4. 自洽可复现路径与测试

- 已锁协议：`configs/g1_backend_recovery_v058.json`，执行驱动 `tools/run_backend_batch_v058.py`，7个预注册case按`flock` GPU互斥、20GiB磁盘门禁、每例唯一目的路径、源SHA保护。
- 物理运行器：`tools/run_placement_strike_v058.py`只修复backend参数转发，保留原控制器物理路径；原`v056`运行器原样保留。
- 独立归因和原始审计：`tools/audit_v058_cpu_contact_truth.py`验证7个原始日志/报告SHA，拒绝无结果CUDA；`tools/summarize_v058_censored_g1.py`区分200Hz物理和终止状态标记，验证原29DoF关节速度/双足支撑/CoM/真实point pair，严格保留缺失后窗口。
- 本地哈希锚点：v58 CPU G1原始JSON SHA256=`4c834f62c23f1fb94ae0bd1d043081f877d0570dad7c4afecd5e54446aa30644`；v58协议SHA256如上；非脚事件路径与Point数据由本地 `reports/v058_independent_cpu_contact_audit.json` 及 `reports/v058_posthoc_independent_cpu_200hz.json` 核查。
- 本版本的纯CPU测试模拟Manifold fixture只是测试判定规则，**不是物理实验真值**；物理结论只来自真实CPU G1报告。

## 5. 物理解释与下一个需冻结的实验

已确认CPU侧存在真实骨盆—右髋自接触，但修复后GPU仍无原始轨迹，**目前不能把此现象归因于特定控制器或人类运动风格，更不能训练安全奖励依赖CPU/GPU混合伪标签**。

下一版本在全新冻结分支，使用原控制器及相同原G1初态，分别进行CPU无native订阅、CPU只读订阅、GPU最小观察器对照；对同后端共同物理字段逐值验证instrumentation parity，并在GPU`READY`到首个`STEP`之间增加进程级里程碑以定位无日志退出。若CPU两种观察器产生偏差，则native订阅只适于诊断而不能作为严格无扰动观测。**训练/优化必须等到正脚背运动资格和稳定的G1原生物理标签成立；原v56未见组不作为调参反馈。**
