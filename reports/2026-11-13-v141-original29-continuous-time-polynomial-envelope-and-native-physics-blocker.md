# KineticKick v0.1.141 — 全29DoF七阶段动作的连续时间多项式包络反证：完整45组冻结QP与原PhysX证据

日期：2026-11-13。**父版本：** `version/v0.1.140`, `df46cf40857f4cde177aa14cae6e93092dc3adff`。**实验前独立协议冻结：** `d0df1a047b058fc20d1c2696300f11ab6c370c5c`。**原29DoF G1 URDF SHA256：** `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`。

**研究结论（必须与能力边界一起阅读）：** 45组不可改动的原v139 QP、v140 1ms准入与本版全时域Bernstein多项式包络准入分别为 **20/45、19/45、19/45**。v141检查了**15,660个**原关节位置/速度/研究性加速度的分段多项式包络，未发现v140额外漏检的候选，但**在已被v140否决的+4/0m/s/0.4s候选中独立检出两项区间内加速度违例，精确定位到右膝与右踝pitch**。这不是正式验证的安全证书，更不是物理射门成功。

## 1. 设备、仓库与证据保全

- 经授权Remote Desktop Commander查询，Tang在线；当前主工作目录是 **dirty** `version/v0.1.64`（原有 `?? =100===` 未追踪文件）。未触碰、覆盖、清理或提交它；v131及其他工作树未改动。
- 用 `git ls-remote --heads` 审查当前所有版本尾部，v140是创建前最新远端；`version/v0.1.141`此前本地与远端均不存在。只从**干净的v140 SHA**建立隔离 `/home/you/projects/kinetickick-wt-v141`。
- Tang根盘可用 **30GiB**（97%已使用），高于20GiB门槛。原v120 pytest **PID 242644** 仍独占RTX4090（260MiB显存，约19小时，持续占用CPU）；既没有终止它，也未启动IsaacLab/PhysX或训练新作业。
- 既有CMU/C3D、权重、原v52/v54真实PhysX、原约30MB JSON/NPZ和媒体均保留。本轮所有原始运行JSON、独立审计JSON、日志及两图留在Tang的Git忽略目录 `reports/`，GitHub提交可复现源代码、协议、测试及这份报告。
- 本轮**新增真正PhysX回合=0，新策略物理成功率=N/A，高水平人类真实正脚背动捕/同步GRF新增真值=0**。不得将CPU数值计算说成真实踢球。

## 2. A/B — 原PhysX 200Hz证据重新SHA认证（没有制造同物理步contact真值）

真实地复跑 `tools/v134_native_contact_vector_gate.py` 中的SHA锁定历史证据复核：

| 原v54物理控制器 | 原真实200Hz状态步 | 执行结局 |
|---|---:|---|
| `frozen_v32` | 2,800 | 14s物理trace完成，不等于严格射门PASS |
| `cmu_10_02` | 2,800 | 14s物理trace完成，不等于严格射门PASS |
| `cmu_10_05` | 1,028 | 5.140s提前终止，不能声称恢复 |
| **总计** | **6,628** | **3组历史回合，非本版新增实验** |

CMU10_05真实trace非脚净力峰值时刻 **5.135s**，最后真实29DoF `q` 在 **5.140s**；命名ContactSensor **5.145s**的记录没有同物理步完整29DoF姿态。左腕—左髋的原生PhysX actor/collider/contact manifold **仍为UNKNOWN**。任何等大反向25.89N和URDF间距都不能冒充接触双方真值。原v52 OFF/v54 ON对照证据不是本次v133/v134全新回调的真实非侵入性证明。

## 3. C/D — 冻结45组全身时域QP，升级从1ms采样到整个[0,14]s多项式包络

前置冻结协议有明确反证意图：不再把v140的1ms数值抽样解释成连续时间安全。保持v139原样重跑**5个共享全身动作幅度（0,+1,+2,+4,-1）×3个踝LINK原点相对速度目标（0,1,3m/s）×3种QP半宽（0.4,0.8,1.2s）=45组**；**不**根据已观察的v54失败修改求解器、时间相位、球位或研究阈值。

技术实现：

1. 按原v135七段五次smoothstep与固定29DoF协同矩阵，解析生成分段五次 `q(t)`；添加原v139四项紧支撑奇次多项式 `s^(1,3,5,7)(1-s²)^4`。按全部七阶段边界及QP支撑边界精确分割时间域，并对每一段构造局部自变量 `z=(t-a)/(b-a)` 的 `q(z), dq/dt, d²q/dt²` 多项式。终点14s亦由闭区间覆盖。
2. 将每段原29DoF的这三类多项式由幂基转换为Bernstein基，其凸包给出整段**数值包络**；对不明确的区间以de Casteljau二分进一步收缩。安全只能由包络进入限额给出，违规必须有实际端点多项式数值见证；最大深度20、最多100,000个节点后保持`UNRESOLVED`而不是假安全或假不可行。
3. 对每段三个非网格内部点与独立v140解析函数交叉验证，差异冻结容差2e-7。本轮最大全29DoF位置/速度/加速度交叉误差 **4.023e-10**，最大递归深度7；共检查**15,660**个分段关节导数约束，遍历**15,846**个Bernstein节点。
4. q/qdot使用原URDF29关节的原始限额，继承v140的后验数值容差0.0003rad与0.0003rad/s；40rad/s²及0.005容差仅是研究性加速度筛查，不是G1实测驱动器极限。Bernstein系数用float64构造并采用1e-8安全边际，**不**是IEEE外向舍入、有理数证明、非线性根/接触/摩擦/力矩安全证明。所有`PASS`只是条件式、数值保守、纯运动学包络通过。
5. 原v139 QP拒绝永不升级，v140任何1ms否决亦不可升级。所有45个网格单元记录QP状态与拒绝原因；独立JSON审计拒绝伪造真人监督、真实接触及物理成功率。

### 45组真实CPU计算结果（不是G1物理回合）

| 合成右踝减左踝link-origin速度目标 | v139 200Hz QP | v140 1ms + 中心精确FK | v141 全时段Bernstein数值包络 |
|---|---:|---:|---:|
| 0m/s负例 | 13/15 | 12/15 | **12/15** |
| 1m/s | 7/15 | 7/15 | **7/15** |
| 3m/s | 0/15 | 0/15 | **0/15** |
| **全部** | **20/45** | **19/45** | **19/45** |

v141通过的19组按半宽分别为0.4s: **4/15**，0.8s: **8/15**，1.2s: **7/15**。原QP25组未准入，内含18组`primal infeasible`、1组`primal infeasible inaccurate`、5组`maximum iterations reached`（**未解决≠证明数学不可行**）、1组求解器状态`solved`但独立解后约束检查拒绝。独立审计第一遍正因误把后者当成成功而报错；修复**审计元数据，不改冻结物理/求解参数**后，完整45组重新执行与独立审核均通过。错误日志及未通过审核的第一次JSON仍只读保留。

### 完整保存的两项隐藏的连续时域违例

同一历史已否决候选 `multiplier=+4, target=0m/s, halfwidth=0.4s`：

| 原29DoF关节 | 解析加速度见证时刻 | 解析加速度 | 允许屏幕阈值 |
|---|---:|---:|---:|
| `right_knee_joint` | 2.9234375s | +40.06628947rad/s² | ±40.005rad/s² |
| `right_ankle_pitch_joint` | 2.921875s | −40.10984549rad/s² | ±40.005rad/s² |

每一项以Bernstein子区间端点值给出；右膝见证还由独立v140解析参考与导数重算单元测试复核，误差小于1e-7rad/s²。该候选此前1ms抽样峰值约40.1234rad/s²。此处**没有新增另一组v140漏检失败**，也没有把v140已否决样本升级为成功。

## 4. E — 代码、测试、局部原始证据

新增研究工具：
- `tools/v141_continuous_envelope.py`：源锁定、完整45组原QP复跑、七阶段多项式全29DoF分段、解析q/qd/qdd、Bernstein递归、独立交叉检查、完整否决见证；
- `tools/v141_independent_envelope_audit.py`：45格唯一性、历史QP`postsolve`拒绝保留、v140准入不可回升、数值见证元数据与伪造物理真值拦截；
- `tools/v141_plot_envelope_grid.py`：独立审查通过后生成两张本地结果图（通过率与45格失败热图）；
- `tests/test_v141_continuous_envelope.py`：**52项新增专项测试**，包括段边界、全29DoF求导、4项扰动多项式、内部尖峰漏采负例、Bernstein二分、不确定状态、原URDF源锁、冻结篡改、独立审计防伪、实际失败样本解析重验。

**最终v124–v141选择性CPU扩展回归：673 passed、2 skipped、0 failed、2条继承自SciPy SLSQP的限界裁剪警告**。没有声明全项目GPU测试通过。

本地忽略证据位置：
- `reports/v141_envelope_trials_20261113_FINAL.json`（45格逐项输出，SHA256 `65e603fcb582e6334b7f0d0620f5db1fbde477c5398b8e14abb53d7ecf8fadad`）
- `reports/v141_independent_20261113_FINAL.json`
- `reports/v141_run_20261113_FINAL.log`
- `reports/v141_selected_cpu_regression_FINAL_20261113T1600.log`
- `reports/v141_historical_v54_sha_recert_20261113T1600.json`
- `reports/v141_20261113_FINAL_admissions.png`, `reports/v141_20261113_FINAL_cells.png`

复现（使用新文件名，已有证据禁止覆盖）：

```bash
cd /home/you/projects/kinetickick-wt-v141
PY=/home/you/conda_env/env_isaaclab/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/v134_native_contact_vector_gate.py --output reports/v141_original_v54_recheck_new.json
$PY tools/v141_continuous_envelope.py --output reports/v141_recheck_new.json
$PY tools/v141_independent_envelope_audit.py --input reports/v141_recheck_new.json --output reports/v141_independent_recheck_new.json
$PY tools/v141_plot_envelope_grid.py --trials reports/v141_recheck_new.json --independent reports/v141_independent_recheck_new.json --stem reports/v141_recheck_new
$PY -m pytest -q tests/test_v12[4-9]*.py tests/test_v13[0-9]*.py tests/test_v140_nonlinear_dense_reaudit.py tests/test_v141_continuous_envelope.py
```

## 5. 论文假设、失败与依赖的下一批

**验证假设H1（有限采样假准入）：** 旧200Hz准入20/45，其中至少1个不满足更细的连续加速度条件；v140已证实1例，v141独立分段多项式检查在完整14s内也否决该例、精确记录两关节的证据。**没有再发现新的v140假准入**，这也必须如实披露。

**研究假设H2（全时域包络而非弱姿态MSE）：** 在工程生成轨迹可表示为分段多项式时，Bernstein包络优于仅靠离散时点检查；但本版仍没有真正的区间认证浮点证明，下一版可转外向舍入有理数/interval arithmetic并验证有实测物理量的支撑/碰撞边界。

**尚未验证的论文主张H3（从优秀人类足球动作学得全身协调/发力）：** v135共享协同函数是手工工程假设，不是职业运动员标注。没有人体正脚背高水平运动链、接触时刻、同步GRF/冲量真值，不允许叫“学成”；当前踝link原点虚拟速度3m/s目标为0/15，更不能说明真实脚背球速不足或已实现爆发力。

依赖下一批（必须真正执行时按资源门禁）：
1. **A优先：** 等既有GPU作业退出、可用空间>=20GiB后，先做已冻结v133/v134原生contact callback已知碰撞/非碰撞正负控制，记录USD actor/collider/face/manifold/冲量向量，确认post-step时钟。再用**原v54三控制器同种子**真正重跑OFF/ON配对物理，全部共享状态逐值比对；不能因名字相似声称腕髋碰撞。
2. **B/C：** 把共享全身协调器与浮动基、真实单/双足底GRF/CoP、摩擦锥、足底滑移、非脚碰撞、原G1电机扭矩/加速度及14s恢复共同优化。预先冻结未见球位/种子；首次真实PhysX试验前不能拿已见历史失败暗中调超参。
3. **D/E：** 未见球位×多种子×原参考/新方法/负控制真实PhysX批次，以**正脚背冲量占比>=95%、合法入门、球速>=3.5m/s、全29DoF qdot、安全接触、完整14s恢复**的严格联合通过率报告，不用“碰球即成功”替代真实目标。

**最终研究状态：** v141解决了v140留下的全时间域**数值**漏检疑问并保留所有负例，没有解决原生PhysX接触双方、真实动力学/训练、人类专业发力数据以及真正高速稳定正脚背射门；这些未完成的门禁不能被新的CPU指标掩盖。
