# KineticKick v0.1.67｜真实原29DoF G1 PhysX：合成全身相位方向对照、严格脚背门禁漏判与4次非脚碰撞失败

**日期**：2026-11-11。**独立分支**：`version/v0.1.67`。**父 SHA**：`d80785015174884246daf6a73b4bb61f7492b236`（v64）。
**实验前已提交源码与协议 SHA**：`14c554844ac51ddd12c5558ff90d2683466263c4`；协议 `configs/g1_phase_directionality_batch_v067.json` SHA256=`c72074b7306275e394f80ecc68bc720f2b16e31bc703581d1dcee5f3d730480e`。 所有子进程原始 JSON、日志、本地 PNG 留在 Tang，不进入 GitHub；源码和结论可复现。

## 最重要的结论：原任务 PASS 并非合法脚背全门禁 PASS

**9/9 次原始 CUDA IsaacLab/PhysX 自由根 G1 29DoF 物理尝试有真实 200Hz 输出；5/9 完整 wrapper 认证并被旧 `stride_shot_success` 判 PASS； 剩余4次真实提前触发 `physical_nonfoot`，原生物理已记录但 wrapper 未完成验证，不能把其标签涂改为有效完整成功回合。**

**独立按照用户完整硬门禁复核只有 4/9 PASS，5/9 FAIL或提前物理删失。** 旧 wrapper 的5个 `stride_shot_success=True` 中，`new_stance_b_synth` 的实际正脚背冲量份额只有 `0.729834707982299`，**严重低于不可放宽的95%**。它虽然球进门，球速代理约4.390m/s，14秒完成、关节未超速， 仍然是**非法脚背比例 FAIL**；不能报道为全门禁成功。 本版本没有修改、宽松处理或事后重写任何原 PhysX 控制器和旧文件，另用只读 `tools/audit_v067_hard_gates.py` 对真实日志做**事后独立硬门禁重新评价**。

### 冻结设计、真实反事实及来源有效性

来源于 v63 的两个**合成**（非人类运动学习）29DoF 时序动作：原顺序 vs 相同预接触每个关节样本在0.86–1.56s内时间倒序，绝对幅度样本集合保持不变；另有原 v32 基线。 三个新扰动真实重置种子6707/6719/6731和三个冻结球状态，9回合=**3个独立物理条件、3种控制**， 不是9个独立人物或9个独立种子。original 29DoF PD、运动专家、场景球/接触几何、 0.005s Physics、0.020s action、20N真实非脚早停和恢复均未改动。 使用原全29qdot、真实 ball 球速（接触后5–110ms窗口峰值作为**非独立校准**代理）、 球门、≥95%请求脚背冲量、14s恢复判定，真正做了串行有锁GPU物理而非离线回放。

## D｜9回合个体硬门禁结果（源JSON SHA均已逐份检查）

| 冻结条件 / 控制 | 原 native 5ms行 | 原 wrapper | 原 `stride_shot_success` | 实测脚背份额 | 接触后球速代理 m/s | 全门禁 | 原始具体负结果 |
|---|---:|---|---|---:|---:|---|---|
| `new_stance_a_v32` | 2800 | VALID_G1 | True | 1.000 | 5.198 | PASS | 所有核实硬门禁通过 |
| `new_stance_a_synth` | 2800 | VALID_G1 | True | 1.000 | 5.231 | PASS | 所有核实硬门禁通过 |
| `new_stance_a_reverse` | 2800 | VALID_G1 | True | 1.000 | 5.171 | PASS | 所有核实硬门禁通过 |
| `new_stance_b_v32` | 741 | INVALID_G1 | None | 1.000 | 4.419 | **FAIL** | original_G1_wrapper_source_certified, no_nonfoot, full_14s, recovered |
| `new_stance_b_synth` | 2800 | VALID_G1 | True | 0.730 | 4.390 | **FAIL** | original_measured_instep_impulse_ge_95pct |
| `new_stance_b_reverse` | 2800 | VALID_G1 | True | 1.000 | 4.432 | PASS | 所有核实硬门禁通过 |
| `new_stance_c_v32` | 329 | INVALID_G1 | None | 0.000 | — | **FAIL** | original_G1_wrapper_source_certified, goal, original_measured_instep_impulse_ge_95pct, ball_speed_3p5mps_5_to_110ms_proxy, no_nonfoot, full_14s, recovered |
| `new_stance_c_synth` | 329 | INVALID_G1 | None | 0.000 | — | **FAIL** | original_G1_wrapper_source_certified, goal, original_measured_instep_impulse_ge_95pct, ball_speed_3p5mps_5_to_110ms_proxy, no_nonfoot, full_14s, recovered |
| `new_stance_c_reverse` | 329 | INVALID_G1 | None | 0.000 | — | **FAIL** | original_G1_wrapper_source_certified, goal, original_measured_instep_impulse_ge_95pct, ball_speed_3p5mps_5_to_110ms_proxy, no_nonfoot, full_14s, recovered |

原 wrapper 5条有效成功中：v32基线为1/1、合成顺序为1/2、合成倒序为2/2符合严格全门禁，**分母极小且3个物理环境不是随机总体采样**。
四条 native 原始 `physical_nonfoot` 不能晋升 wrapper VALID；其失败有原真实事件、接触终止和非脚净力支撑。
 b 的原 v32 在3.700001s提前早停，即使进门、脚背100%、球速约4.419m/s仍无完整14s恢复； c 的三个策略均在1.640001s提前非脚，碰撞强而未能完成合法脚背和后1.25s。

## A/B｜全身真实200Hz运动、接触真值边界、接触窗口删失

- **9次 CUDA 原 G1 真实轨迹**：5条完整2800行至14s（每条原29DoF q/dq/目标、真实CoM、两脚ground-filtered负载和命名nonfoot净力可审计）， b原v32 741行至3.700001s；c三个策略各329行至1.640001s。停止之后**不人工补帧**。
- **同一物理起点的全部3组**：各组v32、顺序、反序的最前100个5ms帧，原25类可比物理字段逐值相同（容差1e-6）， 原生GPU/原始固定29DoF顺序、seed、root_fixed=False、原始路径SHA均由独立只读审计复核。 这证明相同初始物理，不证明后续新策略安全/人类自然。
- 四条提前退出的native JSON没有最终wrapper配置hash/stride verdict，因此冻结runner将其标为`INVALID_G1`， 另起**事后、独立、只读**`tools/audit_v067_censored_native_verified.py` 验证终止日志、 `physical_nonfoot`、实际非脚强度和失去恢复；不把它们改标`VALID_G1`。 首次只读审计因原始`joint_velocity_limits`存储形状(1,29)错误拒绝， 第二次修复后又因变量名`nonfoot`错误拒绝，均有本地保留的原负日志； 最终postregistered第三版审计4/4找到直接物理非脚证据，全部原JSON未改。
- c组3个原生早停的非脚最大净力分别2272.839N/2225.377N/2287.163N； b原v32非脚净力37.709N。**这些是命名刚体聚合净力，不是形状级两刚体真实pair、接触法向或撞击冲量真值**。
- 各回合29DoF实测qdot限速最大比值约0.907–0.914，均≤1。 正式评估完整恢复不能由机器人根高度或倾角替代；只承认真实到14s的记录。
- 同源安全门对左腕/髋只用**link-origin距离代理**； GPU左腕—左髋的真正PhysX接触双方、collider USD paths / manifold、实时mesh signed distance仍然 **UNVERIFIED**。CPU v58另一个pelvis–右髋的原生点，不能冒充CUDA左腕—左髋真值。

## C｜实测全身刺激 & 碰撞安全接口（不可夸大为学习方法成功）

- 四个完整且认证的synthetic全身候选（a/b的顺序和反序）均**真实作用7个20ms控制步**， 顺序峰值0.017375rad、倒序峰值0.01799997rad； 它是原G1动作控制器的真实物理变化，**不是职业人类全身动力学学习**。 方向性反转并不能作为自然性正确时序标签；在仅3组物理条件且c全部非脚失败时，不存在方向效果的可泛化证据。
- 新`tools/stance_safe_temporal_policy_v067.py`给出真正全29DoF目标和支持腿/腰髋/摆腿/双臂4个群组 非零协调、量纲映射、0.006rad/20ms步幅、全部原qdot/关节幅度、支撑边缘、 双脚地面接触、体倾和非脚事件检查接口。没有**当前真正GPU原形状接触证据**时强制返回原动作， 不是已经上机训练/获验证的CBF/MPC安全控制器。

## 有界人类运动数据资格研究（不是伪造真实力学）

六个数据来源按职业/原生3D人体/同步脚球触球/真实冲量/同步GRF五项同时要求核对， `tools/qualify_soccer_sources_v067.py` **0/6符合完整职业正脚背动力学监督资格**。 本地两条CMU subject10 120Hz光学运动仅几何/非真值触球相位； CMU subject11尚未检验原C3D，GroundLink有真实GRF但不是足球球触证据， SoccerKicks视频估计3D非力学GT，职业比赛2D广播视频不能变成3D测力台，机器人数据也不是人。 详见 `docs/V067_SOCCER_DATA_QUALIFICATION.md`及权威资源链接。 不声称优秀真人职业抽射动力链已经学会。

## 不可篡改证据 / 复现路径

- 冻结设计 `docs/V067_DIRECTIONALITY_PREREGISTRATION.md`、 配置`configs/g1_phase_directionality_batch_v067.json`和实验前Git commit`14c5548`； 任何事后独立审计/报告/可视化均存**新增文件**，不修改冻结Runner、配置或历史源码 SHA。
- 原始write-once `reports/v067_new_stance_{a,b,c}_{v32,synth,reverse}.json`及各自`.log` 仅Tang本地。`reports/v067_protocol_lock.json`、`reports/v067_batch_progress.json` 与三份正式后审计JSON本地含校验哈希：`v067_independent_200hz_audit_postregistered.json`、 `v067_postregistered_censored_native_verified.json`、 `v067_independent_hard_gate_audit_postregistered.json`、`v067_all_native_pair_parity_postregistered.json`。
- **图例（仅Tang本地真实物理绘图）**： `media/v067_new_stance_a_measured_directionality.png`、 `media/v067_native_nonfoot_censoring.png`（早停后不补齐14s）。 所有权重/CMU/C3D/NPZ、30MB+JSON及媒体仍由.gitignore留在Tang。
- 运行： `PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/run_v067_directionality_batch.py`（原有证据被hash复查而非覆盖）； `PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/audit_v067_hard_gates.py`（仅新环境可重建write-once审计，现有本地文件禁止覆盖）； `PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/`。

## 下一完整研发批次：不能把 v67 结果包装成论文终点

1. 首先独立隔离/验证**GPU original G1 self collision shape-level**真实两刚体pair、USD collider路径和manifold， 对原3次完整回合做read-only 25字段 parity。当前只有真实聚合net force，缺直接证据，禁止擅自改碰撞filter。
2. 建职业真人真实脚球冲量与同步GRF的数据资格、伦理/许可、跨人训练验证独立测试， 区分光学相位代理和真实物理接触标签；承重脚plant/COM/摆臂/躯干/髋时序标注应带不确定性。
3. 原G1配合实时原生collider距离与足底polygon/支撑边缘认证后， 再学习29DoF全身助跑→近端到远端发力→合法正脚背→随挥恢复的时序policy， 对不同人/动作/场景/种子冻结先验、baseline和打乱时序负控制。
4. **修复未来评测中的硬门禁一致性**，不得把原`stride_shot_success`单独当≥95%脚背； 需要明示脚背≥0.95、球≥3.5m/s、真实进门、每个原关节qdot、非脚安全、 1.25s后窗与14s恢复。v67的b_synth错误PASS应该作为回归阻断样例，而不是用阈值调低。
