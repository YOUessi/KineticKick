# v0.1.7 连续助跑射门执行计划

Spec: docs/superpowers/specs/2026-11-04-runup-sequence-design.md
Branch: version/v0.1.7, base 8c89b9b. 用户已授权直接执行；复用Tang工作区与本地资产。

## Global constraints
不修改上游/旧配置/模型权重，不改变物理限幅，不传送。诊断、控制、验收必须分开。所有发布成功声明依据本次真实物理结果。三种触球面及定点力度仍是最终范围，不能用某一参考单次成功替代全部验收。

### Task 1: Reference and sequence interfaces
Files: tools/runup_sequence.py, tests/test_runup_sequence.py; tools/eval_free_shooting.py.
- RED: manifest每条参考解析顺序、拒绝部分缺失、未知模式、非有限值、边界阶段/平滑速度。
- GREEN: resolve_motion_files(trials, root, default)；可选髋膝参考时序控制，不改无选项旧路径。
- 接入每环境引用，参考输入指纹记录，新入口参数；先3个原生clip真实物理诊断。

### Task 2: Synchronized evidence and gates
Files: tools/runup_metrics.py, tests/test_runup_metrics.py; eval_free_shooting.py.
- 子任务独立实现summarize_sequence(trace,joint_names,contact_time_s,forward,initial_root,requested_mode)。
- RED:缺覆盖/无接触/零运动贡献/阶段结束瞬间不能冒充恢复/非脚接触失败；GREEN:真实数据来源与保守gate。
- 主循环提供关节/质心/ground filter/足端贡献、每阶段时间及termination reasons，旧字段不改语义。

### Task 3: Controller integration and physical acceptance
Files: configs/runup_shooting_v1.json, tools/show_runup_shooting.py.
- 先验证较长参考，依据实际髋膝贡献选择/修改模式；保持策略隐状态与物理状态连续。
- 有界测试不同模式/表面；出现同类失败两次停止盲扫，记录缺口和下一判别实验。
- 成功候选新种子+球位扰动验证，完整视频含助跑、触球、恢复和实际结果。

### Task 4: Verify and publish
Files: README.md, reports/2026-11-04-runup-sequence.md, VERSION_CONTROL.md.
- 完整CPU回归；本次物理结果+媒体逐项审查；一轮独立全分支代码审查，重要问题一次修复并回归。
- 提交/推送version/v0.1.7 upstream，保留main和旧版本；给用户可播放预览与真实范围。

## Review focus
每环境参考对应、物理时间与参考时间、5ms诊断点和policy20ms目标差异、末尾不重置、没有足够贡献证据时拒绝模式声明、预览/矩阵结果一致、没有把估计PD扭矩说成硬件测量。

## Ledger
- branch已创建，本地/远程v0.1.7均此前不存在。4090 Laptop可用。
- 主控复用当前工作区（developer默认）；用户明确本地资源保留，不创建新worktree。
- 阶段一风险：原生clip髋膝多为屈膝，不能凭文件名认定伸膝击球。

- Task 1 complete: explicit per-environment references, support-gated coordination, smooth target/velocity trajectory and measured waypoint inputs. 11 CPU interface tests.
- Task 2 complete: physical COM/ground/all-body-ball telemetry and conservative combined gates. 13 CPU metric tests.
- Task 3 complete for the published three combinations: hip/inside, hip/instep, knee/instep. Final-source seeds1711/1712:18/18; both complete previews pass. Other combinations remain explicitly unavailable.
- Ruling: preserve the user-approved original29DoF model/limits and use short fast stepping references; no unsupported claim of arbitrary-distance running or crossed six-combination completion. Cost: outside and new aim/strength calibration remain outstanding.
- Ruling: mode means forward foot-motion contribution between hip and knee, not energy; all-body contact and final COM/ground support must pass independently. Cost if misinterpreted: overclaiming biomechanical energy contribution; documentation and preview name the motion metric.
- Final review: two important findings fixed in one pass: replace old-source acceptance with final-source18 trials; upstream tree fingerprint regression RED→GREEN. Config identity minor also addressed with explicit trial provenance.
- Final verification:158/158 CPU, final-source18/18 physics,2/2 full previews,432-frame17.28s video decoded successfully. Main/old branches preserved; publish only version/v0.1.7.
