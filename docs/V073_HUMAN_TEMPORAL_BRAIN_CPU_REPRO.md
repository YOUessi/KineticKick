# KineticKick v0.1.73 | 真人120Hz时序预测、跨被试泛化失败与G1离线形态映射

## 科研结果

本批已在Tang真实执行两层GRU时序训练、时间打乱负控制、主体隔离评估、29DoF G1运动学投影和独立数值审计。没有执行任何新真实G1 PhysX回合，也没有得到优秀职业球员射门冲量/GRF/自然度真值；不能将学习模型与离线关节参考称作已能高爆发正脚背抽射。

数据是CMU原始120Hz C3D的处理版：6段共3358帧，两名被试。每帧19个骨架/光学标记导出的骨盆中心化尺度归一化三维坐标，外加8个躯干/四肢几何余弦，共65维。源中没有实测脚球真实首次触碰、真实球冲量、职业球员资质或GRF。

训练原CMU被试10的10_01、10_02、10_03、10_05四段；验证同被试10_06；唯一真正按人完全隔离的测试为被试11的11_01，测试读取始于两模型在验证集完成选择之后。历史24帧(200ms)→预测未来12帧(100ms)，训练343个相关滑窗，验证45、测试47；不能把滑窗数当作独立球员数。

原实测验证3D误差：时序GRU 0.056799、四帧等速外推0.058800、末帧保持0.117563、打乱历史GRU 0.112004。
另一位被试原留出3D误差：时序GRU 0.103993、等速外推0.052509、末帧保持0.097559、打乱历史GRU 0.105697。
因此真正跨人时序泛化失败，而非具备论文所需的自然协调技能。误差量纲是原光学人体的骨盆相对、尺度归一化位置，不是米、G1动作质量、运动美感或球速。

在训练被试10_02上使用GRU递推0.4秒人体动作，通过原G1真实v63时序关节轨迹和URDF可微IK生成两套700x29、50Hz的静态几何参考；右腿6个击球关节保持完全0增量，遵守0.055rad峰值、0.004rad每20ms变化、原URDF关节限和触球前释放。这只证明可生成离线29DoF几何参考，不是G1真实执行、真实接触或安全证书。

## 环境与命令

开发工作树：/home/you/projects/kinetickick-wt-v073
只读原始资料根目录：/home/you/projects/kinetickick
Python：/home/you/conda_env/env_isaaclab/bin/python
所有相关命令都需设置 CUDA_VISIBLE_DEVICES=''、PYTHONPATH=tools、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1 和 MKL_NUM_THREADS=1，不能使用与v72并发的GPU。

可复现执行顺序（新实验必须创建不同输出目录，绝不覆盖旧权重/JSON）：

1. 运行 python -m pytest -q tests/test_v073_human_temporal_brain.py tests/test_v073_cpu_g1_morphology.py。
2. 运行 tools/train_human_temporal_brain_v073.py，参数 --protocol configs/v073_human_temporal_learning.json --source-root /home/you/projects/kinetickick --output /绝对路径/NEW_UNIQUE_BRAIN。
3. 运行 tools/retarget_v073_human_brain_to_g1.py，参数 --source-root /home/you/projects/kinetickick --brain-output /绝对路径/NEW_UNIQUE_BRAIN --proposal-output /绝对路径/NEW_UNIQUE_G1。
4. 运行 tools/audit_v073_human_brain_g1.py，参数 --source-root /home/you/projects/kinetickick --brain /绝对路径/NEW_UNIQUE_BRAIN --proposal /绝对路径/NEW_UNIQUE_G1 --output reports/NEW_UNIQUE_AUDIT.json。
5. 用 tools/write_v073_report.py 将审计结果制成Markdown与本地绘图。已经存在的正式报告不可覆盖。

冻结模型/原始数据/协议源SHA在 configs/v073_human_temporal_learning.json；原成功CPU网络和两份GRU权重在 ignored experiments/v073_cmu_human_temporal_replay1，源数据和原G1全部原始轨迹仍在Tang。生成的离线G1关节参考在ignored experiments/v073_cmu_g1_brain_proposals_replay2。正式数值审计在本地 reports/v073_independent_fullhuman_g1_audit.json，公开报告为 reports/2026-11-11-v073-subject-isolated-CMU-human-temporal-negative.md。本地绘图保存在 media/v073_*.png。原始CMU C3D/AMC、模型checkpoint、G1 30–47MB的PhysX JSON、私有权重都不上传GitHub。

## 保留的运行错误与回归保护

第一次CPU模型预检在拟合启动前因 future_frames/forecast_frames 配置字段不一致失败，不曾读取独立人类测试集。代码与配置以独立新commit记录修正，全部划分、模型容量和超参不变。下一次完整训练完成但写出第一个权重时因相对路径/绝对ROOT不兼容失败；部分权重与日志均保留，用同冻结模型源码、参数、种子、绝对唯一输出目录重新完整执行，未覆盖任何原件。

G1投影第一次因为训练输出JSON字段 heldout 写法不匹配提前退出，第二次因为新worktree无被Git忽略的原URDF停止。经原SHA校验后只在独立v73工作树创建指向原本机URDF的符号链接，在新的不重名输出目录重新执行，不改原物理记录、旧模型或其他分支。

全项目第一次CPU测试25项失败经确认是新worktree缺少原始旧实验只读文件。严格参照已通过v71测试的53份原本机忽略资产创建v73本地符号链接（52个文件SHA验证，另1个目录链接），第二次全项目结果776 passed、69 skipped、51 subtests passed、0 failed。完整测试后重新验证52/52原始文件SHA、6/6人体NPZ源SHA、冻结训练源码SHA均完全不变。

## 下一个实质性科研门槛

不能继续在已经看过的唯一被试11上调参后声称新盲测。需要有更多真实优秀人类球员、明确足球脚背触球/球心运动/支撑与真实GRF数据，按独立运动员划分训练/验证/测试，才能研究髋—躯干—双臂—摆腿—随挥的可泛化时序动作生成。GPU Isaac Kit启动/原生碰撞对仍须单独恢复；随后预注册多个新球位/有效随机种子，统一原G1 29DoF实际qdot、合法正脚背≥95%冲量、真实球速、球门、非脚碰撞、14秒恢复的全部物理门禁，并做自然度独立盲评。

本版结论是：跨人泛化失败、G1 IK只是离线参考、本批新增真正PhysX 0次、默认G1 v32/v5不升级。
