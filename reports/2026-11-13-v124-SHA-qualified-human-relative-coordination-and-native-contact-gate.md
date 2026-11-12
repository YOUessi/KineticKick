# KineticKick v0.1.124 — 冻结人类运动学时序与原生接触真值门禁（CPU有界研发批次）

执行跨 2026-11-12 至 2026-11-13（Asia/Shanghai，Tang 本地）。父提交 `b4da3f1a153a487962b68ddb5e8286a46e100538`；事先独立冻结的研究协议提交 `ef3419db5bf8713f3fd73276c093685bde135504`。**本批没有生成或驱动新的人体式 G1 动作策略，也没有新增 G1 或隔离 PhysX 物理回合；GPU 被已有 v120 全项目 pytest 任务占用，不抢占，不终止。** 这不是自然高速抽射成功报告。

## 1. 安全与完整性起点
- Tang 在线；根盘 916G，剩余约 31GiB（97% 已用，仍 >20GiB）；运行中的原 v120 进程 PID 242644，`/home/you/conda_env/env_isaaclab/bin/python -m pytest -q`，已运行接近3小时，在 GPU 有260MiB占用。严格不启动Isaac/PhysX或第二GPU任务。
- `version/v0.1.123` 工作树起初干净，远端下一 `version/v0.1.124` 尚未存在。创建全新 `/home/you/projects/kinetickick-wt-v124`，不更改基础 `version/v0.1.64`、已有 v120/v121/v122/v123 工作树、实验进程或源数据。
- 协议 `docs/V124_FROZEN_HUMAN_COORDINATION_AND_CONTACT_EVIDENCE.md` 和 `configs/v124_frozen_cpu.json` 先行 commit + push，然后实现/运行审计。预设的120Hz、6 C3D、12 marker、0.35s lag、45有效间隔、零方差拒绝、2名被试不满足3组subject-disjoint的事实不可因结果调参。

## 2. 真正的真人数据资格：6份C3D，2被试，3,358帧
新工具 `tools/v124_human_relative_coordination.py` 在原 v116 SHA
`402b87f019dd7cd31f9d5af980ad38a2b63a7e3f` 的本地只读原 C3D 目录，逐件对原版 `v116_human_qualification.audit.json` 核对 SHA256；同时固定了该资格清单 SHA256
`b1c2116948a227ba4341c1d7aa6a93f0c5eebce3288f999a86457d7c6b19a62b`，禁止清单与原始文件一起改写造成无感漂移。实际校验6/6原始C3D及6/6来源，记录120Hz、毫米坐标、同名marker唯一、各帧残差与缺测掩码。12类身体marker在本次所用原数据均有100%有效覆盖（仅描述该批，不推广到别的数据）。

提取的**不可随意外推**的观测：肩带横向宽度、骨盆横向宽度、二者3D夹角余弦、左右腕相对于各自肩/左右足趾相对于骨盆中心的径向距离、肩宽归一化距离、相邻有效帧的绝对径向距离变化速率。它们在同一帧下的全局平移/任意3D刚性旋转下保持不变，但**径向伸缩速度不等于完整3D关节角速度/脚速，也不是肌肉功**。不假定世界z轴已经与足球场竖直方向校准。

| 真实C3D | 帧数 | 左腕—右趾径向变化相关最佳lag（秒） | 右腕—左趾最佳lag（秒） |
|---|---:|---:|---:|
| 10_01 | 801 | -0.233 | +0.042 |
| 10_02 | 590 | -0.333 | +0.050 |
| 10_03 | 362 | -0.350 | -0.050 |
| 10_05 | 436 | -0.350 | -0.058 |
| 10_06 | 572 | -0.350 | +0.058 |
| 11_01 | 597 | -0.350 | -0.100 |

lag 定义为 `corr(wrist[t], opposite_toe[t + lag])`，这是**探索性幅值相关的最大值位置**，不是高水平球员生物学发力先后顺序。左腕—右趾有 **4/6 卡在-0.35s预定窗口边界**，信号的实际最佳时间差可能根本不在搜索窗口内；右腕—左趾有3个正lag和3个负lag，不提供一致方向。双信号同时反转时间序列后，lag正负反转而相关峰强度基本不变；说明**仅用峰值相关幅度无法认证真实因果方向**。绝不以此启动后训练或作为射门接触标签。

全部6段没有原生力通道。原样保留：`professional/elite identity=unknown`、`true foot-ball impact time=missing`、`GRF/CoP=missing`、`true instep impulse=missing`、`kicking-foot verified=missing`。只有2个被试，三个非空且互不重叠的train/validation/test被试划分**数学上不可能**，不能按帧切分同一个人后宣称独立泛化。基于此，**真实人体高水平全身踢球模型未达到可训练并有效验证的门禁**。准许后续将这些材料用于标注任务和运动学方法开发，不能称为已学会职业爆发力。

完整本地证据：`reports/v124_human_sha_pinned.audit.json`，SHA256
`a3a47e9ea16c65803696e4b4d7ab746afb6f7830899e14767a8388429704da46`（Git忽略，仅Tang留存）。

## 3. PhysX contact 原生真实pair：重新认证6次历史失败，不制造成功
新工具 `tools/v124_native_contact_truth_gate.py` 静态复核 v119 Git
`b57b855433b3fa6f6f266ee565f7e959b333c288`、原始master SHA256
`691ac065ccb6eaaef648b7c5a08ad44f79cc0bc3b572a8a10df8a96a399cd380`，并分别逐文件重新计算 **6份真实源JSON和6份日志SHA**。所涉及历史（并非本轮新实验）共6次隔离CPU/CUDA PhysX、1,080个实际200Hz步。

- CPU positive OFF/ON、CUDA positive OFF/ON：真实碰撞动力学存在，不能将零native header误解释成无接触。
- CPU/CUDA separated ON：作为原先独立的非碰撞负例。
- 各后端ON场景均在rigid body prim上标记 `PhysxContactReportAPI`；实际collider为其子USD prim；这些事实说明代码有接触报告配置，**但仍未获得任何原生actor0/actor1/collider0/collider1/manifold header**。下一次应逐步检查真实binding、运行时初始化与回调步间缓冲。
- 新schema门禁要求真实2个不同的完整actor路径、对应collider路径、可独立验证的USD collider所有权、带物理步和时间戳的点/法向/非零冲量、原始SHA；tensor力和仅净力相反不算。**即使构造了通过schema的合成报文，也强制 `native_pair_qualified=False`，直到真实PhysX运行源与读出事件被独立认证**。

因此左腕—左髋接触双方仍是 UNKNOWN，不能根据v54左右反向25.89N净力推断真值，也不能排除其它自碰撞。绝不把“负报告”当“证明无接触”。

本地最终源认证审计：`reports/v124_native_sha_pinned.audit.json`，SHA256
`8f8728ec6879467bbff39e4e3c025aa0cf159064481503cdb0c43418e0cd53bd`。

## 4. 可复现的程序、图与负结果
本轮实际新增3个可运行CPU脚本（人体相对运动学、原生接触严格schema/旧真值复核、两张真实统计图），以及2个测试文件；图在本地 `reports/v124_human_relative_lag_measured.png`、`reports/v124_native_contact_missing_truth.png`，不会提交媒体至GitHub。内容包含真实源SHA、rigid变换不变性、marker残差缺测、时间延迟符号与反转负控制、false source、actor/collider路径归属、错误接触点和非零冲量的否定测试。

最终相关CPU回归（v116—v124选择的测试集合）：
**149 passed，0 failed**。首次运行9项失败，均因为新工作树缺少Git忽略的原始C3D/权重/v117或v119实验文件；只建立指向未动原资料的本地忽略路径/符号链接后再次测试剩1项模型资源缺失；增加只读 `.pretrained_checkpoints` 引用后全部通过。初次Native主程序曾在写好不可覆盖审计文件后因CLI打印参数错误退出，保留首次输出，修复后写入新文件名，不覆盖原文件。随后额外加入双方源master SHA pin，保留旧输出，再用新的写一次文件名重审。这些都是代码/环境故障与修复，不是额外物理回合。

未运行全项目GPU相关完整回归，不声称“全套1420+通过”；已有v120进程持续占用GPU与CPU，源数据与工作树保持隔离。

## 5. 对论文、全身生成的直接决策
重要结论：真实C3D可用于人体相对运动学与潜在跨肢体时序描述，但当前搜索窗口、身体标记、被试数量**不支持**可认证的人体完整足球动力链、职业速度或独立训练泛化。原 v32/CMU 的物理历史有真实短时控制效应、历史独立条件有严格PASS与FAIL；本轮**新增原29DoF G1正脚背物理射门0次，新增PhysX隔离验证0次，新增human-conditioned motor action0次**。不把KineticChain-29当前论文合同或离线特征包装成已执行的全身动作策略。

下一次GPU真正空闲后：先单独冻结 **原生contact-positive/negative** 校准脚本；验证真正rigid body上的report schema、child collider所有权、订阅存活、callback与每步poll采样窗口；取得actor0/actor1/collider0/collider1和逐点法向与冲量的正控制后，对原G1左腕/左髋与受控相隔负对照进行完整OFF/ON。与此同时建立至少足以分离训练/验证/测试的人体来源，明确专业/脚球接触/GRF标签实际是否取得；只有支持和碰撞认证通过，才能把全身相位规划器接到原29DoF真实电机安全接口，然后按新冻结未知球位/seed跑合法脚背、球门、速度、原29qdot、非脚碰撞、14秒恢复的严格联合门禁。该任务仍未完成。
