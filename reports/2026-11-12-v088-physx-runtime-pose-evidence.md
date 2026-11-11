# KineticKick v0.1.88 — 真实物理接触标定负结果、原G1时序复核

2026-11-12（UTC+8）。基于公开 v0.1.86 commit cb1e63932bafc2b3105cc2552382a9e75a689400；冻结实验源码 commit **54dd223a38e396e95bfe870ec58dc22710103e01**；冻结配置 SHA256 **6ad56c86e71adaf6b8a96f1d00c2a0be4a721399322842b41798c4a1a184cc55**。

## 0. 安全、数据与版本边界

Tang 在线，初查 RTX 4090 无 CUDA compute 进程，可用磁盘约 37GiB。创建独立干净的 version/v0.1.88 worktree；其他 v65–v87 工作树、原CMU/C3D、权重、原G1 PhysX轨迹未覆盖、未删除。全局 GPU fcntl 互斥与磁盘≥20GiB保护已写入批处理器。原始运行 JSON/日志和失败图按 gitignore 留在 Tang。

## A. 冻结后实际执行的独立 PhysX 正控制

冻结四种独立 1kg 球体—地面测试：CPU/wildcard、GPU/wildcard、CPU/authored_empty、GPU/authored_empty，每组200Hz、180步、独立进程、130s watchdog。**这不是原29DoF G1踢球仿真。**

实际执行第一个 CPU/wildcard：180个真实物理step、OS返回码0、历时5.18秒。与v87仅观察USD静态变换不同，这次读取 IsaacLab RigidObject 的 PhysX runtime tensor。球中心高度 **0.849755m → 0.200000m**；第60步竖直速度约 **-2.943m/s**（符合自由落体）；第80步后速度近0且球停在地面上。这是实质的球体碰地动力学证据，明确修正了此前误用USD静态变换的解释。

同时发现两个独立的仪表故障：指定地面过滤的 PhysX rigid-contact tensor view 最大接触力 **0N**、180步内非零接触帧 **0**，尽管动力学物体已经落地；原生 PhysX callback **0**、poll **180**、native header **0**。零观测不能证明没有碰撞，当前 ground filter / native reporting 尚未校准。进程写出 V088_APP_CLOSE_ENTER，却没有写出 V088_APP_CLOSED；虽然子进程OS code为0、watchdog未触发，冻结协议规定必须看到后者才能认证生命周期。

**按预注册门禁：认证独立PhysX测试0次、未认证独立尝试1次、原生接触正例0次，其余3种冻结配置未执行。** 不在运行后擅自更改 force 阈值或退出判断、没有偷偷重试。左腕—左髋 collider pair / contact manifold / contact normal / impulse **仍未取得直接真值**。历史相反方向的25.890153N净力不能冒充碰撞双方证据。

原始未评分运行、日志、逐例审计、总审计均是 Tang 本地 write-once：
- reports/v088_cpu_wildcard.json（SHA256 5ba512084c4ed7d23d5a7f255ca713e553ec39ff577d937afac537465a438c37）
- reports/v088_cpu_wildcard.log / .audit.json
- reports/v088_batch.audit.json

## B. 独立复核原 G1 200Hz 全身动力学

从 v85 原始只读 PhysX 记录重新执行其 SHA 对照审计（v85 原始 batch audit SHA256：b779c351b43764140e13821a605d2d3ff02f4f18abb8e4837e66791ee8d9829e）。三种不同控制条件的真实物理帧数分别为 v32 **2800**、CMU10_02 **2800**、CMU10_05 **1028**；CMU10_05 另有1条终止注释，**不是新物理步**。三个条件均具有真实触球前0.4秒至后1.25秒窗口；CMU10_05 在5.135秒出现≥20N非脚事件，没有14秒恢复数据。

以原G1实测脚—球接触时刻为0，三个控制条件的关节速度组峰值完全相同：支撑左腿 **-0.175s**、右摆腿 **-0.040s**、腰躯干 **-0.215s**、左对侧摆臂 **-0.225s**、右臂 **-0.400s**。这些只能叫机器人关节运动学代理，**不能宣称真人GRF、真实电机功率、近端到远端能量传递或职业球员自然性已认证**。目前既有微弱CMU参考没有可验证地改进相位顺序。

复核输出 reports/v088_v85_three_policy_reaudit.json（SHA256 c71ea1079d12354e9bd5ff1eab21c23ba93764a7d819491947351b09055ceefc）。失败对比图 media/v088_200hz_original_G1_failure_comparison.png（SHA256 3d5f304f032b07eb56ce1b10525773efff29caea71aa26b9bb20f457b35a958e）仍仅本地保存。

## C/D. 新控制研究与严谨门禁

后续要将支撑状态、骨盆、躯干、摆腿、对侧摆臂及随挥落地同时纳入29DoF训练/控制接口，必须有真实碰撞间距约束、全关节速度保护和可靠的接触报告资格。v88本身**没有修改原控制器、没有训练新的真人爆发抽射策略、没有运行任何新的原G1 GPU射门回合**，因此新射门成功率**不可计算**；历史v85的4/6只是历史诊断场景，不是当前模型的泛化指标。保持正脚背冲量≥95%、球门、球速、29DoF实际qdot、非脚碰撞、完整14秒恢复的所有原门禁。

现有六段CMU 120Hz光学几何仍缺少可核验的同步专业射门GRF和真实正脚背球冲量监督，不能当成优秀职业球员的动力学真值。

## E. 实现、回归、失败透明性

新增约336行冻结代码、协议、配置和测试，独立的PhysX runtime pose / contact tensor / native reporter多通道采样器，write-once四组批处理、GPU互斥及fail-closed校验；**8/8针对性测试通过**。完整回归首次因新worktree缺少本地测试原始数据失败；随后仅在新worktree建立指向先前存在的原始JSON/URDF/所需旧日志的符号链接，没有覆盖原文件，并保留所有初次失败日志和测试重试。最终结果：**987 passed、6 skipped、69 subtests passed、0 failed**，见本地 reports/v088_full_suite_retry4.log。

## 后续依赖

下一独立协议应做「真实PhysX地面存在/不存在」配对，检验基于runtime pose+速度的碰撞正控制，并单独排查 ground filter 为什么输出0N、app.close为何缺乏返回标记。只有独立标定原生actor/collider路径后，才能回到**不改变G1控制器**的腕—髋接触取证及off/on无扰动物理一致性实验。真正的自然高速全身协调学习与跨球位泛化仍未完成；全部阴性证据必须保留。
