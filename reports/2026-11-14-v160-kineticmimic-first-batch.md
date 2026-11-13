# v0.1.160 · KineticMimic 首批物理跟踪与人体动作实验

2026 年 11 月 14 日，Tang。基于 v0.1.159 的研究路线 `cb31d1136ce88b98330f30f70bc2f718aeea6220`。本批有 1 个完整的自由根 G1 CPU 物理回合、8 次刚体接触对照、6 段真人动作的全身重定向，以及两种数据源各 3 个种子的成对模型训练。

一条既有 PAiD 策略在 CPU 物理回合中用右脚将球踢进门。末段加载后水平球速 4.709 m/s，球心过线高度 0.110 m。机器人随后跌倒，严格射门与恢复验收 0/1。两组新训练的条件 VAE 均未通过预先规定的比较。新模型物理执行数为 0。

## 一次完整的跟踪与射门回合

- 在读分前指定 PAiD 标准库排序第一条右脚动作 `soccer-standard-001_right`、已有任务的球位和球门、14 秒物理时长。先核验全部 13 段动作（10 条标准、3 条风格化）共 3,193 帧。原 URDF 对具名 29 关节与 30 个刚体做逐帧正向运动学，位置最大差 0.71 微米，旋转最大差 2.45 微弧度。PAiD 源版本和每个文件 SHA 已冻结。[跟踪协议](../configs/v160_tracking_protocol.json) · [上游运行器](https://github.com/TeleHuman/HumanoidSoccer/tree/e72e470230047dedaf66df0983f1d0ab746faeb5/exp)
- SAPIEN 3.0.3 `PhysxCpuSystem` 载入原已解析 URDF，沿用原导入器 28 个圆柱转胶囊的约定和 1 个球碰撞体。物理模型有 40 个链节，29 个可动关节，总质量 33.341 kg。原版 PAiD ONNX 每 20 ms 输出关节指令。物理解算每 5 ms 保留根位姿、6 维根速度、29 关节状态、30 个刚体的位姿、足球运动和原生接触流形。初始化后没有写机器人或足球状态。足地接触用于测量，策略的 160 维输入没有直接的足底力通道。
- 源参考覆盖 0–5.18 s。期间关节角 RMSE 0.09038 rad，关节速度 RMSE 0.72924 rad/s，14 个跟踪刚体去除各自骨盆姿态后的平均位置误差 2.42 cm，世界坐标根位置 RMSE 25.66 cm。该时段关节位置越限数与超过 20 N 的非足接触时步数均为 0。2,800 个实际物理时步有完整记录。独立核算 700 个观测的最大差 2.68×10⁻⁷，重放的全部 ONNX 动作和换算后指令与记录逐值一致。[跟踪实现](../tools/v160_tracking_cpu.py) · [时钟与输出审计](../tools/v160_tracking_independent.py)
- 右脚第 7 个胶囊（0 起算索引 6）与球有两段超过 2 N 的加载。2.335 s 为单步加载，2.340 s 水平球速 2.992 m/s。2.365–2.420 s 再加载 12 步，2.425 s 水平球速 4.709 m/s。2 N 是该形状对所有点所报告的法向冲量向量之和的模除以 5 ms。低于阈值时，两表面仍可能几何接触。原冻结读数保留首个区间。事后分段脚本读取同一原始记录，报告每个区间。球心 2.94939 s 越过门平面，整球后缘 2.97963 s 越线，后者的球心距门中线 +0.27469 m。[分段与过线计算](../tools/v160_tracking_ball_outcomes.py)
- 5.18 s 后，控制器保持参考末帧且继续读取真实球位。9.005 s，左腕 `left_wrist_yaw_link` 与左髋 `left_hip_roll_link` 的胶囊各自索引 0 发生碰撞，单点冲量模/5 ms 为 69.92 N。9.705 s 骨盆低于 0.45 m，9.725 s 倾角超过 60°。末段连续站稳 0 s。关节速度全程没有越限，位置首次越限发生在 9.855 s，累计 699 个时步×关节单元超出预设 10⁻⁶ rad 容差，最大超出 0.0006773 rad。左、右足承重时的接触点切向滑速 RMS 分别为 0.197、0.224 m/s。腕髋碰撞的先后关系有原生证据，引起失衡的控制因素仍需配对试验。
- 上游 MuJoCo 入口首拍参考为零，随后使用上一拍参考，并显式限幅力矩。本移植沿用项目 Isaac 入口的当前参考帧和有力矩上限的位置驱动。本回合未直接测量执行器力矩。足表面比例尚缺本后端认证，原 Isaac CUDA 环境中历史 v54 的腕髋接触对仍待在该环境验证。

### 同一物理记录的触球面与冲量复算

在全部 13 个有载时步内，实际接触均只记录到一个足—球点。13 个报告点均与步前球心沿法线投影的球面点重合，最大误差 5.32×10⁻⁸ m。用原始报告点或按有符号间距投影的足侧点，各配合步前或步末的实测足姿，将位置和外向法线转入足坐标后调用项目已有的 `paid_contact_patch`。四种约定均将 13/13 个点判为脚内侧，正脚背的法向冲量代理份额为 0%。这一结果用于诊断当前 CPU 回合，后端的解剖足面认证仍开放。首次接触点的报告间距为 +8.455 mm。用投影到足侧的点，在步前足姿中定位对应材料点，借触球前 2.325→2.330 s 两个实测足姿估算该材料点速度为 (−0.14754, −3.53880, +0.03394) m/s，模为 3.542 m/s。其它取点与足姿约定得到的两个读数为 3.532 和 3.543 m/s。这个 5 ms 有限差分没有同点瞬时速度真值。

本机 SAPIEN 使用 PhysX `PxContactPair::extractContacts`。实现把每个法向标量乘以接触法线，导出世界 xyz 下的**法向冲量**，没有返回摩擦锚点切向量。初版 `tracking_result.json` 的一个说明字段误称包含切向贡献。保留原始文件与 SHA，在发布代码和本节纠正说明。阈值和所有原始向量保持原定义。左右脚及球的垂直夹具在此接口下基本闭合，高动态斜碰可出现残差。[原足面分类器](../tools/free_shooting_metrics.py) · [PhysX 接触读取接口](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxContactPair.html)

足球所受脚部法向冲量在首段为 (+0.07725, −1.28408, +0.41481) N·s，第二段为 (−0.01721, −0.79824, +0.63648) N·s。从第一次加载前一帧到最后一次加载后，含中间低于 2 N 的时步，足部累积为 (+0.06003, −2.08232, +1.05129) N·s。该窗口球—地的报告法向冲量为零，重力冲量为 (0, 0, −0.379647) N·s。用足球质量乘起止速度差独立对账，首段残差模 1.36×10⁻⁷ N·s，第二段及整窗约 0.07327 N·s。无法把此残差唯一分摊到脚、地面摩擦或其他未输出项。此项属于事后分析，保留原预注册结论。[冲量、足面与材料点复算工具](../tools/v160_tracking_ball_impulse.py)

## 接触读数的独立校准

原 G1 左、右脚各用真实质量、惯量与 7 个胶囊。释放左脚、释放右脚、球落在固定右脚上、球与右脚横向错开四个条件，分别开关原生接触读数，每次 120 个真实 5 ms 物理步。480 对时步的前后 13 维姿态和速度逐值相同。左右脚各有 95 个加载步，落在脚上的球有 50 个，错开条件下脚球间原生点为 0、保守净空为 0.304 m。另一程序只读每个原生点，依照刚体顺序加带符号的法向冲量，再与质量乘速度差扣除重力冲量比较。最大单步残差 0.00010238 N·s，冻结门槛为 0.002 N·s。[接触协议](../configs/v160_cpu_contact.json) · [原生点记录器](../tools/v160_runtime_contact.py)

首次运行已完成两只脚的四条开关记录，随后 SAPIEN 拒绝给静态支架设置惯量。修正静态初始化后沿用前四份记录的 SHA，仅运行其余条件。一个 NumPy 布尔类型导致首次汇总序列化失败，最终汇总直接从八份原始记录重算。没有为生成汇总重采物理步。

本次 14 秒预览见 Tang 的 `media/v160_paid_tracking_reference_vs_physics.mp4`，SHA-256 为 `0636ce23f7962f40655facaffb168ffe56f9bd7ea6e251cb00e60cadabe7a077`。左侧 50 Hz 参考在 5.18 s 后保持最后一帧，右侧从同一回合的 200 Hz 状态渲染 35 个原 G1 网格和实测足球，持续到倒地终态。视频为 H.264、1280×720、25 fps、350 帧。[离线回放工具](../tools/v160_tracking_preview.py)

## 六段真人全身动作

使用两名 CMU 被试的原始 C3D，27 个标记构造骨盆、躯干、手臂、双腿与手足的 18 个三维方向。按人体肢段比例估计尺度，以原 G1 URDF 带关节边界和速度边界作逆运动学。每段保留首尾、原 120 Hz 时间和缺测掩码。3,358 个源帧共生成 1,401 个 50 Hz G1 参考帧。训练为 `10_01/02/03/05`，验证为同一人的 `10_06`，测试为另一人的 `11_01`。每个片段 29 个关节的幅度均超过 5°。按帧加权，拟合的 18 个目标方向平均误差为 1.483°，同骨盆中立姿态为 37.076°。该误差衡量对 IK 目标的拟合，尚缺独立的全身姿态真值。[输入、目标与边界](../configs/v160_human_reference.json) · [CMU Graphics Lab 数据库](https://mocap.cs.cmu.edu/)

关节位置都在 URDF 限内，相邻帧关节速度峰值为额定上限的 98%。每段开头用一个常量注册地面，六段均出现足底胶囊低于此平面超过 1 cm，共 202 帧。最深为验证段 `10_06` 的 4.174 cm。各段在源标记低且慢的时步里，踝链节水平速度的 95 分位为 0.128–0.296 m/s。该指标依赖几何代理，人体实际足底滑速、地面反力、精确触球、分期和职业资质均缺测。两段较连续的出球标记 `10_02/10_05` 来自同一人。六段新参考的 G1 物理跟踪尚待验证。

此次查阅的补充来源如下。

- OmniContact 公布前、左、右三个踢球方向各 30 个原始捕捉，提供配对的 G1 参考、人体 BVH 和物体轨迹。数据卡为 CC BY-NC 4.0，文件访问需要账号持有人同意共享 Hugging Face 账户信息。本批只核查公开数据卡，没有将这些片段计入训练或独立人数。运动员水平、精确击球面和同步实测力仍待核实。[官方数据卡](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)
- Zhang 等的正脚背试验报告了 14 名足球专业学生、32 名初学者的髋膝踝协调。作者说明原数据可按请求提供。本批使用其研究设计作方法参考，未取得原始试次。[原论文及数据可得性说明](https://www.mdpi.com/2306-5354/12/11/1151)
- SoccerBot 已公开以人体足球动作的隐空间和分层强化学习驱动 G1 的研究，是后续评估相关方法的直接比较对象。本批对其项目说明作了核查，未执行其策略。[作者项目页](https://soccerbot-humanoid.github.io/)

## 行为隐空间的首轮对照

条件 VAE 从过去 8 帧状态预测未来 10 帧的 29 个关节和骨盆平移/旋转，预测时只使用条件先验。输入还允许 PAiD 已标的踢球侧，CMU 侧别输入为空。两次实验分别训练模型，每种数据源固定 3 个种子，每模型每种子 180 个 Adam 更新。CVAE 共 74,174 参数，推理使用 46,574。确定性 MLP 共 74,411 参数。首要指标在每条完整测试片段内求关节绝对误差，再对片段与种子等权平均。[冻结划分与比较规则](../configs/v160_latent_trial.json) · [训练与评估实现](../tools/v160_latent_trial.py)

- PAiD 训练/验证/测试为 8/2/3 条完整片段，来源球员身份未知。测试 MAE 依次为 CVAE **0.037231**、MLP **0.037447**、保持末帧 **0.042865**、末速度外推 **0.048853** rad。CVAE 仅在 2/3 种子优于 MLP。置零隐变量为 0.038495，片内轮换隐变量为 0.042331。相对同一目标的新增关节位置越限为 CVAE 14、MLP 31 个，共 54,810 个种子×目标时刻×关节单元。
- CMU 分别有 424/23/24 个训练/验证/测试窗口。测试只有被试 11 的一个片段。CVAE **0.079735**、MLP **0.094454**、保持末帧 **0.074083**、末速度外推 **0.081672** rad。三个种子上保持末帧均优于 CVAE。置零隐变量为 0.078851，片内轮换为 0.081948。20,880 个单元中，CVAE 有 156 个新增位置越限、0 个新增速度越限，MLP 为 74 和 3。被试 11 与本次训练隔离，项目 v73 已评估过此人，不能算全项目首次盲测。[v73 的历史划分](2026-11-11-v073-subject-isolated-CMU-human-temporal-negative.md)
- 双方训练、验证和测试的单位都是完整片段。窗口数、关节数、随机种子均不能增加独立被试数。两组均未满足「每个种子同时优于 MLP、末帧和等速外推」的预注册标准。重载已保存权重，首个测试窗口预测复算最大差小于 9×10⁻⁸。实时支撑/球位/目标条件化和生成动作的物理执行仍待实施。

## 证据与复现

- 协议及实现冻结点为 W0/W3 `0f78838989857d991ae84f6f38e51635bc30f4f0`，W0 静态夹具修正 `bfe5dd9a1272f6cae853b5a6903478f58b942d8f`，W2 原始 C3D 重定向 `1a7649698ec81070323f91add21622c6bb099dc7`，W1 物理回合 `98f9f34b377cce52c6c337edebd293c0b84a18e3`。后登记的只读审计与预览使用已封存的记录。
- Tang 在 06:09（UTC+8）读取到 RTX 4090 Laptop 总显存 16,376 MiB。旧 v120 `pytest` PID 242644 持有 260 MiB 的 GPU 计算上下文，81 线程，1.10 秒内 CPU 计时增加 115 tick。现存日志为空，系统拒绝读取内核栈，无法判断具体测试进度。磁盘可用 30,667,804,672 字节。本批沿用项目的 GPU 单进程约定，保留旧进程。
- Tang 的本批记录在工作树 `/home/you/.codex/worktrees/kineticmimic-first-batch/kinetickick` 的 `experiments/v160_runtime`、`experiments/v160_tracking/official_001_cpu`、`experiments/v160_human`、`experiments/v160_latent/{paid,cmu}`。原机器人、参考数据和权重读取自 `/home/you/projects/kinetickick` 下的本地归档。实验文件和预览受 `.gitignore` 管理。

主要记录的 SHA-256 如下，可在 Tang 直接核对。

```text
9c9135dfb9b361707346cec7ca164bb869ce73dcc0b056accad1cae158492e06  v160_runtime/device_snapshot.json
f0ea87929578089cb358b25c590e3b228e7c76ed3961b35e525e6d509d0443f2  v160_runtime/independent_summary/native_raw_rebuild.json
341cae3ea15b905df337525e8a6e747562715fe21d71acc02f329d7a4e0118c9  v160_tracking/official_001_cpu/source_inventory.json
fd59cfa12ee67561f1c0eea5a8cbe895031cd82368bf1b4711e276db5aaf0ba4  v160_tracking/official_001_cpu/rollout.npz
3280621a0b0c4c1dfdeb374f1bce1da69cf827500ef3b2e7db330f103c309ad2  v160_tracking/official_001_cpu/native_contacts.jsonl.gz
f73d4430290dbd467f7bef75d066433fd2363cf198fb1e183c805da39120df58  v160_tracking/official_001_cpu/tracking_result.json
26f07966d610a295e49479ebbf2d24ad48e0e3ed3fe1d1d09e82b1e5a178f8b8  v160_tracking/official_001_cpu/independent_audit.json
054ace4b8b4bd9deed65a362f4851aa22e18c351c70faf1ae3f444f5759ab61a  v160_tracking/official_001_cpu/ball_outcomes.json
d8e437bf1d3aa2220fef49226c0fc45ad2af913e524a6270998fac42fbf6b8b9  v160_tracking/official_001_cpu/ball_impulse_posthoc.json
6900628d49f0701159315acafbca7473016515a5942be145dcdd3d9fd6532f4f  v160_human/manifest.json
037dde92adb336814f076ac6e961e23ae327609af9c3730e36673cdfc648ea38  v160_latent/paid/results.json
22e2b4d926358f39bb31149ad6f21cb24071e9ab5b7024345b16c9060424dec8  v160_latent/cmu/results.json
```

在带 Tang 本地资产的同版本新检出目录中复跑。各工具要求输出目录尚不存在，CMU 预测器读取本检出下的 `experiments/v160_human`。

```bash
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
physics_py=/home/you/conda_env/maniskill/bin/python
learning_py=/home/you/conda_env/env_isaaclab/bin/python
lock_sha=0f78838989857d991ae84f6f38e51635bc30f4f0
"$physics_py" tools/v160_runtime_contact.py --output experiments/v160_reproduction/contact
"$physics_py" tools/v160_tracking_cpu.py --output experiments/v160_reproduction/tracking
"$learning_py" -s tools/v160_tracking_independent.py --run experiments/v160_reproduction/tracking --assets /home/you/projects/kinetickick --output experiments/v160_reproduction/tracking/independent_audit.json
/usr/bin/python3 tools/v160_tracking_preview.py --run experiments/v160_reproduction/tracking --output media/v160_reproduction.mp4
"$learning_py" -s tools/v160_human_reference.py --output experiments/v160_human
"$learning_py" -s tools/v160_latent_trial.py --source paid --frozen-commit "$lock_sha" --output experiments/v160_reproduction/latent_paid
"$learning_py" -s tools/v160_latent_trial.py --source cmu --frozen-commit "$lock_sha" --output experiments/v160_reproduction/latent_cmu
```

专项与必要回归共 **26 项通过**。测试覆盖接触对顺序、空载、测量时钟与旋转坐标、人体方向雅可比、时间掩码、预测因果性与集合隔离、冲量及球门几何。没有运行包含 Isaac 启动的全项目测试。

下一批最能区分原因的两项实验是在冻结基线上只改变末段站立参考交接，同时保持真实球位输入。另一项是在训练/验证动作上加入足底支撑几何约束，测量穿地、滑移与全身拟合误差的变化。原 Isaac 运行窗口与碰撞对认证、更多独立球员及持出的未见球位，决定后续对真实正脚背与跨情景泛化的评价范围。
