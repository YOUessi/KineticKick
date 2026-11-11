# KineticKick v0.1.80｜原29DoF G1 200Hz 原生接触轮询、三条件物理配对与人体数据资格审计

2026-11-11。本批确实执行**6次新增GPU原29DoF G1 PhysX回合**，三个固定源策略分别进行完全相同的观测关闭（off）与现有PhysX接触流只读逐帧轮询（poll），共**3/3组原物理状态完全一致**。四条完整14秒严格通过，另两条相同CMU10_05条件在真实非脚事件后提前终止、严格失败。**尽管失败物理确有25.890153N左腕—左髋反向净力，410次原生contact report轮询仍得到0条headers、0个真实manifold：直接collider pair仍为UNKNOWN，绝非已认证自碰撞或无自碰撞。**

## 0. 源、机器、协议与安全

- GitHub唯一源码：https://github.com/YOUessi/KineticKick；本版独立工作树/分支 `version/v0.1.80`，从干净的v79 `6b1b1a6c731d96722f231ed8274ec85e205b8549` 新建。原主树、其余工作树/dirty分支完全不动，不force push。
- 实验前协议SHA `2898b8aca9c168e87a07967d7ccfce14448ffbf693e1bd3265e8bbf7327e978f`，协议冻结commit `2301a4b93bf21d7a9cdd9456cc95b3f02593dca2`，GPU运行源码冻结commit `1fde4dfa6e32039b8242065ca9853305485d7a43`。runner逐SHA检查原v79输入3条、原v54物理3条及全部源控制器和配置；失败即停止。
- Tang Ubuntu/RTX4090在线，启动时GPU无计算进程，磁盘预检38.25GiB可用（>20GiB）。使用现存全局`flock`独占GPU，全部6个IsaacLab/CUDA PhysX子进程**串行**且watchdog 180s，输出独立文件write-once。
- 原数据/CMU/C3D/29DoF模型、ONNX与历史30MB JSON没有移动、覆写或上传。新实验自身6个原JSON+日志+SHA侧车均保留Tang本地，GitHub只上传协议、程序、测试和文字报告。
- 200Hz即5ms积分，原29DoF自由根G1，固定seed1201，原v32/CMU10_02/CMU10_05三控制、原非脚20N与14秒恢复、脚背>=95%、出球速度>=3.5m/s proxy、原29qdot与球门门禁全部不改。停止/重置产生的+1μs事件记录不冒充额外5ms积分帧。

## 1. 6组真实物理与严格独立门禁

| 原控制器 | 观察器 | 实际真实物理 | 严格全门禁 | 状态 |
|---|---|---|---|---|
| v32 | off | 2800帧，14.000s | PASS | 完整恢复 |
| v32 | poll | 2800帧，14.000s | PASS | 与off全部预注册物理字段逐值一致 |
| CMU10_02 | off | 2800帧，14.000s | PASS | 完整恢复 |
| CMU10_02 | poll | 2800帧，14.000s | PASS | 与off全部预注册物理字段逐值一致 |
| CMU10_05 | off | 1028帧 + 1μs终止证据，物理结束5.140s | FAIL | `physical_nonfoot` |
| CMU10_05 | poll | 1028帧 + 1μs终止证据，物理结束5.140s | FAIL | 与off完全一致，同样`physical_nonfoot` |

新真实G1执行**6/6有可认证原200Hz物理轨迹**，严格门禁**4/6 PASS**、**2/6 FAIL**，三对只读物理零扰动**3/3 PASS**。这是相同seed下三种策略的成对复现实验，**有效独立控制条件仅3个**，不能把4/6包装成未见种子泛化率、训练改进或职业暴力射门成功率。失败组的两个FAIL没有因为1.57s已进球而被改成PASS，真实14秒恢复窗口不存在。

全部六组实际触球t=1.57s，真实记录均覆盖触球前0.4s和后1.25s；四条有完整14秒后恢复，两个5.14s右截尾。逐帧原29关节实测`max|qdot|/limit`均约**0.908129**，严格失败者非脚传感净力峰**25.8901538849N**；v32和CMU10_02全程非脚峰0N。足底法向净力不是人体测力板GRF、双足支撑判据不是CoP/ZMP、人体动力学自然性未从G1零碰撞自动得到认证。

## 2. PhysX真实contact pair：原报告和碰撞力证据严格隔离

此版只增加一个独立执行入口`tools/v080_run_poll_strike.py`：与冻结`v076_run_placement_strike.py`逐行diff，**只有7行新增只读observer挂接**；原控制器源码、策略参数、USD schema与contact threshold均不修改。新`tools/v080_contact_poll.py`通过已有`get_contact_report()`，对4.70–5.50s窗口每个5ms物理步读取单步native contact headers/points，且从未`PhysxContactReportAPI.Apply`、创建新接触view或写机器人状态。

| Native poll实测 | 原v32 | CMU10_02 | CMU10_05失败组 |
|---|---:|---:|---:|
| 5ms轮询次数 | 160 | 160 | 90 |
| 已有原物理Prim上ContactReportAPI | 左腕/左髋均True | 均True | 均True |
| 原生contact headers/points | 0/0 | 0/0 | 0/0 |
| API调用异常 / 记录溢出 | 0/0 | 0/0 | 0/0 |
| off/on物理逐字段一致 | PASS | PASS | PASS |
| 双collider path+point+normal+positive impulse直接证明 | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |

**410次有效poll并不意味着接触不存在**。先前v79的native callback同样没有headers；本次直接查询也没有。尤其在失败组，独立命名传感器仍在t≈5.145s读到左髋净力`[-21.220844,-14.821829,-0.537769]N`和左腕完全相反的向量，单端模长25.890153N，之后`physical_nonfoot`终止。此证据支持“需要检查腕—髋碰撞”，**但净力对称不足以证明两个对应网格就是一对接触collider**。工具的`witness()`要求同一原生header含左右两个独立actor或collider槽位、法向与位置、非零冲量，漏字段/跨槽合并/零冲量/报告溢出全部fail-closed。没有阳性native事件，就保持UNKNOWN；不制造manifold。

NVIDIA官方文档指出现有报告buffer通常仅在物理当前step有效，报告依赖原Prim的`PhysxContactReportAPI`、注册时机与阈值：参考 https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physx/docs/api/python.html 及 https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html 。本批**没有证明**Tang运行时缺失消息的根因，不能用最新SDK不同版本的教程取代本机真值。

## 3. 实测全身动力学、失败图与历史证据保护

`tools/v080_independent_dynamics.py`逐个SHA核对6份原GPU JSON、子进程日志和sidecar，不使用原动作评测的`stride_shot_success`作为唯一成功依据；实际计算29DoF原qdot/限速、COM高度变化、双足载荷占空比、非脚力峰、所有触球相对窗口以及右截尾。独立结果与六条原物理记录保存于本地`reports/v080_independent_real_200hz.audit.json`。新增三个原物理轨迹时间图在`media/v080_original_g1_poll_pair_*_poll_true_fullbody_dynamics.png`，原始大JSON和PNG根据.gitignore不上传。

原CMU10_05的失败窗口**确有左腕/左髋两个命名刚体的25.890153N对向净力**，不能借此证明全身已经跌倒；5.14s后没有任何真实14s恢复采样，所以不补齐。三个原控制条件的非脚事件因果机制仍需原生真实collider/manifold或物理可信的几何contact witness。

## 4. 六段真实CMU C3D再次独立按原字节资格审查

`tools/v080_human_data_qualification.py`直接只读打开本地六份原`CMU soccer 120Hz C3D`并逐SHA登记：目录主体ID10下5段、ID11下1段；每段333个marker、120Hz、均出现三枚命名Soccerball标记；原文件有**0模拟量通道、Force platform USED=0**，没有同步平台GRF或真实脚背冲量；有效帧长度分别801/590/362/436/572/597。marker含少量无效/遮挡帧，已记录比例，禁止插值伪造力学真值。保存本地`reports/v080_original_six_c3d_qualification.audit.json`。

这6段可作为**弱几何/相位探索**，不可当作职业运动员高爆发触球监督。职业身份没有可核验认证，跨主体研究仅具5:1的稀疏目录划分；不能将同目录不同轨迹随机拆成独立训练/测试、也不能声称学到了真实冲量。

## 5. 下一可执行研究方法与门禁（方案，不是已训练成功结果）

真正论文假设为**支撑约束条件化的人类全身时序协同生成**：输入经认证的人体跑动—支撑落脚—髋/腰旋转—摆腿—对侧摆臂—足背触球—随挥/站稳连续片段，以人体动力链（髋和躯干角动量传递、摆臂抵消、支撑脚载荷）建立全身体相位隐变量，而不是6个姿态特征、脚部局部残差或1mrad弱参考。可执行接口应输出29DoF `(q_target, dq_target)`、接触相位/支撑脚约束与反向摆臂目标，原50Hz策略更新/200Hz原PhysX执行，并将支撑多边形和COM可行性、原关节位置/实测qdot/扭矩、当前帧实际collision mesh间隙组成投影与拒绝门禁。已有`tools/stance_safe_temporal_policy_v067.py`实现29DoF分组、限步和无mesh witness拒绝，但**不是已验证的真实形状级安全控制屏障**，本轮在未取得原生碰撞真值前未冒险放开执行。

冻结的下一方法性实验必须进行真实人类数据准入和主体级train/validation/holdout；新seed/新球位提前锁定，比较原v32、整身时序、去对侧摆臂、相位乱序、仅脚部弱参考、反向/错位负控制，所有场景用同一严格脚背>=95%/球门/球速/全部原qdot/非脚20N/14s恢复硬门禁，并独立对自然性、支撑步态、躯干—腿—臂协调、角动量和动作质量进行人体盲评或可信动力学评价。失败样本不作holdout调参。**目前无合格专业人类GRF/触球冲量，且G1原生碰撞几何witness仍缺失，因此不声称新全身学习完成或安全控制已部署。**

## 6. 复现、测试与消极结果

- 可复现源码：`configs/v080_frozen_native_poll_matrix.json`、`tools/v080_native_poll_batch.py`、`tools/v080_run_poll_strike.py`、`tools/v080_contact_poll.py`、`tools/v080_independent_dynamics.py`、`tools/v080_human_data_qualification.py`；本地原始结果需用相应SHA凭证核对，绝不在GitHub上传30MB JSON/NPZ/媒体。
- `env_isaaclab/bin/python -m pytest -q tests`：**853 passed、67 skipped、51 subtests passed、0 failed**；新增16项原生真值拒绝、未见证据/不足冲量/异常数据、预注册SHA与控制器源锁定回归。
- 动力学首次分析遇到原关节速度限值`(1,29)`形状（应展平）而失败；首次成功生成图但JSON写出遇到numpy scalar序列化错误，保留初次/第二次失败日志和未完成audit文件改名记录；第三次成功写出完整独立审计，不覆盖既有原始物理或PNG。人体C3D初读错误把少量遮挡marker当完整文件损坏，第二次允许缺失并记录缺失比例，无物理修改。负例与故障记录未删除。
- 仿真结束GPU空闲，本批严格射门结果**4/6**（三种独立控制条件各一次paired off/on），3/3只读零扰动，全原生wrist—hip collider pair认证**0/3 UNKNOWN**；下一批重点验证本机PhysX contact报告注册时间、配置的报告阈值、真实step buffer生命周期和只读filtered actor-collider查询，不改变原控制器/碰撞过滤，也不把人体弱数据包装成优秀职业球员力学真值。
