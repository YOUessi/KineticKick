# KineticKick v0.1.126 — 原29DoF G1腕髋碰撞静态资格 × 八球位成对精确统计反证

**执行**：2026-11-13 01:46–02:00（Tang/Asia-Singapore）；父分支 v125 `e1b67700ca922bf37116d1c8a8527c15c11e2aff`。GPU受既有 v120 pytest 占用，没有新真实物理实验：**新增G1 PhysX 0、隔离PhysX 0、新学习策略电机驱动0、未测得新物理成功率**。

## A. 机器与证据保护

- Tang在线；主项目根目录 `/home/you/projects/kinetickick` 处于既有 `version/v0.1.64` 且有未跟踪 `=100===` 文件，完全未触碰。v125来源工作树干净；新建独立 `version/v0.1.126` 与 `/home/you/projects/kinetickick-wt-v126`。未覆盖其他版本、原始CMU/C3D、源URDF、模型权重或PhysX JSON。
- 磁盘剩余约30GiB（未跌破20GiB）。GPU已有 `PID 242644`、`/home/you/conda_env/env_isaaclab/bin/python -m pytest -q`，已连续运行约4小时50分钟、260MiB显存占用、CPU约103%，v120日志仍为空。未抢占、停止或并行启动新IsaacLab/PhysX。
- 正式分析前独立push预冻结协议 SHA `69b3d1718b92d9755d45a3900b9882d40256ece9`。首次CPU测试暴露冻结config中的历史证据SHA手工转录时截短成54位：测试 **21 pass，4 fail**。这不是物理负结果。**在有效源数据分析前**保留可追溯修正提交 `8e489a00506c06441cf30885ab6a71578e8da261`（原始证据SHA完整64位），推送校正协议；代码固定读取修正SHA，不能静默偏离。校正后25/25专项全部通过。
- 真实历史证据 `/home/you/projects/kinetickick-wt-v125/reports/v125_original_g1_timeline_sha.audit.json`：SHA256 `2a4348cd0c5bfe3878c255f23d181d1cf224a32d5a4df5cc83a2008eac9fc8a3`；v125来源HEAD `e1b6770...` 校验通过。该证据此前独立校验了29个真实原G1 CUDA PhysX回合、81,200帧200Hz数据，含24独立条件和5次只读观察器重复。**v126复核的是这份来源校验的摘要，不把历史29次重复当作本批新物理，也不声称在v126重新跑了29次PhysX。**

## B. 新结果一：腕—髋存在原始URDF碰撞几何，但原生接触仍为 UNKNOWN

首次对原 v116 真实29DoF **PAID resolved URDF** 用SHA锁定后进行只读树遍历（HEAD `402b87f019dd7cd31f9d5af980ad38a2b63a7e3f`，URDF SHA256 `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`）：

| 冻结来源核查 | 实测静态结果 |
| --- | --- |
| Links / joints / actuated | **40 / 39 / 29** |
| `left_wrist_yaw_link` | 1 cylinder；半径0.05m、长0.05m；碰撞形状本地偏移 `(0.075,0,0)` |
| `left_hip_roll_link` | 1 cylinder；半径0.05m、长0.20m；碰撞形状本地偏移 `(0.02,0,-0.10)` |
| 两目标连通拓扑 | **12条关节边**，经手腕、肘、肩、腰到髋，是不同的非邻接刚体链 |
| URDF `disable_collisions` | 0项 |
| 原 `robots/g1.py` 只读配置 | `enabled_self_collisions=True` |

**解释**：原始URDF并非一侧根本没有碰撞形状；模型代码明确请求自身碰撞。这使5.145秒腕髋25.890153N等幅反向净力所提示的自碰撞假说仍然合理、值得继续调查。**但** URDF静态碰撞体不是导入USD后的确切collider路径；没有证明该Pair在动态场景未被过滤、没有取得actor/collider同一接触manifold，也没有取得contact point、normal、impulse的原生报告。因此不是正面认证，`native_physx_contact_pair_found=false`，准确结论仍是 **UNKNOWN**，也不能由没有header证明不存在接触。

相关接口依据：NVIDIA Omni Physics文档 `https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html`。未来应以原生ContactReportAPI在真实刚体上绑定USD，先隔离接触正/分离负控制，再核查原G1 OFF/ON逐字段一致，不对既有控制器改动力学。

## C. 新结果二：旧CMU小参考的成功增幅在场景配对检验下没有统计证据

使用 **8个AG–AN球位/seed为唯一统计单位**，每场景 `v32`、`cmu02`、`cmu05` 三控制器完整成对；5次OFF/ON只读观察器重复排除。原严格PASS总计 **13/24**，其他 **11/24 FAIL** 保留7次29 qdot超速、3次未进球、1次脚背冲量不足。严格成功必须同时满足原进球、脚背≥95%、球速代理≥3.5m/s、所有29个关节qdot原限、非脚净力<20N、真实14秒恢复。

| 预冻结比较 | 原v32 | 候选 | 仅候选胜 | 仅基线胜 | 精确双侧p | Holm校正p | 逐一留出一个球位的成功率差范围 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMU10_02 vs v32 | 4/8 | **5/8** | 2（AH、AK） | 1（AJ） | **1.0** | **1.0** | 0 至 +0.2857 |
| CMU10_05 vs v32 | 4/8 | **4/8** | 1（AK） | 1（AJ） | **1.0** | **1.0** | −0.1429 至 +0.1429 |

精确检验是按discordant pairs条件化二项分布的双侧McNemar，不把24次当iid样本、不把81,200帧当实验独立次数；两比较采用Holm调整。结果**不支持「CMU弱参考提高了G1严格射门成功率」**，但样本量小，不能证明两者等效或证明下一代真实全身学习策略必然无效。8场景是预见过的冻结集合，不代表未知球位分布，不能据此声称泛化能力。

针对真实负例：AG三组皆超速，AN三组皆射偏，AJ的旧v32为PASS而两组CMU都关节超速；AK的旧v32脚背合法性失败但两组CMU均PASS。**这样的双向变化正是只看单独一次改善会误导的原因。**

## D. 研发实际实现与代码验收

- `tools/v126_collision_topology_and_paired_inference.py`：双SHA来源资格、冻结协议历史读取、原URDF几何及关节拓扑解析、29DoF检查、原8场景/seed配对独立性、精确二项p和Holm、leave-one-scene-out、缺证fail-closed；不导入IsaacLab、不写USD、不发电机命令。
- `tests/test_v126_collision_topology_and_paired_inference.py`：25个专项测试，覆盖缺失collider、29DoF来源、未验证native不可变真值、非完整14s、伪重复、seed不匹配、两比较精确统计、缺来源SHA等负例。
- `tools/v126_plot_static_contact_and_paired.py`：由真实SHA认证的统计摘要生成8场景PASS/FAIL矩阵；从原始URDF关节路径生成腕—髋拓扑图，标题明确「**NOT a PhysX contact witness**」。
- `docs/V126_WHOLEBODY_COORDINATION_NEXT_METHOD_CONTRACT.md`：定义大幅升级的KineticChain-29共享全身时序生成、静止起步及真实支撑/角动量/碰撞安全的准入门禁、合格人类源数据和全方法消融矩阵；**还不是已训练或可执行的机器人全身策略**。

本地write-once审计：`reports/v126_static_original_29dof_and_scene_paired.audit.json` SHA256 `bced60099c98acc88186d185f9c89ffac431c7aad7caed7041a9132e5983d9a0`（按.gitignore保留Tang本地）。真实失败矩阵图：`reports/v126_real_scene_paired_strict_gates.png` SHA256 `24bbdc98caa5a6878d67ee31293053379d53bf5970445d4b76f894b6df454c98`。URDF拓扑图：`reports/v126_original_G1_wrist_hip_static_topology_NOT_CONTACT.png` SHA256 `7b7f6fcf59dfffc8852fcac9b6730a80ad01a54d8917607b735cd13efa9a0c70`。

执行 `python3 -m pytest -q tests/test_v126_collision_topology_and_paired_inference.py tests/test_v125_original_g1_failure_timeline.py tests/test_v124_native_contact_truth_gate.py tests/test_v123_initial_runup_observability.py tests/test_v122_full_precontact_chain.py`：**91 passed, 0 failed**。这是CPU相关回归，不是全项目GPU回归。Matplotlib绘图有非致命的3D Axes导入警告，但两个2D证据图成功输出。

## E. 下一批优先级与当前不通过项目

1. v120既有GPU pytest持续占用，优先非破坏式追查为何4h+无进度日志；不得擅自终止、覆盖原结果或并行IsaacLab。若资源释放后仍有>20GiB空间，再进行原生PhysX刚体正负控制与原G1 OFF/ON collider pair/manifold认证。
2. 原29DoF全身时序生成的真实安全门禁继续关闭。先完成可追溯的人体足球动作资格与专业射门标签审核（被试独立、碰球时刻、真实动力学数据有据可查），再冻结新球位/种子和训练-盲测边界，开发及优化完整全身协调策略，而非旧1mrad参考。
3. 当前缺失：原生PhysX actor/collider/contact manifold真值；合格职业足球力学时序；新策略29DoF原物理闭环、多种子、严格球速/脚背/恢复对照。**不宣称这些门禁通过。**

本批 GitHub 分支 `version/v0.1.126`，代码和报告完成提交后需再核验远端commit SHA与干净工作树。研究方向的改变是：原URDF双碰撞体确实存在，但不能认证腕髋碰撞；CMU的单场景改善不能成为论文优势依据。下一步聚焦合格原生碰撞证据和真正的全身协调生成。
