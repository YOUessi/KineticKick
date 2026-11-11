# KineticKick v0.1.77｜原G1启动依赖阻塞的负实验 + 9段200Hz关节/支撑时序联合审计

2026-11-11；冻结父版 `1571bf817eef2de74c447926c7b13a2905121148`；协议提交 `83bc80fb5ac7668e097e2b5be5bc8a4868b7f4b1`；协议文件SHA256 `f299cae1a9d044d99e2c936f3c75c02d10605c264e87a134719ba1d25ef2a928`。

## 环境与完整性

Tang Ubuntu / RTX4090 Laptop GPU（16GB）；起始剩余磁盘约39GiB、没有CUDA计算PID。检查了已经存在的v64～v76 worktree，v74有未跟踪文件，完全未修改其内容。所有新源码在独立`version/v0.1.77`上提交。验证3份原v54 PhysX和8份控制/配置源的SHA无漂移，补齐58个只读原有实验资料软链接；随后为后续CPU全测试补齐原`third_party`和`onnxruntime`目录只读链接。原数据/模型及其他worktree保持不变。

## 先冻协议的独立GPU诊断：一次尝试，没有真正G1物理回合

新建`tools/v077_g1_stack_child.py`、`tools/v077_g1_liveness_batch.py`。仅在独立子进程注册SIGUSR1 faulthandler，不改G1控制器/URDF/PhysX或用户IsaacLab。共用GPU互斥锁，写入唯一命名原始log/audit；已冻结信号[20,50,80]s、105s上限、完成一次失败即停止后续物理的规则。

本次实际启动1个GPU预备子进程，0.50s exit(1)，尚未进入Isaac AppLauncher/环境、也未做任何G1 PhysX step，`physics_episode_certified=false`，`NO_CERTIFIED_EPISODE`。原始日志明确定位异常：`targeted_shooting.controller_provenance()`抛出`ValueError: Missing upstream controller sources`；原因是新worktree最初缺少整棵忽略于Git的`third_party`源目录软链接。它是**实验部署依赖错误，不是控制策略射门失败、不是碰撞、不是原先的仿真hang**。log SHA256 `3996c1176024c3a84cba52e03481b4e2cc69c5a5ec6bcd838a7d687af4b5e201`，输出原始物理JSON不存在。

严格遵守本版冻结失败停止规则：修复只读依赖链接后，没有在本版偷换既定输出名或直接重复GPU实验。原生左腕—左髋 collider path/contact manifold依然未证明。

## 历史原G1 200Hz联合审计（仅复核，不算新增试验）

另开发`tools/v077_kinematic_coupling_audit.py`，逐条锁定v67原9条200Hz PhysX SHA，检验完整29关节实测qdot和限值、支持腿/双足法向净力、躯干—对侧手臂和摆腿髋膝**带符号的时间滞后相关**、接触前400ms及后250ms/1250ms窗口、不完整恢复窗口和单独脚背95%门禁。全9条均验原SHA；其中5条完整14s、6条有完整接触窗口、3条触球窗口截尾，4/9为历史独立全硬门禁PASS（不是v77成功率）。

旧v67 stance A三个控制均原正脚背冲量100%且14s完整，但在预触球窗口 `waist_yaw` 对 `left_shoulder_yaw` 同步相关约0.742～0.746，三者非常接近，**不能证明**真实人体时序参考带来显著协调改善。stance B基线在3.70s遭非脚终止；合成控制虽然进球站到14s，脚背冲量仅**0.7298347**而FAIL；反序控制历史为PASS。stance C三方案均在1.64s非脚终止，非脚净力峰值约2225～2287N。这些历史观察是已见样本，严禁用于调参后伪造独立验证。回归中专门修正了四条提前终止记录附带的**最后1微秒非物理终止样本**：不把该点冒充200Hz积分步，不抹除它的存在。

双足净力不是测力板GRF，也没有真实足底CoP或稳定支撑多边形；相关系数不是跨关节力矩传递的因果证明。左髋/左腕形状对碰撞仍缺真PhysX双方。历史6段CMU未有同步职业运动员足—球冲量、真实GRF，新训练模型的自然性假设仍未成立。

## 验证与交付

针对性测试`10 passed`，首次全套CPU因本新worktree缺少原`onnxruntime`本地依赖获得814 passed / 4 failed；只以只读原目录链接恢复依赖后复测，**818 passed、67 skipped、51 subtests passed，0 failed**。先期失败如实保留在本地日志。新增代码524行及协议/报告，所有原始PhysX/CMU/NPZ/ONNX/原日志保存在Tang，不提交大数据。

本版已对失败原因做可复现归因，而没有违反协议继续尝试新G1物理。下一**独立冻结版本**先把`third_party`和本地ONNX运行库加入完整源依赖预检，再运行一次有界SIGUSR1堆栈实验；若能启动但未正常产生G1输出，则提供真正堆栈阶段真值；若出现完整物理回合，仍须单独的新协议做native off/on三组逐字段parity和硬门禁审计。

论文主张仍是**跨运动员可泛化的人体全身协调学习**，与历史轻微合成相位不是一回事；当前不宣称自然高速度正脚背抽射已学成。
