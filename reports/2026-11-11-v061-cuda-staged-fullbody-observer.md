# KineticKick v0.1.61｜12次冻结原29DoF G1 PhysX：GPU全身观察器故障的首次精细定位

日期：2026-11-11。父SHA 9510d9d24b5f10d5949b5ca123353e044c3e40c2。先冻结的协议源码commit e311cb65f0961e4969cd3be405508251c3bb2723。协议SHA256 974d1bb05f7aba6312152765057d4178fedbc6945969866fd3fac38e8c805db2，实际物理之前写入 reports/v061_protocol_lock.json。历史G1物理JSON/NPZ、CMU/C3D、媒体和权重只留Tang，GitHub不上传。

## 真实PhysX实验：10/12有原始数据，2次无输出不得评分

12次独占执行原29DoF自由根Unitree G1/Isaac PhysX真实脚球物理；10次有可审计原5ms轨迹，2次CUDA联合观察后无JSON。已冻结原v32/CMU10_05的政策、场景、种子1201、执行器、脚面、≥3.5m/s、全部原qdot限制、非脚20N、安全和14s恢复；只读观察器是唯一同backend同策略对照变量。12次不是独立随机条件。

CUDA基线v32：empty、mass、com_pos、com_vel、body_pos、body_vel、forces、report独立8次各有2800条真实5ms PhysX轨迹，严格8次PASS；这八个PASS是相同基线、相同seed和完全相同物理过程的重复确认，只相当于一个独立场景，不可报告为80%或大样本泛化。CUDA v32 full_compute没有可评分G1输出。CPU v32 full_compute有333行（332个5ms+终止微小时间戳），1.66s非脚接触终止，严格FAIL。CUDA CMU10_05 mass有1029行，5.14s physical_nonfoot FAIL；CMU10_05 full_compute无输出。全批原G1 10个有效严格8 PASS、2 FAIL，外加2个不可评分；**没有训练新控制器或提高自然度**。

独立审计 tools/audit_staged_probe_v061.py 对CUDA v32前8次及CPU v32 full_compute共9组进行了与v060相同策略、设备、种子的所有每帧25类物理字段逐值对照：全部0处差异，最大绝对数值差0。只读单一stage不会改变原G1物理。跨CPU/GPU不可用于无扰动性断言。原始write-once源SHA/完整严评在Tang reports/v061_independent_audit_20261111.json；GPU互斥flock；磁盘≥20GiB；无覆盖历史数据。

## 新发现：多项GPU读数分别可行，但质量缓存与首帧COM读取组合失败

新 tools/staged_body_probe_v061.py 在每个可疑原始物理张量访问前后打印 V061_PROBE_ENTER/OK 精确边界。GPU单独root_physx_view.get_masses及归一化sum无异常（2800真实帧），GPU单独读取 body_com_pos_w、body_com_lin_vel_w、body_pos_w、body_lin_vel_w、非脚net contact forces、现有get_contact_report也各自完成14s原始轨迹，与off对照0差异。

CUDA full_compute复现两控制器第一帧失败：日志完整打印 mass_complete、constructor_complete、FREE_SHOOTING_READY、V061_PROBE_ENTER full_compute body_com_pos frame 0；但没有该字段读取完毕标记，也没有G1 JSON。此区间同时包括body_com_pos_w读取、GPU到CPU张量读和(pos*mass_weights).sum(0)，**仅能说与质量缓存+COM首帧联合路径高度相关，不能归因单一CUDA指令**。CPU同操作成功，不能因此将CPU和GPU动力学混为一谈。下一批需独立测试mass+com不加权、mass克隆后com、先com后mass、独立质量CPU副本，以及从原已记录whole_body_com_xyz读取代替重复质量计算的无扰动稳定完整观察器。

GPU原生报告0头不等于没有实际自碰撞，左腕/左髋真实GPU actor/collider paths和manifold仍未验证；此前CPU骨盆/右髋直接pair证据不能替代。绝不私改碰撞过滤、降低nonfoot门禁。

## 真实200Hz全身运动与失败窗口

tools/analyze_v061_coupling.py 对CUDA v32、CUDA CMU10_05、CPU v32三条真实冻结源读取原29DoF q/qdot、全部原始限速、全身COM、双足支撑过滤法向力、非脚contact净力；首次真实脚球PhysX碰撞对齐 t=0；区分前0.4s、后完整1.25s和至14s/提前终止，缺失恢复窗口不填补。原始SHA和分组数值在Tang reports/v061_true_200hz_29dof_coupling_r1.json。

CUDA v32：2800帧完整14s PASS；前0.4s实际29关节最大qdot/原限约0.825，接触后1.25s约0.749；相应非脚净力为0。CUDA CMU10_05：前0.4s及后1.25s均完整，但5.14s左腕/左髋非脚候选净力约25.89N、提前终止；有1.25s观察窗不代表14s恢复。CPU v32：触球1.58s、1.66s提前终止，只有0.08s真实后窗，不能制造其余1.25s；非脚净力峰值约894N。

基线CUDA v32右踝pitch、右膝、右髋pitch在触球前0.4s窗口内各自绝对角速度峰值相对真实触球约-0.25/-0.22/-0.04s。这并不是人类正确的近端到远端发力链证据，反而提示需直接训练并验证真正全身时序。双足过滤法向力不等于专业人体GRF或CoP/ZMP，身体link-origin距离不等于碰撞网格signed distance。

两张真实PhysX图：media/v061_200hz_human_proxy_caution_joint_coupling.png、media/v061_real_nonfoot_failure_and_support.png；均按.gitignore保留Tang本地，源码 tools/plot_v061_coupling.py 可复现。

## 论文核心仍未完成

计划研究方法：脚球接触校准的人体全身相位编码器 → 真实支持腿承载感知的29DoF联合时序生成器（助跑、支撑、髋躯干旋转、摆臂抵消、摆腿、随挥落脚）→ 经原PD、全部关节qdot/扭矩、足地接触可行域和非脚碰撞网格距离约束的低层控制 → 完整14s真实PhysX安全/射门/动作自然度盲评。必须先固定主体级训练/验证/未见测试，人体专业高速正脚背触球/GRF目前为0条合格真值；现有CMU10_02/10_05只可做弱相位标记，禁止冒充专业式高爆发已学成；不得把旧1mrad/18mrad参考或离线余弦MSE包装成成果。未获得实时可信碰撞pair前，新控制器的自碰撞安全门禁必须fail-closed，不能升级默认模型。

## 复现与测试

源/协议在GitHub本分支，原物理资料Tang write-once。使用 tools/run_staged_gpu_v061.py 在有GPU互斥、磁盘≥20GiB且旧source hashes匹配时重核已有12条源SHA，不覆盖；独立审计 tools/audit_staged_probe_v061.py --output reports/NEW_UNIQUE_v061_audit.json；全量回归 pytest tests。实际回归749 passed,69 subtests passed。相应文档：https://docs.nvidia.com/isaaclab/concepts/sensors/contact-sensor 和 https://docs.nvidia.com/isaaclab/concepts/isaac-sim-phys-x（注意文档版本与Tang实际环境差异，不当作本机证明）。
