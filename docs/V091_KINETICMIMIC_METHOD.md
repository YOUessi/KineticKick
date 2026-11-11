# KineticMimic v0.1.91 — 已实现因果全身观察器、未实现的运动模仿假设与证据等级

## 1. 要改进的不是脚部“碰球”，而是全身接触约束下的运动生成

论文主假设：一个保留人体支撑模式、髋—躯干—摆腿—对侧摆臂相位及实际冲量门禁的模仿策略，能够在**未见人体被试和未见G1物理球位**上提高自然性/全身协调，并且保证全部动作物理安全与正脚背高速有效射门。这个假设**尚未得到实验验证**。

不可混淆四层证据：

1. 真人专业3D运动和真GRF/足球冲量监督（当前缺失）。
2. 物理确证的原29DoF G1运动及支撑（已有真实200Hz PhysX）。
3. 当前v91因果全身短时 **observation-only** 预测器（已实现训练和独立场景测试）。
4. 真正的人体条件化、29DoF动作生成 + 可验证物理约束的主动策略（当前**未训练、未部署**）。

## 2. 本轮真正实现的学习算法及输入合同

以200Hz原G1物理日志为唯一学习数据源，50Hz采样时刻t，预测t+100ms的归一化全部29个关节速度、4类左右足支撑模式和未来100ms内首个实际足球接触标签。

所有模型输入只含截至当前帧可观测的真实机器人joint q/dq、身体重心与根相对位置、球与根相对位置、右足局部点速度、时钟。Contact模型再加入当前实际左右脚载荷、支撑状态onehot和持续时间，自从上次观测到的支撑模式变更。真实未来碰撞时间、未来GRF、未来冲量、原策略下一动作及所有测试标签**绝不作为模型输入**。

固定训练对象：v84的C_v32_off/C_cmu02/D_v32/D_cmu02。无数据增强虚构GRF；同一场景的只读观察器ON复制不重复计数。固定lambda=10 ridge和训练集标准化。完整负对照包括速度持续、时钟岭回归、无Contact状态岭回归、完整Contact岭回归以及训练Contact序列打乱版本。留出新冻结E/F球位的原控制器轨迹作严格场景测试；不看新场景后修改lambda、特征与门禁。

所有报告均包含完整可观察物理时序以及以首个**真实机器人**触球为中心的[-0.4,+1.25]s窗口，强制标记提前终止的窗口为censored，不以仅触球前的片段伪称完整随挥。

模型目标与直接原G1驱动无关：
```
qdot_29 + state + stance/contact-observation
  -> causal 100ms full29 joint-velocity / stance / near-impact prediction
  -> coordination and timing diagnostic
  -> NOT motor action (actuation hard BLOCKED)
```

## 3. 只作为后续算法研究设计：Contact/Coordination/Impact多目标模仿

候选学习损失（当前未训练，不是实验结果）：

```
L = L_motion(retargeted human fullbody kinematics)
  + λc L_coord(event-relative group phase lags)
  + λs L_support(measured/contact-conditioned stance)
  + λi L_impact(real robot foot-ball impulse and ball outcome)
  + λr L_recovery(14s physical episode)
```

人体阶段可学到相对运动时序及有资格的GRF；**足球冲量若仅有机器人仿真真值，只能作为机器人任务端目标**，不能称“人类真实冲量拟合”。模型必须受限于实际29个电机位置/速度/扭矩、全身自碰撞几何间距、支撑多边形和失稳恢复。缺少原生可认证collider路径/距离时执行器仍然返回原始动作（fail closed），不允许以净接触力为完整自碰撞证明。

## 4. 本轮可证实的结果与否定性结论

E/F共4个新原G1场景-策略物理执行：严格射门**0/4**。这是原v32/CMU参考控制，不是本模型动作策略表现。因果观察器对完整接触前后窗口的E两个case，其Contact-conditioned平均normalized-qdot MAE=0.031685，state-only=0.031933；提升仅约0.78%，宏F1反降；group-peak timing MAE同为0.104s。F两case仅320个真物理step、触球后不足1.25s，必须仅报告**censored预测诊断**，不得纳入完整随挥效果的结论。

因此，**尚无证据证明Contact-conditioned观察器优于state-only的全身自然协调生成**，更没有真实人类式高爆发射门。当前主要贡献是严格实证协议、直接真实物理的负例与安全边界，为将来的方法提供对照和诊断标准。

后续执行上的依赖顺序：
- 校准PhysX原生contact/collider几何正控制；认证脚球和腕/髋/骨盆真实pair；
- 取得合格多被试真人足球全身相位数据及可用真实支撑动力学；
- 训练真正条件化的动作生成器，执行 support-first / arms-frozen / reversed-phase / foot-only / fullbody 的公平消融；
- 冻结未见球位与重置seed，严格交叉衡量自然性（人体动作距离、协调延迟、独立盲审）与全部原G1 PhysX安全门禁；
- 结果不佳则保留失败、变更新版本协议，绝不调参污染已冻结holdout。
