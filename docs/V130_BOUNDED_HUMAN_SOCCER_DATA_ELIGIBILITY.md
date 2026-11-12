# KineticKick v130 — 有界真人足球数据资格调研（2026-11-13）

性质：**元数据/文档资格审核**，不是已下载数据的真实标签验证；本轮没有为训练下载大数据，也没有制造接触/GRF真值。该调研是预冻结物理分析之外的探索性数据盘点，不宣称其阈值经过预注册。

## 目标：区分四种容易混淆的监督

1. **测量的三维人体运动**：实验室标记点或校准多机位数据；单目视频人体重建的三维姿态是估计量，不能升级为测量真值。
2. **真实足球脚球接触时间**：应有同步高速影像/接触传感器或等效可审计标注；动作主题写“kick”不等于接触帧已确认。
3. **脚底地面反力/压力中心**：必须有同步force plate/GRF/CoP和左右足可分离证据；足球冲量、足底GRF、机器人接触tensor不是同一种量。
4. **独立被试和专业身份**：仅真人足球运动不等于职业或高水平样本；跨训练/验证/测试不重叠被试，两个被试无法拆成三个独立非空集合。

## 实际检索到的候选

| 数据集 | 公开说明及可核实内容 | 当前严格资格结论 |
|---|---|---|
| CMU Soccer Subject #10/#11 | CMU网站标记“kick soccer ball”，提供120Hz C3D；本地v124已核6段、2名被试，3,358帧 | **运动学探索合格**；没有认证职业球员、同步GRF/CoP、脚球接触点真值，不能构造三组被试完全独立划分 |
| SoccerKicks（nayariml，2021） | GitHub说明38段定点球视频，2D AlphaPose/OpenPose + HMMR重建3D人体 | **足球视觉姿态参考候选**；重建三维不是实验室真实3D测量；无经核实同步地面反力/足球冲量；真实人高水平身份未认证 |
| GroundLink（SIGGRAPH Asia 2023） | 368段动作、19类运动，同步高频真实GRF/CoP与全身MoCap、左右足划分；包含多种动作如网球 | **可作通用支撑/载荷迁移的动力学预训练或校验候选**；不可冒充足球正脚背脚球接触监督 |
| Wilmes等 football IMU/VICON（Zenodo 2020） | IMU验证和VICON相关数据包，约987.4MB压缩包；公开说明为运动强度量化 | **足球场景运动学候选**；公开页面未认证与高速足球正脚背接触同步，亦未证明GRF/CoP、职业身份和逐帧姿态均可用 |

来源和许可确认入口：
- CMU Soccer Subject #10: https://mocap.cs.cmu.edu/search.php?subjectnumber=10&trinum=6
- CMU Subject #11: https://mocap.cs.cmu.edu/search.php?subjectnumber=11
- SoccerKicks: https://github.com/nayariml/SoccerKicks
- GroundLink: https://csr.bu.edu/groundlink/ 和 https://arxiv.org/abs/2310.03930
- Wilmes IMU/VICON: https://zenodo.org/records/3732988

## 已执行的筛选与结论

- 对每个候选逐项检查页面是否宣称提供 **实测全身3D、真实脚球接触、GRF/CoP、足球冲量、专业身份、受试者身份、复用许可**。没有文档证据的项目保持UNKNOWN，不填“是”。
- **没有一个候选已经满足“高水平真人正脚背实测时序 + 接触 + 真实力量链”全部要求。** 这不是宣布不存在其他数据，而是这轮有界候选审核结果。
- 当前磁盘余量30GiB且独立GPU任务在运行，没有下载987MB包或未知巨型媒体；明确保留下载和源码检验为下一阶段资格实验，不把网页描述写成数据内容验证。
- 下一步优先联系人类运动生物力学实验团队获取带高帧率球接触标注的合法全身运动数据；将GroundLink的真实GRF/CoP用于支撑侧的独立预训练/检验，不能输出职业式冲击真值；SoccerKicks仅用于视觉动作先验并单列2D/单目3D误差。

审计结果：**qualified_human_elite_soccer_instep_force_truth = false；已训练/验证KineticChain-29 = false；无人体足球冲量伪标签。**
