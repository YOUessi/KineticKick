# v0.1.67｜人体正脚背动作数据资格：六源有界核对（2026-11-11）

此次审查将“有真实人类时序几何”“真实职业足球正脚背触球动力学”分开。
规则写在 `configs/soccer_source_qualification_v067.json`，经
`tools/qualify_soccer_sources_v067.py` 显式认证：职业身份可核、人体3D真测、
同步脚球接触帧真值、实际球脚力/冲量、同步真实GRF **五项全部具备**才能
作为职业式强力正脚背动作动力学监督。**Unknown 不等于 True。**

实际在公开源说明及 Tang 已有资产下完成六源初筛，**0/6 符合完整标准**：

- 本地 CMU subject 10 的 10_02/10_05：仅同一标定 subject 的两段 120Hz 人体光学动作；足球时刻是从几何/球标记估计的代理，缺真实接触冲量、GRF、职业身份。只能用于受限几何时序研究。CMU 官方目录列 subject 10 共五条 soccer kick（1、2、3、5、6），尚有未纳入的 clip，不可把更多同 subject 的 clip 当新被试。[CMU subject10](https://mocap.cs.cmu.edu/search.php?subjectnumber=10)
- CMU subject 11_01：官网列 120Hz soccer kick，可成为独立标定骨架的后续审查候选；**不同 subject number 不保证不同受试者**，官网亦提示同一人在不同 subject number 出现过。尚未下载/核实其 contact/force，更不能视为跨人验证。[CMU subject11](https://mocap.cs.cmu.edu/search.php?subjectnumber=11) · [CMU FAQ](https://mocap.cs.cmu.edu/)
- GroundLink：对不同运动真实同步测量的 force-plate GRF/CoP 与人体动作，有 368 个处理过的实验 trial，其中包括网球挥拍、步行等；**不等同足球正脚背—足球冲量真值**。可作为支撑动力学辅助领域，不是踢球配对人体 GT。[GroundLink](https://csr.bu.edu/groundlink/)
- SoccerKicks：38 个踢球视频，HMMR 从单目视频估计 3D 人体关节；可作弱视觉时序/相位标注研究，不是测力台/运动学光学真值。[SoccerKicks](https://github.com/nayariml/SoccerKicks)
- 2026-01-22 发布的 EPL penalty 数据：25fps 转播视频抽取 2D YOLOv8 姿态（12 关键点），球触有 `last_touch` 末帧标记。职业比赛来源不能转化为同步真实3D人体关节、脚背冲量、GRF。[Mendeley DOI:10.17632/brx9bsxnpx.1](https://data.mendeley.com/datasets/brx9bsxnpx/1)
- Booster Dataset：机器人足球动作，不能当职业人类动作，不能偷换研究目标。[Booster](https://huggingface.co/datasets/SaiResearch/booster_dataset)

**下一阶段带预算边界的数据资格工作**：只在 source 许可证允许时核实 CMU 10_01/10_03/10_06 和 11_01 的实际 motion 文件/骨架、
检查 subject ID、对每段抽取 *未知* 脚球接触概率区间和左支撑/右摆腿标签，
以 `uncertain_contact_start/end` 注释；没有测力台/高帧率碰撞传感数据的条目不得由算法补出
“真实接触帧”或球脚 N·s。采集职业球员动作必须明确商业授权及身份/匿名化、
同步 GRF/CoP 和至少 200Hz 近足/球接触证据，按人和动作划分训练/验证/独立测试。
未满足条件时，方法只能标为 **synthetic control** 或 **human optical geometry proxy**。

该有界清单是根据公开描述和本地可见材料的资格判断，**不是已经下载并全面检测六个原始数据集**。
