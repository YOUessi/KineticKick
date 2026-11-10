#!/usr/bin/env python3
"""v51 cross-morphological actual CMU human vs actual G1 FK descriptive report."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/cmu_g1_cross_morph_v051.json"
SOURCE="data/research_human_kicks/cmu_g1_cross_morph_v051/cross_morphology_audit.json"
DEST="reports/2026-11-10-v051-real-human-G1-coordination-baseline.md"
def sha(f):return hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
def main():
 dst=ROOT/DEST
 if dst.exists():raise FileExistsError("No retrospective report overwrite")
 cfg=json.loads((ROOT/STUDY).read_text())
 z=json.loads((ROOT/SOURCE).read_text())
 if z["study_sha256"]!=sha(STUDY) or len(z["comparisons"])!=4:
  raise ValueError("Incomplete/changed source actual CMU-G1 experiment")
 for path,expected in cfg["source_sha256"].items():
  if sha(path)!=expected:raise ValueError("Source changed: "+path)
 lines=[
 "# KineticKick v0.1.51 — 真实120Hz人体足球动捕与 G1 29DoF 射门的跨本体协调时序",
 "",
 "**实际执行：** 利用 v50 已下载的真实 CMU 120Hz 足球踢球三维骨架，以及 v48 完整G1 PhysX原冻结/新学习策略两条真实脚球接触前运动轨迹，建立6个名称、向量方向和相位明确的全身段落方向余弦对照。所有数值来自真实文件与原G1 URDF正运动学，不是生成的新机器人动作。实验仍不提供合格的真实人体第一脚球接触力学真值。",
 "",
 "## 1. 为什么只能比较六个通道而不是强制重定向29DoF",
 "",
 "真实人体光学3D包含骨盆/左右髋膝踝/脚尖/肩肘腕的标记与模型虚拟点；G1 URDF只有自己机器人结构定义的29个关节与腿/臂连杆原点。两者绝对关节角、骨长/比例、旋转轴、关节中心不等价。仅将具有明确几何构造的6类3D向量夹角作为共同语义代理：左/右膝弯曲、左/右肘弯曲、左/右大腿相对躯干。人体脚踝到脚尖的2个方向余弦没有G1可靠对应的脚趾段，**严格排除**，没有用机器人踝关节角冒充人体踝/足背角度。",
 "",
 "两条人体源CMU10_02、CMU10_05具有原足球标记簇连续高速运动，而非正式脚面力/碰撞传感器，因此将观察到的球运动突然开始作为仅供对齐的t=0；G1 v32/v48的t=0是原PhysX力接触的真实物理击球时刻。两种t=0定义不同，不能称为真正同步人体-机器人第一接触对照。",
 "",
 "## 2. 两个人体片段 × 两个G1策略的20帧对照",
 "",
 "每个序列从各自上述代理/真实事件向前取0.40秒至前0.02秒，重采样到20个50Hz时刻。原人体120Hz cosine不跨缺失帧插值，G1使用真实物理root+29DoF并在原独立URDF FK中重建6种段落向量。比较绝对6D余弦MSE及相对自身第一帧的运动变化MSE——它们均不是正确动作评分。",
 "",
 "| 真实人体原动捕 | G1原物理控制策略 | 绝对6D余弦MSE | 相对自身第一帧的变化MSE | G1/人体自身姿态变化RMS比 |",
 "| --- | --- | ---: | ---: | ---: |"]
 for r in z["comparisons"]:
  lines.append(f"| {r['human_source']} | {r['robot_source']} | {r['absolute_cross_morphology_cosine_MSE']:.5f} | {r['within_own_body_first_frame_relative_motion_MSE']:.5f} | {r['relative_motion_amplitude_ratio_G1_to_human']:.3f} |")
 lines+=["",
 "本批次两段人体相对特征RMS幅度约0.279/0.332，冻结G1约0.195；在这些代理段落几何定义下G1人体相对运动幅度比约0.70/0.59。新v48 PPO与原v32的相关MSE/变化幅度几乎相同，反映新策略实际仅在小的动作窗口里微调原动作。",
 "",
 "**不能把比值0.59–0.70解释为机器人只具有人体60%的发力、身体更僵硬、动作难看或低爆发力。** 人体CMU与G1骨架/关节原点各自不同、实际脚球接触也不是真同步；它只说明六个选定、无量纲的相对3D姿态夹角时间变化不同。",
 "",
 "## 3. 已建立并数值验收的跨本体接口",
 "",
 "源码在 tools/cmu_g1_semantic_bridge_v051.py。人体源读取 v50 C3D派生8维cosine，仅选0、1、4、5、6、7共6个；机器人源读取原真实G1 PhysX root与29DoF关节角，重建独立URDF FK（左/右髋膝踝、肩肘腕、骨盆/肩中点）求6个对应几何夹角。与另一路NumPy/SciPy独立G1 FK对照，六通道数值误差低于约1e-12；只比几何夹角，不强行将人体29关节角发给G1 PD。",
 "",
 "全部4对原对照的时序矩阵是20×6。若某帧原人体标记无效、球运动质量缺少持续高速证据、输入SHA不匹配或G1没有真正原PhysX接触，程序会直接拒绝，而不是在无证据的时刻生成专业抽射标签。",
 "",
 "## 4. 下一次真正应训练什么，以及当前硬阻塞",
 "",
 "要从CMU的真实人类动作中学习自然高爆发协调，需要首先确定源动作里的真实冲击瞬间、脚面方向、左右脚支撑和人体骨架的解剖坐标/动力学；目前只完成了人体姿态→机器人姿态的**语义共同比较**，并没有实施人体动作生成、动态重定向或G1模仿强化学习，更没有解决专业运动员动作美感和高冲量的监督。",
 "",
 "下一研究若进行跨形态全身生成，必须以人体自身的相对时间变化而非绝对骨长/关节角为先验，将接触/落脚/原G1 actuator真实速度/地面反力纳入物理控制；从真实样本中标注或核验足面/球体接触，至少独立保留按人划分的测试集。现有2个被试中只有一人包含这两段高速球运动，根本不足以支持职业动作泛化。",
 "",
 "应将此接口作为可复现的研究基线，重点用来发现原G1模板和真实人体相对时序的不一致，而不是因出现MSE就宣称更自然、更强力。",
 "",
 "## 5. 文件、模型/媒体与禁止夸大条件",
 "",
 "- 固定研究协议：configs/cmu_g1_cross_morph_v051.json，源/目标所有文件SHA一次锁定。",
 "- 真实跨本体6D几何与20帧事件时间对齐：tools/cmu_g1_semantic_bridge_v051.py；单测 tests/test_cmu_g1_bridge_v051.py。",
 "- 图：tools/plot_cmu_g1_bridge_v051.py，Tang media/v051_real_human_vs_G1_six_segment_cosines.png 和 media/v051_human_G1_relative_motion_amplitudes.png。",
 "- Tang data/research_human_kicks/cmu_g1_cross_morph_v051 保留原20×6人体与G1时序NPZ以及全部试验SHA/对照JSON。CMU原数据、G1原完整PhysX文件和原模型权重都保持本地，不上传GitHub。",
 "",
 "**结论：已从真实人体120Hz动作走到了真实G1 29DoF的语义时序对照，确认之前的触球前小残差PPO没有显著改变全身时间结构。但职业球员高爆发射门的生物力学真值依旧缺失，因此当前仍不是自然大力抽射的最终论文方法。**"
 ]
 dst.write_text("\n".join(lines)+"\n")
 print("V051_CMUSOCCER_G1_REPORT_DONE",DEST,len(lines),"pairs",len(z["comparisons"]),flush=True)
if __name__=="__main__":main()
