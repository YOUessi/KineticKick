#!/usr/bin/env python3
"""Render v42 human-readable research result directly from certified PhysX.

No hand-copied table, no task reinterpretation, no empirical model selection.
"""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"configs/novel_ball_generalization_v042.json"
OUTPUT=ROOT/"reports/2026-11-09-v042-novel-ball-generalization.md"

def yes(v):return "PASS" if v else "**FAIL**"
def num(v):return "N/A" if v is None else f"{v:.4f}"
def run():
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 audit=json.loads((ROOT/"reports/v042_paired_novel_ball_audit.json").read_text())
 study=json.loads(STUDY.read_text())
 if len(audit["trials"])!=6 or audit["protocol_sha256"]!=hashlib.sha256(STUDY.read_bytes()).hexdigest():
  raise ValueError("Pre-registered source and actual audit do not match")
 mech=ROOT/"reports/v042_mechanics_detail.json"
 detail=json.loads(mech.read_text()) if mech.exists() else None
 lines=[
 "# v0.1.42：6组从未作为命名开发条件使用的足球初始状态，v32/v41严格配对物理对照",
 "",
 "**完整结果由冻结的PhysX JSON自动计算，不允许后验修改测试条件或重新训练。**",
 "",
 "## 1. 预先注册且未用于调参的真实实验",
 "",
 "从已推送version/v0.1.41提交0100770，隔离v0.1.42分支；固定研究配置文件 configs/novel_ball_generalization_v042.json，SHA256 "+
 hashlib.sha256(STUDY.read_bytes()).hexdigest()+
 "，由随机种子4201分层洗牌并固定微扰得到6个此前未在有名称的训练/开发球位表中出现过的精确三元组：ball x/y偏移、G1前进y方向初始速度差。",
 "对每个三元组只进行2个原始14秒G1/PhysX仿真：冻结v32原控制和冻结v41人体姿态代理全身参考控制。"
 "  原始29DoF G1浮动根部、PD执行器真实速度/努力上限、足球接触体、球物理、原真实全球门线判据及恢复策略全部不改。",
 "12个物理实验事前固定全部执行，不根据中途好坏终止，也不从测试球位微调任何策略、增益或权重。",
 "",
 "以下新组合**并非另一个不同的真实足球比赛分布**：它们仅在既有足球环境参数范围内以未测试的连续数值组合产生。"
 "  它们也不是高帧率专业人体动作的独立验证集。"
 "  v41人体来源仍只有3DSP训练子集中的一段视频估计3D姿态，未验证正脚背脚面接触、运动员动力学或足端冲量。",
 "",
 "## 2. 六组配对实测结果",
 "",
 "| 场景 | 球位偏移XY (m) | 根部y速度差 (m/s) | 原v32严格通过 | v41严格通过 | 原/新出球 (m/s) | v41真正加权动作步数 | 原/新qdot超限样本 |",
 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for row in audit["trials"]:
  old=row["original_v32"];new=row["style_v41"]
  x,y=row["ball_offset_xy_m"]
  v=row["root_velocity_y_delta_mps"]
  lines.append(f"| {row['id']} | ({x:+.3f}, {y:+.3f}) | {v:+.3f} | {yes(old['strict_functional_pass'])} | {yes(new['strict_functional_pass'])} | {num(old['ball_outgoing_xy_mps'])} / {num(new['ball_outgoing_xy_mps'])} | {new['style_applied_decisions']} | {old['qdot_speed_limit_violation_samples']} / {new['qdot_speed_limit_violation_samples']} |")
 lines+=["",
 f"**按原始完整射门＋安全条件严格通过：旧v32 {audit['old_success_count']}/6，新v41 {audit['new_success_count']}/6。** "
 f"在六组中新风格真正产生非零关节参考的场景数为 {audit['new_style_actuated_count']}/6；"
 f"既实际参与又完全通过的场景数为 {audit['new_style_engaged_and_success_count']}/6。",
 f"原控制失败而v41严格通过：{audit['old_fail_new_pass']}组；原控制通过而v41失败：{audit['old_pass_new_fail']}组。",
 "",
 "执行的严格门槛由 tools/audit_novel_ball_v042.py 原始物理采样给出：完整14秒/2800个5ms物理步、合法正脚背、完整足球越过原门线、无非脚碰撞/实际关节速度超限、成功恢复站立、跑动及射门连续性、水平出球速度至少3.5m/s。"
 "只进球但真实qdot超速时必须判FAIL。若原基线本身也超速，同样判FAIL，不为任一方法改变限值。",
 "",
 "六个独立数值条件各只跑了一次。即使可报告Wilson描述性区间，也**不能**把这个小样本判作可靠人口成功率、独立训练种子泛化或所有现实球位的鲁棒性证据。",
 "冻结的6组v32/v41原始PhysX输出由 study 对应路径和每文件 SHA-256关联；所有JSON与checkpoint留在Tang的.gitignore中，GitHub只上传源码、协议、测试、文字结果。",
 "",
 "## 3. 实际身体动力学与局限",
 ""]
 if detail:
  if detail["source_paired_data_sha256"]!=hashlib.sha256((ROOT/"reports/v042_paired_novel_ball_audit.json").read_bytes()).hexdigest():
   raise ValueError("Supplemental qdot dynamics original audit changed")
  lines += [
   "工具 tools/analyze_novel_ball_dynamics_v042.py 从全部5ms PhysX采样提取最严重实际qdot限速点、其相对首次触球时刻和同时间原控制的相同关节速度。",
   "",
   "| 场景 | v41关节真实峰值与限值 | 原控制同时间相同关节速度 | 最差事件相对真实触球 |",
   "| --- | ---: | ---: | ---: |"]
  for rec in detail["records"]:
   b=rec["mechanics"]["style_v41"]
   p=rec["mechanics"]["paired_speed_at_candidate_worst_time"]
   dt=b["time_of_peak_qdot_relative_first_ball_contact_s"]
   t="N/A" if dt is None else f"{dt:+.3f}s"
   lines.append(f"| {rec['id']} / {b['actual_max_speed_joint']} | {b['measured_joint_speed_rad_s']:.3f}/{b['original_joint_speed_limit_rad_s']:.3f} rad/s | {p['original_v32_abs_speed_rad_s']:.3f} rad/s | {t} |")
  lines+=["",
   "关节功率仅是原PhysX测得qdot×裁剪PD估计力矩的代理，并非真实电机连续输出或人体肌肉功率；分析髋-膝-踝速度峰值的时序也不能单独证明人体近端到远端动量传递。",
   ""]
 lines +=[
  "## 4. 下一阶段科研判断",
  "",
  "这个研究不仅检查球是否被踢中，还同时测试高动态发力后身体能否安全恢复。"
  "  原v41的人体协调先验只给很小的动作修正，之前3个开发条件的胜利不足以说明其未见状态可靠；"
  "  本版新条件揭示的具体失败模式应保留作为下一版的条件安全/全身协调设计动机。",
  "",
  "不能依据这些样本修改已经固定的v41策略或追加对同一六组测试的调参。"
  "下一项训练需要真正独立的条件动作生成与支撑/摆腿/收势联合动力学约束，"
  "并获得有精确接触、足面、支撑与高速3D时序的专业人体射门数据。"
  "未取得这些证据前，既不能称为职业球员式暴力抽射，也不能称为完整的论文贡献。",
 ]
 OUTPUT.write_text("\n".join(lines)+"\n")
 print("V042_RESEARCH_REPORT_WRITTEN",OUTPUT,len(lines),"lines")
if __name__=="__main__":run()
