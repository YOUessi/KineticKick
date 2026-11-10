#!/usr/bin/env python3
"""Auto-write complete v47 G1 PPO 14-second original PhysX negative/positive report."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT="reports/2026-11-09-v047-real-ppo-contact-recovery.md"
STUDY="configs/learned_contact_recovery_study_v047.json"
AUDIT="reports/v047_independent_original_physx_audit.json"
def sha(f):return hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
def fmt(x):return "N/A" if x is None else f"{x:.4f}"
def passing(x):return "PASS" if x else "**FAIL**"

def main():
 study=json.loads((ROOT/STUDY).read_text())
 audit=json.loads((ROOT/AUDIT).read_text())
 result=json.loads((ROOT/study["training"]["output_directory"]/"result.json").read_text())
 if len(audit["records"])!=6 or audit["frozen_training_study_sha256"]!=sha(STUDY):
  raise ValueError("Cannot present partial or drifted physics results")
 for f,h in study["source_sha256"].items():
  if sha(f)!=h:raise ValueError("Original frozen source drift "+f)
 output=ROOT/OUT
 if output.exists():raise FileExistsError(output)
 lines=[
  "# v0.1.47 — 真实29DoF G1接触后PPO微调：6组新球位完整PhysX配对",
  "",
  "**全部训练/物理结果由已经执行的原始IsaacLab数据生成，不把训练中的代理奖励当作真正的合法安全射门。**",
  "",
  "## 1. 真实PPO学习实验",
  "",
  "从冻结G1 v0.1.32的29关节后射门恢复PPO Actor完整参数作为初始化，而不是随机初始的简单位置调节器；原G1自由根部、足球接触几何、原PD执行器真实速度/力矩限制和冻结原跑动/触球专家不变。",
  "训练任务为在真实5ms PhysX观测中最小化后接触关节超速风险与落脚支撑/站立失衡。原有姿态、支撑、碰撞、落脚接近、终止和动作平滑奖励全部保留；额外惩罚任一真实关节速度超过原限的70%后迅速增长的速度裕度风险（平方、固定权重2.5）。这只是新的工程形状奖励而非CBF控制屏障。",
  "训练观察与Actor输出维度分别为154/29，所有29关节真实物理执行；训练只有18个已经在此前阶段见过的球位初始状态，在训练前冻结新的6个测试三元组，不能训练后再次挑选测试分布。",
  f"固定训练seed {study['training']['seed']}，{study['training']['num_envs']}个G1 PhysX并行环境，上限{study['training']['iterations']}次PPO更新与{study['training']['max_seconds']}秒。实测完成{result['updates']}次更新、训练用时{result['training_seconds']:.2f}秒；唯一被评估的是最终有限检查点，不根据新球位再选权重。",
  f"原v32 Actor SHA-256 {study['frozen_recovery_actor_start_SHA256']}；新后训练Actor SHA-256 {study['actual_final_trained_checkpoint_sha256']}。旧权重完整保留、没有升级默认v5。",
  "训练期间回合重置统计只能说明训练探索确实发生，不等于独立任务成功率。",
  "",
  "## 2. 六组训练前固定的新参数配对：严格14秒真实物理",
  "",
  "| 新固定条件 | 球位偏移XY(m)/初始速度差y(m/s) | 原v32严格通过 | 新PPO严格通过 | 原/新实际关节超速采样 | 原/新出球(m/s) | 原/新稳定站立(s) |",
  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
 for row in audit["records"]:
  old=row["old"];new=row["learned"]
  x,y,v=row["exact_parameter_tuple"]
  lines.append(f"| {row['id']} | ({x:+.3f},{y:+.3f}) / {v:+.3f} | {passing(old['strict_original_full_task_pass'])} | {passing(new['strict_original_full_task_pass'])} | {old['original_qdot_speed_exceedance_sample_joints']} / {new['original_qdot_speed_exceedance_sample_joints']} | {fmt(old['original_outgoing_ball_speed_xy_mps'])} / {fmt(new['original_outgoing_ball_speed_xy_mps'])} | {fmt(old['original_standing_seconds'])} / {fmt(new['original_standing_seconds'])} |")
 lines+=["",
    f"原v32全部严格门槛通过 {audit['original_v32_strict_success']}/6；新PPO通过 {audit['new_finetuned_recovery_strict_success']}/6。",
    f"旧失败转新通过{audit['old_fail_new_pass']}组，旧通过转新失败{audit['old_pass_new_fail']}组。",
    "",
    "原始PhysX硬门槛：自由根部14秒完整回合、真正合法右正脚背触球和整个球过线、真实5ms任一G1关节qdot未超过原速度上限、原跑动+射门连续、最终可靠站立≥1秒、出球水平速度≥3.5m/s。不能因球被击出或机器人站立更多秒就忽略射偏或超速。",
    "",
    "## 3. 所有未通过条件的具体原始原因",
    ""]
 for row in audit["records"]:
  failed=[k for k,v in row["learned"]["original_physical_gates"].items() if not v]
  if failed: lines.append("- "+row["id"]+" 新训练PPO未达到："+", ".join(failed)+"。")
 lines+=["",
    "新恢复PPO的激活发生在原真实触球/离脚之后，它改变实际29关节恢复控制动作，但无法倒流时间纠正已经发生的脚球接触方向、错误脚面或足球第一段出射方向。因此更稳定的落地，不一定增加严格射门成功数。",
    "本实验只包含6个已注册新连续参数条件，每组新旧策略各跑一次；它们仍属于同一足球场、同一G1本体和同一运动模板，不能推断职业人类大力抽射、自然度提升或跨全新环境泛化。",
    "",
    "## 4. 后续论文/工程决策",
    "",
    "本版证明了可以通过物理后训练而非手工补丁改变原G1后射门全身恢复行为。即使该模型能改善某些落脚指标，也必须与冻结v32按相同的全任务硬门槛比较，未来才有资格成为真正可信的接触阶段小脑。",
    "要解决用户最终方向：需要同时学习高动态正脚背射门的触球前身体摆臂/骨盆/髋膝踝发力、冲击和后续稳定；专业足球真实高频3D/击球力学数据、自然度盲评、足端与足球角速度、球速和多种子大规模PhysX全任务验证仍然缺失。当前PPO后训练只是较小的独立子方向。",
    "",
    "## 5. 真实可复现源码/证据",
    "",
    "- 固定训练/球位协议：configs/learned_contact_recovery_study_v047.json，训练专用配置configs/state_conditioned_strike_v18_train.json，验证配置configs/state_conditioned_strike_v18.json。",
    "- 实际PPO物理训练：tools/train_contact_phase_rl_v047.py，新后接触奖励tools/post_shot_objective_v047.py，原冻结Actor源tools/post_shot_recovery.py。",
    "- 原真实14秒G1配对运行与冻结checkpoint：tools/run_contact_recovery_rl_v047.py、tools/audit_contact_recovery_rl_v047.py；原默认G1物理runner未改。",
    "- 性能表与实际qdot图：tools/plot_contact_rl_v047.py，图片在Tang media/v047_safety_margin_ppo_paired_ball_physx.png。",
    "- 真正训练后的checkpoint、训练日志、原12次PhysX JSON、SHA及完整审计留Tang于experiments/contact_phase_recovery_v047/与reports/v047_文件中，不上传原体育视频、数据集或未授权第三方数据。",
    "",
    "**是否升级默认控制器必须以新6组原真实物理严格结论为准，不因PPO网络参数确实更新或训练奖励改善而自动批准。**"]
 output.write_text("\n".join(lines)+"\n")
 print("V047_REAL_PHYSX_PPO_REPORT",OUT,len(lines),
       "baseline",audit["original_v32_strict_success"],
       "learned",audit["new_finetuned_recovery_strict_success"],flush=True)
if __name__=="__main__":main()
