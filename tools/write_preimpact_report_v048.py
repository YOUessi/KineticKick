#!/usr/bin/env python3
"""Write v48 true 14s original G1/PhysX research results from frozen audit."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/learned_preimpact_wholebody_study_v048.json"
EVIDENCE="reports/v048_preimpact_full_physics_audit.json"
OUTPUT="reports/2026-11-10-v048-learned-preimpact-ppo-physx.md"
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def number(v):return "N/A" if v is None else f"{v:.4f}"
def state(v):return "PASS" if v else "**FAIL**"

def write():
    dest=ROOT/OUTPUT
    if dest.exists():raise FileExistsError(dest)
    study=json.loads((ROOT/STUDY).read_text())
    result=json.loads((ROOT/EVIDENCE).read_text())
    train=json.loads((ROOT/study["training_spec"]["output"]/"result.json").read_text())
    if result["original_study_sha256"]!=sha(STUDY) or not result["zero_physics_equals_original"] or len(result["cases"])!=6:
        raise ValueError("Incomplete original true G1/PhysX evidence")
    for f,h in study["source_sha256"].items():
        if sha(f)!=h:raise ValueError("Frozen original source changed "+f)
    lines=[
      "# KineticKick v0.1.48 — 首次真实训练触球前 29DoF 全身协调 PPO 与完整物理对照",
      "",
      "**研究范围：** 原自由根部 G1 29DoF、原PD关节真实速度/力矩限、原足球/脚部PhysX接触、球门与球体判据不变，在触球前施加学得的全身动作，而射后仍由原v32已经训练的PPO处理。真实14秒独立验证依据原5ms物理采样，而不是看训练奖励或踢球演示视频。",
      "",
      "## 1. 架构、训练与数据诚实边界",
      "",
      "新Actor感知160维：154维原G1全身状态、脚部载荷/姿态和29关节实际速度，再附加3维球相对机器人位置与3维真实球体速度；动作输出29DoF。限制真实触球前1.02–1.54秒、正确支撑/摆腿脚负载、球距及原限速90%以下才可干预，单关节参考最大0.018rad，每20ms变化最多0.003rad。原v32恢复Actor冻结并独立控制触球后身体，不曾从已经评估的六组选择模型。",
      "训练奖励包含原完整后接触稳定性、原5ms真实关节速度惩罚、球体实际出射方向/速度，以及从3DSP职业比赛训练片段获得的一条弱跨部位方向变化参考。该数据为广播视频估计的低频3D关键点，而非专业高帧率动作捕捉、真实脚球接触/支撑反力/足端能量链真值，不提供真正的专业抽射动作自然度监督。",
      f"事先固定 seed {study['training_spec']['seed']}、并行 G1 环境 {study['training_spec']['num_envs']}、最大 {study['training_spec']['iterations']} PPO 更新/ {study['training_spec']['max_seconds']} 秒预算；实际执行 {train['updates']} 次更新，真实训练用时 {train['training_seconds']:.2f} 秒。",
      "18个原本已用于开发的足球初始条件组成训练分布；六组与此前精确参数完全不重合的足球XY偏移/G1根部初速度组合在训练前锁定。它们仍在相同Isaac G1物理任务范围内，不是新足球比赛/机器人硬件环境。",
      "",
      "## 2. 禁用新模块时的物理完整性",
      "",
      f"从原v32中心场景对比的新零干预回合，完整持续{result['zero_result']['physical_5ms_steps']}个5ms真实物理步，共有{result['zero_result']['original_matched_fields']}个物理字段逐步一致。",
      "原射门、接触、真实关节、球速、机器人恢复与原PPO动作未改变；完整两份JSON可能含不同只读诊断字段，因此不宣称完整文件字节相同。",
      "",
      "## 3. 六组预注册新球位14秒真实物理配对",
      "",
      "| 球位 | XY偏移(m)/G1初速差y(m/s) | 原v32严格 | 新v48严格 | 新真实全身动作步 | 旧/新出球(m/s) | 新最大实测qdot/原限制 | 旧/新超速样本 |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in result["cases"]:
        orig=v["original_v32"];p=v["learned_preimpact_v48"]
        x,y=v["ball_offset_xy_m"];speed=v["velocity_y_delta_mps"]
        lines.append(f"| {v['id']} | ({x:+.3f},{y:+.3f})/{speed:+.3f} | {state(orig['strict_physical_pass'])} | {state(p['strict_physical_pass'])} | {p['new_preimpact_policy_nonzero_control_steps']} | {number(orig['true_outgoing_ball_speed_xy_mps'])}/{number(p['true_outgoing_ball_speed_xy_mps'])} | {number(p['max_original_joint_velocity_limit_ratio'])} | {orig['qdot_speed_overlimit_sample_joint_count']}/{p['qdot_speed_overlimit_sample_joint_count']} |")
    lines.extend([
      "",
      f"**固定原v32完整合法通过 {result['original_v32_strict_pass']}/6；新触球前全身PPO 完整通过 {result['learned_preimpact_v48_strict_pass']}/6。**",
      f"原失败转为新通过 {result['repaired_old_failures']} 个，原通过退化 {result['regressed_old_successes']} 个。真正非零动作改变的全新球位 {result['learned_actor_actually_engaged_conditions']}/6。",
      "全任务硬判据一律同时要求完整14秒、原29关节的任意真实qdot样本不超限、合法右正脚背≥95%接触冲量、整球入门、出球≥3.5m/s、正常助跑/击球、无非法身体触球、最终连续站稳；绝不把碰球算进球、把进球算自然或者把训练回合数算成功率。",
      "",
      "## 4. 真实失败条件和科研含义",
      ""])
    for c in result["cases"]:
        p=c["learned_preimpact_v48"]
        failures=[key for key,ok in p["strict_gates"].items() if not ok]
        if failures:lines.append("- "+c["id"]+": "+", ".join(failures)+".")
    lines.extend([
      "",
      "新PPO第一次改变真实触球前29DoF动作，仍继承原射门专家作为底层动作模板。当前数据尚不能说明专业运动员式的近端到远端关节加速度序列、髋-膝-踝发力效率、右足/足球法向冲量和身体自然美感是否有任何提升。",
      "六组条件每组每方法只有一个物理回合，样本量小，已有的后续训练不得将本组六组再次当作未知分布；不能给出可靠人口成功率或真实硬件安全承诺。若新模型偶然进球，但原qdot超限或摔倒，同样严格判失败并拒绝晋升。",
      "",
      "## 5. 复现和归档",
      "",
      "真实模型训练：tools/train_preimpact_wholebody_v048.py；G1全身学习控制：tools/preimpact_wholebody_v048.py；原14秒独立物理运行：tools/run_preimpact_wholebody_v048.py；批次调度：tools/run_preimpact_study_v048.py；真实物理审计：tools/audit_preimpact_wholebody_v048.py。",
      "实验配置及源代码/数据SHA：configs/learned_preimpact_wholebody_study_v048.json、configs/state_conditioned_strike_v19_train.json、configs/state_conditioned_strike_v19_zero.json、configs/state_conditioned_strike_v19.json；测试tests/test_preimpact_wholebody_v048.py。",
      "原始训练检查点、失败/训练日志、完整PhysX数据和第三方体育视频都保留在Tang的Git忽略目录；GitHub只提交代码、测试、冻结配置和真实研究报告。",
      "",
      "**结论严格限制：本版是带真实物理反馈的预触球全身小脑探索，不是已经证明自然高爆发正脚背抽射的论文；专业数据、视觉自然度盲评、跨摩擦/球参数分布和多个独立训练种子仍是下一阶段必须完成的工作。**"
    ])
    dest.write_text("\n".join(lines)+"\n")
    print("V048_RESEARCH_REPORT_SAVED",OUTPUT,"lines",len(lines),
          "base_pass",result["original_v32_strict_pass"],
          "new_pass",result["learned_preimpact_v48_strict_pass"],flush=True)
if __name__=="__main__":write()
