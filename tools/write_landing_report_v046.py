#!/usr/bin/env python3
"""Report v46 full original G1/PhysX evidence, not selected successes."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/landing_transfer_study_v046.json"
AUDIT="reports/v046_landing_transfer_final_audit.json"
OUT="reports/2026-11-09-v046-anticipatory-landing-physx.md"
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def fmt(n):return "N/A" if n is None else f"{n:.4f}"
def mark(p):return "PASS" if p else "**FAIL**"
def main():
    output=ROOT/OUT
    if output.exists():raise FileExistsError(output)
    study=json.loads((ROOT/STUDY).read_text())
    a=json.loads((ROOT/AUDIT).read_text())
    if a["original_study_sha256"]!=sha(STUDY) or len(a["results"])!=6:
        raise ValueError("Cannot publish incomplete frozen trial")
    for src,h in study["source_sha256"].items():
        if sha(src)!=h:raise ValueError("Unexpected source code change "+src)
    lines=[
     "# v0.1.46：真实触球后的全身预期落地过渡控制 — 严格原G1 PhysX评测",
     "",
     "全部数字由原始真实PhysX关节状态、球轨迹、接触及支撑数据自动生成，不得把一次进球当作完整物理成功。",
     "",
     "## 1. 实验合同与控制结构",
     "",
     "从v0.1.45/qdot反速度调节始终零动作的真实失败出发，新研究在已经观测到真实右脚触球后、摆腿脚再次着地之前执行有限的全身过渡参考。",
     "原G1 29DoF自由根部、原PD关节速度/扭矩限值、球/脚接触模型、冻结v32 PPO恢复及v41训练集单段3DSP人体视频姿态弱参考全部不改；不能将它称为专业人类高爆发抽射动作生成器。",
     "本版使用真实左/右支撑脚载荷、右足底碰撞胶囊最低点高度及下降速度、29个真实关节速度来判断触球后+0.06至+0.62s的支撑/落地过渡，输出作用于左右腿、躯干和手臂的有界反向速度位置参考。单关节参考绝对幅度≤0.016rad、每20ms变化≤0.002rad，但不能作为物理限速的安全保证。",
     "",
     "原程序冻结协议在任何主动试验开始前发现调度器字段active_config/no_adaptive_gain_search与注册表不一致。原始失败启动没有产生主动物理数据。仅更改调度器读取源表的JSON键，保留原协议全文/初始错误日志、原球位与增益不变；在真正主动测试前修订了其SHA并写明协议修订历史。",
     "",
     "固定试验数量：1个中心零模块、2个已知旧球位诊断、6组全新精确连续球位和初速度各配对旧v41与新v46，共15个G1 PhysX回合。全部结果必须保存，遇到任务失败不得重新选球、调参数或降低原真实关节限速。",
     "",
     "## 2. 禁用时的原始物理等价性",
     "",
     f"新模块关闭后，双方共有的{a['zero_parity']['shared_physical_fields_checked']}项原始物理字段在全部{a['zero_parity']['physics_steps']}个5ms物理子步中逐值完全一致，700个原关节动作、接触/原进球结果相同。旧instrumented记录多出7个只读字段，因此完整JSON不宣称字节一致。",
     "",
     "## 3. 两个旧开发条件的真正动作与限速",
     "",
     "| 既有条件 | 新v46全任务 | 新参考非零20ms步 | 实测最大qdot/原上限 | 新出球水平速度m/s |",
     "| --- | --- | ---: | ---: | ---: |"]
    for row in a["old_diagnostics"]:
        x=row["result"]
        lines.append(f"| {row['id']} | {mark(x['original_strict_pass'])} | {x['landing_actual_nonzero_control_steps']} | {fmt(x['original_qdot_ratio_peak'])} | {fmt(x['outgoing_ball_speed_xy_mps'])} |")
    lines+=["","旧条件只能用于证明新增动作是否真的改变原机器人、以及是否修复已知失稳，不是新分布泛化。",
           "","## 4. 六组未重复精确球位/初速度配对","",
           "| 场景 | 足球XY偏移m / G1初始速度差m/s | 原v41严格任务 | 新v46严格任务 | 原/新实际qdot超限 | 新全身参考非零步 | 原/新出球m/s | 新真实PhysX步数 |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in a["results"]:
        old=row["original_v41"];new=row["landing_v46"]
        x,y=row["ball_offset_xy_m"];v=row["velocity_y_delta_mps"]
        lines.append(f"| {row['id']} | ({x:+.3f},{y:+.3f}) / {v:+.3f} | {mark(old['original_strict_pass'])} | {mark(new['original_strict_pass'])} | {old['original_qdot_excess_samples']} / {new['original_qdot_excess_samples']} | {new['landing_actual_nonzero_control_steps']} | {fmt(old['outgoing_ball_speed_xy_mps'])} / {fmt(new['outgoing_ball_speed_xy_mps'])} | {new['actual_physics_steps']} |")
    lines+=["",f"严格完整射门通过旧v41 {a['fixed_original_v41_full_pass_count']}/6，新v46 {a['new_landing_v46_full_pass_count']}/6。",
            f"原失败→新通过{a['improved_cases']}组，原通过→新失败{a['regressed_cases']}组；新控制真实额外动作球位{a['new_landing_actual_nonzero_case_count']}/6。"]
    for row in a["results"]:
        bad=[k for k,v in row["landing_v46"]["original_task_gates"].items() if not v]
        if bad:lines.append(f"- {row['id']} 新控制被拒的真实门槛："+", ".join(bad))
    lines+=["","所有试验原门槛：完成14秒/2800个5ms真实物理步，右正脚背合法冲量，整个足球越过球门，实际29关节速度都未超原限值，跑动接触连续且最终稳站，出球≥3.5m/s。只有全部同时成立才算任务成功。","",
     "## 5. 科学解释与下一阶段",
     "",
     "此控制器是接触后、脚落地前的全身运动约束工程原型，不是已经由职业人类正脚背爆发式运动标注学习出的新型协调策略。即便新旧严格成功数变化，也无法从每个只跑1次的6个新球位推断总体鲁棒率。",
     "后续科研必须引入高帧率、可校准的真实人体职业正脚背抽射3D姿态、脚球真实接触时刻、支撑脚和足面方向/角速度、冲量和球速；还需动作自然度盲评和机器人真实力矩/接触风险。不能再用微小参考动作或进球代替自然爆发力证据。",
     "",
     "原始G1重量模型、PD电机、几何接触、默认v5、v32 PPO和v41参考全都保持冻结；本版数据与影片原件仅在Tang本地。",
     "",
     "## 6. 源码、真实数据与可复现入口",
     "",
     "- 实际控制：tools/contact_landing_transfer_v046.py。",
     "- 物理运行：tools/run_contact_landing_v046.py、tools/run_landing_transfer_study_v046.py。",
     "- 严格原物理审计：tools/audit_contact_landing_v046.py；真实接触阶段分析：tools/analyze_contact_landing_v046.py。",
     "- 事前冻结协议：configs/landing_transfer_study_v046.json；隔离物理配置：configs/state_conditioned_strike_v17.json、configs/state_conditioned_strike_v17_zero.json。",
     "- 15次原G1 PhysX JSON、独立审计、SHA、真实控制与接触细节在Tang本地reports/v046_前缀下，第三方球员原始图片不上传GitHub。",
     "",
     "项目目标仍是学习职业人类全身协调规律、生成新动作、实现自然有爆发力正脚背抽射；本次只验证实实在在的全身控制与原始物理安全门槛，不声称完成这一论文目标。"]
    output.write_text("\n".join(lines)+"\n")
    print("V046_RESULT_REPORT_CREATED",len(lines),"baseline",a["fixed_original_v41_full_pass_count"],
          "new",a["new_landing_v46_full_pass_count"],
          "real_effect_cases",a["new_landing_actual_nonzero_case_count"])
if __name__=="__main__":main()
