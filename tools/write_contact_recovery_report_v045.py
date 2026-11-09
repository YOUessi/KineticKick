#!/usr/bin/env python3
"""Generate v45 transparent full study report from SHA-verified real PhysX only."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def value(x):return "N/A" if x is None else f"{x:.4f}"
def tag(x):return "**PASS**" if x else "FAIL"
def run():
    a=json.loads((ROOT/"reports/v045_governor_final_independent_audit.json").read_text())
    p=json.loads((ROOT/"configs/contact_recovery_governor_v045.json").read_text())
    if a["study_sha256"]!=sha("configs/contact_recovery_governor_v045.json"):
        raise ValueError("Original frozen v45 study changed")
    if len(a["results"])!=6 or len(a["known_old_development_diagnostics"])!=2:
        raise ValueError("Not all 15 prespecified physical runs audited")
    output=ROOT/"reports/2026-11-09-v045-contact-recovery-governor.md"
    if output.exists():raise FileExistsError(output)
    lines=[
 "# KineticKick v0.1.45 — 触球后29关节速度位置参考调节器：严格物理对照与失败审计",
 "",
 "**来源：** 由真实PhysX原始JSON、原G1 qdot限制及冻结研究脚本计算的六组配对任务结果，而不是主观选球。所有原始源SHA在 configs/contact_recovery_governor_v045.json 中冻结。",
 "",
 "## 一、实验目标、范围与原始物理约束",
 "",
 "v0.1.44的未来1.25秒动作条件MLP对危险关节超速出现大量false-safe，因此没有接入真实控制。v0.1.45重新用独立的后接触控制研究原型：维持原G1 v32 PPO小脑、v41训练集单视频弱人体姿态先验参考和原球/脚/执行器参数。在真实右脚触球、足球分离确认且距触球+0.10s至+1.25s窗口内，用实际29关节速度生成反方向的受限关节**位置目标**残差，幅度≤0.025rad、每20ms残差变化≤0.004rad。真实动力学/速度限值并不由参考幅度保证，最终仍由全部2800个原5ms PhysX子步判断。没有写足球速度、没有重新训练模型、没有使用v44不可靠风险模型或更换碰撞体。",
 "",
 "研究方案事前固定1次关闭模式中心零对照、2个已经观察过的旧球位诊断、随机种子4501产生的6个新的连续球位/初始速度三元组，每个新状态各用原v41和新调节器跑一次，总共15次独立14秒或提早终止的物理实验。任何场景即使失败仍然保留，不能事后调参数或选择性报告成功场景。与其他足球技能/职业人类动捕无关。",
 "",
 "## 二、零控制严格等价与日志差异",
 "",
 "在标准中心球位关闭v45通道时，700次原G1策略动作、原碰撞记录、运动学与原始进球结果和v41严格一致。两份来源不同：旧v41记录通过instrumented runner额外输出7个只读物理诊断字段，新v45使用未加这些附加列的core runner。因此审计只能声明**全部28个双方共有的实际物理字段在每个2800×5ms样本完全相同**，不应宣称两份完整JSON字节一致。禁用时新增目标残差700步全为0。",
 "",
 "## 三、两条旧场景机理探针",
 "",
 "| 旧开发场景 | 旧v41实测qdot是否超限 | 新v45严格成功 | 新调节器真实非零控制步 | 新max(qdot/原限制) | 新出球速度(m/s) |",
 "| --- | --- | --- | ---: | ---: | ---: |"]
    for z in a["known_old_development_diagnostics"]:
        g=z["governor"]
        known_prior_file=next(c["reference"] for c in p["known_development_diagnostics"] if c["id"]==z["case"])
        from audit_p0_dynamic_strike import limit_observations
        from strike_baseline_metrics import validated_episode
        old=json.loads((ROOT/known_prior_file).read_text())
        old_limit=limit_observations(validated_episode(old),old)["actual_speed_exceedance_sample_joint_count"]
        lines.append(f"| {z['case']} | {'是' if old_limit else '否'}（超限采样{old_limit}） | {tag(g['original_strict_pass'])} | {g['actual_governor_nonzero_steps']} | {value(g['real_joint_velocity_max_ratio'])} | {value(g['real_outgoing_ball_speed_xy_mps'])} |")
    lines+=["",
 "## 四、六组未重复精确参数的配对实测",
 "",
 "| 场景 | 足球XY偏移(m) / 根部y速度差(m/s) | 原v41严格任务 | 新v45严格任务 | 原/新实际qdot超限采样 | 原/新出球(m/s) | 新调节器非零控制步 | 新真实物理步 |",
 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in a["results"]:
        old=row["original_style"];now=row["contact_governor"]
        delta=f"({row['ball_offset_xy_m'][0]:+.3f},{row['ball_offset_xy_m'][1]:+.3f}) / {row['root_velocity_y_delta_mps']:+.3f}"
        lines.append(f"| {row['id']} | {delta} | {tag(old['original_strict_pass'])} | {tag(now['original_strict_pass'])} | {old['real_joint_velocity_overlimit_sample_joints']} / {now['real_joint_velocity_overlimit_sample_joints']} | {value(old['real_outgoing_ball_speed_xy_mps'])} / {value(now['real_outgoing_ball_speed_xy_mps'])} | {now['actual_governor_nonzero_steps']} | {now['actual_physx_samples']} |")
    old=a["paired_original_style_strict_pass"];new=a["paired_contact_governor_strict_pass"]
    delta=a["paired_change"]
    actual=sum(x["contact_governor"]["actual_governor_nonzero_steps"]>0 for x in a["results"])
    absent=sum(x["contact_governor"]["first_contact_s"] is None for x in a["results"])
    lines+=["",
 f"冻结的6个精确新参数组中，原v41严格通过 **{old}/6**，新接触后调节器严格通过 **{new}/6**。配对改进 {delta['style_to_governor_improved']}，退化 {delta['style_to_governor_regressed']}。新调节器真正改变关节动作的条件数 **{actual}/6**，完全没有原脚球接触的条件数 **{absent}/6**。",
 "",
 "严格任务成功依然要求：真实右正脚背接触、整个足球进入原球门、原G1关节速度无超限、运动完整14秒/2800物理步、无非法身体碰撞、原助跑与射门连续及最终站稳，另外球速≥3.5m/s。任意一项失败必须判FAIL，不允许用旧的球速或单次进球冲掉关节超速。",
 "",
 "## 五、科学解释与下一代工作",
 "",
 "本版的运动保护设计是接触后使用关节速度给PD关节位置增加小幅反向参考。它只是物理控制实验，不是模型预测控制，更不是职业人体的能量传递学习。5ms物理接触可能瞬间冲击原G1关节，即使关节速度在50Hz控制时刻远低于限制、上一控制步也没有足够快的可观测前兆；此时阈值式控制可能根本不激活。失效前已经提前发生的高度、足端接触几何和支撑转移，不能由事后关节速度反馈代替。",
 "",
 "下一版不能简单放宽限速/碰撞体，而应融合**脚着地时刻预测、质心/支撑脚负载/足端最低点下降速度和全身关节动力学**，提前规划触球后的随挥与落脚动作。应与强任务基线同时比较自然性、真实球速、关节功率和身体协调相位；没有足够职业正脚背3D/脚球接触/支撑反力原始人体数据，不能宣称专业暴力抽射已学成。",
 "",
 "本组新参数只是未用过的具体连续数值，原G1/球场/运动参考/任务仍相同，每个条件只运行一次，不代表统计可靠的跨环境泛化。本项目真实生产控制器和v32/旧v41都未升级。",
 "",
 "## 六、完整研发与证据",
 "",
 "- 控制器：tools/contact_recovery_governor_v045.py，测试：tests/test_contact_recovery_governor_v045.py、tests/test_contact_recovery_governor_v045_physical.py。",
 "- 物理Runner隔离复制：tools/run_contact_recovery_v045.py；不会更改已SHA冻结的历史tools/run_placement_strike.py。",
 "- 运行与记录：tools/run_contact_recovery_study_v045.py、configs/contact_recovery_governor_v045.json、configs/state_conditioned_strike_v16.json及v16_zero.json。",
 "- 独立真实PhysX审计：tools/audit_contact_recovery_v045.py；所有原始JSON/SHA/控制步记录在Tang reports/v045_开头的物理数据与final独立审计。",
 "- v44 false-safe权重没有加载进G1控制器。没有修改旧PPO。第三方人体素材、原始大体积物理JSON和实验视频不上传GitHub。",
 "",
 "**本版保留所有物理负结果，不因训练与单测通过宣称整体类人高爆发抽射技术实现。**"]
    output.write_text("\n".join(lines)+"\n")
    print("V045_RESEARCH_REPORT_CREATED",len(a["results"]),len(lines),"old",old,"new",new,"governor_effective_cases",actual,flush=True)
if __name__=="__main__":run()
