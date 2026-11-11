#!/usr/bin/env python3
"""Evidence-only research markdown; never regenerates/changes raw physics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
report=ROOT/"reports/2026-11-11-v063-real-29dof-synthetic-phase-ablation.md"
audit=ROOT/"reports/v063_independent_recheck.json"
if report.exists():raise FileExistsError(report)
a=json.loads(audit.read_text())
p=json.loads((ROOT/"configs/g1_phase_batch_v063.json").read_text())
progress=json.loads((ROOT/"reports/v063_batch_progress.json").read_text())
lines=[
"# KineticKick v0.1.63｜真实PhysX的合成全身相位协同动作反例 + 200Hz时间对齐审计",
"",
"**日期**：2026-11-11。**父 SHA**："+p["parent_sha"]+"。**分支**：version/v0.1.63。",
"**运行前协议 SHA256**："+a["protocol_sha256"]+"。**预实验代码提交**："+a["frozen_git_sha"]+"。 ",
"",
"## 主要结论与科学边界",
"",
"**本版研发了新的真实G1 29DoF相位耦合控制接口并执行原始物理对照。"
"该候选是有意构造的合成（非真人学习）多关节相位消融，绝不是自然踢球的已训练生成模型或职业人类强射动作。"
"已在全量物理、实际控制输入和不合格样本上按原规则裁决，不能因为球进门而忽略不合法的正脚背冲量。**",
"",
f"- 串行执行原29DoF自由根部Isaac/PhysX：**{a['real_PhysX_attempts']}次尝试，{a['scorable']}次有可审计原200Hz真实物理，{a['strict_successes']}次全门禁PASS**。",
"- 4种球/运动状态各配对原v32和合成全身相位，共4对；首对为已知中心开发状态，余3对采用事前冻结且含有效reset扰动的新种子3107/3119/3137。",
"- 固定原200Hz/20ms控制、原目标策略和恢复PPO、29关节速度限值、全14s、合法右脚正脚背≥95%冲量、球速≥3.5m/s、实际进门以及无非脚碰撞与完成恢复的原有门禁。",
"- 本次合成时间基函数幅度预算最高0.075rad，20ms每关节变化≤0.006rad，但实际安全门控抑制了大部分意图；不能以合成目标幅度当作实际关节动作，也不能以运动学代理当作“自然性”。",
"- 只读观察器沿用GPU v62 mass-free资格路径。原模拟时钟标签比状态提前约10ms；v63用原真实200Hz逐帧同序29关节位置/速度比对给出 state_aligned_time_s，保留 raw_sim_clock_s 原样。**仅后处理时间索引认证，未改PhysX模拟时钟或物理控制**。",
"- 仍未取得GPU原G1左腕—左髋直接原生actor/collider/contact manifold；CPU骨盆—右髋真值不用于替代。安全门中腕—髋连杆原点距离仅是**拒绝式代理**，不是碰撞间隙的证据。",
"",
"## 冻结对照：真实物理与严格结果",
"",
"| 案例 | Seed | 策略 | 200Hz帧 | 终点(s) | 严格结果 | 完成触球后≥1.25s | 原29DoF实测qdot峰值/限值 | 非脚净力峰值(N) |",
"| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: |"
]
for row in a["episodes"]:
    ident=row["id"]
    case=next(x for x in p["cases"] if x["id"]==ident)
    if row["status"]!="VALID_G1":
        lines.append(f"| {ident} | {case['seed']} | {case['policy']} | — | — | 不可评分 | — | — | — |")
        continue
    m=row["actual_motion"]
    lines.append("| {} | {} | {} | {} | {:.3f} | {} | {} | {:.4f} | {:.2f} |".format(
        ident,case["seed"],case["policy"],m["physical_rows"],
        m["end_time_s"],"PASS" if row["strict_pass"] else "FAIL",
        "是" if m["impact_window_post_1p25s_available"] else "否/截断",
        m["max_qdot_ratio"],m["maximum_net_nonfoot_force_N"]))
lines.extend(["","## 成对差异与候选动作真实激活","",
    "| 预注册组 | 配对初始物理认证 | v32严格 | 合成严格 | 实际非零控制步 | 实际最大关节增量(rad) |",
    "| --- | --- | --- | --- | ---: | ---: |"])
for g in a["independent_pairs"]:
    lines.append("| {} | {} | {} | {} | {} | {} |".format(
        g["case"],"PASS" if g["certified_pair"] else "FAIL",
        g.get("original_v32_pass","不可评分"),g.get("synthetic_candidate_pass","不可评分"),
        g.get("candidate_effective_nonzero_steps","—"),
        ("%.5f"%g["candidate_max_joint_applied_rad"]) if "candidate_max_joint_applied_rad" in g else "—"))
lines.extend(["",
"任何配对初始0–0.5s原始25类物理字段不一致，必须拒绝该对策略的因果比较。候选在某个种子成功或失败不证明自然性；自然动作评价仍需高质量人体完整动作真值和盲评。",
"",
"## 关键失败归因",
""])
for row in a["episodes"]:
    if row["status"]!="VALID_G1" or row["strict_pass"]:continue
    shot=row["original_evaluation_record"]
    lines.append("- **{}**：球门 {}，原请求脚背冲量占比 {}，触球时刻 {}s，实际触球面 {}，"
        "完整恢复 {}，终止 {}。失败不能因球已进门而重写PASS。".format(
        row["id"],shot.get("goal"),shot.get("requested_surface_impulse_fraction"),
        shot.get("contact_time_s"),shot.get("dominant_surface"),shot.get("measured_surface"),shot.get("recovered"),
        row["termination"]))
lines.extend(["",
"## 独立检验、仪器与来源记录",
"",
"- 源文件/策略和协议独立SHA冻结。原始大JSON、NPZ、G1模型权重、CMU/C3D源文件只在Tang，结果write-once并经过独立SHA复核，GitHub仅保留代码、协议、测试和本报告。",
"- 物理积分是实测5ms；全身COM和足地支撑力属于G1仿真，而非人类GRF。全身四组髋、躯干、摆腿、摆臂关节的实际qdot RMS与位移幅度仅是机器人的运动学代理，**未验证人体近端到远端发力/碰撞功率**。",
"- v63初版 preregistered audit_v063_phase_batch.py 的25类帧比较接口实参写错：误将frame-list传入要求raw-dict的 compare_raw；**原实验协议与日志继续只读保留**，后新增 audit_v063_post_registered.py、独立测试修正该读取错误，不影响冻结策略、物理动作或原始结果。",
"- 本地绘图 media/v063_*_real_dynamics.png 通过原始真实200Hz和SHA生成，未提交大媒体。",
"",
"## 人体数据资格与后续论文假设",
"",
"本机CMU 120Hz六段踢球，6/6可用3D骨架，只有2段连续球标记运动，0/6有精确专业足背接触/人体GRF真值；本批未进行人体模仿训练，合成相位动作不能替代学得的自然强射。最新数据候选及排除理由见 docs/V063_HUMAN_DATA_QUALIFICATION.md。",
"",
"下一版研究首先应补充真实机器人自碰撞接触对真值和状态可用性，随后建立多相位人体运动生成（助跑—种植脚—髋/躯干—摆腿摆臂—随挥落脚），利用支撑/间距代理预警与原碰撞硬门禁；人体源必须按被试划分训练/验证/未见测试。学习对象是**全身时序协调和动作自然性**，不是单独进球或1mrad小参考。",
"",
"## 复现命令（只读/不覆盖）",
"",
"    /home/you/conda_env/env_isaaclab/bin/python tools/run_v063_phase_batch.py",
"    /home/you/conda_env/env_isaaclab/bin/python tools/audit_v063_post_registered.py",
"    /home/you/conda_env/env_isaaclab/bin/python tools/plot_v063_measured_physics.py",
"    PYTHONPATH=.:tools:third_party/HumanoidSoccer/source/whole_body_tracking /home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests",
"",
"**重要**：第一条命令在全部8组已记录时只验证并保留SHA，不会重新覆盖历史证据；新物理实验必须另开后续version分支和新路径。"
])
lines.extend(["","## 球速与合法脚背接触：原200Hz球轨迹的独立窗口代理","","| 案例 | 第一次触球后5–105ms窗口球速峰值(m/s) | 实测触球分类 | 脚背冲量比例 | 原严格结论 |","| --- | ---: | --- | ---: | --- |"])
for case in p["cases"]:
    path=ROOT/case["output"]
    if not path.exists():continue
    raw=json.loads(path.read_text())
    if len(raw.get("traces",[]))!=1:continue
    record=raw["records"][0];impact=record.get("contact_time_s")
    if impact is None:continue
    speeds=[]
    for f in raw["traces"][0]:
        if impact+.005<=f["time_s"]<=impact+.105:
            speeds.append(sum(float(v)**2 for v in f["ball_velocity"])**.5)
    lines.append("| {} | {} | {} | {:.6f} | {} |".format(
        case["id"],("%.5f"%max(speeds)) if speeds else "不可测",
        record.get("measured_surface"),record.get("requested_surface_impulse_fraction",float("nan")),
        "PASS" if raw["stride_shot_success"] else "FAIL"))
lines.extend(["","这里的5–105ms球速峰值是独立从200Hz逐帧读取的**窗口代理**，原严格评测器是否≥3.5m/s由冻结的原全任务裁决，不应借窗口代理重算或放宽；球进门但脚背比例不合格仍FAIL。"])
report.write_text("\n".join(lines)+"\n")
print(report,len(lines),"lines")
