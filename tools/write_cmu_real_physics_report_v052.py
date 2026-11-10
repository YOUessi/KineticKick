#!/usr/bin/env python3
"""Rebuild v0.1.52 text-only report from immutable original trial/audit files."""
import argparse
import hashlib
import json
from pathlib import Path

from build_cmu_g1_candidate_v052 import ROOT, OUTPUT, CONFIG, digest

P=ROOT/"reports/2026-11-11-v052-real-CMU-phase-controlled-G1-PhysX.md"
BASE=ROOT/"reports/v032_learned_center.json"
ZERO=ROOT/"reports/v052_cmu_10_02_center_zero_physx.json"
CLIPS=("10_02","10_05")


def load(file):return json.loads(file.read_text())


def zero_parity():
    a=load(BASE)
    b=load(ZERO)
    assert a["records"][0]["success"] and b["records"][0]["success"]
    assert len(a["traces"][0])==len(b["traces"][0])==2800
    missing=set(a["traces"][0][0])-set(b["traces"][0][0])
    assert missing=={"segment_quaternions_wxyz","segment_angular_velocity_w_rad_s",
        "support_foot_ground_force_w_N","support_foot_normal_force_w_N",
        "support_foot_friction_force_w_N","support_contact_capacity_exhausted",
        "root_velocity_w"}
    for original,control in zip(a["traces"][0],b["traces"][0]):
        assert all(original.get(k)==v for k,v in control.items())
    for k in ("contacts","kinematics","records","sequence_records",
              "running_evidence","stride_shot_evidence","stride_shot_success"):
        assert a[k]==b[k],k
    assert all(step["human_coordination_reference"]["applied_delta_rad"]==[0.]*29
               for step in b["joint_strike"]["feedback"])
    return missing


def main():
    if P.exists():raise FileExistsError("Do not overwrite a published v52 report")
    cfg=load(CONFIG)
    assert cfg["version"]=="0.1.52" and not cfg["verified_human_contact"]
    missing=zero_parity()
    rows=[]
    for clip in CLIPS:
        a=load(OUTPUT/clip/"candidate_audit.json")
        b=load(ROOT/("reports/v052_cmu_"+clip+"_physics_audit.json"))
        raw=ROOT/("reports/v052_cmu_"+clip+"_center_physx.json")
        assert b["physics_source_sha256"]==digest(raw)
        assert b["source_reference_sha256"]==a["output_sha256"]
        rows.append((clip,a,b))
    x=[]
    x.append("# KineticKick v0.1.52 — 真实 CMU 120Hz 人体相位先验 → G1 29DoF 参考 → 真实 PhysX 严格验收")
    x.append("")
    x.append("**版本基础：** `version/v0.1.51` (`d483886`)。本批次创建 `version/v0.1.52`。")
    x.append("**实验性质：** 两个真实人体 CMU 动捕片段均来自同一人；光学球体标记连续出球只能提供推断运动起点，不能提供真实脚-球碰撞或专业正脚背高冲量力学真值。本研究不把姿态代理得分称作自然度，也不把一次进球称作真实类人爆发抽射。")
    x.append("")
    x.append("## 1. 从上一轮到本轮的真实算法变更")
    x.append("")
    x.append("- 使用原 CMU `10_02`、`10_05` 两段 120Hz 骨架，经 v0.1.51 已审核 8D→6D 跨本体共享的几何方向余弦。分别以光学球标记速度起点为人类相位零点，以原 G1 接触力首触作为 G1 物理相位零点，两者**并不是真实同步触球**。")
    x.append("- 建立保持梯度的可微 G1 29DoF URDF 正运动学六通道映射，数值与 v51 冻结基线正运动学一致；各人体片段围绕自身初始姿态提取相对时间变化，经 `relative_gain=0.14` 加到原真实 G1 已执行射门参考，而不复制人体骨长/绝对关节角。")
    x.append("- 每个片段只执行事前固定的 96 步投影优化，约束左支撑脚、右摆动脚、关节位移、相邻时刻平滑性与足/球接触前提前关闭；`700×29` 控制参考满足 0.006rad 振幅、每20ms 0.0012rad 变化上限及原 URDF 关节范围。")
    x.append("- **没有训练新 PPO 或生成真实人体动量/冲量。** 将两条不同参考分别交给原 v41 独立 STYLE 激活通道；原 G1 脚球碰撞、PD 电机、相同足球初态、真实速度约束、v32 射后平衡策略均不修改。")
    x.append("- 运行额外禁用新参考的 ZERO 物理对照；原始 2800 步物理变量与 v32 基线在共有字段上逐值一致。旧 v32 还记录了 7 个新评估器未输出的附加遥测字段（未伪造），因此不能宣称完整 JSON 字节相同。")
    x.append("")
    x.append("## 2. 离线生成和原始 G1 物理结果")
    x.append("")
    x.append("| 指标 | CMU 10_02 | CMU 10_05 |")
    x.append("| --- | ---: | ---: |")
    fields=[
      ("离线协调目标误差降低",lambda a,b:f'{a["optimization"]["relative_proxy_reduction"]*100:.3f}%'),
      ("生成参考关节峰值",lambda a,b:f'{a["reference_safety"]["reference_peak_rad"]:.6f} rad'),
      ("非零参考控制帧",lambda a,b:str(a["reference_safety"]["active_controller_frames"])),
      ("真实执行新增控制帧",lambda a,b:str(b["nonzero_applied_reference_control_steps"])),
      ("真实控制带来的关节位姿最大变化",lambda a,b:f'{b["actual_robot_joint_pose_peak_delta_from_base_rad"]:.6f} rad'),
      ("实际出球水平速度",lambda a,b:f'{b["ball_xy_outgoing_speed_mps"]:.6f} m/s'),
      ("相对原 v32 球速变化",lambda a,b:f'{(b["relative_ball_xy_speed_vs_baseline"]-1)*100:+.4f}%'),
      ("首次右正脚背冲量",lambda a,b:f'{b["first_true_robot_instep_impulse_Ns"]:.6f} N·s'),
      ("真实关节速度超限样本",lambda a,b:str(b["real_5ms_joint_speed_limit_exceedance_samples"])),
      ("真实物理时长",lambda a,b:f'{b["original_full_14sec_5ms_steps"]*.005:.3f} s'),
      ("成功恢复站立",lambda a,b:f'{b["standing_recovered_s"]:.3f} s'),
      ("全部严格任务门禁",lambda a,b:"**PASS**" if b["task_success_all_gates"] else "**FAIL**"),
    ]
    for title,fn in fields:
        x.append(f'| {title} | {fn(*rows[0][1:])} | {fn(*rows[1][1:])} |')
    x.append("")
    x.append("两组均为真实物理右正脚背首次合法触球且球进入球门，两组都真实激活了新增控制参考（各5个20ms控制时刻），且实测 qdot 没有触碰原本速度上限。成功与否**由完整14秒及射后恢复的联合条件决定，不能因球进门而跳过跌倒。**")
    x.append("")
    x.append("## 3. 关键负结果：10_05 在 5.14 秒失败")
    x.append("")
    x.append("- `10_02` 完成 700 个控制步 / 2800 个真实 5ms 物理步；动作实际变化峰值约 1.30mrad，原物理任务和站立完整通过。")
    x.append("- `10_05` 在 **5.14s** 左右触发 `physical_nonfoot` 非脚部接触终止：仅执行 257 个控制步 / 1029 个物理步，`standing_seconds=0`；即使此前已进球，完整任务不合格。该失败没有被凑成700步或补出缺失物理状态。")
    x.append("- 两组物理六维人体相位代理 MSE 变化为 " +
             f'`{rows[0][2]["physically_realized_proxy_MSE_change"]:+.9f}` 与 ' +
             f"`{rows[1][2]['physically_realized_proxy_MSE_change']:+.9f}`。变化极小且不是人体自然度，更不足以抵消 10_05 的真实跌倒。")
    x.append("- 无单独重复随机种子，无法确定动力学敏感性/参考差异在统计上的鲁棒性；此次不能推销为“人类爆发抽射学习成功”，更不能升级正式控制器。")
    x.append("")
    x.append("## 4. 可复现证据与门禁")
    x.append("")
    x.append("- 预先冻结的19份人体/机器人物理源码和原始数据SHA清单：`configs/cmu_g1_phase_projection_v052.json`。")
    x.append("- 真正的可微6D→29DoF有界投影生成器：`tools/build_cmu_g1_candidate_v052.py`。")
    x.append("- 读取全部物理过程的独立门禁：`tools/audit_real_cmu_physx_v052.py`；失败样本也输出不可抹除的判定与物理首触、非脚部接触证据。")
    x.append("- 单元/集成回归：`tests/test_real_cmu_phase_v052.py`、`tests/test_real_cmu_physx_v052.py`。")
    for name,a,b in rows:
        x.append(f'- `{name}` 真值 SHA：生成参考 `{a["output_sha256"]}`；物理记录 `{b["physics_source_sha256"]}`。')
    x.append("- Tang 原 120Hz C3D、完整 G1 2800步 / 失败1029步记录、权重、PNG与NPZ 均保持本地，**不传入 GitHub 大文件**；GitHub 只保留可复现脚本、锁定条件、严格门禁和这份文字结论。")
    x.append("")
    x.append("## 5. 与最终论文目标的差距及下一个有科学价值的批次")
    x.append("")
    x.append("1. **输入证据不足。** 需要有精确脚球接触时刻/脚面姿态、左右支撑、骨盆-躯干-踢腿-摆臂全身相位及真实球速/旋转的数据；目前的CMU两条样本只是单人的光学球体运动代理，且未知是否属于顶级球员大力正脚背抽射。")
    x.append("2. **风格与安全必须联合建模。** 当前不是一个能主动维持射后平衡的条件动作生成器。下一版应把相位先验和真正可物理实现的动作/触球/射后恢复结合，在相同足球状态测 `t_contact−0.4s` 到 `t_contact+≥1.25s` 的qdot、支撑切换、根姿态、实际冲量和非脚部接触；保留 10_05 为失败回归案例，不能调参后将同一案例称作未见测试。")
    x.append("3. **独立验证自然度。** 需要按人/球位/动作类型划分数据，在固定任务成功与安全前提下做匹配的盲评对照，比较模仿策略、通用风格奖励与明确相位协调先验；进球率仅是功能门禁而不是论文主要创新。")
    x.append("")
    x.append("**本轮科研判决：** 已完成真实CMU人体 → 差异化29DoF G1参考 → 真实非零动作执行的实验链；一个物理PASS、一个物理FAIL。离线代理误差改善不能预测物理恢复，当前方法**暂不满足安全/自然的全身爆发抽射研究终点**。")
    x.append("")
    P.write_text("\n".join(x).rstrip()+"\n")
    print("V052_REPORT",P,"bytes",P.stat().st_size,"sha256",digest(P),
          "cases",[(n,v["task_success_all_gates"]) for n,_,v in rows])
if __name__=="__main__":
    main()
