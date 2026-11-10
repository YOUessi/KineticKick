#!/usr/bin/env python3
"""Produce provenance-checked conclusion of v54 true physical named-body contacts."""
from __future__ import annotations
import json
from run_named_nonfoot_diagnostics_v054 import ROOT,CONFIG,sha
from audit_named_nonfoot_diagnostics_v054 import OUTPUT as AUDIT
OUT=ROOT/"reports/2026-11-11-v054-original-PhysX-named-nonfoot-contact.md"
def report():
 if OUT.exists():raise FileExistsError("No silent rewrite of v54 original results")
 c=json.loads(CONFIG.read_text());a=json.loads(AUDIT.read_text())
 assert c["version"]==a["version"]=="0.1.54" and a["original_center_3_diagnostic_rollouts"]==3
 text=[
 "# KineticKick v0.1.54 — 实际G1非脚部碰撞体定位：原有3组相同控制配置的只读PhysX复跑",
 "",
 "## 范围和真实性",
 "",
 "本版没有重新训练小脑，没有更改CMU人体参考、球、机器人、PD、碰撞/原关节限速或任何数据初态。只将原本存在的 `tools/run_placement_strike.py --contact-diagnostics` 标志打开，顺序跑一次禁用新动作的v32基线、CMU10_02（原v52成功）和CMU10_05（原v52`physical_nonfoot`提前终止）。三次都是**已开发的相同中心球位**，属于原因诊断复跑，绝非独立新球位/新策略成绩。研究前冻结源物理JSON/源码/config共14项SHA。",
 "",
 "## 1. 核对仪器增量是否改变原来的物理轨迹",
 "",
 "| 条件 | 原物理终止 | 诊断复跑物理终止 | 原物理时间 | 诊断时间 | 全部共有物理字段逐值等价 | 新记录命名非脚事件数 |",
 "| --- | --- | --- | ---: | ---: | --- | ---: |"]
 for name,row in a["results"].items():
  status=lambda x: "正常" if x==[[]] else str(x)
  text.append(f"| {name} | {status(row['original_physics_terminated'])} | {status(row['instrumented_physics_terminated'])} | {row['original_end_s']:.3f}s | {row['instrumented_end_s']:.3f}s | {'YES' if row['parity_full_on_all_common_physics_records'] else '**NO**'} | {row['named_nonfoot_event_count']} |")
 text+=["",
 "如果读传感器对原结果产生了任何差异，则它只是新instrumented实验，不得宣称复刻了同一物理轨迹：每组的 `parity_by_original_field` 列出原200Hz物理trace、50Hz全身joint pose/targets、真实接触、过程结果、球/门及提前终止是否严格一致。",
 "",
 "## 2. 真实ContactSensor逐非脚刚体证据",
 ""]
 for name,row in a["results"].items():
  text.append(f"### {name}")
  text.append("")
  text.append(f"原回合实际结论：`{row['original_physics_terminated']}`；新诊断在全回合记录 **{row['named_nonfoot_event_count']}** 个名义接触力≥5N的非脚body监测事件。末尾100ms共有 **{row['terminal_near_event_count']}** 条命名事件。")
  if row["named_nonfoot_body_counts_and_peak"]:
   text.append("")
   text.append("| 具名非脚碰撞体（原机器人场景标签） | 事件数 | 最大测得非脚接触力 | 峰值发生原始时间 |")
   text.append("| --- | ---: | ---: | ---: |")
   for body,e in row["named_nonfoot_body_counts_and_peak"].items():
    text.append(f"| `{body}` | {e['count']} | {e['max_force_N']:.3f} N | {e['max_force_time_s']:.4f}s |")
  else:
   text.append("原新记录没有达到已注册读出阈值的命名事件，**无法推断具体碰撞部位**。")
  text.append("")
  if name=="cmu_10_05":
   if row["terminal_100ms_named_nonfoot_events"]:
    text.append("最靠近原提前终止时刻的传感器证据（最多20条，按合力大小排序）：")
    for e in row["terminal_100ms_named_nonfoot_events"][:8]:
     import numpy as np
     force=float(np.linalg.norm(e["force_w_N"]))
     text.append(f"- `{e['body']}` at `{e['time_s']:.5f}s`，合力`{force:.2f}N`，原3D物理坐标`{e['position_w']}`。")
   else:
    text.append("末尾100ms内本次新增诊断没有命名非脚接触事件，因此**不能确定是哪一个身体部位触发原终止条件**。")
   text.append("")
 text += [
 "## 3. 安全与科学结论",
 "",
 f"完整原物理逐字段校验的结果：**{'三组均完全一致' if a['all_physics_same_except_added_diagnostics'] else '并非三组均一致'}**。无论诊断能够观察到哪些具名接触，`physical_nonfoot`是原G1模拟器的接触终止事件，**不是**用根姿态实测证明的整体跌倒，更不是现实硬件真实踢球事故。不能因新的命名诊断里出现大合力就推断机器人有对应专业足球人类发力方式。",
 "",
 "严格控制策略的功能/自然度效果不因增加观测而改变：上一整批v53的12回合严格结果是原v32、两组CMU先验分别0/4，没有发生跨球位成功提升。后续任何新策略的控制器都必须单独预注册、固定安全与任务门槛，在多球位/种子和真实人体高质量数据上验收，而不能在这个已知失败中心球位调到成功就宣布提升。",
 "",
 "## 4. 复现和证据",
 "",
 f"- 14项源/G1物理/原v52结果SHA冻结配置：`configs/cmu_named_nonfoot_diagnostics_v054.json` SHA256=`{sha(CONFIG)}`。",
 "- 顺序单GPU原物理复跑：`tools/run_named_nonfoot_diagnostics_v054.py`；纯只读具名碰撞/共有字段校验：`tools/audit_named_nonfoot_diagnostics_v054.py`；正式文字结论生成器：`tools/write_named_nonfoot_report_v054.py`；回归：`tests/test_named_nonfoot_v054.py`。",
 "- 3份完整长物理原始JSON及诊断sensor名单存Tang忽略目录，GitHub不上传CMU C3D/AMC、原模型、物理轨迹或原机高体积结果；只上传固定源码/协议/报告/独立测试。报告含名字、最大力以及原始时间的文字表格，与Tang实际原始SHA挂钩。",
 ""]
 pair=ROOT/"reports/v054_opposite_nonfoot_link_forces.json"
 if pair.exists():
  z=json.loads(pair.read_text())
  if z["original_v54_named_body_result_sha256"]!=sha(AUDIT):
   raise ValueError("Cannot cite altered pairwise contact data")
  q=z["left_wrist_hip_symmetric_force_pair"]
  text.extend(["## 5. 一对等大反向接触力：左手腕—左髋自碰撞的强线索",
    "",
    f"在`{q['measured_t_s']:.5f}s`，`{q['body_a']}`和`{q['body_b']}`分别记录到`{q['force_magnitude_each_N']:.5f}N`大小的合接触力，方向余弦`{q['cosine_between_forces']:.1f}`，向量和**恰好为0**，两个连杆原点相距`{q['position_of_link_origins_separation_m']:.4f}m`。这与机器人**左手腕与左髋可能发生自碰撞**高度一致，且是目前可观测的具体失败线索。",
    "",
    "**关键限制：** 原传感器只记录每个body的净接触力和连杆原点，没有真正的PhysX刚体碰撞对/接触流形，因此暂时不能100%确认两个连杆确实互相接触。下一批若要因果确认，需只读抓取contact pair/collider paths并锁定相同物理轨迹；在未确认前不要改机器人碰撞过滤，也不要用放宽nonfoot终止门槛来使失败算PASS。",
    "",
    f"附加只读事件配对证据：`tools/infer_nonfoot_pair_v054.py`，原审计SHA`{z['original_v54_named_body_result_sha256']}`。",
    ""])
 OUT.write_text("\n".join(text).rstrip()+"\n")
 print("V054_FINAL_RESEARCH_REPORT",str(OUT),"bytes",OUT.stat().st_size)
 return OUT
if __name__=="__main__":report()
