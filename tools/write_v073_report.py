#!/usr/bin/env python3
"""Generate original v73 human temporal generalization and G1 offline-IK report.

Reads only independently SHA-validated CPU training+source evidence. All
figures stay Tang local. Does NOT train, simulate, actuate or make new claims.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def report(brain,proposal):
 auditp=ROOT/"reports/v073_independent_fullhuman_g1_audit.json"
 protocolp=ROOT/"configs/v073_human_temporal_learning.json"
 if not auditp.exists():raise FileNotFoundError("Requires independent 6-source, 2-checkpoint, G1 IK audit")
 audit=json.loads(auditp.read_text())
 cfg=json.loads(protocolp.read_text())
 results=json.loads((brain/"human_motion_evaluation.json").read_text())
 proposals=json.loads((proposal/"human_brain_to_G1_reference_audit.json").read_text())
 if audit["source_protocol_SHA256"]!=sha(protocolp) or results["protocol_sha256"]!=sha(protocolp):
  raise ValueError("Human study SHA drift")
 out=ROOT/"reports/2026-11-11-v073-subject-isolated-CMU-human-temporal-negative.md"
 if out.exists():raise FileExistsError("Never overwrite v73 scientific report")
 names={"persistence":"末帧保持","linear_extrapolation":"等速外推","ordered":"时序GRU",
        "history_shuffle_negative":"输入历史打乱GRU"}
 val=results["validation_subject_10_clip_06_per_method"]
 test=results["test_subject_11_per_method"]
 v0=val["ordered"]["normalized_xyz_landmark_euclidean_MAE"]
 vb=val["linear_extrapolation"]["normalized_xyz_landmark_euclidean_MAE"]
 h0=test["ordered"]["normalized_xyz_landmark_euclidean_MAE"]
 hb=test["linear_extrapolation"]["normalized_xyz_landmark_euclidean_MAE"]
 lines=[
 "# KineticKick v0.1.73｜真人120Hz全身时序网络、跨被试失败与G1运动学映射",
 "",
 "**科学判决：两层GRU在同被试验证上略优于等速外推，但在真正独立的另一被试上误差接近其两倍。全身时序网络、两条离线G1关节参考及独立审计真实运行；本轮新真实PhysX回合0，不能据此宣称自然高爆发抽射或任一安全改善。**",
 "",
 "## 真实源与防泄漏划分",
 "",
 "- Tang原始CMU 120Hz光学运动6条，共两名被试、3358个源帧；每帧19个骨架标记衍生相对3D点和8个角度余弦，共65维。源代码/NPZ逐文件SHA冻结。CMU没有可靠职业/优秀运动员资质、足球瞬间冲量、校准球心、同步足底GRF或人体功率真值。",
 "- 被试10中10_01/10_02/10_03/10_05四段用于训练，10_06唯一用于模型早停/验证选择；被试11的11_01仅在两套模型选完之后才从原始NPZ读取评分。训练343个重叠滑窗，验证45、跨被试测试47；这些窗不是独立的343/45/47名球员。",
 "- 输入过去24帧即200ms，输出未来12帧即100ms；训练两层GRU残差预测器，使用同种子、同参数、同优化预算的打乱时序GRU负控制；另以历史末帧保持/固定4帧速度外推作两条传统基线。",
 f"- 冻结协议SHA256: {sha(protocolp)}；测试对象仅一位新被试，没有可证实的统计泛化效力或真人动作自然度标注。",
 "",
 "## 120Hz真人全身位置预测真实对照",
 "",
 "| 方法 | 同被试验证3D MAE | 跨被试11_01 3D MAE | 跨被试8几何余弦 MAE |",
 "|---|---:|---:|---:|"]
 for key in names:
  lines.append(f"| {names[key]} | {val[key]['normalized_xyz_landmark_euclidean_MAE']:.6f} | {test[key]['normalized_xyz_landmark_euclidean_MAE']:.6f} | {test[key]['observed_8_segment_cosine_MAE']:.6f} |")
 lines += ["",
  f"同被试验证中时序GRU对等速基线改善{100*(1-v0/vb):.2f}%，但跨被试11_01的误差为{h0:.6f}，等速基线只有{hb:.6f}，GRU相对差{100*(h0/hb-1):+.2f}%。打乱时序GRU在验证与测试上均表现较差，说明模型尝试利用时间顺序，但跨人泛化失败。禁止在已经观察过的唯一被试11上继续调参并称为新留出结果。",
 "",
 "全部3D误差的单位是原始CMU人体骨盆中心化、形态尺度归一化的相对位置，不是米、G1末端距离、运动美感或球速。自监督人类时序预测不直接提供职业蹬地、脚背接触/GRF或生物力学真值。",
 "",
 "## 从学习到的真人时序到29DoF G1的可执行参考边界",
 "",
 "本版对训练被试10_02，取原光学球标记速度变化前24帧，递推48个真实120Hz人体相位帧，并按既有v64可微G1 URDF/关节限位/末端位置约束转换为700×29个50Hz关节参考；负控制打乱时间历史的GRU使用相同IK、原G1记录和物理参数。",
 "",
 "| 模型 | 原G1 20ms 非零参考帧 | 非零G1关节数 | 最大参考角 | 新增静态位置越界 |",
 "|---|---:|---:|---:|---:|"]
 for name in ("ordered","history_shuffle_negative"):
  r=audit["original_g1_29dof_URDF_and_joint_envelope"][name]
  lines.append(f"| {names[name]} | {r.get('active_G1_20ms_command_reference_frames','—')} | {r.get('active_G1_joints','—')} | {r.get('actual_kinematic_reference_max_delta_rad',0):.6f} rad | {r.get('static_URDF_new_q_margin_violations','—')} |")
 lines += ["",
  "离线验收保留原右腿6个正脚背关节通道完全0增量，保持源0.055rad峰值、0.004rad/20ms变化、源URDF静态关节余量与触球前释放。**这只是几何关节参考，不是G1实际执行成功**，没有真实29DoF执行器/跑步支撑/球碰撞/足底载荷/射后14秒安全验证，也没有真实碰撞对或人类GRF认证。",
 "",
 "## 原始负例、完整实验与代码",
 "",
 "- 首次训练在模型启动前因future_frames与forecast_frames字段不一致停止；没有产生权重或评估测试。按原SHA另行记录字段修正提交，保持人/片段划分和全部超参不变。",
 "- 第二次训练完成但在写检查点时混用绝对/相对路径，保留第一个局部模型及失败日志，在独立绝对路径输出命名空间按相同冻结脚本/参数完整重跑；源训练SHA和6个CMU原NPZ均未覆盖。",
 "- 第一次离线G1投影因分组字段以及缺少独立工作树原URDF资产而无法输出。仅创建SHA一致的本地原URDF只读链接，在独立新输出名空间生成两套参考；原G1真实物理JSON和原工作树未触碰。",
 "- 复现入口：tools/train_human_temporal_brain_v073.py，tools/retarget_v073_human_brain_to_g1.py，tools/audit_v073_human_brain_g1.py，tools/write_v073_report.py；专项tests/test_v073_*.py与版本化训练协议configs/v073_human_temporal_learning.json。",
 f"- 独立审计记录SHA256: {sha(auditp)}；模型checkpoint和原始原CMU文件有独立SHA，自身只保存Tang，不上传GitHub。图保存在Tang本地media/v073_*.png。",
 "",
 "## 下一整批研究门槛",
 "",
 "必须新增具有真实脚球冲量/明确支撑事件、能分选手隔离的高质量足球射门完整身体数据，避免在已经看过的唯一被试11上调参；在真正新的选手/场景验证通过后，再研究髋—躯干—摆臂—摆腿—支撑换足的端到端生成。GPU上的Isaac Kit启动与G1 native碰撞pair仍需独立修复；任何生成控制器必须在独立GPU恢复后执行相同球位、有效种子、真实29DoF qdot、合法正脚背≥95%、球速、球门、非脚碰撞和14秒恢复的完整物理门禁。",
 "",
 "**不可升级结论：跨被试泛化失败，本轮新增真正PhysX 0次；默认v32/v5不能替换，真人专业强力抽射没有实现。**",
 ""]
 out.write_text("\n".join(lines).rstrip()+"\n")
 print("V073_RESEARCH_REPORT",str(out),"SHA",sha(out))
 try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  fig,ax=plt.subplots(figsize=(10,5))
  keys=list(names)
  x=np.arange(len(keys));w=.36
  ax.bar(x-w/2,[val[k]["normalized_xyz_landmark_euclidean_MAE"] for k in keys],w,label="subject10 val")
  ax.bar(x+w/2,[test[k]["normalized_xyz_landmark_euclidean_MAE"] for k in keys],w,label="subject11 test")
  ax.set_xticks(x,list(names.values()))
  ax.set_ylabel("Pelvis-relative scale-normalized 3D error")
  ax.legend();ax.grid(alpha=.25,axis="y")
  fig.tight_layout()
  root=ROOT/"media";root.mkdir(exist_ok=True,parents=True)
  image=root/"v073_human_subject_val_vs_test.png";fig.savefig(image,dpi=130);plt.close(fig)
  print("V073_LOCAL_FIGURE",str(image),"SHA",sha(image))
  fig,ax=plt.subplots(figsize=(10,4))
  for name in ("ordered","history_shuffle_negative"):
   item=proposals["models"][name]
   if item["status"]!="OFFLINE_G1_REFERENCE_ONLY_NOT_PHYSX":continue
   with np.load(ROOT/item["source_generated_G1_29dof_file"],allow_pickle=False) as z:
    q=z["joint_delta_rad"]; t=z["times_s"]
   ax.plot(t,np.max(abs(q),axis=1),label=names[name])
  ax.set_xlim(.85,1.6)
  ax.set_xlabel("G1 controller timeline (s)")
  ax.set_ylabel("Offline original G1 max |q delta| rad")
  ax.set_title("NOT physical G1 motor execution; CPU URDF IK only")
  ax.grid(alpha=.2);ax.legend();fig.tight_layout()
  image=root/"v073_offline_g1_29dof_human_proposals.png";fig.savefig(image,dpi=130);plt.close(fig)
  print("V073_LOCAL_FIGURE",str(image),"SHA",sha(image))
 except ImportError:
  print("V073_OPTIONAL_PLOT_UNAVAILABLE")
 return out

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--brain",type=Path,required=True)
 p.add_argument("--proposal",type=Path,required=True)
 a=p.parse_args()
 report(a.brain,a.proposal)
