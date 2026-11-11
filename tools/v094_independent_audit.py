#!/usr/bin/env python3
"""Independent proof ledger for frozen v94 isolated PhysX controls + historical G1.
Only read data, source-check SHA; never infer actual collider paths from opposite net forces.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def source_sha_check(root:Path, names:list[str], commit:str)->dict:
 out={}
 for name in names:
  original=subprocess.check_output(["git","show",commit+":"+name],cwd=root)
  on_disk=(root/name).read_bytes()
  if original!=on_disk:raise ValueError("FROZEN_PROTOCOL_DRIFT:"+name)
  out[name]=hashlib.sha256(on_disk).hexdigest()
 return out
def exact_parity(a,b):
 x=a.get("runtime_z_vz",[]);y=b.get("runtime_z_vz",[])
 if len(x)!=180 or len(y)!=180: return {"valid":False,"reason":"UNVERIFIED_FRAMES"}
 if any(r[0]!=i+1 for i,r in enumerate(x)) or any(r[0]!=i+1 for i,r in enumerate(y)):
  raise ValueError("NONCONTIGUOUS_PHYSICS_STEPS")
 v=max(abs(float(s[j])-float(t[j])) for s,t in zip(x,y) for j in (1,2))
 return {"valid":True,"frames":180,"max_abs_z_vz":v,"exact_at_9_digits":v==0}
def evaluate(root=ROOT,v91_root=None):
 cfg=json.loads((root/"configs/v094_frozen_matrix.json").read_text())
 summary=json.loads((root/"reports/v094_frozen_contact_batch.audit.json").read_text())
 assert cfg["version"]==summary["version"]=="v0.1.94"
 assert len(cfg["matrix"])==len(summary["trials"])==6 and not summary["unexecuted"]
 assert summary["original_G1_shots"]==cfg["new_original_G1_experiments"]==0
 assert summary["protocol_sha256"]==sha(root/"configs/v094_frozen_matrix.json")
 frozen=[
  "configs/v094_frozen_matrix.json","docs/V094_FROZEN_PROTOCOL.md",
  "tools/v094_box_pair_probe.py","tools/v094_box_batch.py","tests/test_v094_contact_contract.py"]
 hashes=source_sha_check(root,frozen,summary["frozen_commit"])
 p={};cases={}
 for row,expected in zip(summary["trials"],cfg["matrix"]):
  for field in ("case","device","capture","layout","report_mode"):
   assert row[field]==expected[field], "FROZEN_MATRIX_ORDER_MISMATCH"
  case=row["case"];stem=root/"reports"/("v094_"+case)
  out=Path(str(stem)+".json");log=Path(str(stem)+".log")
  side=Path(str(stem)+".audit.json")
  raw=json.loads(out.read_text());a=json.loads(side.read_text())
  assert a==row and sha(out)==row["raw_sha256"] and sha(log)==row["log_sha256"],"SOURCE_DRIFT"
  assert row["recorded_runtime_evidence"] and raw["actual_runtime_frames"]==180
  assert row["physically_confirmed"] and row["rc"]==0 and not row["timeout"]
  assert row["correct_report_api_contract"]
  trace=raw["runtime_z_vz"]
  assert len(trace)==180 and all(r[0]==i+1 and all(math.isfinite(v) for v in r[1:])
    for i,r in enumerate(trace))
  assert bool(raw.get("tensor_pair_witness"))==bool(row["force_positive"])
  assert bool(raw.get("native_pair_witness"))==bool(row["native_positive"])
  phys=raw["observed_collision_dynamics"] if row["layout"]=="colliding" else raw["observed_noncollision_dynamics"]
  assert phys,"RUNTIME_PHYSICS_NOT_ACTUALLY_EVIDENCED"
  points=raw.get("tensor_contact_points",[])
  for z in points:
   assert z["force_magnitude_N"]>0 and len(z["point_w"])==len(z["normal_w"])==3
   assert all(math.isfinite(v) for v in z["point_w"]+z["normal_w"])
  if raw.get("tensor_pair_witness"):
   view=raw["contact_view_shape"]
   assert view["bodies"]==view["filters"]==1
   assert "/World/ProbeBall" in view["sensor_paths"] 
   assert any("/World/ProbeBox" in s for s in view["filter_paths"])
   assert raw["tensor_contact_counts"]>0 and raw["tensor_peak_force_N"]>0 and points
  if row["layout"]=="separated":
   assert not raw.get("tensor_pair_witness") and raw.get("tensor_contact_counts",0)==0
  p[case]=raw
  cases[case]={
   "source_sha256":row["raw_sha256"],"log_sha256":row["log_sha256"],
   "real_physics_steps":len(trace),"physics_confirmed":True,"cuda":row["device"]=="cuda:0",
   "contact_view_shape":raw["contact_view_shape"],"tensor_points_sampled":len(points),
   "tensor_counts":raw["tensor_contact_counts"],"tensor_force_peak_N":raw["tensor_peak_force_N"],
   "tensor_pair_positive":bool(raw["tensor_pair_witness"]),
   "native_direct_collider_manifold_certified":bool(raw["native_pair_witness"]),
   "native_callback_calls":raw["native_counts"]["callback_calls"],
   "native_header_count":raw["native_counts"]["poll_headers"],
   "close_marker":row["lifecycle_marker_app_closed"],
   "z_start":trace[0][1],"z_end":trace[-1][1]}
 parity={
  "CPU_pose_OFF_vs_tensor_ON":exact_parity(p["cpu_pose_off"],p["cpu_tensor_on"]),
  "CUDA_pose_OFF_vs_tensor_ON":exact_parity(p["gpu_pose_off"],p["gpu_tensor_on"]),
  "CUDA_pose_OFF_vs_native_ON":exact_parity(p["gpu_pose_off"],p["gpu_native_on"])}
 assert all(v["valid"] for v in parity.values())
 if not all(v["exact_at_9_digits"] for v in parity.values()):
  raise ValueError("READ_ONLY_INSTRUMENTATION_PHYSICS_PARITY_FAILED")
 historical={}
 if v91_root:
  spec=importlib.util.spec_from_file_location("v091_independent_g1",root/"tools/v091_independent_dynamics.py")
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  m.ROOT=Path(v91_root)
  for case in m.CASES:
   data,_=m.inspect(case)
   historical[case]={
    "original_g1_physx_sha256":data["original_raw_sha256"],
    "real_200hz_frames":data["real_200hz_frames"],
    "terminal_annotation_rows":data["unscored_terminal_annotation_rows"],
    "first_foot_ball_contact_s":data["first_foot_ball_contact_s"],
    "post_1p25s_complete":data["contact_window_post1p25_complete"],
    "true_29dof_velocity_limit_ratio":data["original_max_all29_measured_qdot_ratio"],
    "instep_impulse_fraction":data["computed_true_instep_impulse_fraction_from_raw_surface_bins"],
    "nonfoot_contact_N":data["original_nonfoot_max_net_N"],
    "full_hard_gate_pass":data["independent_gates"]["strict_pass"],
    "hard_gate_failures":data["independent_gates"]["failed"]}
  assert len(historical)==4 and not any(x["full_hard_gate_pass"] for x in historical.values())
 return {
  "version":"v0.1.94","frozen_protocol_sha":summary["protocol_sha256"],
  "frozen_source_sha256":hashes,"original_G1_new_shots":0,
  "new_isolated_real_PhysX_probe_runs":len(cases),
  "true_200hz_probe_steps":sum(x["real_physics_steps"] for x in cases.values()),
  "physical_pair_settling_positive_cases":sum(x["physics_confirmed"] and k!="gpu_separated_on" for k,x in cases.items()),
  "verified_5m_separated_negative_cases":1,
  "tensor_direct_filtered_pair_positive_count":sum(x["tensor_pair_positive"] for x in cases.values()),
  "cpu_tensor_positive":cases["cpu_tensor_on"]["tensor_pair_positive"],
  "gpu_tensor_positive":cases["gpu_tensor_on"]["tensor_pair_positive"],
  "gpu_separated_false_positive":cases["gpu_separated_on"]["tensor_pair_positive"],
  "native_collider_path_pair_positive_count":sum(x["native_direct_collider_manifold_certified"] for x in cases.values()),
  "original_G1_wrist_hip_native_manifold_certified":False,
  "zero_native_headers_does_not_prove_no_contact":True,
  "tensor_witness_is_not_native_original_actor_pair":True,
  "read_only_parity":parity,
  "historical_original_G1_heldout":historical,
  "historical_original_G1_strict_pass_count":sum(x["full_hard_gate_pass"] for x in historical.values()),
  "new_learned_human_coordination_actuation_episodes":0,
  "app_close_returned_count":sum(x["close_marker"] for x in cases.values()),
  "cases":cases}
def plot(root:Path):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,5.5))
 for case in ("gpu_pose_off","gpu_tensor_on","gpu_separated_on","cpu_tensor_on"):
  o=json.loads((root/"reports"/("v094_"+case+".json")).read_text())
  t=[q[0]*.005 for q in o["runtime_z_vz"]]
  z=[q[1] for q in o["runtime_z_vz"]]
  ax.plot(t,z,label=case,linewidth=1.3)
 ax.axhline(.38,linestyle="--",linewidth=.8,label="ball-on-box expected center z")
 ax.set(xlabel="Real simulated time (s) / 200 Hz",ylabel="Real sphere center z (m)",
 title="Independent v94 PhysX pair + negative control: 180 real steps, not original G1 shots")
 ax.legend()
 fig.tight_layout()
 dest=root/"reports/v094_independent_physx_parity.png"
 if dest.exists():raise FileExistsError("WRITE_ONCE_PLOT")
 fig.savefig(dest,dpi=125);plt.close(fig)
 return {"local_plot":str(dest),"plot_sha256":sha(dest)}
if __name__=="__main__":
 a=argparse.ArgumentParser()
 a.add_argument("--v91-root",required=True,type=Path)
 a.add_argument("--output",required=True,type=Path)
 args=a.parse_args()
 if args.output.exists():a.error("WRITE_ONCE")
 result=evaluate(ROOT,args.v91_root)
 result["plot"]=plot(ROOT)
 with args.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print(json.dumps({k:result[k] for k in ("new_isolated_real_PhysX_probe_runs",
  "true_200hz_probe_steps","tensor_direct_filtered_pair_positive_count","native_collider_path_pair_positive_count",
  "historical_original_G1_strict_pass_count","cpu_tensor_positive","gpu_tensor_positive",
  "gpu_separated_false_positive","read_only_parity")},indent=2))
