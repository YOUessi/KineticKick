#!/usr/bin/env python3
"""Read original v54 named G1 net-force vectors, test symmetric body pair hint.

Equal-and-opposite two body net forces at one time are CONSISTENT with a
robot self collision, but ContactSensor net forces alone do not expose exact
contact manifold or prim-pair collision attribution. Do not certify it as a
known frictional pair without the appropriate collision API.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from run_named_nonfoot_diagnostics_v054 import ROOT,CONFIG,sha
from audit_named_nonfoot_diagnostics_v054 import OUTPUT as AUDIT
OUT=ROOT/"reports/v054_opposite_nonfoot_link_forces.json"

def analyze():
 if OUT.exists():raise FileExistsError("Do not overwrite existing conservative pairwise evidence")
 c=json.loads(CONFIG.read_text())
 a=json.loads(AUDIT.read_text())
 assert a["version"]=="0.1.54"
 r=a["results"]["cmu_10_05"]
 raw_file=ROOT/c["original_three_center_policies"]["cmu_10_05"]["instrumented_output"]
 if r["instrumented_sha256"]!=sha(raw_file):
  raise ValueError("Original named body source changed after physical audit")
 entries=r["terminal_100ms_named_nonfoot_events"]
 pairs=[]
 for i,x in enumerate(entries):
  for y in entries[i+1:]:
   if abs(x["time_s"]-y["time_s"])>2e-6:continue
   f1=np.asarray(x["force_w_N"],float)
   f2=np.asarray(y["force_w_N"],float)
   if min(np.linalg.norm(f1),np.linalg.norm(f2))<5.:continue
   cancellation=float(np.linalg.norm(f1+f2))
   ratio=cancellation/max(1e-12,float(np.linalg.norm(f1)))
   cosine=float(f1@f2/(np.linalg.norm(f1)*np.linalg.norm(f2)))
   sep=float(np.linalg.norm(np.asarray(x["position_w"])-np.asarray(y["position_w"])))
   pairs.append(dict(body_a=x["body"],body_b=y["body"],
      measured_t_s=float(x["time_s"]),
      a_force_w_N=x["force_w_N"],b_force_w_N=y["force_w_N"],
      net_force_sum_w_N=(f1+f2).tolist(),
      net_force_cancellation_ratio=ratio,cosine_between_forces=cosine,
      position_of_link_origins_separation_m=sep,
      force_magnitude_each_N=float(np.linalg.norm(f1)),
      body_pair_directly_measured_in_contact_manifold=False,
      possible_self_collision_not_certified=(ratio<1e-4 and cosine<-.9999)))
 expected=("left_hip_roll_link","left_wrist_yaw_link")
 strong=[v for v in pairs if {v["body_a"],v["body_b"]}==set(expected) and v["possible_self_collision_not_certified"]]
 if len(strong)!=1:
  raise ValueError("No exact opposite pair in original physically recorded net contact forces; cannot claim inferred self-collision")
 result=dict(version="0.1.54",direct_original_contact_pair_label_verified=False,
   original_v54_named_body_result_sha256=sha(AUDIT),
   instrumented_real_G1_PhysX_trace_sha256=sha(raw_file),
   human_CMU_true_contact_force_verified=False,
   failure_label="original physical_nonfoot termination; no certified gross robot tumble",
   interpretation="Two distinct G1 non-foot links have exactly equal-and-opposite net forces at the same time, consistent with a shared self-contact pair. ContactSensor net forces alone cannot certify which bodies contacted without pairwise contact manifold data.",
   all_synchronized_nonfoot_force_pairs=pairs,
   left_wrist_hip_symmetric_force_pair=strong[0],
   need_next_step_pairwise_contact_manifold=True)
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 x=strong[0]
 print("V054_BODY_PAIR_EVIDENCE",json.dumps(dict(
     body_a=x["body_a"],body_b=x["body_b"],time_s=x["measured_t_s"],
     force_N=x["force_magnitude_each_N"],force_cos=x["cosine_between_forces"],
     origin_separation_m=x["position_of_link_origins_separation_m"],
     pair_certified=result["direct_original_contact_pair_label_verified"])),flush=True)
 return result

if __name__=="__main__":analyze()
