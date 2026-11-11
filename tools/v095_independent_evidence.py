#!/usr/bin/env python3
"""SHA-locked, independent v95 native-timing/tensor calibration and original
200Hz 29DoF G1 heldout failure audit. No motor writes. No source changes.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
V91=Path("/home/you/projects/kinetickick-wt-v091")
sys.path.insert(0,str(ROOT/"tools"))
import v091_independent_dynamics as orig
orig.ROOT=V91
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for piece in iter(lambda:f.read(1<<20),b""):h.update(piece)
 return h.hexdigest()
def certify_contact(raw,row):
 if raw.get("actual_runtime_frames")!=180:raise ValueError("NOT_180_TRUE_PHYSX_STEPS")
 if raw.get("device")!=row["device"] or raw.get("capture")!=row["capture"]:raise ValueError("WRONG_RUN_FIELDS")
 if raw.get("subscribe_phase")!=row["subscribe_phase"]:raise ValueError("NATIVE_TIMING_DRIFT")
 if raw.get("report_mode")!=row["report_mode"] or raw.get("layout")!=row["layout"]:raise ValueError("CONTRACT_DRIFT")
 if len(raw.get("runtime_z_vz",[]))!=180:raise ValueError("NO_200HZ_TRACE")
 trace=np.asarray(raw["runtime_z_vz"],float)
 if trace.shape!=(180,3) or not np.isfinite(trace).all() or not np.array_equal(trace[:,0],np.arange(1,181)):
  raise ValueError("NOT_180_CONSECUTIVE_200HZ_PHYSX_STEPS")
 if row["layout"]=="colliding" and not raw.get("observed_collision_dynamics"):raise ValueError("PHYSICS_CONTACT_CONTROL_NOT_CONFIRMED")
 if row["layout"]=="separated" and not raw.get("observed_noncollision_dynamics"):raise ValueError("NEGATIVE_CONTROL_NOT_CONFIRMED")
 stage=raw.get("stage",{})
 if stage.get("ball_report") is not (row["report_mode"]=="on") or stage.get("box_report") is not (row["report_mode"]=="on"):
  raise ValueError("OFF_ON_USD_REPORT_API_MISMATCH")
 if not stage.get("all_colliders",{}).get("ProbeBall") or not stage.get("all_colliders",{}).get("ProbeBox"):
  raise ValueError("SOURCE_COLLIDER_PATHS_UNQUALIFIED")
 contacts=raw.get("tensor_contact_points",[])
 tensor=raw.get("tensor_pair_witness",False)
 if tensor:
  if not contacts or not all(len(c["point_w"])==len(c["normal_w"])==3 and c["force_magnitude_N"]>0 and np.linalg.norm(c["normal_w"])>.8 for c in contacts):
   raise ValueError("INVALID_TENSOR_WITNESS")
  if row["device"]!="cuda:0" or row["capture"] not in ("tensor","both") or row["layout"]!="colliding":
   raise ValueError("INVALID_TENSOR_POSITIVE_CONTEXT")
 if row["layout"]=="separated" and (tensor or raw.get("tensor_contact_counts") or raw.get("tensor_peak_force_N")):
  raise ValueError("NEGATIVE_CONTROL_FALSE_POSITIVE")
 # Native actor/collider manifold *is not inferred* from USD shape inventory.
 if raw.get("native_pair_witness") and not raw.get("native_events"):
  raise ValueError("NATIVE_WITNESS_WITHOUT_CONTACT_EVENT")
 return dict(case=row["case"],step_count=180,actual_time_s=.9,
  cuda_tensor_witness=bool(tensor),tensor_positive_contact_points=len(contacts),
  filtered_tensor_total_counts=raw.get("tensor_contact_counts"),
  tensor_peak_force_N=raw.get("tensor_peak_force_N"),
  native_callback_calls=raw.get("native_counts",{}).get("callback_calls"),
  native_poll_headers=raw.get("native_counts",{}).get("poll_headers"),
  direct_actor_collider_manifold_certified=bool(raw.get("native_pair_witness")),
  observed_collision=bool(raw.get("observed_collision_dynamics")),
  observed_separated=bool(raw.get("observed_noncollision_dynamics")),
  original_G1=False)
def strict_parity(a,b):
 if a["layout"]!=b["layout"] or a["layout"]!="colliding" or a["device"]!=b["device"]:raise ValueError("NONMATCHING_MECHANICS")
 x=np.asarray(a["runtime_z_vz"],float)
 y=np.asarray(b["runtime_z_vz"],float)
 if x.shape!=y.shape:raise ValueError("NOT_MATCHING_FRAME_COUNTS")
 return {"steps":len(x),"fields_compared":["step","z","vz","box_position_w"],"max_abs_saved_z_vz":float(np.abs(x-y).max()),
  "exact_saved_numeric_state":bool(np.array_equal(x,y) and np.array_equal(a["box_position_w"],b["box_position_w"])),
  "unmeasured_internal_kinetic_state_claimed_equal":False}
def original_g1():
 items=[]
 for case in ("E_v32","E_cmu02","F_v32","F_cmu02"):
  entry,_=orig.inspect(case)
  items.append(entry)
 return items
def audit(out):
 if out.exists():raise FileExistsError("WRITE_ONCE_INDEPENDENT_AUDIT")
 frozen=json.loads((ROOT/"configs/v095_frozen_matrix.json").read_text())
 batchpath=ROOT/"reports/v095_frozen_contact_batch.audit.json"
 batch=json.loads(batchpath.read_text())
 if batch["frozen_commit"]!="6f6715714a761c4dbb76e7bc3c8df097ee43f744":
  raise ValueError("PROTOCOL_FREEZE_SHA_DRIFT")
 if batch["protocol_sha256"]!=sha(ROOT/"configs/v095_frozen_matrix.json"):
  raise ValueError("PROTOCOL_HASH_DRIFT")
 if len(batch["trials"])!=len(frozen["matrix"]) or len(frozen["matrix"])!=9:
  raise ValueError("MATRIX_COMPLETION_MISMATCH")
 cases={};raws={};ids={}
 for r,ref in zip(batch["trials"],frozen["matrix"]):
  if r["case"]!=ref["case"]:raise ValueError("ORDER_CHANGED")
  path=ROOT/"reports"/("v095_"+r["case"]+".json")
  log=path.with_suffix(".log")
  auditpath=path.with_suffix(".audit.json")
  if any(not p.exists() for p in (path,log,auditpath)):raise ValueError("SOURCE_MISSING")
  if sha(path)!=r["raw_sha256"] or sha(log)!=r["log_sha256"]:raise ValueError("RAW_OR_LOG_SHA_DRIFT")
  if json.loads(auditpath.read_text())!=r:raise ValueError("CHILD_AUDIT_CHANGED")
  if not r["recorded_runtime_evidence"] or not r["physically_confirmed"] or r["rc"]!=0 or r["timeout"]:
   raise ValueError("NOT_CONFIRMED_REAL_PHYSX")
  raw=json.loads(path.read_text())
  cases[r["case"]]=certify_contact(raw,ref)
  raws[r["case"]]=raw
  ids[r["case"]]={"raw_sha":sha(path),"log_sha":sha(log),"audit_sha":sha(auditpath)}
 checks={}
 for case in ("cpu_native_on","cpu_native_after_on","gpu_native_on","gpu_native_after_on","gpu_tensor_on","gpu_both_on"):
  base="cpu_pose_off" if case.startswith("cpu") else "gpu_pose_off"
  checks[base+"_vs_"+case]=strict_parity(raws[base],raws[case])
 if not all(x["exact_saved_numeric_state"] for x in checks.values()):raise ValueError("NONPERTURBATION_FAIL")
 if not cases["gpu_tensor_on"]["cuda_tensor_witness"] or not cases["gpu_both_on"]["cuda_tensor_witness"]:
  raise ValueError("CALIBRATED_CUDA_TENSOR_POSITIVE_MISSING")
 if cases["gpu_separated_on"]["cuda_tensor_witness"]:raise ValueError("NEGATIVE_PAIR_FALSE_POSITIVE")
 historical=original_g1()
 if any(z["independent_gates"]["strict_pass"] for z in historical):
  raise ValueError("MISMATCH_WITH_VERIFIED_HELDOUT_V91_FAILURES")
 # No native positive control => direct shape pair still fail-closed.
 native_count=sum(x["direct_actor_collider_manifold_certified"] for x in cases.values())
 result={"version":"v0.1.95",
  "protocol_frozen_commit":batch["frozen_commit"],"protocol_sha256":batch["protocol_sha256"],
  "batch_audit_sha256":sha(batchpath),"g1_source_root":str(V91),
  "physics_probe_runs":len(cases),"calibrated_cuda_tensor_positive_runs":sum(x["cuda_tensor_witness"] for x in cases.values()),
  "original_g1_physx_shots_new":0,"new_learned_policy_physx_shots":0,
  "native_pair_manifold_calibrated":bool(native_count),"native_pair_manifold_positive_count":native_count,
  "all_true_steps":sum(x["step_count"] for x in cases.values()),
  "probe_cases":cases,"individual_write_once_raw_source_sha256":ids,
  "off_on_exact_saved_state_parity":checks,
  "original_g1_historical_200hz_cases":historical,
  "historical_g1_count":len(historical),
  "historical_g1_physical_strict_pass":sum(z["independent_gates"]["strict_pass"] for z in historical),
  "original_G1_left_wrist_left_hip_native_contact_proven":False,
  "new_fullbody_motor_actuation_authorized":False}
 with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print("V095_INDEPENDENT_VERDICT",json.dumps({"probe_trials":len(cases),
  "true_physx_frames":result["all_true_steps"],
  "native_actor_collider_certifications":native_count,
  "tensor_positive":result["calibrated_cuda_tensor_positive_runs"],
  "off_on_state_parity_pass":sum(x["exact_saved_numeric_state"] for x in checks.values()),
  "historical_g1_strict_pass":result["historical_g1_physical_strict_pass"],
  "historical_g1_count":len(historical),
  "original_G1_shots_new":0}),flush=True)
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,required=True)
 args=p.parse_args();audit(args.output)
