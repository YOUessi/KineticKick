#!/usr/bin/env python3
"""Full original G1 trajectory parity and body-named nonfoot contact audit.

Adding --contact-diagnostics must not alter anything physical. If it does, an
instrumented trace cannot be promoted to exact deterministic reproduction.
Missing named events are honestly missing, not a fabricated anatomical label.
"""
from __future__ import annotations
from collections import defaultdict
import json,hashlib,math
from pathlib import Path
import numpy as np
from run_named_nonfoot_diagnostics_v054 import ROOT,CONFIG,PROGRESS,sha,freeze
OUTPUT=ROOT/"reports/v054_original_named_nonfoot_audit.json"
COMPARE=("traces","contacts","kinematics","records","sequence_records",
   "running_evidence","stride_shot_evidence","stride_shot_success",
   "termination_reasons")
def audit():
 if OUTPUT.exists():raise FileExistsError("Do not replace v54 original read-only evidence")
 protocol=json.loads(CONFIG.read_text())
 freeze(protocol)
 progress=json.loads(PROGRESS.read_text())
 if progress["protocol_sha256"]!=sha(CONFIG) or len(progress["runs"])!=3:
  raise ValueError("No partial/shuffled three-way original physics comparisons")
 results={}
 for name,spec in protocol["original_three_center_policies"].items():
  old_path=ROOT/spec["original"]
  new_path=ROOT/spec["instrumented_output"]
  old=json.loads(old_path.read_text())
  new=json.loads(new_path.read_text())
  if old["root_fixed"] or new["root_fixed"] or old["post_initialization_state_writes"] or new["post_initialization_state_writes"]:
   raise ValueError("G1 free-root original physical state requirement")
  if new["policy_sha256"]!=old["policy_sha256"] or new["joint_strike"]["mode"]!=old["joint_strike"]["mode"]:
   raise ValueError("Instrumented run used different real physics config")
  if new["joint_strike"]["checkpoint_sha256"]!=old["joint_strike"]["checkpoint_sha256"]:
   raise ValueError("Original saved 29DoF PPO weights differ")
  replay_sig={}
  for k in COMPARE:
   replay_sig[k]=dict(equal=(new.get(k)==old.get(k)),
      original_count=len(old[k]) if isinstance(old.get(k),list) else None,
      rerun_count=len(new[k]) if isinstance(new.get(k),list) else None)
  common_equal=all(x["equal"] for x in replay_sig.values())
  events=new.get("nonfoot_contact_diagnostics")
  if not isinstance(events,list):
   raise ValueError("Original sensor must record a named nonfoot event collection, even if empty")
  per_body=defaultdict(lambda: dict(count=0,max_force_N=0.,first_s=None,last_s=None,
                                    max_force_time_s=None))
  event_time_min=float(new["traces"][0][0]["time_s"])
  event_time_max=float(new["traces"][0][-1]["time_s"])
  for e in events:
   t=float(e["time_s"])
   f=np.asarray(e["force_w_N"],float)
   namebody=e["body"]
   if not isinstance(namebody,str) or not namebody or f.shape!=(3,) or not np.isfinite(f).all() or not math.isfinite(t):
    raise ValueError("Invalid original nonfoot body/time/contact force evidence")
   if t<event_time_min-.005 or t>event_time_max+.005:
    raise ValueError("Named nonfoot event must belong to actual original physics episode")
   force=float(np.linalg.norm(f))
   if force<4.999:
    raise ValueError("Read-only recorder stored contact below registered 5N event threshold")
   row=per_body[namebody]
   row["count"]+=1
   row["first_s"]=t if row["first_s"] is None else min(t,row["first_s"])
   row["last_s"]=t if row["last_s"] is None else max(t,row["last_s"])
   if force>row["max_force_N"]:
    row["max_force_N"]=force;row["max_force_time_s"]=t
  terminal_time=float(new["traces"][0][-1]["time_s"])
  around_end=[e for e in events if abs(float(e["time_s"])-terminal_time)<.10]
  peak_near_end=sorted(around_end,key=lambda e:float(np.linalg.norm(e["force_w_N"])),reverse=True)[:20]
  # Instrumentation should not cause an original physical failure. The
  # per-body contact records may occur >5ms from terminal marker because
  # this is an auxiliary sensor threshold on original net-forces.
  row=dict(version="0.1.54",name=name,source_sha256=sha(old_path),
     instrumented_sha256=sha(new_path),original_end_s=float(old["traces"][0][-1]["time_s"]),
     instrumented_end_s=terminal_time,original_physics_terminated=old["termination_reasons"],
     instrumented_physics_terminated=new["termination_reasons"],
     parity_full_on_all_common_physics_records=common_equal,
     parity_by_original_field=replay_sig,
     original_shot_success=bool(old["records"][0]["success"]),
     instrumented_shot_success=bool(new["records"][0]["success"]),
     original_actual_robot_root_z_m=float(old["traces"][0][-1]["root_xyz"][2]),
     original_actual_robot_root_tilt_deg=float(old["traces"][0][-1]["root_tilt_deg"]),
     named_nonfoot_event_count=len(events),
     named_nonfoot_body_counts_and_peak=dict(sorted(per_body.items())),
     terminal_100ms_named_nonfoot_events=peak_near_end,
     terminal_near_event_count=len(around_end),
     data_does_not_measure_human_instestep_impact=True,
     evidence_was_not_collected_in_original_v52=True)
  results[name]=row
 data=dict(version="0.1.54",original_protocol_sha256=sha(CONFIG),
     original_progress_sha256=sha(PROGRESS),
     all_physics_same_except_added_diagnostics=all(r["parity_full_on_all_common_physics_records"] for r in results.values()),
     no_policy_training=True, original_center_3_diagnostic_rollouts=3,
     no_new_heldout_context_or_generalization=True,
     uncertainty="Named nonfoot sensor body identifies a G1 collision body in this original simulated environment, not actual human instep or real hardware fall classification",
     results=results)
 OUTPUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
 print("V054_INSTRUMENTED_NAMED_NONFOOT_AUDIT",json.dumps({k:dict(
   parity=v["parity_full_on_all_common_physics_records"],
   raw_end=v["original_end_s"],new_end=v["instrumented_end_s"],
   original_termination=v["original_physics_terminated"],
   new_termination=v["instrumented_physics_terminated"],
   n_events=v["named_nonfoot_event_count"],
   per_body={p:x["max_force_N"] for p,x in v["named_nonfoot_body_counts_and_peak"].items()},
   last_100ms_events=v["terminal_near_event_count"])
   for k,v in results.items()}),flush=True)
 return data
if __name__=="__main__":audit()
