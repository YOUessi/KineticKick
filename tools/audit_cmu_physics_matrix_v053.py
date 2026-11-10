#!/usr/bin/env python3
"""Independent post-hoc strict 12x original G1 PhysX evidence audit.

All 4 exact ball/velocity triples and 3 original frozen controls were
precommitted. A scored goal cannot override illegal foot surface, qdot,
termination or recovery. A tiny sample is not a naturalness study.
"""
from __future__ import annotations
import json,hashlib,math
from pathlib import Path
import numpy as np
from run_cmu_physics_matrix_v053 import ROOT,CONFIG,PROGRESS,ORDER,digest,check
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations
OUTPUT=ROOT/"reports/v053_cmu_frozen_matrix_audit.json"
def assess(spec,name,study):
 file=ROOT/spec["outputs"][name]
 report=json.loads(file.read_text())
 cfgpath=study["policies"][name]["config"]
 actual=report["joint_strike"]
 if (report["root_fixed"] or report["post_initialization_state_writes"] or
     report["policy_sha256"]!=digest(ROOT/cfgpath) or
     actual["config_sha256"]!=digest(ROOT/cfgpath)):
  raise ValueError("Physical independence/frozen config violated")
 for key,expected in actual["source_sha256"].items():
  if digest(ROOT/"tools"/key)!=expected:
   raise ValueError("Physical source code drift: "+key)
 if abs(actual["velocity_y_delta"]-spec["velocity_y_delta_mps"])>1e-8:
  raise ValueError("Actual original root initial velocity is not registered")
 if not np.allclose(actual["ball_offset_xy"],spec["ball_offset_xy_m"],atol=1e-8,rtol=0):
  raise ValueError("Actual original ball offset differs")
 cfg=json.loads((ROOT/cfgpath).read_text())
 if not np.allclose(report["points"][0],np.asarray(cfg["ball_xy"])+spec["ball_offset_xy_m"],atol=1e-8,rtol=0):
  raise ValueError("Actual original soccer ball differs from protocol")
 if actual["mode"]!="coordination_style_v041":
  raise ValueError("Wrong original G1 controller")
 # The original checkpoint is already SHA-pinned in the frozen physical cfg.
 if actual["checkpoint_sha256"]!=cfg["post_shot_recovery"]["checkpoint_sha256"]:
  raise ValueError("Original G1 postshot expert changed")
 traces=validated_episode(report)
 phys=limit_observations(traces,report)
 result=task_outcome(report)
 contact=first_force_episode(traces)
 shot=report["records"][0]
 speed=None if contact is None else contact["outgoing_speed_xy_mps"]
 full=(len(traces)==2800 and abs(traces[-1]["time_s"]-14.)<1e-6 and
       report["termination_reasons"]==[[]])
 gates=dict(
     real_physx_full_14_seconds=full,
     normal_free_root_no_state_write=True,
     actual_qdot_under_original_limits=(phys["actual_speed_exceedance_sample_joint_count"]==0),
     legal_right_instep_contact=(shot["measured_surface"]=="instep" and
          shot["requested_surface_impulse_fraction"]>=.95 and
          report["sequence_records"][0]["illegal_body_ball_contact"] is False),
     entire_ball_goal=(shot["goal"] is True),
     original_task_outcome=(result["success"] is True),
     outgoing_ball_speed_at_least_3_5=(speed is not None and speed>=3.5),
     genuine_runup_and_shot=(report["running_evidence"]["running_success"] is True and
        report["stride_shot_evidence"]["continuous_running_contact"] is True),
     recovered_and_stable=(shot["recovered"] is True and shot["standing_seconds"]>=1.),
     complete_observation=(len(report["kinematics"][0])==len(actual["feedback"])) )
 applied=0
 last=np.zeros(29)
 errors=[]
 for index,step in enumerate(actual["feedback"]):
  st=step.get("coordination_style_v041")
  hr=step.get("human_coordination_reference")
  if not isinstance(st,dict) or not isinstance(hr,dict):
   raise ValueError("Cannot verify correct state feedback controller")
  delta=np.asarray(hr["applied_delta_rad"],float)
  if delta.shape!=(29,) or not np.isfinite(delta).all():
   raise ValueError("Missing real 29DoF joint motion command")
  if hr["enabled"] is not (name!="frozen_v32") or st["enabled"] is not (name!="frozen_v32"):
   raise ValueError("Wrong true CMU generation ON/OFF state")
  if st["ball_state_write"] is not False or hr["ball_state_write"] is not False:
   errors.append("G1 or ball state write")
  if name=="frozen_v32":
   if np.any(delta!=0):errors.append("nonzero baseline reference")
  else:
   if max(abs(delta))>.006002:errors.append("reference amplitude exceeded")
   if max(abs(delta-last))>.00121:errors.append("reference slew exceeded")
   if index*.02>=1.82-1e-6 and max(abs(delta))>1e-7:
    errors.append("did not release preimpact style")
  applied+=int(max(abs(delta))>1e-7)
  last=delta
 if errors:raise ValueError("Unregistered physical controller action: "+str(sorted(set(errors))))
 if name!="frozen_v32":
  predeclared=study["source_sha256"][
    "experiments/cmu_real_120hz_phase_v052/"+name[-5:]+"/g1_joint_delta_50hz.npz"]
  if cfg["human_coordination_reference"]["trajectory_sha256"]!=predeclared:
   raise ValueError("Unexpected CMU reference / v53 policy tuning detected")
 gates["candidate_reference_actually_nonzero"]=(applied>0) if name!="frozen_v32" else True
 return dict(
  source=str(file.relative_to(ROOT)),sha256=digest(file),
  success=all(gates.values()),gates=gates,
  original_task_failure_reasons=result["failure_reasons"],
  actual_50hz_reference_decisions=applied,physical_samples=len(traces),
  terminal_time_s=float(traces[-1]["time_s"]) if traces else None,
  physical_termination=report["termination_reasons"],
  qdot_overspeed_joint_sample_count=phys["actual_speed_exceedance_sample_joint_count"],
  maximum_qdot_to_original_limit_ratio=float(phys["maximum_actual_speed_ratio"]),
  contact_time_s=shot["contact_time_s"],instep_fraction=shot["requested_surface_impulse_fraction"],
  original_ball_xy_speed_mps=speed,goal=shot["goal"],
  recovery_standing_s=shot["standing_seconds"],
  recorded_robot_initial=report["initial_robot_state"],
  recorded_ball_initial_xy=report["points"][0])

def wilson(k,n):
 z=1.959963984540054
 cen=(k/n+z*z/(2*n))/(1+z*z/n)
 half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
 return [max(0.,cen-half),min(1.,cen+half)]
def audit():
 if OUTPUT.exists():raise FileExistsError("Do not overwrite the original strict batch audit")
 study=json.loads(CONFIG.read_text())
 check(study)
 if not PROGRESS.exists():raise FileNotFoundError("Physical research batch was not run")
 prog=json.loads(PROGRESS.read_text())
 if len(prog["completed"])!=len(study["trials"])*len(ORDER):
  raise RuntimeError("Do not publish partial or selectively sampled results")
 if prog["protocol_sha256"]!=digest(CONFIG):raise ValueError("Protocol changed")
 cases=[]
 for spec in study["trials"]:
  res={name:assess(spec,name,study) for name in ORDER}
  originals=[res[n]["recorded_robot_initial"] for n in ORDER]
  balls=[res[n]["recorded_ball_initial_xy"] for n in ORDER]
  if any(x!=originals[0] for x in originals[1:]) or any(x!=balls[0] for x in balls[1:]):
   raise ValueError("Matched conditions were not actually identical")
  for r in res.values():
   r.pop("recorded_robot_initial")
   r.pop("recorded_ball_initial_xy")
  cases.append(dict(id=spec["id"],ball_offset_xy_m=spec["ball_offset_xy_m"],
    velocity_y_delta_mps=spec["velocity_y_delta_mps"],policies=res))
 summary={}
 n=len(cases)
 for policy in ORDER:
  k=sum(x["policies"][policy]["success"] for x in cases)
  summary[policy]=dict(success_count=k,n=n,descriptive_wilson_95=wilson(k,n),
     number_actuated=sum(x["policies"][policy]["actual_50hz_reference_decisions"]>0 for x in cases),
     nonfoot_stops=sum("physical_nonfoot" in x["policies"][policy]["physical_termination"][0] for x in cases),
     actual_qdot_overspeed_conditions=sum(
       x["policies"][policy]["qdot_overspeed_joint_sample_count"]>0 for x in cases))
 for policy in ORDER[1:]:
  summary[policy]["old_fail_new_pass"]=sum(
   not x["policies"]["frozen_v32"]["success"] and x["policies"][policy]["success"] for x in cases)
  summary[policy]["old_pass_new_fail"]=sum(
   x["policies"]["frozen_v32"]["success"] and not x["policies"][policy]["success"] for x in cases)
 doc=dict(version="0.1.53",frozen_protocol_sha256=digest(CONFIG),
   progress_sha256=digest(PROGRESS),n_original_PhySX_runs=n*len(ORDER),n_ball_conditions=n,
   no_intermediate_hyperparameter_tuning=True,no_controller_promotion=True,
   comparisons_are_tiny_fixed_n_not_statistical_generalization=True,
   two_real_120hz_human_CMU_clips_are_same_person=True,
   CMU_human_true_instestep_impact_verified=False,
   summary=summary,comparisons=cases)
 OUTPUT.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n")
 print("V053_STRICT_MATRIX_EVIDENCE",json.dumps(dict(
  runs=doc["n_original_PhySX_runs"],summary=summary,
  rows=[dict(case=x["id"],**{k:dict(ok=v["success"],engaged=v["actual_50hz_reference_decisions"],
        speed=v["original_ball_xy_speed_mps"],terminal=v["physical_termination"])
        for k,v in x["policies"].items()}) for x in cases])),flush=True)
 return doc
if __name__=="__main__":audit()
