#!/usr/bin/env python3
"""v42: audit six one-shot novel ball/velocity triples under original PhysX.

Each case is paired v32 frozen vs v41 one-clip pose-proxy style controller.
This is one fixed small-sample feasibility test, NOT a claim of natural
professional explosive kicking, high-confidence robustness or hardware success.
"""
from pathlib import Path
import hashlib,json,math,argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations

def sha(name):
 h=hashlib.sha256()
 with (ROOT/name).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def assess(case,side,study):
 p=case["outputs"][side];r=json.loads((ROOT/p).read_text())
 cfgpath=study["controllers"][side]["config"]
 cfg=json.loads((ROOT/cfgpath).read_text())
 if r["root_fixed"] is not False or r["post_initialization_state_writes"] is not False:
  raise ValueError("G1 must be free-root, no physical state edits")
 if r["policy_sha256"]!=sha(cfgpath) or r["joint_strike"]["config_sha256"]!=sha(cfgpath):
  raise ValueError("Invalid original runner configuration provenance")
 if r["joint_strike"]["checkpoint_sha256"]!=study["original_v32_checkpoint_sha256"]:
  raise ValueError("Original post-shot G1 PPO replaced")
 for f,h in r["joint_strike"]["source_sha256"].items():
  if sha("tools/"+f)!=h:raise ValueError("Runtime source drift: "+f)
 if not np.allclose(r["joint_strike"]["ball_offset_xy"],case["ball_offset_xy_m"],rtol=0,atol=1e-8):
  raise ValueError("Ball displacement not registered")
 if abs(r["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-8:
  raise ValueError("Robot initial velocity not registered")
 if not np.allclose(r["points"][0],np.asarray(cfg["ball_xy"])+case["ball_offset_xy_m"],rtol=0,atol=1e-8):
  raise ValueError("Actual ball start not prescribed")
 expected="coordination_style_v041" if side=="style_v41" else "post_shot_residual"
 if r["joint_strike"]["mode"]!=expected:raise ValueError("Wrong original controller")
 if side=="style_v41":
  if (r["joint_strike"]["config"]["human_coordination_reference"]["trajectory_sha256"]
       !=study["reference_generated_q_sha256"]):
   raise ValueError("Human-proxy joint reference changed")
 rows=validated_episode(r)
 lim=limit_observations(rows,r)
 outcome=task_outcome(r)
 record=r["records"][0]
 contact=first_force_episode(rows)
 speed=None if contact is None else contact["outgoing_speed_xy_mps"]
 full=(len(rows)==2800 and abs(rows[-1]["time_s"]-14.)<1e-6 and r["termination_reasons"]==[[]])
 legal=(record["measured_surface"]=="instep" and record["surface_match"] is True
        and record["requested_surface_impulse_fraction"]>=.95
        and r["sequence_records"][0]["illegal_body_ball_contact"] is False)
 recovered=record["recovered"] is True and record["standing_seconds"]>=1.
 gates=dict(goal=record["goal"] is True,
     original_task=outcome["success"] is True,
     legal_instep=bool(legal),
     physical_14s=full,
     qdot_within_original=lim["actual_speed_exceedance_sample_joint_count"]==0,
     recovery_stable=bool(recovered),
     actual_ball_speed_at_least_3_5=(speed is not None and speed>=study["minimum_ball_speed_mps"]),
     running=bool(r["running_evidence"]["running_success"]),
     runup_shot_continuous=bool(r["stride_shot_evidence"]["continuous_running_contact"]))
 count=0
 if side=="style_v41":
  for f in r["joint_strike"]["feedback"]:
   v=f.get("human_coordination_reference");m=f.get("coordination_style_v041")
   if not isinstance(v,dict) or not isinstance(m,dict):raise ValueError("Missing true style-controller telemetry")
   a=np.asarray(v["applied_delta_rad"],float)
   if a.shape!=(29,) or not np.isfinite(a).all() or float(abs(a).max())>.006002:
    raise ValueError("Invalid original 29DoF joint-reference bounds")
   if not m["independent_of_nominal_error_gate"] or v["ball_state_write"] or m["ball_state_write"]:
    raise ValueError("Illegal style control/state write")
   count+=int(float(abs(a).max())>1e-7)
 full_success=all(gates.values())
 return dict(path=p,sha256=sha(p),original_outcome_success=outcome["success"],
     original_failure_reasons=outcome["failure_reasons"],strict_functional_pass=full_success,
     effective_style_and_functional_pass=(full_success and count>0) if side=="style_v41" else None,
     original_physx_gates=gates,style_applied_decisions=count,
     ball_outgoing_xy_mps=speed,contact_time_s=record["contact_time_s"],
     instep_impulse_fraction=record["requested_surface_impulse_fraction"],
     goal=record["goal"],standing_s=record["standing_seconds"],
     physics_steps=len(rows),actual_joint_speed_max_ratio=lim["maximum_actual_speed_ratio"],
     qdot_speed_limit_violation_samples=lim["actual_speed_exceedance_sample_joint_count"],
     early_physical_termination=r["termination_reasons"],robot_initial=r["initial_robot_state"],
     initial_ball_xy=r["points"][0])

def interval(k,n,z=1.959963984540054):
 p=k/n
 q=1+z*z/n
 mid=(p+z*z/(2*n))/q
 half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
 return [float(max(0,mid-half)),float(min(1,mid+half))]

def run():
 ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);args=ap.parse_args()
 dest=ROOT/args.output
 if dest.exists():raise FileExistsError(dest)
 study=json.loads((ROOT/"configs/novel_ball_generalization_v042.json").read_text())
 if study["version"]!="0.1.42" or len(study["trials"])!=6 or not study["no_adaptive_parameter_search"]:
  raise ValueError("Wrong research protocol")
 for f,h in study["source_sha256"].items():
  if sha(f)!=h:raise ValueError("Changed frozen study source: "+f)
 cases=[]
 for spec in study["trials"]:
  base=assess(spec,"original_v32",study)
  style=assess(spec,"style_v41",study)
  if base["robot_initial"]!=style["robot_initial"] or base["initial_ball_xy"]!=style["initial_ball_xy"]:
   raise ValueError("Paired G1 initial conditions differ")
  for r in (base,style):r.pop("robot_initial");r.pop("initial_ball_xy")
  cases.append(dict(id=spec["id"],ball_offset_xy_m=spec["ball_offset_xy_m"],
    root_velocity_y_delta_mps=spec["velocity_y_delta_mps"],
    original_v32=base,style_v41=style))
 n=len(cases)
 old=sum(c["original_v32"]["strict_functional_pass"] for c in cases)
 new=sum(c["style_v41"]["strict_functional_pass"] for c in cases)
 applied=sum(c["style_v41"]["style_applied_decisions"]>0 for c in cases)
 report=dict(version="0.1.42",n_cases=n,protocol_sha256=sha("configs/novel_ball_generalization_v042.json"),
   old_success_count=old,new_success_count=new,
   new_style_actuated_count=applied,new_style_engaged_and_success_count=sum(
    c["style_v41"]["effective_style_and_functional_pass"] for c in cases),
   old_fail_new_pass=sum(not c["original_v32"]["strict_functional_pass"] and c["style_v41"]["strict_functional_pass"] for c in cases),
   old_pass_new_fail=sum(c["original_v32"]["strict_functional_pass"] and not c["style_v41"]["strict_functional_pass"] for c in cases),
   descriptive_old_wilson95=interval(old,n),descriptive_new_wilson95=interval(new,n),
   source_parameter_triples_unseen_in_named_fixed_development_cases=True,
   source_environment_robot_motion_controller_ball_physics_unchanged=True,
   population_robustness_established=False,professional_instep_biomechanics_established=False,
   no_trained_new_controller=True,no_hyperparameter_tuning=True,no_promotion=True,
   limitations=["Six one-shot exact new parameter triples are inside original physics/control task envelope, not a new soccer skill/domain",
       "No replicate random seeds or hardware; Wilson intervals only descriptive for tiny fixed n",
       "Pose prior uses one 3DSP video-estimated training sample without verified foot-ball contact or force",
       "No human naturalness or high-power explosive mechanics measured"],
   trials=cases)
 dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print("V042_NOVEL_PAIRED_RESULT",json.dumps(dict(n=n,original_pass=old,style_pass=new,
    style_engaged_cases=applied,old_fail_new_pass=report["old_fail_new_pass"],
    old_pass_new_fail=report["old_pass_new_fail"],
    rows=[dict(id=c["id"],old=c["original_v32"]["strict_functional_pass"],
      style=c["style_v41"]["strict_functional_pass"],
      style_on=c["style_v41"]["style_applied_decisions"],
      old_ball_speed=c["original_v32"]["ball_outgoing_xy_mps"],
      new_ball_speed=c["style_v41"]["ball_outgoing_xy_mps"],
      new_qdot_violations=c["style_v41"]["qdot_speed_limit_violation_samples"])
       for c in cases]),ensure_ascii=False),flush=True)
if __name__=="__main__":run()
