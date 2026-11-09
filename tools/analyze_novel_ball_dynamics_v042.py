#!/usr/bin/env python3
"""v42 read-only whole-body safety and pre-impact measured robot dynamics.

Reports actual joint velocity overshoot and clipped-PD estimated work for
known baseline/style pairing at 6 sealed new ball conditions. This is
not a human biological power/beauty measure or causal mechanism isolation.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,argparse,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,summarize_strike
from audit_p0_dynamic_strike import limit_observations

JOINTS=("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint",
        "left_knee_joint","waist_yaw_joint")

def check(report):
 rows=validated_episode(report)
 limits=limit_observations(rows,report)
 speed=limits["worst_actual_speed"]
 contact=report["records"][0]["contact_time_s"]
 out=dict(
  actual_qdot_ratio_max=limits["maximum_actual_speed_ratio"],
  actual_limit_exceed_sample_joints=limits["actual_speed_exceedance_sample_joint_count"],
  actual_max_speed_joint=speed["joint"],
  worst_qdot_time_s=speed["time_s"],
  measured_joint_speed_rad_s=speed["absolute_speed_rad_s"],
  original_joint_speed_limit_rad_s=speed["limit_rad_s"],
  time_of_peak_qdot_relative_first_ball_contact_s=None if contact is None else speed["time_s"]-contact,
  complete_14s=len(rows)==2800 and report["termination_reasons"]==[[]],
  first_contact_time_s=contact)
 if contact is None:
  out["pre_impact_original_joint_power"]=None
  return out
 s=summarize_strike(report)
 p=s["precontact"]
 if p is not None:
  z=p["joints"]
  out["pre_impact_original_joint_power"]=dict(
   measured_window_s=p["measured_duration_s"],
   total_positive_clipped_PD_work_J=p["whole_body_positive_work_J"],
   total_negative_clipped_PD_work_J=p["whole_body_negative_work_J"],
   individual_joint_peak_velocity_time_to_contact_s={
       k:z[k]["peak_velocity_time_to_contact_s"] for k in JOINTS},
   individual_peak_abs_velocity_rad_s={
       k:z[k]["peak_abs_velocity_rad_s"] for k in JOINTS},
   caveat="Clipped PD estimated torque times actual measured qdot, NOT certified mechanical motor power or human segmental energy transfer.")
 return out

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--output",required=True)
 args=ap.parse_args()
 out=ROOT/args.output
 if out.exists():raise FileExistsError(out)
 summary=json.loads((ROOT/"reports/v042_paired_novel_ball_audit.json").read_text())
 rows=[]
 for item in summary["trials"]:
  details={}
  for name in ("original_v32","style_v41"):
   r=json.loads((ROOT/item[name]["path"]).read_text())
   details[name]=check(r)
  old=details["original_v32"]
  new=details["style_v41"]
  t=new["worst_qdot_time_s"]
  s=json.loads((ROOT/item["original_v32"]["path"]).read_text())
  trial_physics=validated_episode(s)
  ref_index=int(np.argmin([abs(p["time_s"]-t) for p in trial_physics]))
  joint_index=s["joint_names"].index(new["actual_max_speed_joint"])
  base_speed=abs(float(trial_physics[ref_index]["joint_vel"][joint_index]))
  details["paired_speed_at_candidate_worst_time"]=dict(
   joint=new["actual_max_speed_joint"],physical_time_s=t,
   original_v32_abs_speed_rad_s=base_speed,
   style_v41_abs_speed_rad_s=new["measured_joint_speed_rad_s"],
   original_joint_speed_limit_rad_s=new["original_joint_speed_limit_rad_s"],
   difference_style_minus_baseline_abs_rad_s=new["measured_joint_speed_rad_s"]-base_speed)
  rows.append(dict(id=item["id"],ball_offset=item["ball_offset_xy_m"],
     delta_vy=item["root_velocity_y_delta_mps"],mechanics=details))
 result=dict(version="0.1.42",source_paired_data_sha256=hashlib.sha256((ROOT/"reports/v042_paired_novel_ball_audit.json").read_bytes()).hexdigest(),
  original_physical_5ms_telemetry_only=True,
  all_cases_in_registered_order=[v["id"] for v in rows],
  caution="Measured G1 motor qdot and clipped-PD power proxy, not muscle biomechanics, not a causal identification of safety degradation",
  records=rows)
 out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print("V042_REAL_ROBOT_DYNAMICS_DETAIL",json.dumps([
  dict(id=r["id"],new_qdot=r["mechanics"]["style_v41"]["actual_qdot_ratio_max"],
    new_off_limit=r["mechanics"]["style_v41"]["actual_limit_exceed_sample_joints"],
    worst=r["mechanics"]["paired_speed_at_candidate_worst_time"])
  for r in rows]),flush=True)
if __name__=="__main__":main()
