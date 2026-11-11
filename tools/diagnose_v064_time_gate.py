#!/usr/bin/env python3
"""Falsifiable v64 zero-action time-support diagnosis from original v63 G1 PhysX.

The v63 eligible 20ms frames come from original debug recorded in PREVIOUS
center experimental run, not from posthoc v64 held-out trial optimization.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze():
 frozen=json.loads((ROOT/"configs/g1_human_phase_batch_v064.json").read_text())
 old=ROOT/"reports/v063_center_synth.json"
 ref=ROOT/"experiments/cmu_phase_fit_v064/human_ordered.npz"
 with np.load(ref,allow_pickle=False) as f:
  time=f["controller_time_s"];curves=f["delta_joint_position_rad"]
 active=np.flatnonzero(np.max(abs(curves),axis=1)>1e-5)
 if len(active)<5:raise ValueError("Human fitted 50Hz reference is identically inactive before control")
 data=json.loads(old.read_text())
 feedback=data["joint_strike"]["feedback"]
 original_valid=[i for i,x in enumerate(feedback) if (x.get("phase_coordinated_v063") or {}).get("eligible")]
 if not original_valid:raise ValueError("Old trusted center physical control never eligible")
 intersect=sorted(set(active).intersection(original_valid))
 return dict(
  observation="Historical synthetic v63 center full G1 200Hz physical run with active state safety logic; does not validate real human impact",
  historical_center_PhysX_SHA256=sha(old),
  fitted_human_reference_SHA256=sha(ref),
  pre_registered_v64_protocol_SHA256=sha(ROOT/"configs/g1_human_phase_batch_v064.json"),
  first_human_nonzero_frame=int(active[0]),last_human_nonzero_frame=int(active[-1]),
  human_reference_nonzero_time_s=[float(time[active[0]]),float(time[active[-1]])],
  historical_real_G1_center_original_safety_gate_eligible_time_s=[round(i*.02,2) for i in original_valid],
  intersection_nonzero_human_reference_and_historical_motor_safety_gate_count=len(intersect),
  physically_effective_phase_prior_predicted_to_be_eligible_in_center=bool(intersect),
  preimpact_first_PhysX_ball_foot_contact_s=data["records"][0]["contact_time_s"],
  fatal_missing_groundtruth="No measured professional human contact time, GRF, joint torque or muscle work; all human events are optical proxies",
  interpretation="Fit is nonzero offline but physically inapplicable at the original G1 support/safety phase; this does not prove the fit is natural.",
  no_heldout_trial_tuning_used=True)
if __name__=="__main__":
 out=ROOT/"reports/v064_temporal_support_diagnosis.json"
 if out.exists():raise FileExistsError(out)
 result=analyze()
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print("V064_TEMPORAL_DIAGNOSIS",result)
