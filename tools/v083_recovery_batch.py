#!/usr/bin/env python3
"""Infrastructure-only retry of precisely frozen v83 trials, write-once new namespace.

Original failure (missing upstream symlink) is retained as a negative *infrastructure*
attempt, never a physics trial. All six seed/placement/controls remain frozen.
"""
from __future__ import annotations
import fcntl,json,os
from pathlib import Path
from v083_run_physics_batch import (
 ROOT,PYTHON,cfg_load,resources,run_child,certified_physics,sha)
OUTPUT=ROOT/"reports/v083_recovered_final_real_physx.audit.json"
ORIGINAL_FAILURE_REASON="Missing upstream controller sources"
def preflight(c):
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 for name in ("v083_staged_native_pair_probe","v083_independent_offset_A_v32"):
  stem=ROOT/"reports"/name
  side=json.loads(Path(str(stem)+".audit.json").read_text())
  log=Path(str(stem)+".log")
  if (side.get("raw_sha256") is not None or side.get("output_exists") or
      not side.get("log_sha256")==sha(log) or ORIGINAL_FAILURE_REASON not in log.read_text()):
   raise RuntimeError("RETRY_ONLY_CERTIFIED_PRE_GPU_ASSET_PRECHECK_FAILED")
 if not (ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking/soccer").is_dir():
  raise RuntimeError("ORIGINAL_UPSTREAM_SOURCE_MISSING")
 from targeted_shooting import controller_provenance
 p=controller_provenance(ROOT)
 if not p.get("controller_sources_sha256"):raise RuntimeError("UNTRUSTED_UPSTREAM_SOURCE")
 return p

def execute(c,trial):
 stem="v083_recovered_"+trial["case"]
 out=ROOT/"reports"/(stem+".json")
 args=[PYTHON,str(ROOT/"tools/v080_run_poll_strike.py"),"--config",
       str(ROOT/trial["source_config"]),"--output",str(out),
       "--ball-offset",*[str(v) for v in trial["ball_offset_xy"]],
       "--velocity-delta",str(trial["velocity_delta"]),
       "--contact-diagnostics","--eval-seed",str(trial["seed"])]
 row,rawpath,log,side=run_child(c,stem,args,c["safety"]["trial_timeout_s"])
 raw={}
 if rawpath.exists():
  try:raw=json.loads(rawpath.read_text())
  except Exception as e:row["parse_error"]=repr(e)
 certified,reason=certified_physics(raw,log.read_text(errors="replace"),
                                    trial,row["rc"],row["watchdog"])
 row.update(case=trial["case"],scenario=trial["scenario"],seed=trial["seed"],
            source_config=trial["source_config"],expected_ball_offset_xy=trial["ball_offset_xy"],
            expected_velocity_delta=trial["velocity_delta"],true_physics=certified,
            source_certification=reason,actual_200hz_rows=len(raw.get("traces",[[]])[0]) if raw else 0)
 from audit_v067_hard_gates import physically_scored
 if certified:
  try:score=physically_scored(raw,wrapper_certified=True)
  except Exception as exc:score={"hard_gate_pass":False,"failed_gates":["INDEPENDENT_AUDIT_ERROR"],"error":repr(exc)}
 else:score={"hard_gate_pass":False,"failed_gates":["NO_CERTIFIED_ORIGINAL_G1_GPU_PHYSICS"]}
 row["strict_gates"]=score
 with side.open("x") as fp:json.dump(row,fp,indent=2,ensure_ascii=False);fp.write("\n")
 print("V083_RECOVERED_G1",trial["case"],"real",certified,"rows",row["actual_200hz_rows"],
       "pass",score["hard_gate_pass"],"reason",reason,"failed_gates",score["failed_gates"],flush=True)
 return row

def staged(c):
 trial=c["independent_probe"];stem="v083_recovered_staged_native_pair_probe"
 out=ROOT/"reports"/(stem+".json")
 args=[PYTHON,str(ROOT/"tools/v083_run_staged_strike.py"),"--config",
       str(ROOT/trial["source_config"]),"--output",str(out),"--eval-seed",
       str(trial["seed"]),"--contact-diagnostics","--tensor-wrist-hip-v081"]
 row,p,log,side=run_child(c,stem,args,trial["timeout_s"])
 phases=[x for x in log.read_text(errors="replace").splitlines() if "V083_STAGE" in x]
 row.update(phase_markers=phases,last_phase=phases[-1] if phases else None,
            native_manifold_proven=False,probe_not_scored_as_shot=True)
 with side.open("x") as fp:json.dump(row,fp,indent=2,ensure_ascii=False);fp.write("\n")
 print("V083_RECOVERED_STAGE",row["rc"],"timeout",row["watchdog"],"last_phase",row["last_phase"],flush=True)
 return row

def main():
 c=cfg_load();preflight(c);resources(c)
 lock=Path(c["safety"]["gpu_lock"])
 rows=[]
 with lock.open("r+") as fp:
  try:fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError("OTHER_GPU_JOB_LOCK")
  probe=staged(c)
  # No collateral GPU leak accepted after timed contact probe:
  resources(c)
  for trial in c["trials"]:
   r=execute(c,trial);rows.append(r)
   if not r["true_physics"]:break
 report={"version":"v0.1.83","context":"ONE_INFRA_ASSET_RECOVERY_SAME_FROZEN_CASES",
         "protocol_sha256":"629b8cd8940a8469c1aa81dc62bf33cea2049542aab1a1263449feea9a69c647",
         "probe":probe,"infrastructure_precheck_negatives_already_retained":2,
         "real_original_G1_PhysX_episodes":sum(x["true_physics"] for x in rows),
         "strict_pass":sum(x["strict_gates"]["hard_gate_pass"] for x in rows),
         "strict_fail":sum(x["true_physics"] and not x["strict_gates"]["hard_gate_pass"] for x in rows),
         "unscored_episodes":sum(not x["true_physics"] for x in rows),
         "remaining_frozen_not_executed":[x["case"] for x in c["trials"][len(rows):]],
         "native_collision_pair_truth":"UNKNOWN_UNTIL_ACTOR_COLLIDER_PAIR_MANIFOLD",
         "human_kick_supervision":False,"runs":rows}
 with OUTPUT.open("x") as fp:json.dump(report,fp,indent=2);fp.write("\n")
 print("V083_RECOVERY_BATCH_DONE",json.dumps({k:v for k,v in report.items() if k not in ("runs","probe")}),flush=True)
if __name__=="__main__":main()
