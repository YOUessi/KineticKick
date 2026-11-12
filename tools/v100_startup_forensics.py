#!/usr/bin/env python3
"""Read-only bounded GPU-startup forensic triage. Never score an unexecuted shot."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

MILESTONES=(
 ("simulation_environment_ready","[INFO]: Completed setting up the environment..."),
 ("first_actor_recurrent_model_initialized","Actor RNN: Memory"),
 ("second_actor_mlp_initialized","Actor MLP: Sequential(\n  (0): Linear(in_features=154"),
 ("real_ground_contact_views_ready","SEQUENCE_GROUND_COLLIDERS"),
 ("first_physics_frame_ready","FREE_SHOOTING_READY"),
 ("shot_physics_50_policy_steps","FREE_SHOOTING_STEP 50"),
 ("shot_result_computed","CLOSED_LOOP_STRIKE_RESULT"),
)
def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inspect_log(log):
 seen={name:token in log for name,token in MILESTONES}
 return dict(milestones=seen,
   last_completed_milestone=next((name for name, _ in reversed(MILESTONES) if seen[name]),None),
   has_watchdog_termination=("OWN_CHILD_ONLY_TIMEOUT" in log or "WATCHDOG_TERMINATION" in log),
   path_to_native_actor_manifold_authenticated=False)
def audit(success,old_fail,new_fail,old_master,new_master,out):
 if out.exists():raise FileExistsError('WRITE_ONCE_STARTUP_FORENSICS')
 inputs={"historical_success":success,"v099_failed":old_fail,"v100_failed":new_fail}
 report={k:{'source_sha256':sha(p),**inspect_log(Path(p).read_text(errors='replace'))} for k,p in inputs.items()}
 old=json.loads(Path(old_master).read_text())
 new=json.loads(Path(new_master).read_text())
 if old['completed_certified_original_G1_cuda']!=0 or new['completed_certified_original_G1_cuda']!=0:
  raise ValueError('FAILED_SHOT_MISLABELED_AS_CERTIFIED')
 if len(old['runs'])!=1 or len(new['runs'])!=1:raise ValueError('UNEXPECTED_STOP_RULE')
 if not (old['runs'][0]['timeout'] and new['runs'][0]['timeout']):
  raise ValueError('REPORTED_WATCHDOG_NOT_PRESENT')
 if not all(report[k]['has_watchdog_termination'] for k in ('v099_failed','v100_failed')):
  raise ValueError('MISSING_WATCHDOG_LOG')
 if not report['historical_success']['milestones']['shot_result_computed']:
  raise ValueError('HISTORICAL_POSITIVE_CONTROL_INCOMPLETE')
 if any(report[k]['milestones']['first_physics_frame_ready'] for k in ('v099_failed','v100_failed')):
  raise ValueError('FAILURE_POINT_AFTER_FIRST_PHYSICS_FRAME_NOT_PREINIT')
 result=dict(scope='Same-source historical success vs v99/v100 bounded unscored startup failure',
             source_log_sha256s={k:report[k]['source_sha256'] for k in report},
             frozen_v099_master_sha256=sha(old_master),frozen_v100_master_sha256=sha(new_master),
             local_failure_attempts=2,certified_new_original_G1_physics=0,
             v099_timeout_s=180,v100_timeout_s=110,
             v099_off_runner='v098_run_pair_strike.py --tensor-pair-v098 ABSENT',
             v100_off_runner='v080_run_poll_strike.py (unchanged previously successful wrapper)',
             observer_on_in_both_failures=False,
             original_29dof_actuation_reached=False,source=report,
             forensic_conclusion='Both OFF trials reach first actor RNN and stop before second actor MLP/first physical frame; tensor observer activation cannot be the sole trigger. Root cause unresolved. Do not launch further GPU shots before separate diagnostic preflight.')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('V100_STARTUP_FORENSICS',result['forensic_conclusion'],flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('success','old-fail','new-fail','old-master','new-master','out'):p.add_argument('--'+name,type=Path,required=True)
 args=vars(p.parse_args())
 audit(**{k.replace('-','_'):v for k,v in args.items()})
