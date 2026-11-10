#!/usr/bin/env python3
"""One command reproducible v50+v51 CMU real human → G1 morphology research.

Precondition: 14 official CMU soccer files already downloaded, and original
G1 14s v48 physically measured raw traces present. Neither source is bundled
in the repository. This command never bypasses TLS, retrains a controller,
overwrites data, or fabricates physical contact labels. Exact original source
SHA are checked first and before each stage.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
V50="configs/cmu_soccer_quality_v050.json"
V51="configs/cmu_g1_cross_morph_v051.json"
def digest(name):
 h=hashlib.sha256()
 with (ROOT/name).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def check_study(cfg):
 path=ROOT/cfg
 if not path.exists():raise FileNotFoundError(path)
 s=json.loads(path.read_text())
 for name,h in s["source_sha256"].items():
  if not (ROOT/name).exists():
   raise FileNotFoundError("Original human/robot data only on Tang, not GitHub: "+name)
  if digest(name)!=h:
   raise ValueError("Reject source hash drift: "+name)
 return s

def run_script(name):
 print("CMU_TO_G1_BATCH_RUN",name,flush=True)
 environment=os.environ.copy()
 environment["PYTHONPATH"]=str(ROOT/"tools")
 subprocess.run([sys.executable,str(ROOT/"tools"/name)],cwd=ROOT,check=True,env=environment)

def pipeline(no_test=False):
 cmu=check_study(V50)
 initial=ROOT/"data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/quality_audit.json"
 if not initial.exists():
  run_script("audit_cmu_soccer_v050.py")
 bridge=check_study(V51)
 source_count=len(cmu["source_sha256"])+len(bridge["source_sha256"])
 stages=[
  ("data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/quality_audit.json","audit_cmu_soccer_v050.py"),
  ("data/research_human_kicks/cmu_g1_cross_morph_v051/cross_morphology_audit.json","cmu_g1_semantic_bridge_v051.py"),
  ("reports/2026-11-10-v050-real-120hz-soccer-mocap.md","write_cmu_report_v050.py"),
  ("reports/2026-11-10-v051-real-human-G1-coordination-baseline.md","write_cmu_g1_report_v051.py"),
  ("media/v050_human_soccer_ball_mocap_120hz_quality.png","plot_cmu_soccer_v050.py"),
  ("media/v051_real_human_vs_G1_six_segment_cosines.png","plot_cmu_g1_bridge_v051.py")
 ]
 for file,program in stages:
  if not (ROOT/file).exists():
   check_study(V50);check_study(V51)
   run_script(program)
  else:
   print("CMU_TO_G1_BATCH_RETAIN_EXISTING",file,flush=True)
 a=json.loads((ROOT/stages[0][0]).read_text())
 b=json.loads((ROOT/stages[1][0]).read_text())
 if a["original_human_soccer_kick_trial_count"]!=6 or a["accurate_verified_human_impact_time_truth_sample_count"]!=0:
  raise ValueError("CMU human qualification may not be falsified")
 if a["original_fast_ball_marker_motion_sample_count"]!=2:
  raise ValueError("Source ball motion requalification requires a new study")
 if b["number_of_exploratory_human_robot_pairs"]!=4 or b["new_trained_PPO_policy"]:
  raise ValueError("No unverified trained human→robot controller")
 if b["study_sha256"]!=digest(V51) or a["frozen_study_sha256"]!=digest(V50):
  raise ValueError("Trial source changed")
 if not no_test:
  print("CMU_TO_G1_BATCH_VALIDATING_BOTH_VERSIONS",flush=True)
  subprocess.run([sys.executable,"-m","pytest","-q",
       "tests/test_cmu_soccer_v050.py",
       "tests/test_cmu_g1_bridge_v051.py"],cwd=ROOT,check=True)
 print("CMU_TO_G1_BATCH_SUCCESS",json.dumps({
   "frozen_source_files_checked":source_count,
   "original_human_120hz_frames":a["original_human_C3D_frame_total"],
   "real_human_pose_source_count":a["original_human_pose_prior_pilot_eligible_count"],
   "ball_marker_exit_motion_clips_NOT_impact_truth":a["original_fast_ball_marker_motion_sample_count"],
   "verified_human_instep_groundtruth_count":0,
   "human_G1_exploratory_comparisons":b["number_of_exploratory_human_robot_pairs"],
   "new_PPO_or_PhysX_runs":0
  }),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--run",action="store_true")
 p.add_argument("--no-tests",action="store_true")
 a=p.parse_args()
 if a.run:pipeline(a.no_tests)
 else:
  check_study(V50);check_study(V51)
  print("CMU_TO_G1_FROZEN_STUDIES_VALID",digest(V50),digest(V51))
