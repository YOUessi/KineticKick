#!/usr/bin/env python3
"""Fetch six named 120Hz CMU soccer kicks, original C3D and ASF/AMC only.

CMU Mocap official indexed source: Subjects 10 motions 01/02/03/05/06,
Subject 11 motion 01, all labeled 'soccer - kick ball' at 120fps. Public data
may be used subject to CMU Mocap rules; no raw C3D/BVH or video is committed.
Original CMU TLS server has incomplete certificate chain on the Tang runtime.
No account secrets transmitted, and transport-insecure flag is mandatory.
SHA256 here is LOCAL content integrity, not a signed official CMU checksum.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/research_human_kicks/cmu_soccer_120hz_v050"
MOTIONS={"10":["01","02","03","05","06"],"11":["01"]}
HOST="https://mocap.cs.cmu.edu/subjects"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def fetch_file(subject,name):
 relative=f"subjects/{subject}/{name}"
 url=f"{HOST}/{subject}/{name}"
 folder=OUT/"raw"/subject
 folder.mkdir(parents=True,exist_ok=True)
 path=folder/name
 if path.exists():raise FileExistsError("Do not overwrite official raw mocap file "+str(path))
 part=folder/(name+".partial")
 if part.exists():raise FileExistsError("Partial previous download not audited")
 cmd=["curl","--fail","--location","--silent","--show-error",
      "--insecure","--connect-timeout","12","--max-time","90",
      "--retry","2","--output",str(part),url]
 sub=subprocess.run(cmd,check=False,text=True,capture_output=True)
 if sub.returncode!=0:
  raise RuntimeError(f"CMU fetch failed for {relative}: rc={sub.returncode} details={sub.stderr[:170]}")
 value=part.read_bytes()
 if not 100<len(value)<50_000_000:raise ValueError("Empty/huge raw source "+name)
 if name.endswith(".c3d"):
  if len(value)<512 or value[1]!=0x50:
   raise ValueError("File does not have C3D binary signature: "+name)
 elif name.endswith(".asf"):
  if b":version" not in value[:300] or b":root" not in value:
   raise ValueError("Not valid CMU ASF calibration")
 elif name.endswith(".amc"):
  if b":FULLY-SPECIFIED" not in value[:600] and b":DEGREES" not in value[:600]:
   raise ValueError("Not the official AMC motion text schema")
 part.rename(path)
 result=dict(path=str(path.relative_to(ROOT)),url=url,source_subject=subject,
  bytes=path.stat().st_size,sha256=sha(path),
  transport_certificate_chain_verified=False,
  source_sha256_authority="LOCAL ONLY, NOT PUBLISHED SIGNED CMU")
 print("V050_CMU_REAL_MOCAP_FETCHED",name,result["bytes"],result["sha256"][:15],flush=True)
 return result

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 audit=OUT/"original_source_manifest.json"
 if audit.exists():raise FileExistsError("Refusing repeated original download manifest")
 files=[]
 for subject,motions in MOTIONS.items():
  files.append(fetch_file(subject,f"{subject}.asf"))
  for number in motions:
   files.append(fetch_file(subject,f"{subject}_{number}.amc"))
   files.append(fetch_file(subject,f"{subject}_{number}.c3d"))
 assert len(files)==14
 cfg=dict(version="0.1.50",
  original_source_index_url="https://mocap.cs.cmu.edu/search.php?maincat=4&subcat=7",
  original_CMU_database="https://mocap.cs.cmu.edu/",
  source_subject_trial_pairs={subject:list(motions) for subject,motions in MOTIONS.items()},
  expected_true_mocap_fps=120.,
  source_dataset_licensed_for_research=True,
  original_dataset_may_not_be_resold=True,
  original_source_has_real_physical_ball_contact_or_impulse_truth=False,
  source_is_certified_professional_elite_player=False,
  original_motion_format="C3D marker trajectories plus ASF calibration and AMC joint angles",
  source_based_120fps_motion_not_2D_broadcast_video_lift=True,
  original_file_count=len(files),
  original_c3d_count=sum(x["path"].endswith(".c3d") for x in files),
  original_amc_count=sum(x["path"].endswith(".amc") for x in files),
  original_asf_count=sum(x["path"].endswith(".asf") for x in files),
  source_TLS_issuer_chain_verify_failed_on_device=True,
  transport_used_insecure_TLS_due_to_original_site_certificate=True,
  local_SHA256_not_equivalent_to_signed_upstream_checksum=True,
  files=files,
  training_eligible_for_true_human_motion_pilot=False,
  impact_event_label_available=False,
  high_speed_biomechanics_qualified=False,
  limitation=["True optical human motion capture at officially listed 120fps, but only 6 sampled kick trials across 2 subjects",
             "No ball motion or accurate instep impact time in ASF/AMC/C3D alone",
             "No force plates, power outputs or professional-athlete provenance",
             "Manual inspection and marker/segment calibration necessary before valid human-to-G1 retargeting",
             "CMU site TLS certificate chain validation failed on the connecting machine; no published signed checksum from source was checked",
             "Never push raw motion files to source repository"])
 audit.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
 print("V050_ORIGINAL_REAL_SOCCER_MOCAP_READY",json.dumps(
      {"num_files":len(files),"C3D":cfg["original_c3d_count"],
       "AMC":cfg["original_amc_count"],"ASF":cfg["original_asf_count"],
       "total_bytes":sum(x["bytes"] for x in files),
       "source_manifest_sha256":sha(audit)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
