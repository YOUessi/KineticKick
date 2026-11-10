#!/usr/bin/env python3
"""v50 six CMU 120Hz soccer C3D/AMC human mocap integrity and usable-label audit.

C3D has original 3D human joint/marker-labelled points and three soccerball
WEP markers. Their triad centroid is NOT a calibrated ball center. Reconstruct
only OBSERVED positions, no filling NaNs or inferring exact physical impact.
Measured high ball-marker speed onset is a candidate LAUNCH, not a verified
instep collision, muscle mechanical work, ground reaction or elite biomechanics.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import ezc3d

ROOT=Path(__file__).resolve().parents[1]
RAW="data/research_human_kicks/cmu_soccer_120hz_v050"
CFG="configs/cmu_soccer_quality_v050.json"
LANDMARKS=("LHJ0","RHJ0","LKNE","RKNE","LANK","RANK","LTOE","RTOE",
           "LHEE","RHEE","LMT5","RMT5","STRN","LSHO","RSHO",
           "LELB","RELB","LWRA","RWRA")
RIGHT=("RTOE","RHEE","RMT5","RANK")
LEFT=("LTOE","LHEE","LMT5","LANK")
RELATIONS=("left_knee_cos","right_knee_cos","left_ankle_cos","right_ankle_cos",
           "left_elbow_cos","right_elbow_cos","left_leg_trunk_cos","right_leg_trunk_cos")

def sha(file):
 h=hashlib.sha256()
 with (ROOT/file).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def parse_amc(file):
 frames=[];row=None
 for text in Path(file).read_text(errors="replace").splitlines():
  line=text.strip()
  if not line or line.startswith("#") or line.startswith(":"):continue
  if re.fullmatch(r"\d+",line):
   if row is not None:frames.append(row)
   row=dict(index=int(line),values={})
  else:
   if row is None:raise ValueError("AMC has no frame index")
   parts=line.split()
   value=np.array([float(s) for s in parts[1:]],dtype=float)
   if not np.all(np.isfinite(value)) or parts[0] in row["values"]:
    raise ValueError("Nonfinite/duplicate original AMC channel")
   row["values"][parts[0]]=value
 if row is not None:frames.append(row)
 if not frames or any(x["index"]!=i+1 for i,x in enumerate(frames)):
  raise ValueError("Noncontiguous original AMC")
 if not all(all(name in x["values"] for name in
      ("root","rfemur","rtibia","rfoot","lfemur","ltibia","lfoot")) for x in frames):
  raise ValueError("Human locomotor skeleton motion missing")
 return frames

def marker_index(labels,suffix):
 ids=[i for i,k in enumerate(labels) if k.upper().endswith(":"+suffix.upper())]
 if len(ids)!=1:raise ValueError("Human marker primary label missing or ambiguous "+suffix)
 return ids[0]

def valid_xyz(a):
 return np.isfinite(a).all(axis=-1)&~(np.all(a==0,axis=-1))

def contiguous_high_speed(centroid,ball_valid,rate,threshold=3.):
 if centroid.shape!=(len(ball_valid),3):raise ValueError("Bad ball cluster centroid")
 raw=np.full(len(ball_valid),np.nan)
 dt=np.linalg.norm(np.diff(centroid,axis=0),axis=1)*rate
 raw[1:]=np.where(ball_valid[1:]&ball_valid[:-1],dt,np.nan)
 ids=np.flatnonzero(np.isfinite(raw)&(raw>=threshold))
 runs=[x for x in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(x)]
 return raw,runs

def cosine(a,b):
 dot=(a*b).sum(axis=-1)
 den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
 return np.divide(dot,den,out=np.full_like(dot,np.nan),where=den>1e-6).clip(-1,1)

def pose_semantics(pose):
 if pose.ndim!=3 or pose.shape[1:]!=(len(LANDMARKS),3):
  raise ValueError("Expected 19 named human C3D coordinates")
 p={name:pose[:,i,:] for i,name in enumerate(LANDMARKS)}
 pelvis=.5*(p["LHJ0"]+p["RHJ0"])
 trunk=p["STRN"]-pelvis
 terms=(
  cosine(p["LHJ0"]-p["LKNE"],p["LANK"]-p["LKNE"]),
  cosine(p["RHJ0"]-p["RKNE"],p["RANK"]-p["RKNE"]),
  cosine(p["LKNE"]-p["LANK"],p["LTOE"]-p["LANK"]),
  cosine(p["RKNE"]-p["RANK"],p["RTOE"]-p["RANK"]),
  cosine(p["LSHO"]-p["LELB"],p["LWRA"]-p["LELB"]),
  cosine(p["RSHO"]-p["RELB"],p["RWRA"]-p["RELB"]),
  cosine(trunk,p["LKNE"]-p["LHJ0"]),
  cosine(trunk,p["RKNE"]-p["RHJ0"]))
 return np.stack(terms,axis=-1)

def trial(subject,num,source):
 label=f"{subject}_{num}"
 path=f"{RAW}/raw/{subject}/{label}.c3d"
 amc=f"{RAW}/raw/{subject}/{label}.amc"
 if source[f"{label}.c3d"]!=sha(path) or source[f"{label}.amc"]!=sha(amc):
  raise ValueError("CMU original source drift "+label)
 c=ezc3d.c3d(str(ROOT/path))
 labels=c["parameters"]["POINT"]["LABELS"]["value"]
 data=c["data"]["points"]
 rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
 unit=c["parameters"]["POINT"]["UNITS"]["value"]
 if abs(rate-120.)>1e-8 or unit!=["mm"] or data.shape[0]!=4:
  raise ValueError("Only 120Hz original mm C3D, no invented values")
 xyz=data[:3].transpose(2,1,0)*.001
 length=len(xyz)
 assert length>120
 frames=parse_amc(ROOT/amc)
 if len(frames)!=length:raise ValueError("C3D/AMC human motion sample count mismatch")
 body_indexes=[marker_index(labels,x) for x in LANDMARKS]
 pose=xyz[:,body_indexes].copy()
 body_valid=valid_xyz(pose)
 if np.min(body_valid.mean(axis=0))<.95:
  raise ValueError("Wholebody human source has severe missing data")
 pelvis=.5*(pose[:,0]+pose[:,1])
 scale=float(np.median(np.linalg.norm(pose[:,2]-pose[:,0],axis=-1)))
 if not .2<scale<1.:raise ValueError("Human anatomical scale implausible; no invented metric")
 normalized=(pose-pelvis[:,None])/scale
 semantics=pose_semantics(pose)
 semantics[~body_valid.all(axis=1)]=np.nan
 bcols=[i for i,x in enumerate(labels) if re.fullmatch(r"soccerball:wep[123]",x,re.I)]
 if len(bcols)!=3:raise ValueError("3 unique original ball 3D marker labels required")
 markers=xyz[:,bcols].copy()
 bvalid=valid_xyz(markers)
 full=bvalid.all(axis=1)
 center=np.full((length,3),np.nan)
 center[full]=markers[full].mean(axis=1)
 ball_speed,runs=contiguous_high_speed(center,full,rate)
 right=xyz[:,[marker_index(labels,x) for x in RIGHT]]
 left=xyz[:,[marker_index(labels,x) for x in LEFT]]
 nearR=np.where(full,np.min(np.linalg.norm(right-center[:,None],axis=-1),axis=1),np.nan)
 nearL=np.where(full,np.min(np.linalg.norm(left-center[:,None],axis=-1),axis=1),np.nan)
 near=np.fmin(nearR,nearL)
 if not full.any():raise ValueError("No original visible ball evidence")
 strong=sorted((r for r in runs if len(r)>=8),key=lambda r:(-len(r),r[0]))
 weak=sorted((r for r in runs if len(r)<8),key=lambda r:(-len(r),r[0]))
 quality="ball_motion_unobserved_before_marker_dropout"
 candidate=None
 if strong:
  chosen=strong[0]
  start=int(chosen[0])
  local=np.arange(max(0,start-10),min(length,start+13))
  closest=float(np.nanmin(near[local]))
  if closest<=.30:
   quality="continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth"
   candidate=dict(
     inferred_ball_launch_onset_frame=start,
     inferred_ball_launch_onset_s=round(start/rate,6),
     high_speed_contiguous_raw_120hz_intervals=len(chosen),
     ball_marker_cluster_exit_speed_mps=float(np.nanmedian(ball_speed[chosen])),
     nearest_foot_to_uncalibrated_ball_marker_cluster_m=closest,
     nearest_foot_side="right" if np.nanmin(nearR[local])<np.nanmin(nearL[local]) else "left",
     verified_true_foot_ball_contact=False,
     verified_instestep_surface=False,
     exact_contact_time_accuracy_proven=False)
  else:
   quality="fast_ball_motion_but_no_foot_proximity_within_300mm"
 elif weak:
  chosen=weak[0]
  start=int(chosen[0])
  local=np.arange(max(0,start-10),min(length,start+13))
  quality="very_short_ball_speed_burst_near_dropout_not_reliable_exit_speed"
  candidate=dict(inferred_ball_motion_onset_frame=start,
      inferred_ball_motion_onset_s=round(start/rate,6),
      observed_high_speed_120hz_intervals=len(chosen),
      max_observed_short_ball_speed_mps=float(np.nanmax(ball_speed[chosen])),
      nearest_foot_to_ball_cluster_m=float(np.nanmin(near[local])),
      verified_true_foot_ball_contact=False,
      reliable_exit_speed_available=False)
 record=dict(subject=subject,id=label,original_C3D_frames=length,original_AMC_frames=len(frames),
      original_3D_rate_hz=rate,
      primary_human_marker_count=len(LANDMARKS),
      human_primary_marker_min_valid_fraction=float(np.min(body_valid.mean(axis=0))),
      original_three_ball_markers_all_valid_fraction=float(full.mean()),
      original_three_ball_markers_valid_frames=int(full.sum()),
      last_original_three_ball_markers_visible_s=float(np.where(full)[0][-1]/rate),
      closest_foot_to_ball_cluster_m=float(np.nanmin(near)),
      raw_ball_speed_peak_mps=float(np.nanmax(ball_speed)),
      ball_fast_3mps_raw_interval_count=sum(len(r) for r in runs),
      ball_fast_highest_contiguous_intervals=max((len(r) for r in runs),default=0),
      observed_ball_kinematics_quality=quality,
      inferred_event=candidate,
      original_human_mocap_qualified_for_pose_priors=True,
      original_ball_marker_centroid_is_not_calibrated_true_ball_center=True,
      real_instestep_force_and_contact_truth_qualified=False,
      force_plate_ground_reaction_available=False,
      elite_professional_verified=False,
      verified_impact_s=None,
      source_C3D_sha256=source[f"{label}.c3d"])
 raw_amc_root=np.stack([frame["values"]["root"] for frame in frames])
 derived=dict(
  time_s=np.arange(length)/rate,primary_human_3d_m=pose.astype("float32"),
  primary_marker_valid=body_valid,
  pelvis_relative_scale_normalized_xyz=normalized.astype("float32"),
  marker_derived_human_8_cosine_relations=semantics.astype("float32"),
  raw_three_ball_marker_xyz_m=markers.astype("float32"),
  three_ball_marker_valid=bvalid,
  uncalibrated_ball_marker_cluster_centroid_m=center.astype("float32"),
  raw_ball_marker_cluster_speed_mps=ball_speed.astype("float32"),
  right_foot_marker_to_ball_cluster_distance_m=nearR.astype("float32"),
  left_foot_marker_to_ball_cluster_distance_m=nearL.astype("float32"),
  ASF_AMC_root_UNCALIBRATED_TO_C3D=raw_amc_root.astype("float32"),
  primary_landmark_names=np.asarray(LANDMARKS),
  semantic_relation_names=np.asarray(RELATIONS),
  original_human_subject=np.asarray(subject),
  original_frame_rate_hz=np.asarray(rate),
  true_impact_event_available=np.asarray(False),
  force_plate_data_available=np.asarray(False))
 return record,derived

def main():
 cfg=json.loads((ROOT/CFG).read_text())
 if cfg["version"]!="0.1.50" or cfg["impact_truth_not_available"] is not True:
  raise ValueError("Must not classify candidate onsets as real measured impact")
 for f,h in cfg["source_sha256"].items():
  if sha(f)!=h:raise ValueError("Frozen original source/code drift "+f)
 if sha(RAW+"/original_source_manifest.json")!=cfg["original_source_manifest_sha256"]:
  raise ValueError("CMU download manifest changed")
 manifest=json.loads((ROOT/RAW/"original_source_manifest.json").read_text())
 original={Path(x["path"]).name:x["sha256"] for x in manifest["files"]}
 dest=ROOT/cfg["processed_dir"]
 if dest.exists():raise FileExistsError("No overwriting real human mocap dataset")
 dest.mkdir(parents=True)
 rows=[]
 for subject,ids in cfg["source_subjects"].items():
  for num in ids:
   record,data=trial(subject,num,original)
   name=record["id"]
   npz=dest/(name+".npz")
   np.savez_compressed(npz,**data)
   record["derived_npz_sha256"]=sha(str(npz.relative_to(ROOT)))
   rows.append(record)
   print("V050_TRUE_MOCAP_QUALITY",name,"frames",record["original_C3D_frames"],
      "ball_valid",round(record["original_three_ball_markers_all_valid_fraction"],3),
      "quality",record["observed_ball_kinematics_quality"],
      "candidate",record["inferred_event"],flush=True)
 if len(rows)!=6:raise ValueError("Expected 6 official CMU kick trials")
 out=dict(version="0.1.50",
  frozen_study_sha256=sha(CFG),source_manifest_sha256=sha(RAW+"/original_source_manifest.json"),
  original_human_3d_subject_count=len({x["subject"] for x in rows}),
  original_human_soccer_kick_trial_count=len(rows),
  original_human_C3D_frame_total=sum(x["original_C3D_frames"] for x in rows),
  original_human_pose_prior_pilot_eligible_count=sum(x["original_human_mocap_qualified_for_pose_priors"] for x in rows),
  original_fast_ball_marker_motion_sample_count=sum(x["observed_ball_kinematics_quality"]=="continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth" for x in rows),
  accurate_verified_human_impact_time_truth_sample_count=0,
  accurate_instestep_contact_force_sample_count=0,
  elite_professional_biomechanics_groundtruth_count=0,
  source_motion_subject_disjoint_split=cfg["subject_level_pilot_split"],
  no_missing_ball_track_interpolation=True,
  no_new_G1_policy_training_or_PhysX_simulation=True,
  not_treating_120hz_3D_as_true_force_power_measurement=True,
  training_usable_only_for_unqualified_human_pose_priors=True,
  important_limitations=[
   "120Hz motion capture is genuine C3D 3D tracking, but some labels are skeleton-derived points, not precisely calibrated human anatomical joint centers",
   "All soccerball WEP triplets eventually vanish before end of each trial. Ball marker-cluster centroid is not a certified ball center or surface/collision normal",
   "Only >=8 contiguous raw frame-to-frame 3m/s speed intervals and <=0.30m foot marker proximity count as reliable measured ball EXIT MOTION; not true instep impact or contact time",
   "10_03 has a short last-frames speed burst followed by missing ball data: impossible to certify outgoing speed or exact impact",
   "One subject contributes five motions and the second one: subject-disjoint split is just an exploratory leakage control, never statistical generalization",
   "ASF/AMC subject calibration uses non-C3D raw coordinates; world-space fusion requires explicit transform calibration",
   "No human verified muscle/power/GRF/force plate, real instep surface impulse or professional-athlete credentials",
   "CMU original files locally SHA256-verified but remote server TLS chain failed locally, so not publisher-signed data provenance",
   "CMU original usage terms permit research, products, no direct resale; acknowledge CMU and NSF EIA-0196217 when publishing"],
  motions=rows)
 final=dest/"quality_audit.json"
 final.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
 print("V050_SIX_CMUSOCCER_REAL_MOCAP_AUDITED",json.dumps(dict(
   n_motions=len(rows),total_120hz_frames=out["original_human_C3D_frame_total"],
   pose_source_count=out["original_human_pose_prior_pilot_eligible_count"],
   strong_ball_kinematics_not_impact_count=out["original_fast_ball_marker_motion_sample_count"],
   true_impact_labels=0,certified_human_instep_forces=0,sha256=sha(str(final.relative_to(ROOT)))),
   ensure_ascii=False),flush=True)

if __name__=="__main__":main()
