#!/usr/bin/env python3
"""Investigate observed true PhysX recovery sensitivity around the v52 nonfoot stop.

NO fabricated contact body or claims of an observed humanoid 'fall' from merely
an early physical_nonfoot termination. Five-ms aggregate sensor telemetry
cannot by itself identify the particular body that hit the environment.
"""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"configs/cmu_g1_physics_matrix_v053.json"
OUTPUT=ROOT/"reports/v053_known_center_contact_mechanism.json"
FIGURE=ROOT/"media/v053_center_recovery_5ms_differences.png"
KEYS={"frozen_v32":"reports/v052_cmu_10_02_center_zero_physx.json",
      "cmu_10_02":"reports/v052_cmu_10_02_center_physx.json",
      "cmu_10_05":"reports/v052_cmu_10_05_center_physx.json"}

def digest(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def series(record,key):
 if key=="com_z":return np.asarray([r["whole_body_com_xyz"][2] for r in record],float)
 if key=="support_left":return np.asarray([r["support_foot_force_N"]["left"] for r in record],float)
 if key=="support_right":return np.asarray([r["support_foot_force_N"]["right"] for r in record],float)
 if key=="root_z":return np.asarray([r["root_xyz"][2] for r in record],float)
 return np.asarray([r[key] for r in record],float)

def scan():
 if OUTPUT.exists():raise FileExistsError("Never overwrite preserved mechanism assessment")
 study=json.loads(STUDY.read_text())
 for p in KEYS.values():
  if digest(p)!=study["source_sha256"][p]:
   raise ValueError("v52 known reference source has changed")
 data={k:json.loads((ROOT/v).read_text()) for k,v in KEYS.items()}
 baseline=data["frozen_v32"]
 source=baseline["traces"][0]
 measures={}
 fields=("root_z","root_tilt_deg","com_z","support_left","support_right",
         "nonfoot_contact_N","feet_support_N","root_speed_xy")
 for k,obj in data.items():
  saved_rows=obj["traces"][0]
  raw_ts=np.asarray([r["time_s"] for r in saved_rows])
  # Original runner adds a +1us terminal marker on early termination.
  # Preserve it as a recorded failure, not an invented 200Hz physics step.
  marker=(len(raw_ts)>2 and 0<raw_ts[-1]-raw_ts[-2]<2e-6)
  rows=saved_rows[:-1] if marker else saved_rows
  ts=np.asarray([r["time_s"] for r in rows])
  if not np.allclose(np.diff(ts),.005,rtol=0,atol=2e-6):
   raise ValueError("Original physical 200Hz samples missing")
  n=len(rows)
  if np.max(abs(ts-np.asarray([r["time_s"] for r in source[:n]])))>1e-6:
   raise ValueError("Original and modified real physics timeline mismatch")
  diffs={}
  for attr in fields:
   s=series(rows,attr);b=series(source[:n],attr)
   diffs[attr]=dict(max_absolute_deviation=float(np.max(np.abs(s-b))),
       root_mean_absolute=float(np.mean(np.abs(s-b))),
       at_last_time_candidate=float(s[-1]),at_last_time_frozen=float(b[-1]))
  root_dist=np.linalg.norm(np.asarray([r["root_xyz"] for r in rows])-
                           np.asarray([r["root_xyz"] for r in source[:n]]),axis=-1)
  step=np.where(root_dist>0.005)[0]
  peak_idx=int(np.argmax(root_dist))
  clip=dict(source=KEYS[k],sha256=digest(KEYS[k]),
   time_s_end=float(raw_ts[-1]),real_5ms_samples=n,
   terminal_marker_sample_count=int(marker),
   total_saved_record_samples=len(saved_rows),
   physical_termination=obj["termination_reasons"],
   original_shot_record=obj["records"][0],
   nonfoot_last_value_N=float(rows[-1]["nonfoot_contact_N"]),
   nonfoot_sensor_is_aggregate_not_named_body=True,
   root_z_last_m=float(rows[-1]["root_xyz"][2]),
   root_tilt_last_deg=float(rows[-1]["root_tilt_deg"]),
   root_com_z_last_m=float(rows[-1]["whole_body_com_xyz"][2]),
   max_root_displacement_vs_original_m=float(root_dist.max()),
   max_root_displacement_time_s=float(ts[peak_idx]),
   first_root_displacement_gt_5mm_s=None if not len(step) else float(ts[step[0]]),
   position_and_support_differences=diffs)
  clip["physically_observed_fall_by_tilt_or_root_height"]=bool(
    clip["root_z_last_m"]<.45 or clip["root_tilt_last_deg"]>60.)
  measures[k]=clip
 if measures["cmu_10_05"]["physical_termination"]!=[["physical_nonfoot"]]:
  raise ValueError("Original regression result no longer has the known nonfoot stop")
 if measures["cmu_10_05"]["physically_observed_fall_by_tilt_or_root_height"]:
  raise ValueError("Re-evaluate first-order physical failure mechanism; posture actually went out of bounds")
 if measures["cmu_10_05"]["time_s_end"]>=14:
  raise ValueError("Known early-stop source was replaced")
 result=dict(version="0.1.53",study_sha256=digest(str(STUDY.relative_to(ROOT))),
     issue="CMU 10_05 generated preimpact residual caused later physical_nonfoot termination even though strike was legal and goal scored",
     sensor_interpretation="Cannot infer that robot body visibly fell: at stop root z and tilt stayed within original root-height/orientation bounds; original environment contact sensor triggered nonfoot termination.",
     no_named_contact_body_available_in_current_v52_records=True,
     counterfactual_limit="One baseline + two trajectories at known original center state, not a controlled mechanism-specific intervention; cannot causally isolate a particular body or joint solely from correlations",
     all_research_traces_sha_locked=True,conditions=measures)
 OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print("V053_KNOWN_RECOVERY_MECHANISM",json.dumps({k:dict(
     end_time_s=v["time_s_end"],reason=v["physical_termination"],
     root_z_m=v["root_z_last_m"],tilt_deg=v["root_tilt_last_deg"],
     max_root_dev_m=v["max_root_displacement_vs_original_m"],
     first_5mm_s=v["first_root_displacement_gt_5mm_s"],
     verified_root_tilt_fall=v["physically_observed_fall_by_tilt_or_root_height"])
     for k,v in measures.items()}),flush=True)
 try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
  for name,obj in data.items():
   rows=obj["traces"][0]
   times=np.array([r["time_s"] for r in rows])
   ax[0].plot(times,series(rows,"root_tilt_deg"),label=name)
   ax[1].plot(times,series(rows,"root_z"),label=name)
   ax[2].plot(times,series(rows,"nonfoot_contact_N"),label=name)
  ax[0].set_ylabel("root tilt (deg)")
  ax[1].set_ylabel("root z (m)")
  ax[2].set_ylabel("aggregate non-foot N")
  ax[2].set_xlabel("Real PhysX time (s)")
  for a in ax:a.grid(alpha=.25)
  ax[0].legend()
  fig.suptitle("One center-ball trial: legal kick does not guarantee completion; no fall verified by root pose")
  fig.tight_layout()
  fig.savefig(FIGURE,dpi=140)
  print("V053_MECHANISM_FIGURE",FIGURE)
 except ImportError:
  print("V053_MATPLOTLIB_UNAVAILABLE; raw measured audit retained")
 return result

if __name__=="__main__":scan()
