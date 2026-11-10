#!/usr/bin/env python3
"""Visual quality audit: ORIGINAL 120Hz human optical body and ball marker gaps.

Plots measured centroid speed only where all 3 soccerball WEP points observed.
Candidate lines denote UNVERIFIED launch onset, never true impact or human
instep/collision force. No interpolation of missing ball markers.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050"
REPORT=BASE/"quality_audit.json"

def plot():
 summary=json.loads(REPORT.read_text())
 if summary["original_human_soccer_kick_trial_count"]!=6:raise ValueError("Missing original six")
 fig,axes=plt.subplots(3,2,figsize=(13.8,10),sharey=False)
 for ax,info in zip(axes.flat,summary["motions"]):
  with np.load(BASE/(info["id"]+".npz"),allow_pickle=False) as z:
   t=z["time_s"]
   speed=z["raw_ball_marker_cluster_speed_mps"]
   near=np.fmin(z["right_foot_marker_to_ball_cluster_distance_m"],
                z["left_foot_marker_to_ball_cluster_distance_m"])
   marker_available=z["three_ball_marker_valid"].all(axis=1)
  ax.plot(t,speed,linewidth=1.35,label="Measured 3-marker-cluster speed (m/s)")
  ax.set_ylim(0,max(5.5,min(17.,np.nanmax(speed)+1.4)))
  ax.set_xlim(0,t[-1])
  ax.set_ylabel("Marker-cluster speed (m/s)")
  ax.axhline(3.,linestyle="--",color="grey",linewidth=.9,label="Fixed motion-evidence cutoff: 3 m/s")
  last=float(t[np.flatnonzero(marker_available)[-1]])
  ax.axvline(last,linestyle=":",color="red",linewidth=1.3,label="Last all-3-ball-marker observation")
  event=info.get("inferred_event")
  if event:
   onset=event.get("inferred_ball_launch_onset_s",event.get("inferred_ball_motion_onset_s"))
   ax.axvline(onset,linestyle="-.",color="purple",linewidth=1,label="Inferred ball-motion onset (NOT impact truth)")
  second=ax.twinx()
  second.plot(t,near,color="darkorange",linewidth=.8,alpha=.8,label="Nearest observed human foot marker to WEP cluster")
  second.set_ylim(0,1.25)
  second.set_ylabel("Closest marker distance (m)")
  ax.set_title(info["id"]+" — "+info["observed_ball_kinematics_quality"].replace("_"," "),fontsize=9)
  ax.grid(alpha=.2)
  ax.set_xlabel("Seconds from C3D first recorded motion frame")
  if info["id"]=="10_01":
   aa,bb=ax.get_legend_handles_labels()
   cc,dd=second.get_legend_handles_labels()
   ax.legend(aa+cc,bb+dd,loc="upper left",fontsize=7)
 fig.suptitle("CMU original 120Hz soccer kicks | Real ball-triad visibility, outgoing evidence, foot proximity")
 fig.text(.012,.008,
  "Observed 3-point ball-marker centroid is not a certified soccer ball center; inferred launch onset is NOT ground-truth instep impact; original C3D NaN gaps remain unfilled. Genuine human optical 3D, not broadcast AI pose.",
  fontsize=8)
 fig.tight_layout(rect=(0,.055,1,.963))
 out=ROOT/"media/v050_human_soccer_ball_mocap_120hz_quality.png"
 out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out,dpi=150)
 plt.close(fig)
 print("V050_HUMAN_BALL_MOCAP_QC_FIGURE",out,out.stat().st_size)
 # Inferred event aligned right-knee/ankle/arm pose channels, no exact impact
 ids=("10_02","10_05")
 fig,axs=plt.subplots(2,1,figsize=(11.5,7.4))
 for ax,name in zip(axs,ids):
  info=next(x for x in summary["motions"] if x["id"]==name)
  estimated=info["inferred_event"]["inferred_ball_launch_onset_s"]
  with np.load(BASE/(name+".npz"),allow_pickle=False) as z:
   t=z["time_s"].astype(float)-estimated
   h=z["marker_derived_human_8_cosine_relations"]
   labels=z["semantic_relation_names"].tolist()
  for which in ("right_knee_cos","right_ankle_cos","left_knee_cos","right_elbow_cos"):
   ax.plot(t,h[:,labels.index(which)],label=which,lw=1.2)
  ax.axvline(0,linestyle="--",color="black",lw=.9,label="Inferred soccerball motion onset (not impact)")
  ax.set_xlim(-.6,.35)
  ax.set_ylim(-1.05,1.05)
  ax.set_title(name+": human measured full-body segment-direction cosine features")
  ax.set_ylabel("Measured marker-derived cosine (-1 to 1)")
  ax.set_xlabel("Seconds relative to inferred ball-triad launch onset")
  ax.grid(alpha=.25)
  ax.legend(fontsize=8,ncol=3)
 fig.suptitle("Two CMU soccer kicks with observed outgoing ball-marker movement | Human pose coordination, NOT athlete joint power")
 fig.tight_layout(rect=(0,0,1,.955))
 pos=ROOT/"media/v050_human_pose_knee_ankle_arms_timing.png"
 fig.savefig(pos,dpi=145)
 plt.close(fig)
 print("V050_HUMAN_120HZ_POSE_COORDINATION_FIGURE",pos,pos.stat().st_size)

if __name__=="__main__":plot()
