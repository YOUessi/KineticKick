"""Contact-threshold sensitivity / gait-cycle audit for saved baseline CSV only.

No Isaac Sim, no extra GPU. Extends baseline to avoid declaring physical foot
events based on an arbitrary 45N threshold.
"""
import csv,json,math
from pathlib import Path
from statistics import median
import numpy as np

ROOT=Path("/home/you/projects/kinetickick")
src=ROOT/"reports/human_walk_baseline_v050.csv"
rows=list(csv.DictReader(src.open()))
rows=[{k:(float(v) if k not in ("step","done") else int(float(v))) for k,v in r.items()} for r in rows]
rows=[r for r in rows if r["t_s"]>=0.5 and not r["done"]]
dt=0.02

def segments(flags):
    start=None;out=[]
    for i,b in enumerate(flags+[False]):
        if b and start is None:start=i
        if not b and start is not None:
            out.append((start,i));start=None
    return out

results={}
for t in (5,15,30,45,75,120):
    sides={}
    both=[]
    neither=[]
    for s in ("left","right"):
        flags=[r[f"{s}_contact_N"]>t for r in rows]
        ss=[(b-a)*dt for a,b in segments(flags)]
        ww=[(b-a)*dt for a,b in segments([not q for q in flags])]
        sides[s]={
            "stance_ratio":sum(flags)/len(flags),
            "stance_bouts":len(ss),
            "stance_median_s":median(ss) if ss else None,
            "stance_iqr_s":(float(np.percentile(ss,25)),float(np.percentile(ss,75))) if ss else None,
            "stance_median_abs_knee_rad":median([abs(r[f"{s}_knee"]) for r,b in zip(rows,flags) if b]) if any(flags) else None,
            "swing_median_abs_knee_rad":median([abs(r[f"{s}_knee"]) for r,b in zip(rows,flags) if not b]) if not all(flags) else None,
        }
    left=[r["left_contact_N"]>t for r in rows]
    right=[r["right_contact_N"]>t for r in rows]
    results[str(t)]={
       "sides":sides,
       "double_support_ratio":sum(a and b for a,b in zip(left,right))/len(rows),
       "no_support_ratio":sum(not a and not b for a,b in zip(left,right))/len(rows),
       "note":"Sensor force threshold sensitivity; no calibrated foot sole contact model"
    }
slope_y=np.polyfit([r["t_s"] for r in rows],[r["root_y"] for r in rows],1)[0]
results["summary"]={
    "samples":len(rows),
    "command_vx_mps":0.5,
    "height_median":median([r["root_z"] for r in rows]),
    "left_knee_all_median_abs_rad":median([abs(r["left_knee"]) for r in rows]),
    "right_knee_all_median_abs_rad":median([abs(r["right_knee"]) for r in rows]),
    "lateral_world_drift_mps":float(slope_y),
    "diagnosis":"high knee flexion, rapid contact alternation, weak asymmetric shoulder/hip coordination; only proxies, need human motion benchmark"
}
out=ROOT/"reports/gait_contact_threshold_sensitivity.json"
out.write_text(json.dumps(results,indent=2,ensure_ascii=False))
for t in (5,15,30,45,75,120):
 d=results[str(t)]
 print("THRESHOLD",t,"STANCE_MEDIAN_L/R",
       round(d["sides"]["left"]["stance_median_s"] or 0,3),
       round(d["sides"]["right"]["stance_median_s"] or 0,3),
       "DOUBLE",round(d["double_support_ratio"],3),
       "NO_SUPPORT",round(d["no_support_ratio"],3))
print("SENSITIVITY_DONE",str(out))
