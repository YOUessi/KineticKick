"""Audit actual live Tang fixed-root soccer kicks from Isaac Sim telemetry.

A PHYSICAL_BALL_HIT must precede a GOAL_CROSSING=True in the same shot.
Saved metrics are empirical single-condition demonstrations, NOT free-root RL.
"""
from pathlib import Path
import re,json,collections,datetime

BASE=Path("/home/you/projects/kinetickick")
LOG=BASE/"reports/show_g1_soccer_real_kicks.log"
lines=LOG.read_text(errors="replace").splitlines()
shots={}
for line in lines:
    if line.startswith("SHOT_START "):
        fields=line.split(" ")
        _,i,mode,power=fields[:4]
        shots.setdefault(int(i),{"id":int(i),"mode":mode,"power":power,"contact":None,"crossed":None})
    if line.startswith("PHYSICAL_BALL_HIT "):
        m=re.match(r"^PHYSICAL_BALL_HIT\s+(\d+)\s+(\w+)\s+(\w+)\s+(\{.*\})$",line)
        if not m: continue
        i=int(m.group(1))
        shots.setdefault(i,{"id":i,"mode":m.group(2),"power":m.group(3),"contact":None,"crossed":None})
        shots[i]["contact"]=json.loads(m.group(4))
    if line.startswith("GOAL_CROSSING "):
        m=re.match(r"^GOAL_CROSSING\s+(\d+)\s+(\w+)\s+(\w+)\s+(True|False)\s+(\[.*\])$",line)
        if not m:continue
        i=int(m.group(1))
        if i in shots:
            shots[i]["crossed"]=m.group(4)=="True"
            shots[i]["crossing_xyz"]=json.loads(m.group(5))

valid=[v for v in shots.values() if v["contact"] is not None and v["crossed"] is not None]
by={}
for mode in ("instep_top","medial","lateral"):
    subset=[v for v in valid if v["mode"]==mode and v["power"]=="medium"]
    if not subset: raise AssertionError(f"Missing successful physics verification for mode {mode}")
    first=subset[0]
    hit=first["contact"]
    by[mode]={
      "verified_shots":len(subset),
      "first_shot_id":first["id"],
      "first_impact_time_s":hit["time_s"],
      "first_impact_force_N":hit["first_force_N"],
      "first_ball_velocity_mps_xyz":hit["vxyz"],
      "first_contact_face_estimate":hit["face_est"],
      "first_goal_crossing":first["crossed"],
      "first_goal_xyz":first.get("crossing_xyz"),
      "observed_goal_crossing_success_count":sum(bool(s["crossed"]) for s in subset)
    }
assert all(x["first_goal_crossing"] for x in by.values())
result={
   "recorded_at_local":datetime.datetime.now().isoformat(),
   "scene":"Tang Isaac Lab 37DoF G1 + 0.43kg physics football + training goal",
   "physics_root_fixed":True,
   "robot_free_standing":False,
   "learned_human_motion":False,
   "football_count":1,
   "football_speed_injected_directly":False,
   "shooting_modes":by,
   "number_of_verified_complete_shots":len(valid),
   "limitations":"Foot contact face is inferred from cuboid geometry, not direct PhysX patch contacts. Goal crossing uses position-plane test. This is fixed-root scripted swing; not natural/run-up/balanced humanoid kicking."
}
target=BASE/"reports/real_soccer_kick_live_audit.json"
target.write_text(json.dumps(result,indent=2,ensure_ascii=False))
print("SOCCER_SHOT_AUDIT_OK",json.dumps({k:v["first_goal_crossing"] for k,v in by.items()}),flush=True)
print("FILE",target,"total_checked",len(valid),flush=True)
