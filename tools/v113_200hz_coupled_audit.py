#!/usr/bin/env python3
"""Independent write-once 200Hz 29DoF dynamics, support, and censoring audit.

This produces measurements from certified raw PhysX, never invents GRF/CoP
or actor/collider contact manifold. No training or motor actuation.
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
GROUPS={
  "waist": ("waist_",),
  "left_leg": ("left_hip_","left_knee_","left_ankle_"),
  "right_leg": ("right_hip_","right_knee_","right_ankle_"),
  "left_arm": ("left_shoulder_","left_elbow_","left_wrist_"),
  "right_arm": ("right_shoulder_","right_elbow_","right_wrist_")}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def group_indices(names):
    if len(names)!=29 or len(set(names))!=29:raise ValueError("INVALID_29DOF_NAMES")
    groups={key:[i for i,n in enumerate(names) if any(n.startswith(p) for p in prefixes)]
            for key,prefixes in GROUPS.items()}
    if sorted(i for ix in groups.values() for i in ix)!=list(range(29)):
        raise ValueError("UNCATEGORIZED_OR_DUPLICATED_G1_JOINT")
    return groups
def strip_terminal_event_only(t):
    """Remove *only* duplicate final event marker; never extrapolate its physics."""
    if len(t)<2:return t,0
    prev,terminal=t[-2:]
    delta=float(terminal["time_s"])-float(prev["time_s"])
    if -1e-8<=delta<=2e-6 and terminal.get("terminated") is True and prev.get("terminated") is False:
        keys=set(prev)|set(terminal)
        if keys != set(prev) or any(prev[k]!=terminal[k] for k in keys if k not in ("time_s","terminated")):
            raise ValueError("TERMINAL_MARKER_HAS_NEW_PHYSICAL_CONTENT")
        return t[:-1],1
    return t,0
def require_200hz(t):
    if len(t)<100:raise ValueError("INCOMPLETE_REAL_200HZ")
    x=np.asarray([r["time_s"] for r in t],float)
    d=np.diff(x)
    if not np.isfinite(x).all() or not np.all(np.isclose(d,.005,atol=1e-6)):
        raise ValueError("NON_200HZ_OR_DUPLICATE_TERMINAL_STEPS")
    if any(abs(float(r["physics_dt_s"])-.005)>1e-6 for r in t):
        raise ValueError("NOT_REAL_200HZ_PHYSICS_STEPS")
    return x
def _stats(x):
    x=np.asarray(x,float)
    return {"mean":float(x.mean()),"peak":float(x.max()),"minimum":float(x.min())}
def analyze(raw, certified=True):
    if not certified:raise ValueError("SOURCE_PHYSICS_NOT_CERTIFIED")
    if raw.get("root_fixed") is not False:raise ValueError("NOT_FREE_ROOT")
    names=raw["joint_names"];groups=group_indices(names)
    source_rows=raw["traces"][0]
    t,terminal_event_only=strip_terminal_event_only(source_rows)
    times=require_200hz(t)
    if any(len(r["joint_pos"])!=29 or len(r["joint_vel"])!=29 for r in t):
        raise ValueError("INVALID_ACTUAL_29DOF_OR_COM_SHAPE")
    q=np.asarray([r["joint_pos"] for r in t],float)
    dq=np.asarray([r["joint_vel"] for r in t],float)
    com=np.asarray([r["whole_body_com_xyz"] for r in t],float)
    root=np.asarray([r["root_xyz"] for r in t],float)
    lim=np.asarray(raw["joint_velocity_limits"],float).reshape(-1)
    if q.shape!=(len(t),29) or dq.shape!=(len(t),29) or com.shape!=(len(t),3) or root.shape!=(len(t),3) or lim.shape!=(29,):
        raise ValueError("INVALID_ACTUAL_29DOF_OR_COM_SHAPE")
    if not all(np.isfinite(a).all() for a in (q,dq,com,root,lim)) or (lim<=0).any():
        raise ValueError("INVALID_NONFINITE_MEASURED_STATE")
    support=np.asarray([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in t],float)
    nonfoot=np.asarray([r["nonfoot_contact_N"] for r in t],float)
    if not np.isfinite(support).all() or not np.isfinite(nonfoot).all():raise ValueError("NONFINITE_CONTACT")
    contact=raw["records"][0].get("contact_time_s")
    if contact is None or not math.isfinite(float(contact)):raise ValueError("NO_REAL_BALL_CONTACT_TIME")
    contact=float(contact)
    pre=np.flatnonzero((times>=contact-.4-1e-7)&(times<contact))
    post=np.flatnonzero((times>=contact)&(times<=contact+1.25+1e-7))
    if len(pre)==0 or len(post)==0:raise ValueError("CONTACT_OUTSIDE_REAL_TRACE")
    post_complete=bool(times[-1]>=contact+1.25-1e-6)
    full14=bool(times[0]<=.005+1e-6 and times[-1]>=13.995)
    def phase(ix):
        return {"actual_rows":len(ix),"start_s":float(times[ix[0]]),"end_s":float(times[ix[-1]]),
            "joint_group_qdot_rms_radps":{
              k:float(np.sqrt(np.mean(np.square(dq[ix][:,g])))) for k,g in groups.items()},
            "support_left_N":_stats(support[ix,0]),
            "support_right_N":_stats(support[ix,1]),
            "COM_z_m":_stats(com[ix,2]),
            "root_z_m":_stats(root[ix,2]),
            "COM_horizontal_displacement_m":float(np.linalg.norm(com[ix[-1],:2]-com[ix[0],:2])),
            "nonfoot_peak_N":float(nonfoot[ix].max())}
    nonfoot_events=np.flatnonzero(nonfoot>=20)
    return {"count_real_200hz":len(t),"source_raw_trace_rows":len(source_rows),
      "duplicate_terminal_sentinel_markers":terminal_event_only,"contact_s":contact,
      "has_full_pre_0p4_s":bool(times[pre[0]]<=contact-.4+.005001),
      "has_full_post_1p25_s":post_complete,"full_14s":full14,
      "right_censored":not full14,"end_time_s":float(times[-1]),
      "qdot_peak_all_original29_ratio":float(np.max(np.abs(dq)/lim)),
      "first_nonfoot_ge20N_s":float(times[nonfoot_events[0]]) if len(nonfoot_events) else None,
      "first_nonfoot_ge20N_not_direct_pair_proof":True,
      "pre_contact":phase(pre),"post_contact_observed_only":phase(post),
      "native_actor_collider_manifold_certified":False,
      "true_GRF_CoP_and_foot_slip_witness_certified":False,
      "inferred_natural_human_coordination":False}
def main():
    master=ROOT/"reports/v113_original_g1_batch.audit.json"
    if not master.exists():raise SystemExit("NO_ORIGINAL_G1_BATCH")
    batch=json.loads(master.read_text())
    out=ROOT/"reports/v113_200hz_coupled.audit.json"
    if out.exists() or out.is_symlink():raise FileExistsError("WRITE_ONCE_COUPLED_AUDIT")
    runs=[]
    for e in batch["runs"]:
        if not e["certified_original_G1_cuda_200Hz"]:continue
        rawfile=ROOT/"reports"/("v113_"+e["name"]+".json")
        if sha(rawfile)!=e["raw_sha256"]:raise ValueError("SOURCE_SHA_MISMATCH")
        m=analyze(json.loads(rawfile.read_text()))
        runs.append({"name":e["name"],"sha256":e["raw_sha256"],"actual_real_dynamics":m,
           "all_original_hard_gates":e["hard_gates"]})
    result={"version":"v0.1.112","source_batch_sha256":sha(master),
      "true_scored_original_G1":len(runs),
      "real_200hz_rows":sum(x["actual_real_dynamics"]["count_real_200hz"] for x in runs),
      "terminal_event_sentinels_not_counted_as_steps":sum(x["actual_real_dynamics"]["duplicate_terminal_sentinel_markers"] for x in runs),
      "native_pair_not_certified":True,
      "full_post_contact_windows":sum(x["actual_real_dynamics"]["has_full_post_1p25_s"] for x in runs),
      "full_14s_recovery_observed":sum(x["actual_real_dynamics"]["full_14s"] for x in runs),
      "per_episode":runs}
    with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V113_REAL_COUPLED_AUDIT",json.dumps({k:v for k,v in result.items() if k!="per_episode"}))
if __name__=="__main__":main()
