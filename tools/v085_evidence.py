#!/usr/bin/env python3
"""Independent SHA-bound 200Hz whole-body audit of v84 genuine G1 PhysX.
Descriptive timings are not learned human force or contact manifold truth.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v085_frozen_native_contact.json"
RUNS=ROOT/"reports/v085_g1_physics_final.audit.json"
TARGET=ROOT/"reports/v085_independent_200hz.audit.json"
def sha(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()
def is_number(x):return type(x) in (float,int) and math.isfinite(x)
def audit_episode(raw,expected):
    rows=raw["traces"][0]
    names=raw["joint_names"]
    if len(names)!=29 or len(set(names))!=29:raise ValueError("ORIGINAL_29DOF_ORDER_INVALID")
    if raw["root_fixed"] is not False or raw["seed"]!=expected["seed"]:
        raise ValueError("ORIGINAL_G1_SOURCE_OR_SEED_INVALID")
    if not rows:raise ValueError("NO_REAL_PHYSICS")
    limits=raw["joint_velocity_limits"][0]
    if len(limits)!=29 or any(not is_number(x) or x<=0 for x in limits):
        raise ValueError("ORIGINAL_QDOT_LIMITS_INVALID")
    ratio=0.0;max_by_joint=[0.]*29
    supports=Counter();nonfoot_peak=0.;comz=[];torso_energy=[]
    times=[];min_height=float("inf");max_tilt=0.
    groups={g:[i for i,n in enumerate(names) if match(n)]
        for g,match in {
        "support_leg":lambda x:x.startswith("left_hip") or x.startswith("left_knee") or x.startswith("left_ankle"),
        "swing_leg":lambda x:x.startswith("right_hip") or x.startswith("right_knee") or x.startswith("right_ankle"),
        "waist":lambda x:x.startswith("waist"),
        "arms":lambda x:x.startswith("left_shoulder") or x.startswith("right_shoulder") or "elbow" in x or "wrist" in x,
        }.items()}
    energy={g:[] for g in groups}
    for i,r in enumerate(rows):
        t=r["time_s"];v=r["joint_vel"]
        if len(v)!=29 or abs(r["physics_dt_s"]-.005)>1e-8:
            raise ValueError("NOT_200HZ_29_JOINT_MEASUREMENTS")
        if not is_number(t) or not all(is_number(x) for x in v):
            raise ValueError("NONFINITE_TIME_OR_JOINT_VELOCITY")
        if times and abs((t-times[-1])-.005)>1e-5:raise ValueError("NONCONTIGUOUS_PHYSX")
        times.append(t)
        for j,x in enumerate(v):
            max_by_joint[j]=max(max_by_joint[j],abs(x))
            ratio=max(ratio,abs(x)/limits[j])
        foot=r["support_foot_force_N"]
        left,right=foot["left"],foot["right"]
        if not all(is_number(a) and a>=0 for a in (left,right)):
            raise ValueError("INVALID_MEASURED_SUPPORT")
        supports[("L" if left>25 else "")+("R" if right>25 else "") or "flight"]+=1
        nonfoot_peak=max(nonfoot_peak,r["nonfoot_contact_N"])
        if len(r["whole_body_com_xyz"])!=3:raise ValueError("MISSING_REAL_COM")
        comz.append(r["whole_body_com_xyz"][2])
        min_height=min(min_height,r["root_xyz"][2])
        max_tilt=max(max_tilt,r["root_tilt_deg"])
        for group,indices in groups.items():
            energy[group].append(sum(v[j]*v[j] for j in indices))
    rec=raw["records"][0];contact=rec.get("contact_time_s")
    qualified=is_number(contact) and contact>=0
    if qualified:
        ids=[i for i,t in enumerate(times) if contact-.4-1e-8<=t<=contact+1.25+1e-8]
        before=sum(times[i]<contact for i in ids)
        after=sum(times[i]>=contact for i in ids)
        full=before>=80 and after>=250
    else:ids=[];before=after=0;full=False
    peaks={}
    for group,signal in energy.items():
        idx=max(ids,key=lambda i:signal[i]) if ids else None
        peaks[group]={"peak_time_s":round(times[idx],4) if idx is not None else None,
                      "peak_relative_to_contact_s":round(times[idx]-contact,4) if idx is not None else None,
                      "integrated_sq_qdot":round(.005*sum(signal[i] for i in ids),6)}
    diag=raw.get("nonfoot_contact_diagnostics",[])
    return {
        "frames":len(rows),"time_first_last_s":[times[0],times[-1]],
        "recorded_duration_s":round(len(rows)*.005,4),
        "complete_14s":len(rows)>=2800 and times[-1]>=13.995,
        "contact_time_s":contact,"measured_true_contact_window":bool(full),
        "pre_window_frames":before,"post_window_frames":after,
        "peak_qdot_over_original_limit":round(ratio,8),
        "peak_qdot_by_joint_rad_s":dict(zip(names,[round(x,5) for x in max_by_joint])),
        "support_fraction":{k:round(v/len(rows),5) for k,v in sorted(supports.items())},
        "wholebody_com_z_min_max_m":[round(min(comz),5),round(max(comz),5)],
        "root_min_z_m":round(min_height,5),"max_root_tilt_deg":round(max_tilt,4),
        "nonfoot_net_force_peak_N":round(nonfoot_peak,5),
        "recorded_named_nonfoot_events":len(diag),
        "peak_measured_group_kinetic_proxies":peaks,
        "instep_fraction":rec.get("requested_surface_impulse_fraction"),
        "goal":rec.get("goal"),"recovered_record_flag":rec.get("recovered"),
        "warning":"joint qdot^2 proxy is not joint power; net wrench is not a shape-level pair or GRF ground truth"}
def summarize():
    cfg=json.loads(CFG.read_text())
    batch=json.loads(RUNS.read_text())
    if batch["protocol_sha256"]!=sha(CFG):
        raise ValueError("FROZEN_PROTOCOL_MISMATCH")
    expected={x["name"]:x for x in cfg["cases"]}
    episodes=[]
    for r in batch["runs"]:
        if not r["true_original_200Hz_G1_CUDA_PhysX"]:continue
        rawpath=ROOT/"reports"/("v085_"+r["name"]+".json")
        if sha(rawpath)!=r["raw_sha256"]:
            raise ValueError("RAW_PHYSX_SHA_MISMATCH:"+r["name"])
        phys=json.loads(rawpath.read_text())
        row=audit_episode(phys,expected[r["name"]])
        if row["peak_qdot_over_original_limit"]>cfg["gates"]["original_all_joint_qdot_ratio_max"]:
            if r["gates"]["hard_gate_pass"]:raise ValueError("INVALID_HARD_PASS_QDOT")
        if not row["complete_14s"] and r["gates"]["hard_gate_pass"]:
            raise ValueError("CENSORED_AS_SUCCESS")
        if not row["measured_true_contact_window"] and r["gates"]["hard_gate_pass"]:
            raise ValueError("NO_FULL_CONTACT_WINDOW_SUCCESS")
        episodes.append({"name":r["name"],"scene":r["scene"],
             "policy":r["control"],"raw_sha256":r["raw_sha256"],
             "strict_hard_gate_pass":r["gates"]["hard_gate_pass"],
             "failed_gates":r["gates"]["failed_gates"],**row})
    result={
        "version":"v0.1.84","frozen_commit":batch["frozen_commit"],
        "protocol_sha256":batch["protocol_sha256"],
        "new_certified_true_PhysX_episodes":len(episodes),
        "strict_pass_count":sum(x["strict_hard_gate_pass"] for x in episodes),
        "strict_fail_count":sum(not x["strict_hard_gate_pass"] for x in episodes),
        "real_14s_completed":sum(x["complete_14s"] for x in episodes),
        "complete_pre_minus0p4_post_plus1p25":sum(x["measured_true_contact_window"] for x in episodes),
        "readonly_parity":{k:p.get("pass",False) for k,p in batch.get("paired_readonly_physical_parity",{}).items()},
        "native_actor_pair_manifold_truth":"UNKNOWN unless independently positive native pair witness",
        "CMU_professional_instep_contact_force_truth":False,
        "synthetic_wholebody_policy_tested_in_PhysX":False,
        "episodes":episodes}
    if TARGET.exists():raise FileExistsError(TARGET)
    with TARGET.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V085_INDEPENDENT",json.dumps({k:v for k,v in result.items() if k!="episodes"}))
    return result
if __name__=="__main__":summarize()
