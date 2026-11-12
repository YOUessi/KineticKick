#!/usr/bin/env python3
"""Independent whole-body pre/post strike mechanics from SHA-certified 200Hz G1 data.

All τ are *recorded motor torque estimates*, never measured GRF or a
physically integrated external joint impulse. This does not prove human learning.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from v107_200hz_coupled_audit import analyze,group_indices,require_200hz
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def mechanics(raw):
    baseline=analyze(raw,certified=True)
    tr=raw["traces"][0]
    if len(tr)>baseline["count_real_200hz"]:
        tr=tr[:baseline["count_real_200hz"]]
    time=require_200hz(tr)
    group=group_indices(raw["joint_names"])
    dq=np.asarray([t["joint_vel"] for t in tr],float)
    tq=np.asarray([t["joint_torque_estimate"] for t in tr],float)
    support=np.asarray([[t["support_foot_force_N"][k] for k in ("left","right")] for t in tr],float)
    comv=np.asarray([t["whole_body_com_velocity"] for t in tr],float)
    if tq.shape!=(len(tr),29) or comv.shape!=(len(tr),3):
        raise ValueError("MISSING_PHYSICAL_ESTIMATED_TORQUE_OR_COMVELOCITY")
    if not all(np.isfinite(a).all() for a in (tq,dq,support,comv)):
        raise ValueError("NONFINITE_PHYSICAL_MEASUREMENTS")
    contact=baseline["contact_s"]
    pre=np.flatnonzero((time>=contact-.4-1e-7)&(time<contact))
    post=np.flatnonzero((time>=contact)&(time<=contact+1.25+1e-7))
    if len(pre)<79:raise ValueError("INCOMPLETE_PRE_CONTACT")
    def window(ix):
        # τ_est*qdot is only an actuator power estimate. Not external strike energy.
        power=tq[ix]*dq[ix]
        res={}
        for name,ids in group.items():
            v=power[:,ids]
            positive=float(np.maximum(v,0).sum()*.005)
            negative=float(np.minimum(v,0).sum()*.005)
            weights=np.maximum(v,0).sum(axis=1)
            center=(float(np.dot(time[ix],weights)/weights.sum())
                    if weights.sum()>1e-10 else None)
            res[name]={"estimated_positive_motor_work_J":positive,
                       "estimated_negative_motor_work_J":negative,
                       "positive_motor_power_time_centroid_s":center,
                       "max_abs_estimated_group_torque_Nm":float(np.abs(tq[ix][:,ids]).max())}
        return {"observed_real_steps":int(len(ix)),
          "observed_time_interval_s":[float(time[ix[0]]),float(time[ix[-1]])],
          "segment_estimated_motor_work":res,
          "two_foot_normal_force_proxy_impulse_Ns":
              {"left":float(support[ix,0].sum()*.005),
               "right":float(support[ix,1].sum()*.005)},
          "COM_horizontal_speed_mean_mps":float(np.linalg.norm(comv[ix,:2],axis=1).mean())}
    nearest=int(np.argmin(np.abs(time-contact)))
    contributions=tr[nearest].get("foot_velocity_contributions")
    if not isinstance(contributions,dict) or set(contributions)!=set(("hip","knee","ankle","base")):
        raise ValueError("MISSING_SOURCE_FOOT_VELOCITY_CONTRIBUTIONS")
    contrib={k:np.asarray(v,float) for k,v in contributions.items()}
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in contrib.values()):
        raise ValueError("NONFINITE_FOOT_VELOCITY_CONTRIBUTIONS")
    return {"source":"actual_original_G1_200Hz_not_human",
      "contact_time_s":float(contact),
      "pre_0p4s":window(pre),"post_observed_up_to_1p25s":window(post),
      "post_1p25s_full":bool(baseline["has_full_post_1p25_s"]),
      "full_14s":bool(baseline["full_14s"]),
      "right_censored":bool(baseline["right_censored"]),
      "instep_velocity_contribution_at_contact_mps":
         {k:[float(v) for v in x] for k,x in contrib.items()},
      "contribution_not_proof_of_true_impulse_or_external_power":True,
      "estimated_torque_not_force_plate_or_ground_reaction_true":True,
      "native_actor_collider_manifold_certified":False,
      "qualified_human_sports_kinetics":False}
def main():
    master=ROOT/"reports/v107_original_g1_batch.audit.json"
    out=ROOT/"reports/v107_mechanics_evidence.audit.json"
    if not master.is_file():raise SystemExit("MISSING_OWN_FROZEN_BATCH")
    if out.exists() or out.is_symlink():raise FileExistsError("WRITE_ONCE_MECHANICS")
    batch=json.loads(master.read_text())
    rows=[]
    for e in batch["runs"]:
        if not e["certified_original_G1_cuda_200Hz"]:continue
        p=ROOT/"reports"/("v107_"+e["name"]+".json")
        if sha(p)!=e["raw_sha256"]:raise ValueError("SOURCE_RAW_SHA_DRIFT")
        rows.append({"name":e["name"],"source_raw_sha256":e["raw_sha256"],
          "hard_gate_pass":e["hard_gates"]["hard_gate_pass"],
          "failed_gates":e["hard_gates"]["failed_gates"],
          "source_real_mechanics":mechanics(json.loads(p.read_text()))})
    result={"version":"v0.1.107","source_batch_sha256":sha(master),
        "qualified_real_original_G1":len(rows),
        "verified_motor_torque_estimate_not_human_kinetics":True,
        "native_manifold_certified":False,"rows":rows}
    with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V107_REAL_MECHANICS",json.dumps({k:v for k,v in result.items() if k!="rows"}))
if __name__=="__main__": main()
