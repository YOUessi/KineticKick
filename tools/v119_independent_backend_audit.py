#!/usr/bin/env python3
"""Independent six-case native-backend audit; no PhysX launch and no control writes."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def score(root:Path,master:dict):
    root=Path(root); cases=master["results"]
    if len(cases)!=6 or master["completed_isolated_physx_runs"]!=6:
        raise ValueError("MISSING_FROZEN_CASES")
    expected=("cpu_positive_off","cpu_positive_on","cpu_separated_on",
              "cuda_positive_off","cuda_positive_on","cuda_separated_on")
    if tuple(a["name"] for a in cases)!=expected:raise ValueError("CASE_ORDER_DRIFT")
    data={}
    for r in cases:
        p=root/"reports"/("v119_native_"+r["name"]+".json")
        if not p.exists() or sha(p)!=r["raw_sha"]:raise ValueError("RAW_SHA_MISMATCH:"+r["name"])
        x=json.loads(p.read_text())
        if x.get("actual_runtime_frames")!=180 or x.get("dt_s")!=.005 or x.get("device")!=r["device"]:
            raise ValueError("INVALID_REAL_PHYSICS_STEP:"+r["name"])
        tr=x.get("runtime_z_vz",[])
        if len(tr)!=180 or any(len(a)!=3 or a[0]!=i+1 or not all(math.isfinite(v) for v in a)
                                for i,a in enumerate(tr)):raise ValueError("BAD_200HZ_PHYSICS_TRACE:"+r["name"])
        if "exception" in x or x.get("errors"):raise ValueError("HIDDEN_RUNTIME_ERROR:"+r["name"])
        if r["layout"]=="colliding" and not x.get("observed_collision_dynamics"):
            raise ValueError("MISSING_POSITIVE_CONTACT_DYNAMICS:"+r["name"])
        if r["layout"]=="separated" and not x.get("observed_noncollision_dynamics"):
            raise ValueError("MISSING_NEGATIVE_SEPARATION_DYNAMICS:"+r["name"])
        data[r["name"]]=x
    parity={}
    for dev in ("cpu","cuda"):
        off=data[dev+"_positive_off"];on=data[dev+"_positive_on"];neg=data[dev+"_separated_on"]
        parity[dev]=off["runtime_z_vz"]==on["runtime_z_vz"] and off["box_position_w"]==on["box_position_w"]
        if not parity[dev]:raise ValueError("READONLY_STATE_DRIFT:"+dev)
        if neg.get("tensor_contact_counts")!=0:raise ValueError("SEPARATED_SENSOR_FALSE_POSITIVE:"+dev)
    # Case-level observed native headers must be preserved even if actor pair decode fails.
    native_headers={name:{k:int(obj.get("native_counts",{}).get(k,0)) for k in
                           ("callback_headers","poll_headers")} |
                         {k:int(obj.get("native_full_counts",{}).get(k,0)) for k in
                           ("callback_headers","poll_headers")}
                    for name,obj in data.items()}
    # Explicitly carry both regular/full counts without dict collision.
    actual={name:{
        "regular_callback_headers":obj.get("native_counts",{}).get("callback_headers",0),
        "regular_poll_headers":obj.get("native_counts",{}).get("poll_headers",0),
        "full_callback_headers":obj.get("native_full_counts",{}).get("callback_headers",0),
        "full_poll_headers":obj.get("native_full_counts",{}).get("poll_headers",0),
        "native_valid_colliders":bool(obj.get("native_pair_witness")),
        "tensor_contact_counts":obj.get("tensor_contact_counts",0),
        "tensor_peak_N":obj.get("tensor_peak_force_N",0.0)
    } for name,obj in data.items()}
    return {
        "version":"v0.1.119",
        "source_commit":master["source_sha"],
        "frozen_protocol_sha256":master["protocol_sha256"],
        "actual_physx_runs":len(data),
        "actual_200hz_steps":sum(len(x["runtime_z_vz"]) for x in data.values()),
        "cpu_and_cuda_recorded_off_on_parity_exact":parity,
        "physics_positive_and_separated_controls":{"cpu":True,"cuda":True},
        "reporter_enabled_on_actor_prim_both_backends":all(
            data[k]["stage"]["ball_report"] and data[k]["stage"]["box_report"]
            for k in ("cpu_positive_on","cuda_positive_on")),
        "cpu_tensor_zero_despite_mechanical_contact":data["cpu_positive_on"].get("tensor_contact_counts")==0,
        "cuda_tensor_positive_witness":data["cuda_positive_on"].get("tensor_contact_counts",0)>0,
        "native_actor_collider_manifold_certified":{
            dev:data[dev+"_positive_on"].get("native_pair_witness",False)
            for dev in ("cpu","cuda")},
        "native_header_breakdown":actual,
        "earlier_v118_cuda_positive_raw_identical":(
            (root/"reports/v118_native_full_positive_on.json").exists() and
            sha(root/"reports/v118_native_full_positive_on.json")==
            sha(root/"reports/v119_native_cuda_positive_on.json")),
        "original_G1_shots_this_isolated_batch":0,
        "new_control_policy_motor_actuation":0,
        "cannot_infer_wrist_hip_noncollision":True
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args(); master=json.loads((a.root/"reports/v119_native_backend_master.audit.json").read_text())
    result=score(a.root,master)
    out=a.root/"reports/v119_independent_backend.audit.json"
    with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("INDEPENDENT_NATIVE_BACKEND_RESULT",json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
