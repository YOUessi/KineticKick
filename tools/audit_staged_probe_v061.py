#!/usr/bin/env python3
"""Independent fail-closed v061 12-case 200Hz/29DoF observer qualification."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from audit_v059_observer_parity import compare_raw
from audit_v060_observer_factorial import assert_complete_physics_fields
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()
def audit(root=ROOT):
    cpath=root/"configs/g1_staged_probe_v061.json"
    c=json.loads(cpath.read_text())
    lock=json.loads((root/"reports/v061_protocol_lock.json").read_text())
    progress=json.loads((root/"reports/v061_batch_progress.json").read_text())
    if len(c["cases"])!=12 or len(progress["runs"])!=12:raise ValueError("Incomplete preregistered physical matrix")
    digest=sha(cpath)
    if digest!=lock["protocol_sha256"] or digest!=progress["protocol_sha256"]:
        raise ValueError("Preregistration SHA drift")
    for p,expected in {**c["source_hashes"],**c["policy_hashes"]}.items():
        if sha(root/p)!=expected:raise ValueError("Source drift "+p)
    rows=[];raw={}
    for k,res in zip(c["cases"],progress["runs"]):
        if k["id"]!=res["id"] or res["seed"]!=k["seed"] or res["probe"]!=k["probe"]:
            raise ValueError("Out of order/frozen physics configuration drift")
        path=root/k["output"];log=path.with_suffix(".log")
        if not log.exists() or sha(log)!=res["log_sha256"]:raise ValueError("Original PhysX log modified")
        if path.exists()!=bool(res.get("raw_sha256")):raise ValueError("Original PhysX presence modified")
        if path.exists() and sha(path)!=res["raw_sha256"]:raise ValueError("Original PhysX JSON modified")
        valid=res["status"]=="VALID_G1" and path.exists() and res["code"]==0
        row=dict(id=k["id"],policy=k["policy"],backend=k["backend"],probe=k["probe"],
                 status=res["status"],valid_G1=valid,raw_sha256=res.get("raw_sha256"),
                 last_stage_if_missing=res.get("last_stage_log"),strict_pass=None,
                 measured_200hz_frames=None,post_impact_1p25s_complete=False)
        if valid:
            o=json.loads(path.read_text())
            if o["root_fixed"] is not False or o["seed"]!=1201 or len(o["joint_names"])!=29:
                raise ValueError("Original G1 physics mismatch")
            tr=o["traces"][0]
            assert_complete_physics_fields(tr)
            marker=bool(tr[-1].get("terminated") is True and 0<tr[-1]["time_s"]-tr[-2]["time_s"]<.003)
            frames=tr[:-1] if marker else tr
            if not frames or any(abs(b["time_s"]-a["time_s"]-.005)>2e-4 for a,b in zip(frames,frames[1:])):
                raise ValueError("Not 200Hz")
            probe=o["staged_probe_v061"]
            if probe["capture_count"]<30 or probe["controller_writes"] or probe["physx_writes"]:
                raise ValueError("Probe broken")
            ratio=max(abs(v)/lim for f in frames for v,lim in zip(f["joint_vel"],o["joint_velocity_limits"][0]))
            t=o["records"][0].get("contact_time_s")
            qdot_allowed=ratio<=1
            row.update(strict_pass=o["stride_shot_success"],measured_200hz_frames=len(frames),
                       contact_time_s=t,last_real_time_s=frames[-1]["time_s"],
                       post_impact_1p25s_complete=t is not None and frames[-1]["time_s"]>=t+1.25,
                       legal_instep_fraction=o["records"][0].get("requested_surface_impulse_fraction"),
                       qdot_peak_ratio=ratio,qdot_all_joints_allowed=qdot_allowed,
                       goal=o["records"][0].get("goal"),termination_reasons=o["termination_reasons"],
                       probe_last_success_stage=probe["last_successful_stage"])
            raw[k["id"]]=o
        rows.append(row)
    baseline=root/"reports/v060_cuda_v32_none.json"
    if not baseline.exists():raise ValueError("Frozen baseline unavailable")
    old=json.loads(baseline.read_text())
    comparisons={}
    for k in c["cases"]:
        if k["id"] in raw and k["backend"]=="cuda:0" and k["policy"]=="v32":
            comparisons[k["id"]]=compare_raw(old,raw[k["id"]])
    if "cpu_v32_full_compute" in raw:
        ref=json.loads((root/"reports/v060_cpu_v32_none.json").read_text())
        comparisons["cpu_v32_full_compute"]=compare_raw(ref,raw["cpu_v32_full_compute"])
    nvalid=sum(x["valid_G1"] for x in rows)
    return dict(version="v0.1.61",frozen_protocol_sha256=digest,
                physical_attempts=len(rows),valid_original_29DoF_G1=nvalid,
                unscorable=len(rows)-nvalid,strict_successes=sum(x["strict_pass"] is True for x in rows),
                diagnostic_only_no_new_policy_training=True,
                gpu_nonfoot_wrist_hip_direct_manifold_still_unproven=True,
                cpu_gpu_pair_physics_differ_do_not_compare_them_for_parity=True,
                parity_25_field_comparisons=comparisons,runs=rows)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(ROOT)
    with a.output.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print("V061_AUDIT",{k:v for k,v in result.items() if k not in ("runs","parity_25_field_comparisons")})
    for row in result["runs"]:
        print(row["id"],row["status"],row["strict_pass"],row["last_stage_if_missing"])
    print("V061_PARITY",json.dumps(result["parity_25_field_comparisons"],ensure_ascii=False))
