#!/usr/bin/env python3
"""Read-only, fail-closed independent audit of eight frozen *real* G1 PhysX trials.

CUDA and CPU are never paired as if they were equivalent physical dynamics.
Time labels in the new observer are verified against joint state; do not
silently interpret env.sim.current_time as the state acquisition timestamp.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from audit_v059_observer_parity import compare_raw
from audit_v060_observer_factorial import assert_complete_physics_fields

ROOT=Path(__file__).resolve().parents[1]
DEV=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")
REFERENCES={
    "cuda_v32_safe_full":"v060_cuda_v32_none.json",
    "cpu_v32_safe_full":"v060_cpu_v32_none.json",
    "cuda_cmu05_safe_full":"v060_cuda_cmu05_native_only.json",
    "cpu_cmu05_safe_full":"v060_cpu_cmu05_native_only.json",
    "cuda_v32_mass_no_mul":"v060_cuda_v32_none.json",
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1048576),b""):h.update(part)
    return h.hexdigest()

def real_200hz(trace):
    if len(trace)<32:raise ValueError("Missing actual original G1 physics")
    terminal=bool(trace[-1].get("terminated") is True
                  and 0<trace[-1]["time_s"]-trace[-2]["time_s"]<.003)
    phys=trace[:-1] if terminal else trace
    if any(not .0048<=b["time_s"]-a["time_s"]<=.0052 for a,b in zip(phys,phys[1:])):
        raise ValueError("Non-200Hz original G1 trajectory")
    return phys,terminal

def compare_29dof_sample_alignment(phys,obs):
    """Same-index 29DoF state must match, even if sim clock labels lead state.

    Does not infer timestamps from qdot; records discrepancy for correction.
    """
    if len(obs)!=len(phys):raise ValueError("Observer/physics row count mismatch")
    maxq=maxv=maxoff=0.
    offsets=[]
    for a,b in zip(phys,obs):
        if len(b.get("joint_pos_rad",[]))!=29 or len(b.get("joint_vel_rad_s",[]))!=29:
            raise ValueError("Observer did not capture the whole 29DoF state")
        maxq=max(maxq,max(abs(x-y) for x,y in zip(a["joint_pos"],b["joint_pos_rad"])))
        maxv=max(maxv,max(abs(x-y) for x,y in zip(a["joint_vel"],b["joint_vel_rad_s"])))
        offsets.append(b["t"]-a["time_s"])
    if maxq>1e-6 or maxv>1e-6:raise ValueError("Observer and trace are not synchronized by index")
    if max(offsets)-min(offsets)>1e-4:raise ValueError("Variable observer clock offset")
    return dict(joint_position_max_abs_diff=maxq,joint_velocity_max_abs_diff=maxv,
                observer_clock_minus_actual_physics_seconds=round(sum(offsets)/len(offsets),8),
                observer_time_must_not_be_used_as_measured_state_time=True)

def audit(root=ROOT):
    cfgpath=root/"configs/g1_massfree_probe_v062.json"
    cfg=json.loads(cfgpath.read_text())
    prog=json.loads((root/"reports/v062_batch_progress.json").read_text())
    lock=json.loads((root/"reports/v062_protocol_lock.json").read_text())
    digest=sha(cfgpath)
    if cfg["version"]!="v0.1.62" or len(cfg["cases"])!=8 or len(prog["runs"])!=8:
        raise ValueError("Frozen protocol incomplete")
    if digest!=prog["protocol_sha256"] or digest!=lock["protocol_sha256"]:
        raise ValueError("Frozen protocol SHA drift")
    for path,expected in {**cfg["source_hashes"],**cfg["policy_hashes"]}.items():
        if sha(root/path)!=expected:raise ValueError("Frozen source hash drift: "+path)
    rows=[]; parity={}
    for c,p in zip(cfg["cases"],prog["runs"]):
        if (c["id"],c["backend"],c["probe"],c["policy"],c["seed"])!=(
            p["id"],p["backend"],p["probe"],p["policy"],p["seed"]):
            raise ValueError("Physical trial ordering or configuration drift")
        target=root/c["output"];log=target.with_suffix(".log")
        if not log.exists() or sha(log)!=p["log_sha256"]:
            raise ValueError("Raw PhysX stdout altered")
        detected=DEV.findall(log.read_text(errors="replace"))
        if not detected or detected[0]!=c["backend"] or detected[0]!=p["observed_backend"]:
            raise ValueError("Different runtime PhysX device")
        if target.exists()!=bool(p.get("raw_sha256")):
            raise ValueError("Raw evidence existence altered")
        if target.exists() and sha(target)!=p["raw_sha256"]:
            raise ValueError("Original PhysX raw data modified")
        valid=p["status"]=="VALID_G1" and p["code"]==0 and target.exists()
        row=dict(id=c["id"],backend=c["backend"],policy=c["policy"],probe=c["probe"],
                 seed=c["seed"],status=p["status"],original_G1_valid=valid,
                 strict_pass=None,original_physx_sha256=p.get("raw_sha256"),
                 last_operation_if_no_G1=p.get("last_stage_log"))
        if valid:
            raw=json.loads(target.read_text())
            if raw.get("seed")!=1201 or raw.get("root_fixed") is not False or len(raw.get("joint_names",[]))!=29:
                raise ValueError("Not original free-root 29DoF")
            if raw.get("policy_sha256")!=cfg["policy_hashes"][cfg["policies"][c["policy"]]]:
                raise ValueError("Policy drift")
            trace=raw["traces"][0]
            assert_complete_physics_fields(trace)
            phys,terminal=real_200hz(trace)
            o=raw.get("safe_body_v062",{})
            if o.get("mode")!=c["probe"] or o.get("actuator_writes") or o.get("physical_writes"):
                raise ValueError("Observer invalid/physical mutation")
            if o.get("actual_200hz_rows")!=len(phys) or len(o.get("rows",[]))!=len(phys):
                raise ValueError("Observer capture misses original frame")
            limits=raw["joint_velocity_limits"][0]
            if len(limits)!=29 or min(limits)<=0:raise ValueError("Missing original limits")
            ratio=max(abs(v)/lim for f in phys for v,lim in zip(f["joint_vel"],limits))
            r0=raw["records"][0]; impact=r0.get("contact_time_s")
            row.update(strict_pass=raw["stride_shot_success"],physics_200hz_frames=len(phys),
                       terminal_marker_excluded=terminal,
                       contact_s=impact,last_physics_s=phys[-1]["time_s"],
                       legal_instep_fraction=r0.get("requested_surface_impulse_fraction"),
                       goal=r0.get("goal"),recovered=r0.get("recovered"),
                       terminations=raw.get("termination_reasons"),
                       peak_original_29dof_qdot_over_limit=ratio,
                       full_pre_0p4=impact is not None and phys[0]["time_s"]<=impact-.4+.0051,
                       full_post_1p25=impact is not None and phys[-1]["time_s"]>=impact+1.25,
                       full_14s_observed=phys[-1]["time_s"]>=13.999)
            if c["probe"]=="safe_full":
                align=compare_29dof_sample_alignment(phys,o["rows"])
                dist=[b["left_wrist_hip_link_origin_distance_m"] for b in o["rows"]]
                if len(dist)!=len(phys) or not all(math.isfinite(x) and x>=0 for x in dist):
                    raise ValueError("Missing measured wrist/hip geometry")
                row.update(measured_min_wrist_hip_LINK_ORIGIN_m=min(dist),
                           contact_pair_not_identified=True,observed_state_clock=align,
                           no_observer_joint_target_writes=True)
            ref=REFERENCES.get(c["id"])
            if ref:
                reference=root/"reports"/ref
                if not reference.exists():raise ValueError("Missing historical baseline")
                prev=json.loads(reference.read_text())
                if prev.get("seed")!=raw.get("seed") or prev.get("policy_sha256")!=raw.get("policy_sha256"):
                    raise ValueError("Historical parity not same policy/seed")
                cmp=compare_raw(prev,raw)
                if not cmp.get("matches_at_abs_tol_1e_6") or cmp.get("common_field_count")!=25:
                    raise ValueError("Frozen same-backend 25-field physics parity FAIL "+c["id"])
                parity[c["id"]+"_vs_"+ref]=cmp
        else:
            if target.exists() or p["status"]!="NO_G1_OUTPUT":
                raise ValueError("Unscorable missing case misclassified")
            if c["probe"] in ("mass_clone","mass_after") and "weighted_com frame 0" not in p["last_stage_log"]:
                raise ValueError("Unsupported last-stage claim")
        rows.append(row)
    n=sum(x["original_G1_valid"] for x in rows)
    return dict(version="v0.1.62",frozen_protocol_sha256=digest,
                real_PhysX_attempts=len(rows),valid_original_G1=n,
                cuda_valid=sum(x["original_G1_valid"] and x["backend"]=="cuda:0" for x in rows),
                cpu_valid=sum(x["original_G1_valid"] and x["backend"]=="cpu" for x in rows),
                unscorable=len(rows)-n,
                strict_PASS_of_valid=sum(x["strict_pass"] is True for x in rows if x["original_G1_valid"]),
                not_independent_seed_or_new_policy_training=True,
                suspected_cuda_weighted_com_operation_not_proven_single_cause=True,
                fail_closed_gpu_wrist_hip_actual_manifold_unproven=True,
                measured_200hz_state_clock_offset_not_properly_calibrated=True,
                parity_all_25_original_physics_fields=parity,runs=rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.root.resolve())
    with a.output.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n")
    print("SUMMARY",json.dumps({k:v for k,v in result.items() if k not in ("runs","parity_all_25_original_physics_fields")},ensure_ascii=False))
    for r in result["runs"]:print(r["id"],r["status"],r["strict_pass"],r.get("last_operation_if_no_G1"))
    print("25_FIELD_PARITY",len(result["parity_all_25_original_physics_fields"]), "ALL_ZERO")
if __name__=="__main__":main()
