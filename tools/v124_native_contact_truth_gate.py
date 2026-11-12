#!/usr/bin/env python3
"""Offline PhysX *native* actor/collider contact witness validator.

No contact manifold is invented or reconstructed from net tensor forces.
Validation of synthetic packets demonstrates only the parser, never physics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION="v0.1.124"
SRC_119="b57b855433b3fa6f6f266ee565f7e959b333c288"
SRC_119_MASTER_SHA="691ac065ccb6eaaef648b7c5a08ad44f79cc0bc3b572a8a10df8a96a399cd380"
PAIR=("left_wrist_yaw_link","left_hip_roll_link")
SHA=re.compile(r"^[0-9a-f]{64}$")

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def nonzero_finite_vec(value,*,positive=False):
    if not isinstance(value,(list,tuple)) or len(value)!=3:return False
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool)
               and math.isfinite(x) for x in value):return False
    return not positive or sum(x*x for x in value)>1e-12

def canonical_path(path):
    return isinstance(path,str) and path.startswith("/") and not (
        "//" in path or path.endswith("/") or ".." in path)

def qualified_native_packet(event,expected_pair,collider_owner,source_sha):
    """Inspect a native callback packet copied within its own simulation step.

    Header actor and collider full USD paths must have explicit actor-collider
    ownership from a trustworthy stage traversal. No string-substring matching.
    """
    reasons=[]
    if not isinstance(source_sha,str) or not SHA.fullmatch(source_sha):
        reasons.append("SOURCE_SHA_INVALID")
    if not isinstance(event,dict):
        return dict(native_pair_qualified=False,reasons=["BAD_PACKET"])
    if event.get("packet_origin") not in ("native_callback","native_full_callback",
                                           "native_poll"):
        reasons.append("TENSOR_OR_SYNTHETIC_UNTRUSTED_ORIGIN")
    if not isinstance(event.get("step"),int) or event["step"]<1:
        reasons.append("NO_REAL_STEP")
    if not isinstance(event.get("dt_s"),(int,float)) or not math.isfinite(
        event["dt_s"]) or event["dt_s"]<=0:
        reasons.append("NO_STEP_DURATION")
    else:
        time=event.get("timestamp_s")
        if (not isinstance(time,(int,float)) or not math.isfinite(time) or
                abs(time-event["step"]*event["dt_s"])>1e-6):
            reasons.append("STEP_TIME_INCONSISTENT")
    actors=event.get("actors"); colliders=event.get("colliders")
    if not isinstance(actors,list) or len(actors)!=2 or not all(map(canonical_path,actors)):
        reasons.append("BAD_ACTOR_PATHS")
    if not isinstance(colliders,list) or len(colliders)!=2 or not all(map(canonical_path,colliders)):
        reasons.append("BAD_COLLIDER_PATHS")
    if (isinstance(actors,list) and len(actors)==2 and
        isinstance(colliders,list) and len(colliders)==2 and
        all(map(canonical_path,actors+colliders))):
        if actors[0]==actors[1] or colliders[0]==colliders[1]:
            reasons.append("DUPLICATE_ACTOR_OR_COLLIDER")
        if set(actors)!=set(expected_pair):
            reasons.append("WRONG_BODY_PAIR")
        if any(collider_owner.get(c)!=a for a,c in zip(actors,colliders)):
            reasons.append("UNPROVEN_COLLIDER_TO_ACTOR_OWNERSHIP")
    if event.get("native_header_count",0)<1 or event.get("native_contact_count",0)<1:
        reasons.append("NO_NATIVE_HEADERS_OR_POINTS")
    points=event.get("points")
    if not isinstance(points,list) or len(points)<1:
        reasons.append("NO_NATIVE_MANIFOLD_POINTS")
    else:
        valid=False
        for p in points:
            if not isinstance(p,dict):continue
            if not nonzero_finite_vec(p.get("position")):continue
            if not nonzero_finite_vec(p.get("normal"),positive=True):continue
            if not nonzero_finite_vec(p.get("impulse"),positive=True):continue
            normal_mag=math.sqrt(sum(x*x for x in p["normal"]))
            if not .5<=normal_mag<=1.5:continue
            valid=True
        if not valid:reasons.append("NO_POINT_NORMAL_AND_NONZERO_IMPULSE")
    if isinstance(points,list) and isinstance(event.get("native_contact_count"),int):
        if event["native_contact_count"]!=len(points):
            reasons.append("CONTACT_POINT_COUNT_MISMATCH")
    return dict(packet_schema_valid=len(reasons)==0,native_pair_qualified=False,
                reasons=sorted(set(reasons)),
                conclusion="SCHEMA_VALID_BUT_UNATTESTED" if not reasons else "UNKNOWN_OR_UNQUALIFIED",
                requires_independent_physx_runtime_and_source_attestation=True,
                all_torque_and_joint_safe_from_this_event=False)

def audit_v119(root):
    """SHA-reconcile authentic negative control; do not repurpose it as G1 truth."""
    root=Path(root)
    commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if commit!=SRC_119:raise ValueError("V119_SOURCE_HEAD_DRIFT")
    master=root/"reports/v119_native_backend_master.audit.json"
    if digest(master)!=SRC_119_MASTER_SHA:
        raise ValueError("V119_SOURCE_MASTER_SHA_DRIFT")
    m=json.loads(master.read_text())
    if m["version"]!="v0.1.119" or len(m["results"])!=6:
        raise ValueError("V119_EXPECTED_SIX_SOURCE_RUNS")
    rows=[]
    for case in m["results"]:
        stem="v119_native_"+case["name"]
        raw=root/"reports"/(stem+".json")
        log=root/"reports"/(stem+".log")
        if digest(raw)!=case["raw_sha"] or digest(log)!=case["log_sha"]:
            raise ValueError("V119_RAW_OR_LOG_SHA_DRIFT:"+stem)
        a=json.loads(raw.read_text())
        stage=a.get("stage",{})
        report=a.get("report_mode")=="on"
        # Calibrate only the isolated probe. Stage collision shapes are children,
        # while reporter API applies to /World/ProbeBall and /World/ProbeBox.
        rows.append(dict(name=case["name"],device=a["device"],
                         report_enabled=report,
                         positive_dynamics=bool(a.get("observed_collision_dynamics")),
                         separated_negative=bool(a.get("observed_noncollision_dynamics")),
                         real_200hz_steps=int(a["actual_runtime_frames"]),
                         body_api_applied=bool(stage.get("ball_report") and stage.get("box_report")),
                         rigid_body_api_both=bool(stage.get("ball_rigid") and stage.get("box_rigid")),
                         collider_paths=stage.get("all_colliders"),
                         native_header_count=int(a.get("native_counts",{}).get("callback_headers",0)),
                         native_pair_witness=bool(a.get("native_pair_witness")),
                         native_pair_qualified=False,
                         raw_sha256=case["raw_sha"]))
    # No non-event conclusion: empty reports mean only an unobserved pair.
    return dict(version=VERSION,source_version="v0.1.119",source_commit=commit,
                master_sha256=digest(master),actual_isolated_physx_runs=len(rows),
                actual_physics_steps=sum(r["real_200hz_steps"] for r in rows),
                native_pair_positive_certifications=sum(r["native_pair_qualified"] for r in rows),
                cannot_conclude_separated_or_collision_free_from_empty_header=True,
                original_G1_wrist_hip_pair_unknown=True,
                original_G1_new_physx_episodes=0,
                no_new_native_manifold_capture=True,
                report_generation_cpu_only=True,per_case=rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="/home/you/projects/kinetickick-wt-v119")
    p.add_argument("--output",type=Path,
                   default=ROOT/"reports/v124_native_truth_gate.audit.json")
    args=p.parse_args()
    result=audit_v119(args.source)
    with args.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="per_case"}),flush=True)
    for case in result["per_case"]:
        print("V124_NATIVE",case["name"],case["positive_dynamics"],
              case["body_api_applied"],case["native_header_count"])
if __name__=="__main__":main()
