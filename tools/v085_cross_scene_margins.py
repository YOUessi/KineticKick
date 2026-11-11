#!/usr/bin/env python3
"""Postregistered cross-scene 29DoF G1 margin audit; never pools dependent repeats."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLDER=ROOT.parent/"kinetickick-wt-v084"
OUT=ROOT/"reports/v085_cross_scene_margins.audit.json"
THRESHOLDS=(3.5,5.,10.,15.)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def score_episode(run,batch,root):
    if not run.get("true_original_200Hz_G1_CUDA_PhysX",False):return None
    name=run["name"]
    path=root/"reports"/(batch+"_"+name+".json")
    if sha(path)!=run["raw_sha256"]:raise ValueError("PHYSX_SOURCE_SHA_MISMATCH:"+str(path))
    g=run["gates"]
    speed=g.get("ball_speed_peak_mps_post_5_to_110ms_proxy")
    ratio=g.get("original_29_qdot_max_ratio")
    if speed is not None and not (isinstance(speed,(float,int)) and math.isfinite(speed)):
        raise ValueError("INVALID_BALL_SPEED")
    if ratio is None or not math.isfinite(ratio):raise ValueError("MISSING_ORIGINAL_QDOT")
    return dict(name=name,scene=run["scene"],control=run["control"],
                repeated_readonly_run=bool(name.endswith("_on")),
                raw_sha256=run["raw_sha256"],strict_pass=bool(g["hard_gate_pass"]),
                failed_gates=g["failed_gates"],observed_14s=g["whole_14s_observed"],
                measured_29dof_qdot_ratio=ratio,qdot_margin_ratio=1.-ratio,
                post_contact_5_to_110ms_ball_speed_proxy_mps=speed,
                speed_margin_above_3p5_mps=(speed-3.5 if speed is not None else None),
                descriptive_speed_ladders={str(t):bool(speed is not None and speed>=t) for t in THRESHOLDS},
                instep_fraction=g.get("measured_instep_fraction"),
                nonfoot_max_net_N=g.get("measured_nonfoot_max_net_N"))

def report():
    source_versions=[("v084",OLDER),("v085",ROOT)]
    result={"version":"v0.1.85","type":"POSTREGISTERED_SHA_BOUND_MARGINS",
        "thresholds_mps_descriptive_only":list(THRESHOLDS),
        "strict_hard_gate_mps_unchanged":3.5,
        "not_true_human_power":"Ball speed proxy is not professional instep impulse or realistic naturalness",
        "not_independent_replicates":"OFF/ON and controls on same scene/seed share physical initial conditions",
        "versions":{}}
    for version,root in source_versions:
        audit=root/"reports"/(version+"_g1_physics_final.audit.json")
        b=json.loads(audit.read_text())
        unique={}
        rows=[]
        for row in b["runs"]:
            outcome=score_episode(row,version,root)
            if outcome is None:continue
            rows.append(outcome)
            key=(row["scene"],row["control"])
            if key not in unique or not outcome["repeated_readonly_run"]:
                unique[key]=outcome
        diffs={}
        by_scene={}
        for outcome in unique.values():
            by_scene.setdefault(outcome["scene"],{})[outcome["control"]]=outcome
        for scene,policies in by_scene.items():
            baseline=policies.get("v32")
            if baseline is None:continue
            for control in ("cmu02","cmu05"):
                test=policies.get(control)
                if not test:continue
                a=baseline["post_contact_5_to_110ms_ball_speed_proxy_mps"]
                bb=test["post_contact_5_to_110ms_ball_speed_proxy_mps"]
                diffs[scene+"_"+control]=dict(
                    speed_proxy_delta_mps=(bb-a if a is not None and bb is not None else None),
                    baseline_strict=baseline["strict_pass"],proxy_strict=test["strict_pass"])
        result["versions"][version]=dict(source_manifest_sha256=sha(audit),
            total_physical_executions=len(rows),
            unique_scene_control_pairs=len(unique),
            strict_pass=sum(v["strict_pass"] for v in rows),
            strict_fail=sum(not v["strict_pass"] for v in rows),
            unique_strict_pass=sum(v["strict_pass"] for v in unique.values()),
            descriptive_speed_ladder_unique={str(t):sum(
                v["descriptive_speed_ladders"][str(t)] for v in unique.values()) for t in THRESHOLDS},
            cmu_proxy_delta_vs_original_v32=diffs,
            episodes=rows)
    if OUT.exists():raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V085_CROSS_SCENE",json.dumps({k:{n:v for n,v in value.items() if n!="episodes"}
        for k,value in result["versions"].items()},ensure_ascii=False))
if __name__=="__main__":report()
