#!/usr/bin/env python3
"""v37 fixed paired contact-force diagnostic, WITHOUT a control parameter fit.

Requires three individually verified exact-replay audits and a previously
verified v33 signed ball-force audit. Same fixed 1.45..1.72s 55-bin window.
Descriptive normal/tangent decomposition only, not friction-off simulation.
"""
from pathlib import Path
import hashlib,json,argparse,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_ball_impulse import heading,heading_change,sha

def vec(v):return np.asarray(v,dtype=float)
def vfoot(p):
    m=p["mechanism"]
    j=m["full_window_right_foot_impulses_world_Ns"]
    normal=vec(j["normal"]); tangent=vec(j["tangent"])
    return dict(normal=normal,tangent=tangent,total=normal+tangent)
def old_v33():
    path=ROOT/"reports/v033_ball_impulse_signed_audit.json"
    audit=json.loads(path.read_text())
    if not audit["diagnostic_verified"] or not audit["checks"]["exact_original_replay"]["passed"]:
        raise ValueError("Original v33 signed-raw audit never verified")
    x=audit["windows"]["full_declared_window"]
    if x["passed"] is not True or len(x["bins"])!=55:
        raise ValueError("Old v33 momentum window incomplete")
    m=x["aggregate"]
    v=m["pair_impulses_Ns"]["right_foot"]
    return dict(normal=vec(v["normal"]),tangent=vec(v["tangent"]),
         total=vec(v["normal"])+vec(v["tangent"])),audit
def record(d):
    p=d["acquisition"]; f=vfoot(p);m=p["mechanism"]
    target=m["world_heading_deg"]["goal_center"]
    total_heading=heading(f["total"])
    return dict(normal_w_Ns=f["normal"].round(9).tolist(),
        tangent_w_Ns=f["tangent"].round(9).tolist(),
        sum_w_Ns=f["total"].round(9).tolist(),
        normal_heading_deg=heading(f["normal"]),
        combined_heading_deg=total_heading,
        normal_to_combined_deg=heading_change(f["normal"],f["total"]),
        target_heading_deg=target,
        combined_minus_target_deg=heading_change(vec([math.sin(math.radians(target)), -math.cos(math.radians(target)),0.]),f["total"]),
        actual_ball_heading_deg=heading(vec(m["full_window_velocity_after_world_mps"])),
        max_5ms_balance_residual_Ns=m["total_window_balance"]["worst_5ms_residual_Ns"],
        exact_original_replay=p["read_only_exact_replay"],
        legal_outcome=p["original_outcome"],
        first_contact_friction_anchor_count=m["first_contact_friction_anchor_count"],
        initial_friction_anchor_velocity_directions=[
          {"foot_heading_deg":x["heading_foot_deg"],"relative_heading_deg":x["heading_relative_deg"]}
          for x in m["first_contact_raw_friction_anchors"]])

def paired(base,candidate):
    f0=vfoot(base["acquisition"]);f1=vfoot(candidate["acquisition"])
    target=base["acquisition"]["mechanism"]["world_heading_deg"]["goal_center"]
    h0=heading(f0["total"]);h1=heading(f1["total"])
    return dict(
        target_deg=target,
        initial_normal_heading_deg=heading(f0["normal"]),
        new_normal_heading_deg=heading(f1["normal"]),
        normal_heading_change_deg=heading_change(f0["normal"],f1["normal"]),
        initial_total_heading_deg=h0,new_total_heading_deg=h1,
        total_heading_change_deg=heading_change(f0["total"],f1["total"]),
        initial_tangent_heading_effect_deg=heading_change(f0["normal"],f0["total"]),
        new_tangent_heading_effect_deg=heading_change(f1["normal"],f1["total"]),
        normal_impulse_vector_delta_Ns=(f1["normal"]-f0["normal"]).round(9).tolist(),
        tangent_impulse_vector_delta_Ns=(f1["tangent"]-f0["tangent"]).round(9).tolist(),
        total_impulse_vector_delta_Ns=(f1["total"]-f0["total"]).round(9).tolist(),
        change_in_heading_error_deg=(
          ((h1-target+180)%360-180)-((h0-target+180)%360-180)
        ),
        note="Measured paired differences under original physics; prestrike actor changed the physical trajectory, so normal/tangent effects are not individually causal interventions.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((ROOT/"configs/strike_contact_mechanics_v037.json").read_text())
    for name,digest in protocol["source_sha256"].items():
        if sha(ROOT/name)!=digest:raise ValueError("Source drift: "+name)
    files=[ROOT/f for f in (
        "reports/v037_audit_v36_left_near_slow.json",
        "reports/v037_audit_v36_near_slow_17.json",
        "reports/v037_audit_v32_near_slow_17.json")]
    audits=[json.loads(f.read_text()) for f in files]
    if [r["index"] for r in audits]!=[0,1,2]:
        raise ValueError("Required frozen new physical order missing")
    if any(r["protocol_sha256"]!=sha(ROOT/"configs/strike_contact_mechanics_v037.json") for r in audits):
        raise ValueError("Observed study protocol mismatch")
    if any(not r["acquisition"]["read_only_exact_replay"] or
           r["acquisition"]["mechanism"]["total_window_balance"]["worst_5ms_residual_Ns"]>.02
           for r in audits):
        raise ValueError("Invalid physical/sign reconstruction or original parity")
    previous=protocol["previous_v32_left_case_diagnostic"]
    if sha(ROOT/previous["signed_audit"])!=previous["signed_audit_sha256"]:
        raise ValueError("Previously validated v33 signed evidence changed after study freeze")
    older,audit33=old_v33()
    current_a,current_b,old_b=audits
    target_a=current_a["acquisition"]["mechanism"]["world_heading_deg"]["goal_center"]
    old_a=dict(acquisition=dict(mechanism=dict(full_window_right_foot_impulses_world_Ns={
            "normal":older["normal"].tolist(),"tangent":older["tangent"].tolist()},
            world_heading_deg=dict(goal_center=target_a))))
    out_data=dict(
        version="0.1.37",protocol_sha256=sha(ROOT/"configs/strike_contact_mechanics_v037.json"),
        analyzed_reports={str(f.relative_to(ROOT)):sha(f) for f in files},
        old_signed_audit_sha256=sha(ROOT/"reports/v033_ball_impulse_signed_audit.json"),
        all_readonly_parity=True,all_momentum_balances_verified=True,
        recorded_cases=[
          dict(case="left_near_slow",controller="v32",source="v33 previously signed-audited original full-window raw",
               aggregate_Ns={k:v.round(9).tolist() for k,v in older.items()},
               tangent_direction_effect_deg=heading_change(older["normal"],older["total"]),
               total_heading_deg=heading(older["total"])),
          dict(case="left_near_slow",controller="v36",**record(current_a)),
          dict(case="near_slow_17",controller="v32",**record(old_b)),
          dict(case="near_slow_17",controller="v36",**record(current_b))],
        matched_pairs=dict(left_near_slow=paired(old_a,current_a),
                           near_slow_17=paired(old_b,current_b)),
        original_full_ball_goal_results=dict(
            v36_A=True,v36_B=False,
            interpretation="Only original full sphere scorer, not heading-only success"),
        notes=[
            "All cases were already observed development/training conditions, not unseen test samples",
            "The 1.45-1.72s bin window and certified sign were pinned before acquisition",
            "Normal/tangent are measured impulses; the separate terms are not friction-free causal rollouts",
            "Raw friction anchors and nominal precontact point kinematics do not certify exact substep work",
            "No new controller, human coordination learner, power calibration, or fresh-condition pass is claimed"
        ])
    out.write_text(json.dumps(out_data,ensure_ascii=False,indent=2)+"\n")
    for k,v in out_data["matched_pairs"].items():
        print("V037_PAIR",k,json.dumps({x:v[x] for x in
            ("target_deg","initial_total_heading_deg","new_total_heading_deg",
             "total_heading_change_deg","initial_tangent_heading_effect_deg",
             "new_tangent_heading_effect_deg",
             "normal_impulse_vector_delta_Ns","tangent_impulse_vector_delta_Ns")},ensure_ascii=False),flush=True)
    print("V037_FINAL_PAIRED_ANALYSIS",out,flush=True)

if __name__=="__main__":main()
