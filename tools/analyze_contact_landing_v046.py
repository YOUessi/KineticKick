#!/usr/bin/env python3
"""v46 read-only original 5ms landing/support/29DoF phase dynamics.

Every score uses original G1/PhysX raw qdot and original unchanged joint
velocity limits. This cannot prove a force-generation pathway or a
human-physiological causal mechanism. No simulator API or state modifications.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PHASES=[("precontact",-.25,0.),
        ("very_early_after",0.,.15),
        ("anticipatory_landing",.15,.40),
        ("landing_peak",.40,.65),
        ("recovery_1s",.65,1.00),
        ("recovery_tail",1.00,1.25),
        ("long_after",1.25,3.)]

def sha(f):return hashlib.sha256((ROOT/f).read_bytes()).hexdigest()

def measure(path):
    r=json.loads((ROOT/path).read_text())
    impact=r["records"][0]["contact_time_s"]
    rows=r["traces"][0]
    times=np.asarray([x["time_s"] for x in rows],float)
    qdot=np.asarray([x["joint_vel"] for x in rows],float)
    lim=np.asarray(r["joint_velocity_limits"][0],float)
    if qdot.shape!=(len(rows),29) or lim.shape!=(29,) or not np.isfinite(qdot).all() or np.any(lim<=0):
        raise ValueError("Actual original 29DoF qdot trace invalid")
    fraction=np.abs(qdot)/lim[None]
    vertical=np.asarray([[x["support_foot_force_N"][side] for side in ("left","right")] for x in rows],float)
    root=np.asarray([x["root_xyz"] for x in rows],float)
    data=dict(source=path,sha256=sha(path),physical_steps=len(rows),
        first_valid_foot_ball_contact_s=impact,full_original_g1_qdot_ratio_peak=float(fraction.max()),
        true_g1_qdot_exceeding_actual_limits=int(np.sum(np.abs(qdot)>lim[None]+1e-6)),
        right_instep=bool(r["records"][0]["measured_surface"]=="instep"),
        original_goal=bool(r["records"][0]["goal"]),
        original_standing_s=float(r["records"][0]["standing_seconds"]))
    if impact is None:
        data["event_aligned_phases"]=None
        return data
    relative=times-float(impact)
    data["event_aligned_phases"]={}
    for name,lo,hi in PHASES:
        mask=(relative>=lo-1e-9)&(relative<hi-1e-9)
        if not mask.any():
            data["event_aligned_phases"][name]=None
            continue
        ratio=fraction[mask]
        first=int(np.argmax(ratio.max(axis=1)))
        idmax=int(np.argmax(ratio[first]))
        sample_time=float(times[mask][first])
        data["event_aligned_phases"][name]=dict(
            physical_samples=int(mask.sum()),
            original_max_joint_speed_over_limit=float(ratio.max()),
            joint_at_peak=r["joint_names"][idmax],
            original_physical_time_s=sample_time,
            sample_time_relative_true_impact_s=sample_time-float(impact),
            number_of_qdot_limit_violation_sample_joints=int((np.abs(qdot[mask])>lim[None]+1e-6).sum()),
            left_support_peak_N=float(vertical[mask,0].max()),
            right_support_peak_N=float(vertical[mask,1].max()),
            mean_root_z_m=float(root[mask,2].mean()))
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if (ROOT/a.output).exists():raise FileExistsError(a.output)
    report=json.loads((ROOT/"reports/v046_landing_transfer_final_audit.json").read_text())
    rec=[]
    for row in report["results"]:
        x=dict(id=row["id"])
        x["original_v41"]=measure(row["original_v41"]["source_path"])
        x["landing_v46"]=measure(row["landing_v46"]["source_path"])
        rec.append(x)
    known=[dict(id=c["id"],landing_v46=measure(c["result"]["source_path"])) for c in report["old_diagnostics"]]
    out=dict(version="0.1.46",original_audit_sha256=sha("reports/v046_landing_transfer_final_audit.json"),
         identical_original_speed_limits=True,
         retrospective_original_g1_5ms_sampling=True,
         affected_phases=PHASES,
         physics_5ms_samples_are_not_independent_conditions=True,
         no_unseen_hardware_safety_certificate=True,
         original_risk_1p25s_scope_does_not_cover_full_14s=True,
         paired=rec,known_diagnostics=known)
    (ROOT/a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("V046_LANDING_PHASES_AUDITED",json.dumps([
      dict(case=x["id"],
        old_peak=x["original_v41"]["full_original_g1_qdot_ratio_peak"],
        new_peak=x["landing_v46"]["full_original_g1_qdot_ratio_peak"],
        old_landing=x["original_v41"]["event_aligned_phases"]["landing_peak"]["original_max_joint_speed_over_limit"] if x["original_v41"]["event_aligned_phases"] and x["original_v41"]["event_aligned_phases"]["landing_peak"] else None,
        new_landing=x["landing_v46"]["event_aligned_phases"]["landing_peak"]["original_max_joint_speed_over_limit"] if x["landing_v46"]["event_aligned_phases"] and x["landing_v46"]["event_aligned_phases"]["landing_peak"] else None)
        for x in rec]),flush=True)

if __name__=="__main__":main()
