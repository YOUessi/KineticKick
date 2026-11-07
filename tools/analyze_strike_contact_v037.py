#!/usr/bin/env python3
"""Strict paired signed ball impulse and foot-contact velocity comparison.

Only reads frozen original PhysX JSON + new read-only acquisition. Requires
byte-exact original 2800-sample replay and fixed independent v029 friction sign.
No fitting or inferred causal counterfactual. Fresh physical cases excluded.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_ball_impulse import (sha,compare_original,observer_parameters,collect_bins,balance_bins,
                                _decode_signed_pair,heading,heading_change,WINDOW)
from strike_baseline_metrics import validated_episode,summarize_strike

def vec(x):
    a=np.asarray(x,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError("Finite xyz required")
    return a

def impulse_v(x):
    return {k:np.asarray(v,dtype=float) for k,v in x.items()}

def aggregate_mechanism(normalized,mass,gravity,report):
    complete=balance_bins(normalized,mass,gravity)
    if not complete["passed"]:raise ValueError("Measured ball momentum closure failed")
    active=[i for i,row in enumerate(normalized)
        if any(np.linalg.norm(vec(row["pairs"]["right_foot"][kind]))>.001 for kind in ("normal","tangent"))]
    if not active:raise ValueError("No right foot-ball contact in declared window")
    impact=normalized[active[0]:min(active[-1]+2,len(normalized))]
    short=balance_bins(impact,mass,gravity)
    if not short["passed"]:raise ValueError("Impact window momentum balance failed")
    row=normalized[active[0]]
    matched=next(t for t in report["traces"][0] if abs(t["time_s"]-row["step_end_s"])<1e-8)
    detail=matched["ball_contact_detail"]
    side=detail["pairs"]["right_foot"]
    if side.get("force_on_ball_valid") is False:
        side=_decode_signed_pair(side,1)
    frictions=side["friction_anchors"]
    foot=detail["feet_before"]["right"]
    origin=vec(foot["position_w_m"])
    linear=vec(foot["link_origin_linear_velocity_w_mps"])
    angular=vec(foot["angular_velocity_w_rad_s"])
    ball=vec(detail["ball_before"]["com_linear_velocity_w_mps"])
    speeds=[]
    for a in frictions:
        point=vec(a["point_w"])
        tip=linear+np.cross(angular,point-origin)
        relative=tip-ball
        speeds.append(dict(point_w_m=point.tolist(),foot_velocity_w_mps=tip.tolist(),
            ball_velocity_before_w_mps=ball.tolist(),foot_relative_velocity_w_mps=relative.tolist(),
            heading_foot_deg=heading(tip),heading_relative_deg=heading(relative),
            precontact_foot_angular_speed_rad_s=float(np.linalg.norm(angular))))
    pairs=short["aggregate"]["pair_impulses_Ns"]
    n=vec(pairs["right_foot"]["normal"])
    t=vec(pairs["right_foot"]["tangent"])
    total=n+t
    heading_goal=heading(vec([report["goals"][0][0]-report["points"][0][0],
                               report["goals"][0][1]-report["points"][0][1],0]))
    got=short["aggregate"]
    return {
     "window_end_bin_s":list(WINDOW),"first_active_foot_contact_bin_end_s":normalized[active[0]]["step_end_s"],
     "last_active_foot_contact_bin_end_s":normalized[active[-1]]["step_end_s"],
     "right_foot_active_bin_count":len(active),
     "total_window_balance":dict(passed=complete["passed"],worst_5ms_residual_Ns=complete["maximum_bin_residual_Ns"],
                                 aggregate_residual_Ns=complete["aggregate"]["residual_norm_Ns"]),
     "foot_contact_balance":dict(passed=short["passed"],worst_5ms_residual_Ns=short["maximum_bin_residual_Ns"],
                                 aggregate_residual_Ns=short["aggregate"]["residual_norm_Ns"]),
     "right_foot_impulses_world_Ns":dict(normal=n.tolist(),tangent=t.tolist(),total=total.tolist()),
     "full_window_right_foot_impulses_world_Ns":dict(
         normal=complete["aggregate"]["pair_impulses_Ns"]["right_foot"]["normal"],
         tangent=complete["aggregate"]["pair_impulses_Ns"]["right_foot"]["tangent"]),
     "full_window_velocity_after_world_mps":complete["aggregate"]["velocity_after_w_mps"],
     "world_heading_deg":dict(goal_center=heading_goal,right_foot_normal=heading(n),
           right_foot_normal_plus_tangent=heading(total),
           normal_to_total=heading_change(n,total),
           outgoing_after_contact=heading(vec(got["velocity_after_w_mps"]))),
     "ball_velocities_world_mps":dict(before=got["velocity_before_w_mps"],after=got["velocity_after_w_mps"]),
     "other_pairs_world_Ns":{k:pairs[k] for k in ("left_foot","other_robot","ground")},
     "first_contact_raw_friction_anchors":speeds,
     "first_contact_friction_anchor_count":len(speeds),
     "sensor_stage_warning":"Solver raw anchors and pre-integration foot state; no exact instantaneous local material-work claim."
    }

def controlled_semantic_equivalence(reference,candidate,summary,protocol):
    """Explicit, narrowly bounded exceptions for the archived v32 source wrapper.

    All physical samples, actual actions/feedback and task results MUST be exact.
    The old v32 wrapper predates the read-only observer and v35 inactive
    prestrike dispatcher. We verify both Git versions and certified force
    probe bytes rather than silently ignoring their different SHA fields.
    """
    allowed={"joint_strike","dynamics_telemetry.probe_path","dynamics_telemetry.probe_sha256"}
    if set(summary["changed_fields"])!=allowed or summary["missing_fields"]:
        raise ValueError("Unapproved original field mismatch: "+repr(summary["changed_fields"]))
    if not all(v["equal"] for v in summary["streams"].values()):
        raise ValueError("Physical or contact/kinematic streams changed")
    a,b=reference["joint_strike"],candidate["joint_strike"]
    for key in set(a)|set(b):
        if key!="source_sha256" and a.get(key)!=b.get(key):
            raise ValueError("Original joint actions/debug changed: "+key)
    a_src,b_src=a["source_sha256"],b["source_sha256"]
    if set(a_src)!=set(b_src) or set(k for k in a_src if a_src[k]!=b_src[k])!={"run_placement_strike.py"}:
        raise ValueError("More than archived runner source metadata changed")
    import subprocess
    def historical_sha(path):
        value=subprocess.check_output(["git","show","version/v0.1.32:"+path],cwd=ROOT)
        return hashlib.sha256(value).hexdigest()
    if a_src["run_placement_strike.py"]!=historical_sha("tools/run_placement_strike.py"):
        raise ValueError("Old wrapper does not match frozen v32 Git bytes")
    if b_src["run_placement_strike.py"]!=protocol["source_sha256"]["tools/run_placement_strike.py"]:
        raise ValueError("Current wrapper differs from v37 prespecified bytes")
    old=reference["dynamics_telemetry"]
    new=candidate["dynamics_telemetry"]
    exclude={"probe_path","probe_sha256","runner_source_sha256"}
    for k in old:
        if k not in exclude and old[k]!=new.get(k):
            raise ValueError("Nonexempt original telemetry metadata differs: "+k)
    if old["runner_source_sha256"]!=historical_sha("tools/run_instrumented_strike.py"):
        raise ValueError("Old observer runner is not v32 archived source")
    if new["runner_source_sha256"]!=protocol["source_sha256"]["tools/run_instrumented_strike.py"]:
        raise ValueError("New observer runner not pinned")
    def probe(path):
        p=Path(path)
        if not p.is_absolute():p=ROOT/p
        return p
    old_probe=probe(old["probe_path"])
    new_probe=probe(new["probe_path"])
    if sha(old_probe)!=old["probe_sha256"] or sha(new_probe)!=new["probe_sha256"]:
        raise ValueError("Original or new force probe bytes no longer match recorded SHA")
    a_probe=json.loads(old_probe.read_text())
    b_probe=json.loads(new_probe.read_text())
    if ({k:v for k,v in a_probe.items() if k!="source_sha256"}
       !={k:v for k,v in b_probe.items() if k!="source_sha256"}):
        raise ValueError("Force certificate/observations not equivalent")
    certificate=a_probe["certificate"]
    if certificate!=b_probe["certificate"] or certificate["verified"] is not True or certificate["friction_sign"]!=1:
        raise ValueError("Force-sign certificates not independently equal/valid")
    if {k:v for k,v in a_probe["source_sha256"].items() if k!="probe_contact_force_semantics.py"} != {
       k:v for k,v in b_probe["source_sha256"].items() if k!="probe_contact_force_semantics.py"}:
        raise ValueError("Probe semantics secondary sources changed")
    return [dict(field="joint_strike.source_sha256.run_placement_strike.py",
                 old_sha=a_src["run_placement_strike.py"],new_sha=b_src["run_placement_strike.py"],
                 validation="Both frozen Git source versions verified; all actions, physics and feedback byte-identical"),
            dict(field="dynamics_telemetry.runner_source_sha256",
                 old_sha=old["runner_source_sha256"],new_sha=new["runner_source_sha256"],
                 validation="Both runner source revisions pinned; read-only observer introduced later"),
            dict(field="dynamics_telemetry.force_probe",
                 old_sha=old["probe_sha256"],new_sha=new["probe_sha256"],
                 validation="Full certified independent probe contents exactly equal excluding an archived source file digest; both signs +1")]


def audit_one(row,protocol):
    original=ROOT/row["source_report"]
    observed=ROOT/row["candidate_report"]
    if sha(original)!=row["source_sha256"]:raise ValueError("Read-only reference SHA drift")
    if not observed.is_file():raise FileNotFoundError(observed)
    reference=json.loads(original.read_text());candidate=json.loads(observed.read_text())
    same=compare_original(reference,candidate,ROOT)
    provenance_exceptions=[] if same["passed"] else controlled_semantic_equivalence(reference,candidate,same,protocol)
    seen=candidate["dynamics_telemetry"]["ball_contact_detail"]
    source=protocol["source_sha256"]
    if candidate["dynamics_telemetry"]["ball_contact_detail"]["source_sha256"]!=source["tools/ball_contact_observer.py"]:
        raise ValueError("Read-only acquisition source mismatch")
    if (candidate["dynamics_telemetry"]["runner_source_sha256"]!=source["tools/run_instrumented_strike.py"]
        or seen["indexed_entries_source_sha256"]!=source["tools/ground_contact_observer.py"]
        or seen["probe_sha256"]!=protocol["fixed_ball_contact_probe_sha256"]):
        raise ValueError("Acquisition provenance mismatch")
    assert tuple(protocol["window_bin_end_s"])==WINDOW
    if candidate["dynamics_telemetry"]["registered_case"]!=row["id"]:
        raise ValueError("Acquisition case identity mismatch")
    ctrl=candidate["joint_strike"]["config"]
    target="configs/state_conditioned_strike_v12.json" if row["controller"]=="v36" else "configs/state_conditioned_strike_v9.json"
    if candidate["joint_strike"]["config_sha256"]!=source[target]:
        raise ValueError("Controller config/model differs from frozen protocol")
    obs=observer_parameters(candidate,decode_signed_raw=True)
    if not obs["passed"]:raise ValueError("Invalid physical observer semantics")
    bins,code=collect_bins(candidate,1,decode_signed_raw=True)
    mech=aggregate_mechanism(bins,obs["mass_kg"],obs["effective_gravity_w_mps2"],candidate)
    shot=summarize_strike(candidate)
    if shot["outcome"]!=summarize_strike(reference)["outcome"]:
        raise ValueError("Original acceptance differs from read-only replay")
    return dict(id=row["id"],controller=row["controller"],reference_sha256=row["source_sha256"],
          observed_sha256=sha(observed),
          physical_action_streams_exact=all(x["equal"] for x in same["streams"].values()),
          full_original_metadata_identical=not provenance_exceptions,
          documented_source_exceptions=provenance_exceptions,
          read_only_exact_replay=True,original_outcome=shot["outcome"],
          signed_normal_decoded_count=len(code["offline_signed_raw_decoded_pairs"]),
          force_semantics=dict(mass_kg=obs["mass_kg"],gravity_world=obs["effective_gravity_w_mps2"],linear_damping=obs["damping"]["linear_per_s"]),
          first_contact_s=shot["contact"]["first_force_s"],
          outgoing_xy_mps=shot["contact"]["outgoing_speed_xy_mps"],mechanism=mech)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True,choices=[0,1,2])
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    protocol_path=ROOT/"configs/strike_contact_mechanics_v037.json"
    protocol=json.loads(protocol_path.read_text())
    if protocol["version"]!="0.1.37" or protocol["max_new_complete_physical_replays"]!=3:
        raise ValueError("Wrong fixed study")
    for file,h in protocol["source_sha256"].items():
        if sha(ROOT/file)!=h:raise ValueError("Frozen source drift: "+file)
    if sha(ROOT/"reports/v029_force_probe.json")!=protocol["fixed_ball_contact_probe_sha256"]:
        raise ValueError("Invalid original force-sign probe")
    row=protocol["replay_order"][args.index]
    result=audit_one(row,protocol)
    report=dict(version="0.1.37",schema=1,protocol_sha256=sha(protocol_path),
                index=args.index,acquisition=result)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    m=result["mechanism"]
    print("V037_SIGNED_CONTACT_AUDIT",json.dumps({
        "case":result["id"],"controller":result["controller"],
        "read_only_exact":result["read_only_exact_replay"],
        "old_outcome":result["original_outcome"],"decoded_bins":result["signed_normal_decoded_count"],
        "normal_world_Ns":m["right_foot_impulses_world_Ns"]["normal"],
        "tangent_world_Ns":m["right_foot_impulses_world_Ns"]["tangent"],
        "headings":m["world_heading_deg"],
        "first_anchor_count":m["first_contact_friction_anchor_count"],
        "max_balance_error_Ns":m["total_window_balance"]["worst_5ms_residual_Ns"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
