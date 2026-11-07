#!/usr/bin/env python3
"""v0.1.36: independent fixed-order audit of v35 prestrike checkpoint.

Reads existing PhysX JSON and the SHA-pinned v32 baselines. This is *not*
a training reward, a video score, or a random population success estimate.
The v35 protocol stops before testing center/strong/fresh if either old miss
does not become an originally valid goal.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

import numpy as np

from strike_baseline_metrics import summarize_strike, validated_episode
from audit_p0_dynamic_strike import limit_observations
from strike_cases import load_cases
from free_shooting_metrics import ShotTracker

ROOT=Path(__file__).resolve().parents[1]
ORDER=("heldout_v028_left_near_slow","heldout_v028_near_slow_17")
FILES=("reports/v036_prestrike_left_near_slow.json","reports/v036_prestrike_near_slow_17.json")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def actual_goal_plane(rows,goal_y,contact_s):
    previous=None
    for row in rows:
        if row["time_s"]<=contact_s: continue
        if previous is not None:
            a=np.asarray(previous["ball_xyz"],float)
            b=np.asarray(row["ball_xyz"],float)
            if a[1]>goal_y>=b[1] and a[1]!=b[1]:
                f=(goal_y-a[1])/(b[1]-a[1])
                return {"time_s":round(previous["time_s"]+f*(row["time_s"]-previous["time_s"]),6),
                        "xyz_m":(a+f*(b-a)).round(6).tolist()}
        previous=row
    return None

def audit(name,path,config,study,cases):
    r=json.loads((ROOT/path).read_text())
    info=next(x for x in study["cases"] if x["id"]==name)
    baseline_path=Path(info["baseline"]["path"])
    if not baseline_path.is_absolute():baseline_path=ROOT/baseline_path
    assert digest(baseline_path)==info["baseline"]["sha256"],"Baseline changed"
    b=json.loads(baseline_path.read_text())
    source_config=ROOT/"configs/state_conditioned_strike_v12.json"
    config_sha=digest(source_config)
    shot=r["records"][0]
    details=r["joint_strike"]
    phase=r["joint_strike"]["feedback"]
    expected=cases[name]
    requested=details["ball_offset_xy"]
    expected_speed=expected["root_velocity_y_delta_mps"]
    limits=limit_observations(validated_episode(r),r)
    rows=validated_episode(r)
    old=summarize_strike(b)
    new=summarize_strike(r)
    old_speed=old["contact"]["outgoing_speed_xy_mps"]
    current=new["contact"] or {}
    new_speed=current.get("outgoing_speed_xy_mps")
    ratio=None if not new_speed or not old_speed else float(new_speed/old_speed)
    samples=[]
    first=None
    maximum=0.0
    stepmax=0.0
    previous=np.zeros(29)
    viol=[]
    for i,decision in enumerate(phase):
        delta=np.asarray(decision["prestrike_feedback"]["actual_delta_rad"],float)
        if delta.shape!=(29,) or not np.all(np.isfinite(delta)):
            raise ValueError("Invalid applied joint delta")
        t=float(decision["time_s"])
        if np.any(abs(delta)>1e-8) and first is None:first=t
        maximum=max(maximum,float(np.max(abs(delta))))
        stepmax=max(stepmax,float(np.max(abs(delta-previous))))
        previous=delta
        if t < .6-1e-6 and float(np.max(abs(delta)))>1e-8:
            viol.append("early_actor_action")
        if float(decision["prestrike_feedback"]["context_gate"])==0. and np.any(abs(delta)>1e-8):
            viol.append("nominal_guard_violated")
    if len(phase)!=700:viol.append("policy_step_coverage")
    if maximum>.04+3e-6:viol.append("amplitude")
    if stepmax>.008+3e-6:viol.append("slew")
    if first is not None and first>=shot["contact_time_s"]:
        viol.append("actor_only_after_contact")
    orig=r["initial_robot_state"]
    replay=ShotTracker(initial_root=orig["root_position"][0],
        initial_ball=[*r["points"][0],.113],goal=r["goals"][0],
        forward=r["trials"][0]["goal_forward"],style=r["trials"][0]["style"],
        width=r["goal_width_m"],height=r["goal_height_m"])
    for row in r["traces"][0]:replay.update(row)
    replay_match=replay.summary()==shot
    bs=validated_episode(b)
    prefix_by_name={}
    for k in ("root_xyz","ball_xyz","joint_pos","joint_vel"):
        x=np.asarray([v[k] for v in rows if v["time_s"]<=.6000001],float)
        y=np.asarray([v[k] for v in bs if v["time_s"]<=.6000001],float)
        prefix_by_name[k]=None if x.shape!=y.shape else float(np.max(abs(x-y)))
    goal=np.asarray(r["goals"][0][:2],float)
    ball=np.asarray(r["points"][0][:2],float)
    outgoing=np.asarray(current.get("outgoing_velocity_mps") or [0.,0.,0.],float)[:2]
    target=goal-ball
    angle=None
    if np.linalg.norm(outgoing)>1e-6 and np.linalg.norm(target)>1e-6:
        angle=float(math.degrees(math.atan2(target[0]*outgoing[1]-target[1]*outgoing[0],
                                            target@outgoing)))
    contact_s=shot["contact_time_s"] or 0.
    crossing=actual_goal_plane(rows,float(goal[1]),contact_s)
    full_plane=actual_goal_plane(rows,float(goal[1])-.11,contact_s)
    left_bound=float(goal[0]-r["goal_width_m"]/2+.11)
    right_bound=float(goal[0]+r["goal_width_m"]/2-.11)
    lateral_margin=None if full_plane is None else min(
        float(full_plane["xyz_m"][0]-left_bound),float(right_bound-full_plane["xyz_m"][0]))
    gates={
        "source_contract":r["joint_strike"]["config_sha256"]==config_sha
            and r["joint_strike"]["prestrike_checkpoint_sha256"]==config["prestrike_feedback"]["checkpoint_sha256"]
            and r["joint_strike"]["checkpoint_sha256"]==config["post_shot_recovery"]["checkpoint_sha256"]
            and all(digest(ROOT/"tools"/k)==v for k,v in details["source_sha256"].items()),
        "registered_context":r["dynamics_telemetry"]["registered_case"]==name
            and np.allclose(requested,expected["ball_offset_xy_m"],atol=1e-8)
            and abs(details["velocity_y_delta"]-expected_speed)<1e-8,
        "physical_no_state_writes":r["root_fixed"] is False
            and r["post_initialization_state_writes"] is False
            and r["dynamics_telemetry"]["physical_writes"] is False,
        "full_14_seconds":len(rows)==2800 and abs(rows[-1]["time_s"]-14.)<1e-6,
        "legal_instep":shot["surface_match"] is True
            and shot["measured_surface"]=="instep"
            and shot["requested_surface_impulse_fraction"]>=.95
            and r["sequence_records"][0]["illegal_body_ball_contact"] is False,
        "goal":shot["goal"] is True,
        "running_and_recovery":shot["recovered"] is True and shot["approached"] is True
            and r["running_evidence"]["running_success"] is True
            and r["stride_shot_evidence"]["continuous_running_contact"] is True,
        "no_nonfoot_termination":shot["failed"] is False and r["termination_reasons"]==[[]],
        "real_joint_velocity_limit":limits["actual_speed_exceedance_sample_joint_count"]==0,
        "baseline_speed_retention":ratio is not None and ratio>=study["minimum_speed_ratio"],
        "original_task_outcome":new["outcome"]["success"] is True,
        "record_replay_matches":replay_match,
        "prestrike_residual_contract":len(viol)==0,
        "protected_first_600ms":all(x is not None and x<2e-5 for x in prefix_by_name.values()),
    }
    return dict(case=name,case_registry=expected,source_sha256=digest(ROOT/path),
        original_task=new["outcome"],gates=gates,passed=all(gates.values()),
        out_speed_mps=new_speed,baseline_speed_mps=old_speed,speed_ratio=ratio,
        contact_time_s=shot["contact_time_s"],instep_fraction=shot["requested_surface_impulse_fraction"],
        standing_seconds=shot["standing_seconds"],max_actual_joint_speed_ratio=limits["maximum_actual_speed_ratio"],
        actual_joint_speed_violations=limits["actual_speed_exceedance_sample_joint_count"],
        applied_residual=dict(nonzero=first is not None,first_nonzero_s=first,peak_rad=maximum,
                              maximum_step_rad=stepmax,errors=sorted(set(viol))),
        prefix_differences=prefix_by_name,
        ball_center_at_goal_plane=crossing,
        ball_center_at_full_sphere_goal_crossing=full_plane,
        full_sphere_lateral_limits_m=[left_bound,right_bound],
        full_sphere_lateral_margin_m=lateral_margin,
        initial_ball_xy_m=ball.tolist(),goal_xy_m=goal.tolist(),
        signed_outgoing_heading_error_deg=angle)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(str(out))
    study=json.loads((ROOT/"configs/prestrike_study_v1.json").read_text())
    for file,expected in study["source_sha256"].items():
        assert digest(ROOT/file)==expected,("frozen v35 source drift",file)
    config=json.loads((ROOT/"configs/state_conditioned_strike_v12.json").read_text())
    assert digest(ROOT/config["prestrike_feedback"]["checkpoint"])==config["prestrike_feedback"]["checkpoint_sha256"]
    assert digest(ROOT/config["post_shot_recovery"]["checkpoint"])==config["post_shot_recovery"]["checkpoint_sha256"]
    cases=load_cases(ROOT)
    results=[audit(name,path,config,study,cases) for name,path in zip(ORDER,FILES)]
    all_pass=all(r["passed"] for r in results)
    doc=dict(research_version="v0.1.36",frozen_study_version="v0.1.35",
        source_commit="07ddb6ad6e05624f7e69dcacf69e5c5b93c7e31c",
        candidate_model_sha256=config["prestrike_feedback"]["checkpoint_sha256"],
        original_v32_recovery_sha256=config["post_shot_recovery"]["checkpoint_sha256"],
        audit_sha256=digest(Path(__file__)),evaluated_order=list(ORDER),
        candidate_config_sha256=digest(ROOT/"configs/state_conditioned_strike_v12.json"),
        gates_executed=len(results),both_old_misses_fixed=all_pass,
        stop_protocol_applied=not all_pass,
        new_fresh_conditions_accessed=False,
        center_or_strong_run=False,
        not_deployed=True,
        interpretation="Fixed-condition PhysX tests, not random success rate or human-coordination learning. "
            "Original physical gates remain strict; any failed goal stops further v35 trial.",
        results=results)
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n")
    print("V036_INDEPENDENT_AUDIT",json.dumps({"all_pass":all_pass,"file":str(out),
       "results":[{"id":r["case"],"passed":r["passed"],"gates_failed":[k for k,v in r["gates"].items() if not v],
                   "speed":r["out_speed_mps"],"heading_deg":r["signed_outgoing_heading_error_deg"],
                   "center_plane":r["ball_center_at_goal_plane"],"full_sphere_plane":r["ball_center_at_full_sphere_goal_crossing"],"full_sphere_margin_m":r["full_sphere_lateral_margin_m"],"applied_peak_rad":r["applied_residual"]["peak_rad"]}
                  for r in results]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
