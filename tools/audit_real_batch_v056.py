#!/usr/bin/env python3
"""Independent v56 strict whole-task PhysX audit; fail-closed, no data rewriting.

Use frozen v53 independent scoring and v55 real 200 Hz dynamics analysis. Report
the effective number of distinct reset conditions, not just advertised seeds.
"""
import json, math
from collections import Counter,defaultdict
from pathlib import Path
from run_frozen_batch_v056 import ROOT,CFG,PROTO,PROGRESS,sha,verify
from audit_cmu_physics_matrix_v053 import assess
from audit_pair_dynamics_v055 import measure_200hz
from audit_named_nonfoot_diagnostics_v054 import COMPARE

OUTPUT=ROOT/"reports/v056_independent_strict_audit.json"
SUMMARY=ROOT/"reports/v056_independent_strict_readable.json"
POLICIES=("frozen_v32","cmu_10_02","cmu_10_05")

def effective_resets(rows,read_report):
    """Seed is a label. Require state-vector differences for distinct physics."""
    groups=defaultdict(list)
    for case in rows:
        if not case["perturb_reset"]:continue
        d=read_report(case)
        initial=d["initial_robot_state"]
        groups[(case["policy"],case["id"].split("_s")[0])].append(
            (case["id"],case["seed"],initial,
             d.get("traces",[])[0][:2]))
    comparisons=[]
    for (policy,scene),a in sorted(groups.items()):
        if len(a)!=2:raise ValueError("Every exploratory policy/scene must have exactly two seeds")
        x,y=sorted(a,key=lambda v:v[1])
        comparisons.append(dict(policy=policy,scene=scene,seeds=[x[1],y[1]],
            physical_initial_state_different=x[2]!=y[2],
            actual_200hz_trace_different=x[3]!=y[3],
            not_independent_if_equal=bool(x[2]==y[2])))
    return comparisons


def measure_if_actual_contact(report):
    # A non-foot termination before a ball strike is a valid negative, and
    # must not be coerced into a fabricated time-of-impact window.
    if report["records"][0].get("contact_time_s") is not None:
        return measure_200hz(report)
    import numpy as np
    trace=report["traces"][0]
    names=report["joint_names"]
    dq=np.asarray([r["joint_vel"] for r in trace],dtype=float)
    times=np.asarray([r["time_s"] for r in trace],dtype=float)
    limits=np.asarray(report["joint_velocity_limits"][0],dtype=float)
    force=np.asarray([[r["support_foot_force_N"][f] for f in ("left","right")]
                      for r in trace],dtype=float)
    if (len(names)!=29 or dq.shape!=(len(trace),29) or limits.shape!=(29,)
        or np.any(limits<=0) or not np.isfinite(dq).all()
        or not np.isfinite(force).all()):
        raise ValueError("Missing original 29DoF, joint limits, or support telemetry")
    step=np.diff(times)
    if not np.all(step>0) or not np.all(np.abs(step[:-1]-.005)<.00001):
        raise ValueError("Original 200Hz physics cadence invalid")
    qr=np.abs(dq)/limits
    return dict(
        preimpact_400ms_complete=False,
        postimpact_1250ms_complete=False,
        early_physical_stop_recovery_window_censored=True,
        support_dynamics={"observed_to_terminal_without_ball_contact":{
            "samples":len(trace),
            "left_ground_normal_N_mean":float(force[:,0].mean()),
            "right_ground_normal_N_mean":float(force[:,1].mean())}},
        all_qdot_over_limit_sample_joint_count=int((qr>1+1e-6).sum()),
        max_actual_qdot_to_original_limit_ratio=float(qr.max()),
        nonfoot_max_original_sensor_N=float(max(x["nonfoot_contact_N"] for x in trace)),
        peak_joint_group_timings={
            "observed_precontact_only":{
                "peak_qdot_joint":names[int(np.unravel_index(np.abs(dq).argmax(),dq.shape)[1])],
                "no_observed_foot_ball_contact":True}},
        missing_ball_contact_truth=True,
        terminal_sample_count=len(trace),
        terminal_s=float(times[-1]),
        interpretation="True no-contact before original non-foot termination; contact relative windows undefined")

def episode_score(case,cfg,study,source):
    d=json.loads(source.read_text())
    spec=dict(ball_offset_xy_m=case["ball_offset_xy_m"],
      velocity_y_delta_mps=case["velocity_y_delta_mps"],
      outputs={case["policy"]:case["output"]})
    strict=assess(spec,case["policy"],study)
    physical=measure_if_actual_contact(d)
    report=d.get("physx_native_report_v056")
    native=None
    if case["native_report"]:
        if not isinstance(report,dict):raise ValueError("Contact report missing")
        native=dict(callback_calls=report["callback_calls"],
                    reported_native_headers=report["native_headers"],
                    report_apis_present_on_original=report["report_api_present_on_original_bodies"],
                    captured_path_events=len(report["saved_interesting_events"]),
                    raw_pair_proven=report["rigid_pair_certified"],
                    wrist_hip_events=report["wrist_hip_direct_path_events"],
                    pelvis_right_hip_events=report["pelvis_right_hip_direct_path_events"],
                    dropped=report["dropped_events"],errors=report["errors"])
    original_parity=None
    if "prior" in case:
        prev=json.loads((ROOT/case["prior"]).read_text())
        original_parity={k:(d.get(k)==prev.get(k)) for k in COMPARE}
    summary=dict(id=case["id"],policy=case["policy"],cohort=case["cohort"],
        original_sha256=sha(source),seed=case["seed"],
        strict_success=strict["success"],failed_gates=[k for k,v in strict["gates"].items() if not v],
        gates=strict["gates"],original_failure_reasons=strict["original_task_failure_reasons"],
        real_physx_frames=strict["physical_samples"],terminal_time_s=strict["terminal_time_s"],
        physics_termination=strict["physical_termination"],
        max_qdot_over_limit_ratio=strict["maximum_qdot_to_original_limit_ratio"],
        contact_time_s=strict["contact_time_s"],
        instep_impulse_fraction=strict["instep_fraction"],
        actual_ball_speed_xy_mps=strict["original_ball_xy_speed_mps"],
        entire_ball_goal=strict["goal"],recovery_standing_s=strict["recovery_standing_s"],
        pre_400ms_complete=physical["preimpact_400ms_complete"],
        post_1250ms_complete=physical["postimpact_1250ms_complete"],
        recovered_window_right_censored=physical["early_physical_stop_recovery_window_censored"],
        support_dynamics=physical["support_dynamics"],
        all_29dof_qdot_observed=True,
        max_original_nonfoot_sensor_N=physical["nonfoot_max_original_sensor_N"],
        measured_qdot_limit_exceedance_joint_samples=physical["all_qdot_over_limit_sample_joint_count"],
        group_peak_timings=physical["peak_joint_group_timings"],
        native_pair_diagnostic=native,
        exact_physx_field_parity_with_prior=original_parity,
        no_professional_human_ground_force_or_collision_truth=True)
    return summary

def run():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("Never overwrite primary physical evaluation")
    cfg=json.loads(CFG.read_text())
    protocol_sha=verify(cfg)
    if json.loads(PROTO.read_text())["config_sha256"]!=protocol_sha:
        raise ValueError("Missing frozen SHA protocol")
    progress=json.loads(PROGRESS.read_text())
    if progress["protocol_sha256"]!=protocol_sha or len(progress["runs"])!=len(cfg["cases"]):
        raise ValueError("Batch incomplete; do not selectively report")
    study=json.loads((ROOT/"configs/cmu_g1_physics_matrix_v053.json").read_text())
    admitted=[];invalid=[]
    for expected,record in zip(cfg["cases"],progress["runs"]):
        if expected["id"]!=record["id"]:raise ValueError("Case order drift")
        source=ROOT/expected["output"]
        if record["status"]!="PHYSICAL_REPORT_VALIDATED":
            invalid.append(dict(id=expected["id"],status=record["status"],
                                exit_code=record["exit_code"],
                                log_sha256=record["log_sha256"]))
            continue
        if sha(source)!=record["out_sha256"]:
            raise ValueError("Physical source SHA changed")
        admitted.append(episode_score(expected,cfg,study,source))
    cases={c["id"]:c for c in cfg["cases"]}
    seed_pairs=effective_resets([cases[x["id"]] for x in admitted],lambda c:
        json.loads((ROOT/c["output"]).read_text())) if len([x for x in admitted if x["cohort"]!="frozen_3x_readonly_parity"])==12 else []
    native=[r for r in admitted if r["native_pair_diagnostic"] is not None]
    parity=[r for r in native if r["exact_physx_field_parity_with_prior"] is not None]
    strict_summary={}
    exploratory=[r for r in admitted if r["cohort"]!="frozen_3x_readonly_parity"]
    for p in POLICIES:
        sub=[r for r in exploratory if r["policy"]==p]
        fails=Counter(reason for r in sub for reason in r["failed_gates"])
        strict_summary[p]=dict(strict_success=sum(r["strict_success"] for r in sub),
            completed_real_PhysX=len(sub),signed_failure_gates=dict(fails),
            nonfoot_early_stops=sum(r["physics_termination"]!=[[]] for r in sub),
            complete_recovery_windows=sum(r["post_1250ms_complete"] for r in sub),
            real_joint_overspeed_cases=sum(r["measured_qdot_limit_exceedance_joint_samples"]>0 for r in sub),
            min_ball_speed_mps=min((r["actual_ball_speed_xy_mps"] or 0 for r in sub),default=None))
    report=dict(version="0.1.56",frozen_protocol_sha256=protocol_sha,
        completed_admissible_original_200Hz_physx=len(admitted),
        attempted_physical_evaluations=len(cfg["cases"]),
        unscorable_negatives_preserved=invalid,
        distinct_exploratory_scenarios=2,seed_pair_comparisons=seed_pairs,
        effective_seed_independence=bool(seed_pairs and all(x["physical_initial_state_different"]
          and x["actual_200hz_trace_different"] for x in seed_pairs)),
        original_three_center_reader_parity_all_fields=bool(len(parity)==3 and
          all(all(x["exact_physx_field_parity_with_prior"].values()) for x in parity)),
        native_wrist_hip_pair_proven=bool(any(r["native_pair_diagnostic"]["raw_pair_proven"] for r in native)),
        no_observed_pair_is_not_negative_pair_proof=True,
        control_fitted_or_tuned_on_exploratory_failures=False,
        learned_professional_human_explosive_instep_skill_proven=False,
        strict_summary=strict_summary,results=admitted)
    with OUTPUT.open("x") as fd:json.dump(report,fd,indent=2,ensure_ascii=False);fd.write("\n")
    readable={key:value for key,value in report.items() if key!="results"}
    with SUMMARY.open("x") as fd:json.dump(readable,fd,indent=2,ensure_ascii=False);fd.write("\n")
    print("V056_REAL_PHYSX_AUDIT",json.dumps(dict(
        n=len(admitted),unscorable=len(invalid),summary=strict_summary,
        different_seeds=report["effective_seed_independence"],
        center_pair_proven=report["native_wrist_hip_pair_proven"],
        parity=report["original_three_center_reader_parity_all_fields"])),flush=True)
    return report

if __name__=="__main__":run()
