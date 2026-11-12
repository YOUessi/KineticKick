#!/usr/bin/env python3
"""SHA-qualified 29DoF G1 true-physics chronology; *analysis*, not a new motor rollout."""
from __future__ import annotations
import argparse, json, subprocess, math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import v121_29dof_power_chain as prior
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/v125_frozen_surrogates_and_recovery.json"
FROZEN_COMMIT="65c69521d6a3767c80fa714ed727d0c6ffae93e8"

def assert_frozen(cfg):
    x=subprocess.check_output(["git","-C",str(ROOT),"show",FROZEN_COMMIT+":configs/v125_frozen_surrogates_and_recovery.json"])
    if CFG_PATH.read_bytes()!=x or cfg["version"]!="v0.1.125":raise ValueError("FROZEN_PROTOCOL_DRIFT")
    if cfg["no_motor_command"] is not True or cfg["robot_dt_s"]!=0.005:raise ValueError("MOTOR_OR_RATE_REJECTED")

def windows(t,impact,cfg):
    dt=cfg["robot_dt_s"];pre=cfg["precontact_window_s"];post=cfg["postcontact_window_s"]
    if t.ndim!=1 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):raise ValueError("BAD_TIME_GRID")
    ranges={
        "precontact_400ms":(t>=impact-pre-1e-7)&(t<impact-1e-7),
        "follow_through_first1250ms":(t>=impact-1e-7)&(t<impact+post-1e-7),
        "late_recovery":(t>=impact+post-1e-7)&(t<=cfg["recovery_end_s"]+1e-7),
    }
    if int(ranges["precontact_400ms"].sum())!=int(round(pre/dt)):
        raise ValueError("CENSORED_PRECONTACT_WINDOW")
    if int(ranges["follow_through_first1250ms"].sum())!=int(round(post/dt)):
        raise ValueError("CENSORED_POSTCONTACT_WINDOW")
    if sum(int(x.sum()) for x in ranges.values())>len(t):raise ValueError("OVERLAPPING_PHASE_WINDOWS")
    if not ranges["late_recovery"].any():raise ValueError("MISSING_LATE_RECOVERY")
    return ranges

def stage_metrics(p,mask,joint_names):
    t=p["t"];qdot=p["vel"];lims=p["limits"];force=p["support"]
    ix=np.flatnonzero(mask)
    if not len(ix):raise ValueError("EMPTY_OBSERVED_WINDOW")
    ratio=np.abs(qdot[ix])/lims[None,:]
    location=np.unravel_index(int(np.argmax(ratio)),ratio.shape)
    which=int(ix[location[0]]);j=int(location[1])
    impulse=force[ix].sum(axis=0)*.005
    total=float(impulse.sum())
    balanced=float((impulse[0]-impulse[1])/total) if total>1e-6 else None
    com=p["com"][ix]
    return dict(observed_frames=len(ix),from_s=round(float(t[ix[0]]),6),
        to_s=round(float(t[ix[-1]]),6),
        maximum_original29_qdot_ratio=round(float(ratio[location]),6),
        max_ratio_joint=joint_names[j],max_ratio_time_s=round(float(t[which]),6),
        any_original_joint_qdot_over_limit=bool((ratio>1+1e-6).any()),
        left_foot_net_force_proxy_integral_Ns=round(float(impulse[0]),6),
        right_foot_net_force_proxy_integral_Ns=round(float(impulse[1]),6),
        proxy_left_minus_right_over_sum=None if balanced is None else round(balanced,6),
        com_xy_displacement_m=round(float(np.linalg.norm(com[-1,:2]-com[0,:2])),6),
        nonfoot_body_net_force_peak_N=round(float(np.max(p["nonfoot"][ix])),6))

def locate_first_violation(p,names,cfg):
    r=np.abs(p["vel"])/p["limits"][None,:]
    ix=np.argwhere(r>cfg["original_qdot_limit_ratio_max"]+1e-6)
    if not len(ix):return dict(found=False,time_s=None,joint=None,stage=None,max_ratio=None)
    i,j=map(int,ix[0]);t=float(p["t"][i]);contact=p["impact"]
    if t<contact-1e-7:stage="precontact"
    elif t<contact+cfg["postcontact_window_s"]-1e-7:stage="follow_through_first1250ms"
    else:stage="late_recovery"
    return dict(found=True,time_s=round(t,6),joint=names[j],
                stage=stage,max_ratio=round(float(r[i,j]),6))

def independently_score(g,cfg):
    if not isinstance(g,dict):raise ValueError("INVALID_HARD_GATES")
    must=("original_native_wrapper_source_certified","original_goal_entered",
          "recovered","whole_14s_observed","measured_surface","measured_instep_fraction",
          "ball_speed_peak_mps_post_5_to_110ms_proxy","original_29_qdot_max_ratio",
          "measured_nonfoot_max_net_N","original_physical_nonfoot_terminated",
          "hard_gate_pass","failed_gates")
    if any(k not in g for k in must):raise ValueError("MISSING_ORIGINAL_HARD_GATE_FIELD")
    fraction=g["measured_instep_fraction"];speed=g["ball_speed_peak_mps_post_5_to_110ms_proxy"]
    for val in (fraction,speed,g["original_29_qdot_max_ratio"],g["measured_nonfoot_max_net_N"]):
        if not isinstance(val,(int,float)) or isinstance(val,bool) or not math.isfinite(val):
            raise ValueError("UNQUALIFIED_NUMERIC_HARD_GATE_EVIDENCE")
    safe=bool(g["original_native_wrapper_source_certified"]
        and g["original_goal_entered"] and g["recovered"] and g["whole_14s_observed"]
        and g["measured_surface"]=="instep"
        and float(fraction)>=cfg["instep_share_threshold"]
        and float(speed)>=cfg["ball_speed_proxy_min_mps"]
        and float(g["original_29_qdot_max_ratio"])<=cfg["original_qdot_limit_ratio_max"]+1e-6
        and float(g["measured_nonfoot_max_net_N"])<cfg["nonfoot_net_force_N"]
        and g["original_physical_nonfoot_terminated"] is False)
    if safe!=g["hard_gate_pass"]:raise ValueError("DISAGREEMENT_WITH_FROZEN_ORIGINAL_HARD_GATE")
    return safe

def audit_episode(raw,source_row,cfg):
    n=source_row["actual_traced_200hz_rows"]
    p=prior.validated_physics(raw,n)
    if n!=2800 or abs(float(p["t"][-1])-cfg["recovery_end_s"])>1e-5:
        raise ValueError("NOT_FULL_14S_ORIGINAL_ROLLOUT")
    if tuple(raw["joint_names"])!=tuple(G1_JOINT_ORDER):
        raise ValueError("ORIGINAL_G1_29_JOINT_ORDER_DRIFT")
    names=raw["joint_names"]
    slices=windows(p["t"],p["impact"],cfg)
    results={k:stage_metrics(p,v,names) for k,v in slices.items()}
    first=locate_first_violation(p,names,cfg)
    observed_peak=float(np.max(np.abs(p["vel"])/p["limits"][None,:]))
    h=source_row["hard_gates"]
    if abs(observed_peak-float(h["original_29_qdot_max_ratio"]))>1e-6:
        raise ValueError("ORIGINAL_QDOT_GATE_CHANGED")
    if abs(float(np.max(p["nonfoot"]))-float(h["measured_nonfoot_max_net_N"]))>1e-5:
        raise ValueError("NONFOOT_EVIDENCE_CHANGED")
    passed=independently_score(h,cfg)
    if not passed and not h["failed_gates"]:raise ValueError("UNATTRIBUTED_NEGATIVE")
    return dict(impact_s=round(p["impact"],6),original_29_joint_names=names,
        physical_steps=n,strict_hard_gate_pass=passed,
        failed_gates=h["failed_gates"],first_qdot_violation=first,
        within_original_all29_qdot_limit=not first["found"],
        root_initial_state_is_left_censored=True,
        observed_complete_14s=True,
        per_observed_window=results,
        native_collider_pair_certified=False,foot_force_is_net_proxy_not_certified_GRF=True)

def run(cfg,source_prefix="/home/you/projects/kinetickick-wt-v{}"):
    assert_frozen(cfg)
    physical=[];repeats=[]
    independent_by_scene=defaultdict(dict)
    raw_by_case={}
    for record in cfg["robot_sources"]:
        version=record["version"]
        folder=Path(source_prefix.format(version))
        head=subprocess.check_output(["git","-C",str(folder),"rev-parse","HEAD"],text=True).strip()
        if head!=record["commit"]:raise ValueError("ORIGINAL_GIT_SHA_DRIFT")
        master=folder/"reports"/f"v{version}_original_g1_batch.audit.json"
        prior.verify_file(master,record["master_sha256"])
        manifest=json.loads(master.read_text())
        if manifest["version"]!=f"v0.1.{version}":raise ValueError("ORIGINAL_MASTER_VERSION_DRIFT")
        for item in manifest["runs"]:
            if item["certified_original_G1_cuda_200Hz"] is not True:raise ValueError("UNQUALIFIED_PHYSX_RUN")
            fp=folder/"reports"/f"v{version}_{item['name']}.json"
            prior.verify_file(fp,item["raw_sha256"])
            payload=json.loads(fp.read_text())
            if int(payload["seed"])!=int(item["seed"]) or payload["root_fixed"] is not False:
                raise ValueError("SCENE_ORIGINAL_G1_DRIFT")
            result=audit_episode(payload,item,cfg)
            k=(version,item["scene"],item["control"])
            if item["observer"]:
                repeats.append((k,item,payload,result))
            else:
                if item["control"] in independent_by_scene[(version,item["scene"])]:
                    raise ValueError("DUPLICATE_INDEPENDENT_SCENE_POLICY")
                independent_by_scene[(version,item["scene"])][item["control"]]=result
                raw_by_case[k]=payload
                physical.append(dict(version=version,scene=item["scene"],control=item["control"],
                    seed=item["seed"],raw_sha256=item["raw_sha256"],**result))
    parity=[]
    for k,item,payload,result in repeats:
        base=raw_by_case.get(k)
        if base is None:raise ValueError("MISSING_MATCHED_OBSERVER_OFF")
        if base["traces"]!=payload["traces"] or base["records"]!=payload["records"]:
            raise ValueError("READONLY_OBSERVER_CHANGED_REAL_PHYSICS")
        parity.append(dict(version=k[0],scene=k[1],control=k[2],
                           actual_frames=len(payload["traces"][0]),all_shared_trace_and_record_exact=True))
    if len(physical)!=cfg["require_independent_seed_scene_count"] or len(repeats)!=cfg["observer_repeats_excluded"]:
        raise ValueError("PHYSICAL_COHORT_DRIFT")
    pair_summary=[]
    for cell,controls in sorted(independent_by_scene.items()):
        if set(controls)!={"v32","cmu02","cmu05"}:raise ValueError("INCOMPLETE_PAIR_CELL")
        for name in ("cmu02","cmu05"):
            a=controls["v32"];b=controls[name]
            pair_summary.append(dict(version=cell[0],scene=cell[1],compare=name,
                baseline_pass=a["strict_hard_gate_pass"],candidate_pass=b["strict_hard_gate_pass"],
                baseline_max_29qdot=max(x["maximum_original29_qdot_ratio"] for x in a["per_observed_window"].values()),
                candidate_max_29qdot=max(x["maximum_original29_qdot_ratio"] for x in b["per_observed_window"].values())))
    by_stage=Counter(x["first_qdot_violation"]["stage"] for x in physical if x["first_qdot_violation"]["found"])
    reasons=Counter(reason for x in physical for reason in x["failed_gates"])
    return dict(version=cfg["version"],frozen_protocol_sha=FROZEN_COMMIT,
        true_original_G1_physx_episodes_reaudited=len(physical)+len(repeats),
        original_independent_scene_controller_cells=len(physical),
        strict_pass=sum(x["strict_hard_gate_pass"] for x in physical),
        strict_fail=sum(not x["strict_hard_gate_pass"] for x in physical),
        first_actual_qdot_violation_by_phase=dict(by_stage),
        original_failed_gate_counts=dict(reasons),observer_repeats_not_independent=len(repeats),
        observer_off_on_exact_parity=all(x["all_shared_trace_and_record_exact"] for x in parity),
        pairwise_variant_vs_v32=pair_summary, observer_parity=parity,
        new_G1_physx_episodes=0,real_human_motor_policy_used=False,
        human_force_or_original_native_contact_pair_certified=False,
        physical_motor_command_issued=False, per_episode=physical)

def main():
    a=argparse.ArgumentParser();a.add_argument("--output",type=Path,required=True);args=a.parse_args()
    cfg=json.loads(CFG_PATH.read_text())
    result=run(cfg)
    with args.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print("V125_G1_TIMELINE",json.dumps({k:v for k,v in result.items()
               if k not in ("per_episode","pairwise_variant_vs_v32","observer_parity")}),flush=True)
    for r in result["per_episode"]:
        print("V125_REAL_PHYSICS",r["scene"],r["control"],r["strict_hard_gate_pass"],
              r["failed_gates"],r["first_qdot_violation"],flush=True)

if __name__=="__main__":main()
