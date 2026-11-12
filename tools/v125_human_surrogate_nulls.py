#!/usr/bin/env python3
"""Preregistered 120Hz C3D counterfactual lag audit. CPU; NEVER a soccer motor policy."""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
import ezc3d
import v124_human_relative_coordination as prev

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/v125_frozen_surrogates_and_recovery.json"
FROZEN_COMMIT="65c69521d6a3767c80fa714ed727d0c6ffae93e8"

def frozen_config():
    cfg=json.loads(CFG_PATH.read_text())
    if cfg["version"]!="v0.1.125" or cfg["parent_sha"]!="04fa7d89d5bb988dfecea0d20f899dd96652243d":
        raise ValueError("UNFROZEN_V125_CONFIG")
    frozen=subprocess.check_output(["git","-C",str(ROOT),"show",FROZEN_COMMIT+":configs/v125_frozen_surrogates_and_recovery.json"])
    if CFG_PATH.read_bytes()!=frozen:raise ValueError("PREREGISTRATION_MODIFIED")
    if cfg["no_motor_command"] is not True or cfg["no_new_real_physics_if_gpu_busy"] is not True:
        raise ValueError("MOTOR_OR_PHYSICS_FORBIDDEN")
    return cfg

def shift_candidates(n,rate,separation_s,max_count):
    if n<2 or rate<=0 or not np.isfinite(rate):raise ValueError("BAD_SERIES")
    dist=int(math.ceil(rate*separation_s))
    start,end=dist,n-dist
    if end<start:raise ValueError("NO_SEPARATED_SURROGATES")
    # Deterministic, distinct, fixed budget; no random seed or post-result selection.
    if end-start+1<=max_count:return list(range(start,end+1))
    return sorted({int(x) for x in np.linspace(start,end,max_count)})

def paired_split_lag(a,b,cfg):
    if len(a)!=len(b):raise ValueError("BAD_PAIR_SHAPE")
    cut=len(a)//2
    halves=[]
    for x,y in ((a[:cut],b[:cut]),(a[cut:],b[cut:])):
        halves.append(prev.lag_scan(x,y,cfg["kinematic_rate_hz"],cfg["max_lag_s"],cfg["min_overlap_intervals"]))
    best=[x["best"] for x in halves]
    ok=all(x["qualified"] for x in halves)
    sign_consistent=bool(ok and np.sign(best[0]["lag_frames"])==np.sign(best[1]["lag_frames"]))
    stable=bool(sign_consistent and abs(best[0]["lag_seconds"]-best[1]["lag_seconds"])<=cfg["lag_stability_max_disagreement_s"]+1e-9)
    return dict(halves=best, both_qualified=bool(ok), same_sign=sign_consistent,
                stable_under_contiguous_split=stable)

def surrogate_audit(a,b,cfg):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError("BAD_PAIR_SHAPE")
    obs=prev.lag_scan(a,b,cfg["kinematic_rate_hz"],cfg["max_lag_s"],cfg["min_overlap_intervals"])
    if not obs["qualified"]:return dict(observed=obs, qualified=False,reason="UNIDENTIFIABLE_OR_CONSTANT")
    best=obs["best"]
    edge=abs(best["lag_frames"])==int(round(cfg["max_lag_s"]*cfg["kinematic_rate_hz"]))
    shifts=shift_candidates(len(a),cfg["kinematic_rate_hz"],cfg["min_shift_separation_s"],
                            cfg["max_circular_shift_surrogates"])
    magnitudes=[]
    for s in shifts:
        result=prev.lag_scan(a,np.roll(b,s),cfg["kinematic_rate_hz"],cfg["max_lag_s"],cfg["min_overlap_intervals"])
        if result["qualified"]:magnitudes.append(abs(result["best"]["correlation"]))
    if not magnitudes:raise ValueError("ALL_SURROGATES_INVALID")
    v=np.asarray(magnitudes,float)
    observed_abs=abs(best["correlation"])
    exceed=int(np.count_nonzero(v>=observed_abs-1e-12))
    # This is a descriptive empirical rank, not an inferential causal p-value.
    rank=(1+exceed)/(len(v)+1)
    halves=paired_split_lag(a,b,cfg)
    desc=bool(not edge and halves["stable_under_contiguous_split"] and
              rank<cfg["descriptive_surrogate_exceedance_threshold"])
    return dict(observed=obs, observed_lag_on_boundary=edge,
        surrogate_design="CIRCULAR_SHIFT_SECOND_SERIES_EXCLUDE_HALF_SECOND",
        surrogates_attempted=len(shifts),surrogates_qualified=len(v),
        surrogate_peak_abs_corr_median=round(float(np.median(v)),6),
        surrogate_peak_abs_corr_p95=round(float(np.quantile(v,.95)),6),
        surrogate_peak_ge_observed_count=exceed,
        descriptive_rank_exceedance=round(float(rank),6),
        temporal_split=halves, descriptive_lag_robustness_pass=desc,
        source_kind="HUMAN_RELATIVE_RADIAL_KINEMATICS_ONLY",
        causal_lag_or_force_transfer_certified=False,
        elite_soccer_or_foot_ball_contact_certified=False)

def run(cfg,source_root=None):
    src=Path(source_root or prev.json.loads(prev.CONFIG.read_text())["human_source_worktree"])
    head=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
    if head!=cfg["human_source_sha"]:raise ValueError("HUMAN_HEAD_DRIFT")
    qual=src/"reports/v116_human_qualification.audit.json"
    if prev.sha(qual)!=cfg["human_qualification_sha256"]:raise ValueError("HUMAN_MANIFEST_DRIFT")
    manifest=json.loads(qual.read_text())
    rows={r["trial"]:r for r in manifest["trials"]}
    old_cfg=json.loads(prev.CONFIG.read_text())
    if old_cfg["human_trials"]!=cfg["human_trials"]:raise ValueError("TRIAL_COHORT_DRIFT")
    pairs=tuple(cfg["relative_pairs"])
    trial_rows=[]
    for trial in cfg["human_trials"]:
        subject,stem=trial.split("/")
        if stem not in rows or rows[stem]["subject"]!=subject:raise ValueError("BAD_SUBJECT_PROVENANCE")
        r=rows[stem]
        fp=src/old_cfg["human_source_rel"]/subject/(stem+".c3d")
        if prev.sha(fp)!=r["sha256"]:raise ValueError("RAW_C3D_SHA_DRIFT")
        if r["analogs"]!=0:raise ValueError("UNEXPECTED_FORCE_CHANNEL")
        mark,rate,valid=prev._vectors_from_c3d(ezc3d.c3d(str(fp)),old_cfg)
        if rate!=cfg["kinematic_rate_hz"] or len(next(iter(mark.values())))!=r["frames"]:
            raise ValueError("HUMAN_C3D_TIMING_DRIFT")
        _,signals=prev.intrinsic_features(mark,old_cfg)
        measures={}
        for key in pairs:
            x,y=key.split("__")
            if x not in signals or y not in signals:raise ValueError("BAD_FROZEN_RELATIVE_PAIR")
            measures[key]=surrogate_audit(signals[x],signals[y],cfg)
        trial_rows.append(dict(trial=stem,subject=subject,frames=r["frames"],
            source_sha256=r["sha256"],valid_frame_fraction=round(valid,6),pairs=measures))
    all_meas=[row["pairs"][k] for row in trial_rows for k in pairs]
    by_subject={s:[x["trial"] for x in trial_rows if x["subject"]==s]
                for s in sorted({x["subject"] for x in trial_rows})}
    return dict(version=cfg["version"], frozen_protocol_sha=FROZEN_COMMIT,
        original_source_head=head,original_qualification_sha=prev.sha(qual),
        source_kind="REAL_HUMAN_MARKER_KINEMATICS_ONLY",raw_trials=len(trial_rows),
        independent_subjects=len(by_subject),by_subject=by_subject,
        subject_disjoint_train_val_test_possible=False,
        trial_pair_tests=len(all_meas),
        edge_peak_count=sum(bool(x.get("observed_lag_on_boundary")) for x in all_meas),
        descriptive_lag_robustness_count=sum(bool(x.get("descriptive_lag_robustness_pass")) for x in all_meas),
        not_independent_subject_replicates=True, all_tests_exploratory_descriptive=True,
        true_impact_labels=0,certified_elite_player_trials=0,
        synchronized_grf_cop_trials=0,
        new_real_G1_physx_episodes=0,physical_motor_command_issued=False,
        trials=trial_rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    cfg=frozen_config()
    result=run(cfg)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print("V125_HUMAN_SUMMARY",json.dumps({k:v for k,v in result.items() if k!="trials"}),flush=True)
    for r in result["trials"]:
        print("V125_REAL_HUMAN",r["trial"],[(k,x["observed"]["best"],
            x.get("descriptive_rank_exceedance"),x.get("descriptive_lag_robustness_pass"))
            for k,x in r["pairs"].items()],flush=True)

if __name__=="__main__":main()
