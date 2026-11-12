#!/usr/bin/env python3
"""SHA-locked six-C3D measured human *relative radial* coordination audit.

CPU only. The 120Hz marker set has no certified elite identity, true ball contact,
impact impulse, force plate, calibrated world up or human torque labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import ezc3d
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/v124_frozen_cpu.json"
# Independently sealed original v116 qualification manifest; no moving baseline.
HUMAN_QUAL_MANIFEST_SHA="b1c2116948a227ba4341c1d7aa6a93f0c5eebce3288f999a86457d7c6b19a62b"
NAMES=("RANK","RTOE","LANK","LTOE","RSHO","LSHO",
       "RWRA","LWRA","LBWT","RBWT","RTHI","LTHI")
PAIRS=(("LWRA","LSHO"),("RWRA","RSHO"),("RTOE","PELVIS"),
       ("LTOE","PELVIS"))
LABELS=(("LWRA_LSHO","RTOE_PELVIS"),("RWRA_RSHO","LTOE_PELVIS"))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def finite_adjacent(signal, rate=120.):
    """Abs radial range-change rate, with explicit missing adjacent masks."""
    y=np.asarray(signal,dtype=float)
    if y.ndim!=1 or len(y)<2: raise ValueError("BAD_RADIAL_SERIES")
    diff=np.abs(np.diff(y))*rate
    mask=np.isfinite(y[1:]) & np.isfinite(y[:-1])
    return np.where(mask,diff,np.nan)

def pair_corr(a,b,lag,min_overlap,eps=1e-8):
    """corr(a[t],b[t+lag]); positive lag denotes b following a."""
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.ndim!=1 or a.shape!=b.shape:raise ValueError("BAD_PAIR_SHAPE")
    if abs(lag)>=len(a):return None
    aa=a[max(0,-lag):min(len(a),len(a)-lag)]
    bb=b[max(0,lag):min(len(b),len(b)+lag)]
    mask=np.isfinite(aa)&np.isfinite(bb)
    if int(mask.sum())<min_overlap:return None
    x=aa[mask]; y=bb[mask]
    sx=np.std(x); sy=np.std(y)
    if sx<=eps or sy<=eps:return None
    return float(np.clip(np.mean(((x-x.mean())/sx)*((y-y.mean())/sy)),-1.,1.))

def lag_scan(a,b,rate,max_lag_s,min_overlap,eps=1e-8):
    k=int(round(rate*max_lag_s))
    rows=[(lag,pair_corr(a,b,lag,min_overlap,eps)) for lag in range(-k,k+1)]
    valid=[(lag,r) for lag,r in rows if r is not None]
    zero=dict(lag_frames=0,pearson=next((r for t,r in rows if t==0),None))
    if not valid:return dict(best=None,zero_lag=zero,qualified=False,
                             overlap_min_intervals=min_overlap)
    # Prespecified deterministic ties: smaller abs(lag) then smaller signed lag.
    lag,corr=sorted(valid,key=lambda t:(-abs(t[1]),abs(t[0]),t[0]))[0]
    return dict(best=dict(lag_frames=int(lag),lag_seconds=round(lag/rate,6),
                          correlation=round(corr,6)),
                zero_lag=dict(lag_frames=0,pearson=(
                    round(zero["pearson"],6) if zero["pearson"] is not None else None)),
                qualified=True,overlap_min_intervals=min_overlap)

def _vectors_from_c3d(c3d,cfg):
    labels=c3d["parameters"]["POINT"]["LABELS"]["value"]
    if list(c3d["parameters"]["POINT"]["UNITS"]["value"])!=[cfg["marker_unit"]]:
        raise ValueError("UNQUALIFIED_C3D_UNITS")
    rate=float(c3d["header"]["points"]["frame_rate"])
    if abs(rate-cfg["human_marker_rate_hz"])>1e-9:raise ValueError("UNQUALIFIED_C3D_RATE")
    points=np.asarray(c3d["data"]["points"],float)
    if points.ndim!=3 or points.shape[0]!=4:raise ValueError("BAD_C3D_XYZ_RESIDUAL")
    if points.shape[2]<cfg["min_required_frames"]:raise ValueError("TOO_SHORT_C3D")
    out={}
    for name in NAMES:
        ix=[i for i,v in enumerate(labels) if v.split(":")[-1]==name]
        if len(ix)!=1:raise ValueError("MISSING_OR_AMBIGUOUS_MARKER:"+name)
        xyz=points[:3,ix[0],:].T / 1000.
        # ezc3d residual=-1 marks missing; NaN and Inf separately rejected.
        residual=points[3,ix[0],:]
        invalid=(~np.isfinite(xyz).all(axis=1) | ~np.isfinite(residual) | (residual<0))
        xyz[invalid,:]=np.nan
        out[name]=xyz
    fraction=float(np.mean(np.stack([np.isfinite(v).all(axis=1)
                                      for v in out.values()]).all(axis=0)))
    if fraction<cfg["min_valid_frame_fraction"]:
        raise ValueError("INSUFFICIENT_VALID_REAL_MARKERS")
    return out,rate,fraction

def safe_magnitude(x):
    # norm returns NaN for any missing element, preserving full 3D masking.
    return np.linalg.norm(x,axis=1)

def finite_stats(y):
    a=np.asarray(y,float); good=a[np.isfinite(a)]
    if not len(good):return dict(valid=0,median=None,p95=None,max=None)
    return dict(valid=int(len(good)),median=round(float(np.median(good)),6),
                p95=round(float(np.quantile(good,.95)),6),
                max=round(float(np.max(good)),6))

def intrinsic_features(markers,cfg):
    """Invariant to a COMMON rigid rotation and translation per frame.

    Scalar distances / relative cosines only; distances are not normalized
    across subjects, and radial rates exclude tangential limb motion.
    """
    m={k:np.asarray(v,float) for k,v in markers.items()}
    if set(NAMES)-set(m):raise ValueError("MISSING_REQUIRED_MARKER")
    shapes={a.shape for a in m.values()}
    if len(shapes)!=1 or next(iter(shapes))[1]!=3:raise ValueError("BAD_MARKER_SHAPE")
    pelvis=(m["RBWT"]+m["LBWT"])*.5
    transverse_pelvis=m["RBWT"]-m["LBWT"]
    transverse_shoulder=m["RSHO"]-m["LSHO"]
    shoulder_width=safe_magnitude(transverse_shoulder)
    pelvis_width=safe_magnitude(transverse_pelvis)
    valid=(np.isfinite(shoulder_width)&np.isfinite(pelvis_width)
           &(shoulder_width>cfg["min_body_width_m"])
           &(pelvis_width>cfg["min_body_width_m"]))
    dot=np.einsum("ij,ij->i",transverse_shoulder,transverse_pelvis)
    cos=np.clip(dot/np.where(valid,shoulder_width*pelvis_width,np.nan),-1,1)
    bases={**m,"PELVIS":pelvis}
    radial={f"{a}_{b}":safe_magnitude(bases[a]-bases[b]) for a,b in PAIRS}
    for key in radial:radial[key][~valid]=np.nan
    for v in (shoulder_width,pelvis_width,cos):v[~valid]=np.nan
    features={"shoulder_width_m":shoulder_width,"pelvis_width_m":pelvis_width,
              "shoulder_pelvis_transverse_cos":cos,**radial}
    # Body-scale-normalized radial *distance*; calibration-free angular gait
    # speed or force cannot be inferred from this.
    features.update({k+"_over_shoulder_width":v/shoulder_width
                     for k,v in radial.items()})
    rates={name:finite_adjacent(value,cfg["human_marker_rate_hz"])
           for name,value in radial.items()}
    return features,rates

def analyze_trial(c3d,cfg):
    vectors,rate,valid=_vectors_from_c3d(c3d,cfg)
    features,rates=intrinsic_features(vectors,cfg)
    lag={}
    for wrist,toe in LABELS:
        a,b=rates[wrist],rates[toe]
        original=lag_scan(a,b,rate,cfg["max_lag_s"],
                          cfg["min_overlap_intervals"],cfg["zero_variance_tol"])
        reversed_both=lag_scan(a[::-1],b[::-1],rate,cfg["max_lag_s"],
                               cfg["min_overlap_intervals"],cfg["zero_variance_tol"])
        lag[wrist+"__"+toe]=dict(original=original,
                                both_time_reversed_null=reversed_both)
    return dict(frames=len(next(iter(vectors.values()))),rate_hz=rate,
                all_required_marker_valid_frame_fraction=round(valid,6),
                feature_statistics={k:finite_stats(v) for k,v in features.items()},
                relative_radial_rate_statistics={k:finite_stats(v) for k,v in rates.items()},
                exploratory_opposite_limb_radial_rate_lag=lag,
                human_contact_time_label_available=False,grf_cop_available=False,
                elite_skill_identity_verified=False,
                laboratory_vertical_or_ball_heading_calibrated=False,
                full_joint_angular_velocity_or_kinetic_chain_inferred=False)

def audit(cfg,source_root):
    if cfg["version"]!="v0.1.124" or cfg["parent_sha"]!=(
        "b4da3f1a153a487962b68ddb5e8286a46e100538") or not cfg["no_train_or_motor_actuation"]:
        raise ValueError("UNFROZEN_PROTOCOL")
    source=Path(source_root)
    head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if head!=cfg["human_source_commit"]:raise ValueError("SOURCE_GIT_SHA_DRIFT")
    qa_path=source/"reports/v116_human_qualification.audit.json"
    if sha(qa_path)!=HUMAN_QUAL_MANIFEST_SHA:
        raise ValueError("HUMAN_QUALIFICATION_MANIFEST_SHA_DRIFT")
    manifest=json.loads(qa_path.read_text())
    if manifest["version"]!="v0.1.116" or manifest["unique_trials"]!=6 or manifest["unique_subjects"]!=2:
        raise ValueError("HUMAN_SOURCE_QUALIFICATION_DRIFT")
    expected={r["trial"]:r for r in manifest["trials"]}
    trials=[]
    for name in cfg["human_trials"]:
        subject,stem=name.split("/")
        row=expected.get(stem)
        if row is None or row["subject"]!=subject:raise ValueError("MISSING_FROZEN_HUMAN_TRIAL")
        file=source/cfg["human_source_rel"]/subject/(stem+".c3d")
        if sha(file)!=row["sha256"]:raise ValueError("C3D_SHA_DRIFT:"+stem)
        if row["analogs"]!=0:raise ValueError("FORCE_CHANNEL_QUALIFICATION_REQUIRED")
        actual=analyze_trial(ezc3d.c3d(str(file)),cfg)
        if actual["frames"]!=row["frames"]:raise ValueError("C3D_FRAME_DRIFT")
        trials.append(dict(subject=subject,trial=stem,source_sha256=row["sha256"],**actual))
    n_subject=len({t["subject"] for t in trials})
    need=cfg["required_subject_partition_count"]
    return dict(version=cfg["version"],protocol_sha256=sha(CONFIG),source_commit=head,
                source_qualification_sha256=sha(qa_path),
                qualified_real_human_c3d_trials=len(trials),
                independent_subjects=n_subject,marker_frames=sum(x["frames"] for x in trials),
                subject_disjoint_train_validation_holdout_possible=bool(n_subject>=need),
                required_disjoint_partitions=need,
                cannot_claim_independent_train_validation_test=bool(n_subject<need),
                certified_elite_instep_subjects=0,certified_true_ball_impulse_trials=0,
                certified_synchronized_grf_cop_trials=0,
                real_foot_ball_contact_label_trials=0,
                original_G1_physx_episodes_new=0,physical_motor_command_issued=False,
                outputs_are_intrinsic_marker_kinematic_descriptors_only=True,
                per_trial=trials)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"reports/v124_qualified_human_coordination.audit.json")
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    result=audit(cfg,cfg["human_source_worktree"])
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="per_trial"},ensure_ascii=False))
    for x in result["per_trial"]:
        l={k:v["original"]["best"] for k,v in x["exploratory_opposite_limb_radial_rate_lag"].items()}
        print("V124_TRIAL",x["trial"],x["frames"],x["all_required_marker_valid_frame_fraction"],l,flush=True)
if __name__=="__main__":main()
