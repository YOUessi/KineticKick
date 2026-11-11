#!/usr/bin/env python3
"""Independent v72 six-C3D optical data qualification, never force truth.

Original CMU 120 Hz point marker WEP(s) and foot markers only. Distances are
geometric candidates, not genuine foot-ball contact, instep plane, or GRFs.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C3D_ROOT=Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050/raw")
OUT=ROOT/"reports/v072_six_cmu_optical_suitability.json"
FOOT=("RTOE","RMT5","RHEE","LTOE","LMT5","LHEE")
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def geom_candidates(points, labels, rate, *, min_separation_frames=30):
    ix={n.rsplit(":",1)[-1]:i for i,n in enumerate(labels)}
    foot_idx={s:ix[s] for s in FOOT if s in ix}
    ball_idx=[i for i,n in enumerate(labels) if n.lower().startswith("soccerball:")]
    if len(foot_idx)<6 or len(ball_idx)!=3:raise ValueError("Not a validated soccerball+six-foot-marker optical stream")
    if points.ndim!=3 or points.shape[0]!=4:raise ValueError("Expected 4D C3D marker array")
    n=points.shape[2]
    xyz=np.asarray(points[:3],dtype=float).transpose(2,1,0)/1000
    residual=np.asarray(points[3],dtype=float).T
    good=(np.isfinite(xyz).all(axis=2)&np.isfinite(residual)&(residual>=0)&(np.abs(xyz).sum(axis=2)>1e-7))
    d=np.full(n,np.nan)
    which=[None]*n
    for f in range(n):
        if good[f,ball_idx].sum()<2:continue
        balls=xyz[f,np.array(ball_idx)[good[f,ball_idx]]]
        nearest=[]
        for label,i in foot_idx.items():
            if good[f,i]:nearest.append((np.linalg.norm(balls-xyz[f,i],axis=1).min(),label))
        if nearest:
            dst,label=min(nearest)
            d[f]=float(dst);which[f]=label
    finite=np.flatnonzero(np.isfinite(d))
    chosen=[]
    for i in sorted(finite,key=lambda i:(d[i],i)):
        if any(abs(i-j)<min_separation_frames for j in chosen):continue
        chosen.append(int(i))
        if len(chosen)==3:break
    return dict(
        usable_2_ball_marker_frames=int((good[:,ball_idx].sum(axis=1)>=2).sum()),
        coverage_2_ball_marker_fraction=float((good[:,ball_idx].sum(axis=1)>=2).mean()),
        valid_foot_marker_fractions={s:float(good[:,idx].mean()) for s,idx in foot_idx.items()},
        top_optical_candidates=[dict(t_s=round(j/rate,6),frame_120hz=j,
                                    nearest_foot_marker=which[j],minimum_point_distance_m=float(d[j]),
                                    verified_impact=False) for j in chosen],
        number_frames=n)

def qualification(path):
    import ezc3d
    x=ezc3d.c3d(str(path))
    prm=x["parameters"]
    labels=list(map(str,prm["POINT"]["LABELS"]["value"]))
    hz=float(prm["POINT"]["RATE"]["value"][0])
    if not math.isclose(hz,120,rel_tol=0,abs_tol=1e-6):raise ValueError("Source is not 120Hz")
    units=prm["POINT"].get("UNITS",{}).get("value")
    if units!=["mm"]:raise ValueError("C3D spatial units not original millimetres")
    pts=np.asarray(x["data"]["points"])
    geo=geom_candidates(pts,labels,hz)
    analog_names=prm.get("ANALOG",{}).get("LABELS",{}).get("value",[])
    forceplates=prm.get("FORCE_PLATFORM",{}).get("USED",{}).get("value",[0])
    has_grf=bool(analog_names or int(forceplates[0])>0)
    return dict(name=path.name,sha256=SHA(path),subject_group=path.name.split("_",1)[0],
                true_observed_frame_rate_hz=hz, **geo,
                original_analog_label_count=len(analog_names),force_plate_count=int(forceplates[0]),
                synchronised_grf_channel_present=has_grf,
                validated_professional_athlete=False,validated_instap_contact_impulse=False,
                human_surface_contact_force_groundtruth=False,
                suitable_for="geometric phase candidate discovery and held-out-by-subject optical representation only",
                NOT_suitable_for="instep impulses, collision manifolds, true foot-ball contact force, force-calibrated policy")
def run():
    paths=sorted(C3D_ROOT.rglob("*.c3d"))
    if len(paths)!=6:raise ValueError("Unexpected original optical C3D source count")
    rows=[qualification(p) for p in paths]
    groups=sorted(set(r["subject_group"] for r in rows))
    if any(r["synchronised_grf_channel_present"] for r in rows):
        raise ValueError("New force-platform metadata requires independent scientific requalification")
    data=dict(version="v0.1.72",research_type="bounded source data license/biomechanics suitability, not new human truth",
              six_sha_locked_original_optical_clips=len(rows),
              distinct_subject_groups=groups,subject_independent_split_required=True,
              observed_force_calibrated_clips=0,observed_native_foot_ball_manifold_clips=0,
              no_professional_level_claim=True,no_true_instep_impulse_label=True,
              no_true_GRFs=True,clips=rows)
    with OUT.open("x") as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V072_CMU",len(rows),"GROUPS",groups,
          "GEOMETRIC_CLIPS",sum(x["coverage_2_ball_marker_fraction"]>0 for x in rows),
          "FORCE_VALID",0,"SHA",SHA(OUT))
    for x in rows:print(x["name"],x["number_frames"],x["top_optical_candidates"][:1])
    return data
if __name__=="__main__":run()
