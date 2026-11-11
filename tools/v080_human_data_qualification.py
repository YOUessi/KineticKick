#!/usr/bin/env python3
"""Bounded, read-only qualification of the six original 120Hz CMU C3Ds.

No marker geometry is called real impact impulse or human ground reaction force.
Professionality, athlete identity and contacts are UNKNOWN without provenance.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np
ORIG=Path("/home/you/projects/kinetickick")
ROOT=Path(__file__).resolve().parents[1]
RAW=ORIG/"data/research_human_kicks/cmu_soccer_120hz_v050/raw"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def qualify_one(p):
    import ezc3d
    raw=ezc3d.c3d(str(p))
    marker=raw["data"]["points"]
    analog=raw["data"]["analogs"]
    labels=raw["parameters"]["POINT"]["LABELS"]["value"]
    alabel=raw["parameters"].get("ANALOG",{}).get("LABELS",{}).get("value",[])
    fps=float(raw["parameters"]["POINT"]["RATE"]["value"][0])
    force=raw["parameters"].get("FORCE_PLATFORM",{})
    platform_used=force.get("USED",{}).get("value",[])
    force_count=int(platform_used[0]) if len(platform_used) else None
    source_subject=p.parent.name
    point_counts=marker.shape
    if fps<119 or marker.ndim!=3 or marker.shape[0]!=4 :
        raise ValueError("INVALID_ORIGINAL_C3D")
    invalid=np.any(~np.isfinite(marker[:3]),axis=0) | (marker[3]<0)
    ball=[str(l) for l in labels if re.search("ball",str(l),re.I)]
    foot=[str(l) for l in labels if re.search("foot|ank|heel|toe|metat|to[ei]",str(l),re.I)]
    analog_channels=int(analog.shape[1])
    has_force_plate=bool(force_count and force_count>0)
    synchronized_grf_qualified=bool(has_force_plate and analog_channels>0)
    # Having force plates is necessary but not sufficient for a VERIFIED force trace.
    professional_athlete_verified=False
    observed_foot_ball_impulse_available=False
    qualified_for_professional_instep_force_supervision=bool(
        professional_athlete_verified and synchronized_grf_qualified and
        observed_foot_ball_impulse_available)
    return dict(file=str(p.relative_to(RAW)),source_actor_id_from_dataset_dir=source_subject,
        sha256=sha(p),bytes=p.stat().st_size,
        marker_rate_hz=fps,frames=int(marker.shape[2]),
        duration_s=float(marker.shape[2]/fps),raw_marker_count=int(marker.shape[1]),
        invalid_marker_frame_fraction=float(invalid.mean()),
        ball_named_markers=ball,foot_marker_labels_sample=foot[:16],
        analog_channel_count=analog_channels,analog_labels_count=len(alabel),
        force_platform_count_in_original_parameter=force_count,
        synchronized_grf_qualified=synchronized_grf_qualified,
        verified_professional_athlete_identity=professional_athlete_verified,
        original_foot_ball_contact_impulse_measured=observed_foot_ball_impulse_available,
        qualified_for_professional_instep_force_supervision=qualified_for_professional_instep_force_supervision,
        allowed_use="exploratory geometry / approximate phase only, subject-disjoint splits",
        prohibited_use="professional power / true instep impact / lab GRF ground truth")
def main():
    paths=sorted(RAW.rglob("*.c3d"))
    if len(paths)!=6:raise RuntimeError("EXPECTED_EXACT_SIX_ORIGINAL_FILES")
    out=[qualify_one(p) for p in paths]
    ids=sorted(set(r["source_actor_id_from_dataset_dir"] for r in out))
    manifest=dict(version="v0.1.80",source_root_read_only=str(RAW),
        source_actors_unverified=[{"directory_id":sid,
             "clip_count":sum(r["source_actor_id_from_dataset_dir"]==sid for r in out)}
             for sid in ids],count=len(out),
        source_files_sha256_immutable=True,professional_verified_count=sum(
        r["qualified_for_professional_instep_force_supervision"] for r in out),
        learning_rule="only exploratory kinematics; no shot power GRF/impulse labels",
        no_actor_reidentification_claim=True,clips=out)
    dest=ROOT/"reports/v080_original_six_c3d_qualification.audit.json"
    with dest.open("x") as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write("\n")
    print("V080_CMU_QUALIFICATION",json.dumps(
        {"clips":len(out),"actor_directory_count":len(ids),
         "pro_force_eligible":manifest["professional_verified_count"]}),flush=True)
if __name__=="__main__":main()
