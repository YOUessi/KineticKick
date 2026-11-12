#!/usr/bin/env python3
"""Read-only qualification of six unique 120Hz real CMU C3D files; no kinetics invented."""
import hashlib, json
from pathlib import Path
import ezc3d
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL="data/research_human_kicks/cmu_soccer_120hz_v050/raw"
TRIALS=(("10","10_01"),("10","10_02"),("10","10_03"),("10","10_05"),("10","10_06"),("11","11_01"))

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def eligibility(*,frames,rate,markers,analogs):
    if frames<=0 or rate<=0 or markers<=0 or analogs<0:raise ValueError("INVALID_C3D_DIMENSIONS")
    return dict(kinematics_only=True, real_force_plate_grf_cop=False,
                verified_instep_impulse=False, elite_skill_verified=False,
                eligible_professional_impact_supervision=False)

def inspect(root=ROOT):
    trials=[]
    for subject,stem in TRIALS:
        path=root/REL/subject/(stem+".c3d")
        if not path.is_file():raise FileNotFoundError(path)
        obj=ezc3d.c3d(str(path))
        xyz=np.asarray(obj["data"]["points"][:3],float)
        ana=np.asarray(obj["data"]["analogs"])
        frames,markers=xyz.shape[2],xyz.shape[1]
        rate=float(obj["header"]["points"]["frame_rate"])
        channels=int(obj["parameters"]["ANALOG"]["USED"]["value"][0])
        finite_fraction=float(np.isfinite(xyz).mean())
        if ana.shape[1]!=channels or finite_fraction<0.9 or abs(rate-120)>1e-9:
            raise ValueError("C3D_NATIVE_CHANNELS_OR_POINTS_INCONSISTENT")
        if channels:raise ValueError("FORCE_CHANNELS_REQUIRE_INDEPENDENT_CALIBRATION")
        trials.append(dict(subject=subject, trial=stem, sha256=digest(path),
            frames=frames,rate_hz=rate,markers=markers,analogs=channels,
            finite_marker_xyz_fraction=finite_fraction,
            eligibility=eligibility(frames=frames,rate=rate,markers=markers,analogs=channels)))
    if len({t["sha256"] for t in trials})!=6:raise ValueError("DUPLICATE_HUMAN_C3D_SHA")
    return dict(version="v0.1.116", unique_trials=len(trials),
        unique_subjects=len({t["subject"] for t in trials}),
        marker_frames=sum(t["frames"] for t in trials),
        analog_channels=sum(t["analogs"] for t in trials),
        professionally_qualified_force_trials=0,
        qualified_true_ball_impulse_trials=0,trials=trials)

if __name__=="__main__":
    result=inspect()
    with (ROOT/"reports/v116_human_qualification.audit.json").open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print("HUMAN_DATA_QUALIFICATION",json.dumps({k:v for k,v in result.items() if k!="trials"}))
