#!/usr/bin/env python3
"""SHA-qualified interpolation of real ball center crossing original goal y-plane.

This is diagnostic of *why* the original goal gate failed. Original engine
goal truth is still the authority; no interpolation is used to override it.
"""
import json
from pathlib import Path
import numpy as np
from v118_independent_200hz_audit import sha
ROOT=Path(__file__).resolve().parents[1]

def interpolate_crossing(trace, goal):
    if len(trace)<2:raise ValueError("TOO_FEW_TRUE_FRAMES")
    goal_x,goal_y=map(float,goal)
    t=np.asarray([r["time_s"] for r in trace],float)
    b=np.asarray([r["ball_xyz"] for r in trace],float)
    if b.shape!=(len(trace),3) or not np.isfinite(b).all() or not np.isfinite(t).all():
        raise ValueError("INVALID_BALL_TRACE")
    if not np.allclose(np.diff(t),.005,atol=1e-6,rtol=0):
        raise ValueError("NOT_GENUINE_200HZ_TRACE")
    ix=np.flatnonzero((b[:-1,1]>=goal_y)&(b[1:,1]<goal_y))
    if len(ix)!=1:
        return dict(crossing_observed=False,crossing_count=len(ix),extrapolated=False)
    j=int(ix[0])
    y0,y1=b[j,1],b[j+1,1]
    alpha=float((goal_y-y0)/(y1-y0))
    if not 0<=alpha<=1:raise ValueError("INVALID_INTERPOLATION")
    pt=(1-alpha)*b[j]+alpha*b[j+1]
    return dict(crossing_observed=True,crossing_count=1,
       true_bracketing_frame_indices=[j,j+1],time_s=float((1-alpha)*t[j]+alpha*t[j+1]),
       ball_center_at_original_goal_plane_xyz_m=pt.tolist(),
       ball_center_lateral_error_from_original_goal_center_m=float(pt[0]-goal_x),
       extrapolated=False)

def audit(root=ROOT):
    master=root/"reports/v118_original_g1_batch.audit.json"
    a=json.loads(master.read_text())
    rows=[]
    for row in a["runs"]:
        if not row["certified_original_G1_cuda_200Hz"] or row["observer"]:continue
        p=root/"reports"/("v118_"+row["name"]+".json")
        if sha(p)!=row["raw_sha256"]:raise ValueError("ORIGINAL_BALL_TRACE_SHA_DRIFT")
        raw=json.loads(p.read_text())
        if len(raw.get("goals",[]))!=1:raise ValueError("AMBIGUOUS_ORIGINAL_GOAL")
        if raw.get("goal_width_m",0)<=0:raise ValueError("MISSING_GOAL_WIDTH")
        crossing=interpolate_crossing(raw["traces"][0],raw["goals"][0])
        lateral=None
        if crossing["crossing_observed"]:
            x=crossing["ball_center_at_original_goal_plane_xyz_m"][0]
            center=float(raw["goals"][0][0]);hw=float(raw["goal_width_m"])/2
            lateral=dict(original_goal_left_edge_x_m=center-hw,
                original_goal_right_edge_x_m=center+hw,
                ball_center_outside_goal_lateral_interval_m=float(max(center-hw-x,x-(center+hw),0)),
                ball_center_inside_original_goal_width=bool(center-hw<=x<=center+hw))
        rows.append(dict(scene=row["scene"],control=row["control"],raw_sha256=row["raw_sha256"],
             original_goal_truth=row["hard_gates"]["original_goal_entered"],
             original_strict_pass=row["hard_gates"]["hard_gate_pass"],
             original_ball_speed_proxy_mps=row["hard_gates"]["ball_speed_peak_mps_post_5_to_110ms_proxy"],
             crossing=crossing,lateral_geometry_diagnostic=lateral,
             ball_center_diagnostic_does_not_override_original_goal_truth=True))
    return dict(version="v0.1.118",master_sha256=sha(master),real_independent_shots=len(rows),
        interpolation_not_engine_goal_decider=True,examples=rows)
if __name__=="__main__":
    result=audit()
    with (ROOT/"reports/v118_goal_plane_evidence.audit.json").open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print("V118_REAL_GOAL_PLANE",json.dumps(result),flush=True)
