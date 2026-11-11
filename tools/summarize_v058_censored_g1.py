#!/usr/bin/env python3
"""Independent, read-only 200 Hz original-G1 CPU dynamics and censoring summary.

This is NOT a validation of GPU equivalence, human GRF, or human contact truth.
Output must be a new write-once path. Raw evidence is never rewritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"pelvis", "right_hip_roll_link"}
ROBOT_PREFIX = "/World/envs/env_0/Robot/"

def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pair_points(report: dict, pair: set[str] = PAIR) -> list[dict]:
    if report.get("errors") or report.get("dropped_events") or not report.get("report_capture_complete"):
        return []
    result = []
    for ev in report.get("saved_interesting_events", []):
        actors, shapes = ev.get("actor_paths", []), ev.get("collider_paths", [])
        if len(actors) != 2 or len(shapes) != 2:
            continue
        if not all(a.startswith(ROBOT_PREFIX) and s.startswith(a + "/") for a, s in zip(actors, shapes)):
            continue
        if {a[len(ROBOT_PREFIX):] for a in actors} != pair:
            continue
        for pt in ev.get("point_sample", []):
            loc, normal = pt.get("point_w"), pt.get("normal_w")
            if not isinstance(loc, list) or not isinstance(normal, list) or len(loc) != 3 or len(normal) != 3:
                continue
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in loc + normal):
                continue
            if math.sqrt(sum(v*v for v in normal)) < 0.1:
                continue
            result.append({"t_s": ev["time_s"], "actors": actors, "colliders": shapes,
                           "point_w": loc, "normal_w": normal, "separation_m": pt.get("separation_m"),
                           "impulse_xyz": pt.get("impulse_xyz")})
    return result

def summarize(raw: dict) -> dict:
    if raw.get("v057_requested_physics_backend") != "cpu" or raw.get("root_fixed") is not False:
        raise ValueError("Not original free-root CPU G1")
    names = raw["joint_names"]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Not complete unique original 29DoF")
    lim = raw["joint_velocity_limits"]
    if len(lim) != 1 or len(lim[0]) != 29 or not all(
        isinstance(v, (int, float)) and 0 < v < math.inf for v in lim[0]
    ):
        raise ValueError("Invalid original 29DoF joint velocity limits")
    all_frames = raw["traces"][0]
    dyn = raw["physx_pair_dynamics_v055"]["frames"]
    if len(all_frames) < 2 or len(dyn) < 2:
        raise ValueError("Missing physical trajectory")
    # Upstream appends one terminal status at last physics time + 1 microsecond;
    # this is a STOP MARKER, never claim it is a 200 Hz physics substep.
    terminal_marker = bool(all_frames[-1].get("terminated") is True
        and 0 < all_frames[-1]["time_s"] - all_frames[-2]["time_s"] < .003)
    frames = all_frames[:-1] if terminal_marker else all_frames
    times = [f["time_s"] for f in frames]
    if any(not math.isfinite(v) for v in times) or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("Invalid strictly increasing timestamps")
    if any(not 0.003 <= b-a <= 0.007 for a,b in zip(times,times[1:])):
        raise ValueError("Not genuine 200Hz trajectory")
    if any(len(f["joint_vel"]) != 29 or any(not math.isfinite(x) for x in f["joint_vel"]) for f in frames):
        raise ValueError("Incomplete original qdot")
    ct = raw["records"][0]["contact_time_s"]
    if ct is None or not math.isfinite(ct):
        raise ValueError("No measured contact time")
    peak = max((abs(v)/limit, i, f["time_s"]) for f in frames
               for i,(v,limit) in enumerate(zip(f["joint_vel"],lim[0])))
    pre = [f for f in frames if ct-.4 <= f["time_s"] < ct]
    post = [f for f in frames if f["time_s"] >= ct]
    precomplete = times[0] <= ct - .4 + .0051
    postcomplete = times[-1] >= ct + 1.25
    dpre = [f for f in dyn if ct-.4 <= f["time_s"] < ct]
    dpost = [f for f in dyn if f["time_s"] >= ct]
    def forces(window):
        if not window:
            return None
        return {side:{"max_N":max(f["support_foot_force_N"][side] for f in window),
                      "mean_N":sum(f["support_foot_force_N"][side] for f in window)/len(window)}
                for side in ("left","right")}
    def com_z(window):
        return None if not window else [min(f["com_w"][2] for f in window),max(f["com_w"][2] for f in window)]
    native = raw["physx_native_report_v056"]
    hip = pair_points(native)
    wrist = pair_points(native, {"left_wrist_yaw_link","left_hip_roll_link"})
    strict = raw["stride_shot_success"]
    if not isinstance(strict, bool):
        raise ValueError("Unknown full-task success outcome")
    return {"backend":"cpu", "original_29dof":True, "joint_names":names,
            "trace_frames_including_terminal_marker":len(all_frames),
            "terminal_status_marker_not_200hz":terminal_marker,
            "trace_frames":len(frames),"dynamics_frames":len(dyn),"time_start_s":times[0],
            "time_end_s":times[-1],"impact_time_s":ct,
            "pre_0p4_complete":precomplete,"pre_frames":len(pre),
            "post_1p25_complete":postcomplete,"post_frames":len(post),
            "post_observed_s":round(times[-1]-ct,6),"censored_by_early_stop":not postcomplete,
            "qdot_peak_limit_ratio":peak[0], "qdot_peak_joint":names[peak[1]],"qdot_peak_time_s":peak[2],
            "pre_support":forces(pre),"post_support":forces(post),
            "pre_com_z_range_m":com_z(dpre),"post_com_z_range_m":com_z(dpost),
            "pre_wrist_hip_origin_min_m":min((f["left_wrist_hip_link_origin_distance_m"] for f in dpre),default=None),
            "post_wrist_hip_origin_min_m":min((f["left_wrist_hip_link_origin_distance_m"] for f in dpost),default=None),
            "native_callback_calls":native.get("callback_calls"),
            "native_capture_complete":native.get("report_capture_complete"),
            "native_pelvis_right_hip_points":hip,
            "native_left_wrist_hip_points":wrist,
            "native_G1_pelvis_right_hip_certified":bool(hip),
            "native_G1_left_wrist_hip_certified":bool(wrist),
            "nonfoot_force_peak_N":max(f["nonfoot_contact_N"] for f in frames),
            "termination_reasons":raw["termination_reasons"],
            "instep_impulse_fraction":raw["records"][0]["requested_surface_impulse_fraction"],
            "goal":raw["records"][0]["goal"],
            "strict_success":strict,
            "limitations":["Only CPU backend physically observed; cannot infer GPU pair truth",
                           "Post-impact recovery censored after original nonfoot terminal",
                           "No professional athlete kick GRF or time-synchronized physical contact truth",
                           "Net forces are not pair-level attributions; native paths and contact points are"]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"reports/v058_v058_cpu_frozen_v32.json")
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    raw = json.loads(a.input.read_text())
    result = summarize(raw)
    result["raw_sha256"] = digest(a.input)
    with a.output.open("x") as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("joint_names","native_pelvis_right_hip_points","native_left_wrist_hip_points","limitations")},indent=2))

if __name__ == "__main__":
    main()
