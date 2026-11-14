"""Register CMU's visible ball cluster to a G1 retarget and measure pre-launch foot gaps."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from v163_diagnose_motion import FEET, rotation, sha, xyz


def measure_foot_gaps(urdf, ref, center, radius, event):
    window = (ref["time_s"] >= event - 0.24) & (ref["time_s"] <= event + 0.11)
    indices = np.flatnonzero(window)
    if not indices.size:
        raise ValueError("Reference does not cover the source event")
    result = {}
    for foot in FEET:
        idx = ref["body_names"].tolist().index(foot)
        pose = rotation(ref["body_quat_w"][indices, idx])
        local = pose.inv().apply(center - ref["body_pos_w"][indices, idx])
        gaps = []
        for collision in urdf.find(f"link[@name='{foot}']").findall("collision"):
            cylinder = collision.find("geometry/cylinder")
            origin = collision.find("origin")
            axis = Rotation.from_euler("xyz", xyz(origin, "rpy")).as_matrix()[:, 2]
            diff = local - xyz(origin, "xyz")
            length = float(cylinder.get("length")) / 2
            nearest = diff - np.clip(diff @ axis, -length, length)[:, None] * axis
            gaps.append(
                np.linalg.norm(nearest, axis=1) - float(cylinder.get("radius")) - radius
            )
        gap = np.min(gaps, axis=0)
        i = int(np.argmin(gap))
        hit = np.flatnonzero(gap <= 0.0)
        result[foot] = {
            "minimum_m": float(gap[i]),
            "minimum_time_s": float(ref["time_s"][indices[i]]),
            "first_overlap_s": float(ref["time_s"][indices[hit[0]]])
            if len(hit)
            else None,
        }
    return result


def register(repo, assets, clip):
    proc = assets / "data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050"
    protocol = json.loads((repo / "configs/v160_tracking_protocol.json").read_text())
    meta = json.loads((repo / "experiments/v160_human/manifest.json").read_text())[
        "clips"
    ][clip]
    audit = next(
        a
        for a in json.loads((proc / "quality_audit.json").read_text())["motions"]
        if a["id"] == clip
    )
    source = proc / (clip + ".npz")
    refpath = repo / "experiments/v162_human/references" / (clip + ".npz")
    if (
        sha(source) != meta["source_processed_sha256"]
        or sha(source) != audit["derived_npz_sha256"]
    ):
        raise ValueError("Source file changed")
    if (
        audit["observed_ball_kinematics_quality"]
        != "continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth"
    ):
        raise ValueError("No continuous qualified outgoing ball-marker record")
    with (
        np.load(source, allow_pickle=False) as human,
        np.load(refpath, allow_pickle=False) as ref,
        np.load(
            repo / "experiments/v160_human" / (clip + ".npz"), allow_pickle=False
        ) as orig,
    ):
        names = orig["body_names"].tolist()
        markers = human["primary_landmark_names"].tolist()
        hips = human["primary_human_3d_m"][
            0, [markers.index(f"{s}HJ0") for s in "LR"]
        ].mean(0)
        anchor = orig["body_pos_w"][
            0, [names.index(f"{s}_hip_pitch_link") for s in ("left", "right")]
        ].mean(0)
        cluster = (human["uncalibrated_ball_marker_cluster_centroid_m"] - hips) * meta[
            "geometry"
        ]["map_scale"] + anchor
        onset = audit["inferred_event"]["inferred_ball_launch_onset_frame"]
        pre = slice(onset - 53, onset - 5)
        if (
            onset < 53
            or not human["three_ball_marker_valid"][pre].all()
            or not human["three_ball_marker_valid"][onset : onset + 7].all()
        ):
            raise ValueError("Unobserved prelaunch or outgoing ball markers")
        center = np.median(cluster[pre], axis=0)
        xy = cluster[onset + 6, :2] - cluster[onset, :2]
        length = float(np.linalg.norm(xy))
        if length < 0.001 or not np.all(np.isfinite(xy)):
            raise ValueError("Cannot infer an outgoing direction from source markers")
        xy /= length
        event = float(
            np.interp(human["time_s"][onset], ref["source_time_s"], ref["time_s"])
        )
        radius = protocol["ball_radius_m"]
        center[2] = radius
        urdf = ET.parse(assets / "configs/paid_original/paid_main_resolved.urdf")
        gaps = measure_foot_gaps(urdf, ref, center, radius, event)
        return {
            "source_clip": clip,
            "source_sha256": sha(source),
            "reference_sha256": sha(refpath),
            "source_event_s": event,
            "source_ball_xyz_m": center.tolist(),
            "source_direction_xy": xy.tolist(),
            "origin_window": [pre.start, pre.stop - 1],
            "source_near_foot": audit["inferred_event"]["nearest_foot_side"],
            "reference_fps": 50,
            "source_fps": 120,
            "qualifying_source": audit["observed_ball_kinematics_quality"],
            "gaps": gaps,
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument(
        "--assets", type=Path, default=Path("/home/you/projects/kinetickick")
    )
    args = p.parse_args()
    print(json.dumps(register(args.repo, args.assets, args.clip), indent=2))
