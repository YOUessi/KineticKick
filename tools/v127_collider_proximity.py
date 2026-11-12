#!/usr/bin/env python3
"""v127 original G1 URDF collider capsule lower-bound audit (CPU, read-only).

Positive capsule separation can geometrically EXCLUDE collision for these URDF
cylinders. Nonpositive separation is *not* a detected PhysX contact pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v127_frozen_collider_kinematics.json"
FROZEN_SHA = "8ce91b872f8f7702dec9020281c42dd22e03cf88"
DATA_ROOT = Path("/home/you/projects")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def frozen():
    b = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":" +
         str(CONFIG.relative_to(ROOT))])
    if CONFIG.read_bytes() != b:
        raise ValueError("FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(b)
    if cfg["version"] != "v0.1.127" or not cfg["offline_read_only_no_actuation"]:
        raise ValueError("INVALID_FROZEN_CONTRACT")
    if cfg["native_physx_pair_certification_permitted"] or not cfg["no_parameter_fitting"]:
        raise ValueError("FORBIDDEN_ACTUATION_OR_TUNING")
    return cfg


def require_source(path: Path, expected: str):
    if (not path.is_file() or path.is_symlink() or len(expected) != 64
            or sha(path) != expected):
        raise ValueError("SOURCE_SHA256_DRIFT:" + str(path))


def numbers(s: str | None, default: str, size: int = 3) -> np.ndarray:
    tokens = (s if s is not None else default).split()
    try:
        a = np.array([float(x) for x in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError("BAD_URDF_VECTOR") from exc
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError("BAD_URDF_VECTOR")
    return a


def rpy_matrix(rpy) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], dtype=float)


def axis_rotation(axis, angles) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    v = np.asarray(angles, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all() or axis.shape != (3,):
        raise ValueError("INVALID_AXIS_ANGLE")
    size = np.linalg.norm(axis)
    if not np.isfinite(size) or size < 1e-12:
        raise ValueError("ZERO_JOINT_AXIS")
    x, y, z = axis / size
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    a = np.eye(3)
    return (a[None, :, :] + np.sin(v)[:, None, None] * K[None, :, :] +
            (1-np.cos(v))[:, None, None] * (K @ K)[None, :, :])


def parse_tree(path: Path):
    xml = ET.parse(path).getroot()
    if xml.tag != "robot":
        raise ValueError("NOT_URDF")
    links = {}
    for link in xml.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError("LINK_NAME_ERROR")
        shapes = []
        for c in link.findall("collision"):
            g = c.find("geometry")
            if g is None or len(g) != 1:
                raise ValueError("INVALID_COLLISION")
            origin = c.find("origin")
            shapes.append({
                "type": g[0].tag,
                "radius": float(g[0].get("radius")) if g[0].tag == "cylinder" else None,
                "length": float(g[0].get("length")) if g[0].tag == "cylinder" else None,
                "xyz": numbers(origin.get("xyz") if origin is not None else None, "0 0 0"),
                "R": rpy_matrix(numbers(origin.get("rpy") if origin is not None else None, "0 0 0")),
            })
        links[name] = shapes
    joints = []
    children = set()
    for j in xml.findall("joint"):
        typ = j.get("type")
        if typ not in ("fixed", "revolute"):
            raise ValueError("UNSUPPORTED_KINEMATIC_JOINT")
        p, c = j.find("parent"), j.find("child")
        if p is None or c is None:
            raise ValueError("MISSING_JOINT_LINK")
        pn, cn = p.get("link"), c.get("link")
        if not pn or not cn or pn not in links or cn not in links or cn in children:
            raise ValueError("BAD_TREE")
        children.add(cn)
        o = j.find("origin")
        ax = j.find("axis")
        joints.append({
            "name": j.get("name"), "type": typ, "parent": pn, "child": cn,
            "xyz": numbers(o.get("xyz") if o is not None else None, "0 0 0"),
            "R": rpy_matrix(numbers(o.get("rpy") if o is not None else None, "0 0 0")),
            "axis": numbers(ax.get("xyz") if ax is not None else None, "1 0 0")
        })
    roots = set(links) - children
    if len(roots) != 1 or len(joints) != len(links)-1:
        raise ValueError("NOT_SINGLE_ROOT_TREE")
    root = next(iter(roots))
    ordered = []
    remaining = joints[:]
    reached = {root}
    while remaining:
        ready = [j for j in remaining if j["parent"] in reached]
        if not ready:
            raise ValueError("LOOP_OR_DISCONNECTED_TREE")
        for j in ready:
            remaining.remove(j)
            reached.add(j["child"])
            ordered.append(j)
    actuators = [j["name"] for j in ordered if j["type"] == "revolute"]
    if set(actuators) != set(G1_JOINT_ORDER) or len(actuators) != 29:
        raise ValueError("NOT_ORIGINAL_29_JOINT_NAMES")
    return {"root": root, "links": links, "joints": ordered}


def fk_targets(tree, qs: np.ndarray, targets: set[str]):
    """Return root-relative (rotation,translation) for specified links; all N frames."""
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != 29 or not np.isfinite(qs).all():
        raise ValueError("INVALID_Nx29_ORIGINAL_POSES")
    n = len(qs)
    index = {s:i for i,s in enumerate(G1_JOINT_ORDER)}
    rel = {tree["root"]: (
        np.broadcast_to(np.eye(3), (n, 3, 3)),
        np.zeros((n, 3), dtype=float))}
    for j in tree["joints"]:
        Rp, pp = rel[j["parent"]]
        Ro = Rp @ j["R"]
        pos = pp + np.einsum("nij,j->ni", Rp, j["xyz"])
        R = (Ro @ axis_rotation(j["axis"], qs[:, index[j["name"]]])
             if j["type"] == "revolute" else Ro)
        rel[j["child"]] = (R, pos)
    if not targets <= rel.keys():
        raise ValueError("MISSING_TARGET_LINKS")
    return {k:rel[k] for k in targets}


def cylinder_axis_endpoints(tree, poses, name):
    shapes = tree["links"].get(name)
    if shapes is None or len(shapes) != 1 or shapes[0]["type"] != "cylinder":
        raise ValueError("NOT_SINGLE_CYLINDER_COLLIDER:" + name)
    sh = shapes[0]
    if sh["radius"] <= 0 or sh["length"] <= 0:
        raise ValueError("INVALID_COLLISION_CYLINDER")
    R, p = poses[name]
    ctr = p + np.einsum("nij,j->ni", R, sh["xyz"])
    axis = np.einsum("nij,j->ni", R, sh["R"][:, 2])
    half = 0.5 * sh["length"] * axis
    return ctr-half, ctr+half, sh["radius"]


def segment_distance(p0, p1, q0, q1):
    """Exact Euclidean minimum distance of two finite, nondegenerate 3D segments."""
    pp = [np.asarray(v, dtype=float) for v in (p0,p1,q0,q1)]
    if len({x.shape for x in pp}) != 1 or pp[0].ndim != 2 or pp[0].shape[1] != 3:
        raise ValueError("BAD_SEGMENTS")
    if not all(np.isfinite(x).all() for x in pp):
        raise ValueError("NONFINITE_SEGMENTS")
    p0,p1,q0,q1 = pp
    d1 = p1-p0; d2 = q1-q0; r = p0-q0
    a = np.einsum("ij,ij->i", d1,d1)
    e = np.einsum("ij,ij->i", d2,d2)
    if np.any(a < 1e-16) or np.any(e < 1e-16):
        raise ValueError("DEGENERATE_SEGMENT")
    b = np.einsum("ij,ij->i", d1,d2)
    c = np.einsum("ij,ij->i", d1,r)
    f = np.einsum("ij,ij->i", d2,r)
    denom = a*e-b*b
    s = np.clip(np.divide(b*f-c*e,denom,out=np.zeros_like(denom),
                          where=denom>1e-14*a*e),0,1)
    t = np.clip((b*s+f)/e,0,1)
    s = np.where(t <= 0, np.clip(-c/a,0,1),
                 np.where(t >= 1, np.clip((b-c)/a,0,1), s))
    closest = r+d1*s[:,None]-d2*t[:,None]
    return np.linalg.norm(closest,axis=1)


def clearance_for_episode(tree, qs, pairs):
    names = {s for pair in pairs for s in pair}
    poses = fk_targets(tree,qs,names)
    endpoints = {n:cylinder_axis_endpoints(tree,poses,n) for n in names}
    results = {}
    for a,b in pairs:
        p0,p1,ra = endpoints[a]
        q0,q1,rb = endpoints[b]
        results[a+"__"+b] = segment_distance(p0,p1,q0,q1)-ra-rb
    return results


def windows(t, impact, cfg):
    pre=(t>=impact-cfg["pre_contact_s"]-1e-7)&(t<impact-1e-7)
    post=(t>=impact-1e-7)&(t<impact+cfg["post_contact_s"]-1e-7)
    late=(t>=impact+cfg["post_contact_s"]-1e-7)
    if (pre.sum()!=80 or post.sum()!=250 or
            (pre&post).any() or (pre&late).any() or (post&late).any()):
        raise ValueError("INCOMPLETE_CONTACT_WINDOWS")
    return {"precontact_400ms":pre,"postcontact_1250ms":post,"late_recovery":late}


def load_episode(path, expected_sha, joint_names):
    require_source(path, expected_sha)
    raw=json.loads(path.read_text())
    if tuple(raw["joint_names"]) != tuple(joint_names) or raw["root_fixed"] is not False:
        raise ValueError("ORIGINAL_JOINT_ORDER_OR_ROOT_CHANGED")
    if len(raw["traces"])!=1 or len(raw["traces"][0])!=2800:
        raise ValueError("NOT_FULL_200HZ_14S")
    rows=raw["traces"][0]
    t=np.asarray([row["time_s"] for row in rows],dtype=float)
    q=np.asarray([row["joint_pos"] for row in rows],dtype=float)
    vel=np.asarray([row["joint_vel"] for row in rows],dtype=float)
    nonfoot=np.asarray([row["nonfoot_contact_N"] for row in rows],dtype=float)
    com=np.asarray([row["whole_body_com_xyz"] for row in rows],dtype=float)
    support=np.asarray([[row["support_foot_force_N"][s] for s in ("left","right")]
                        for row in rows],dtype=float)
    if (q.shape!=(2800,29) or vel.shape!=(2800,29) or com.shape!=(2800,3)
            or support.shape!=(2800,2) or not all(np.isfinite(x).all()
                for x in (t,q,vel,nonfoot,com,support))):
        raise ValueError("TRUNCATED_OR_NONFINITE_29DOF_PHYSICS")
    if not np.allclose(t, .005*np.arange(1,2801),rtol=0,atol=2e-5):
        raise ValueError("INVALID_200HZ_TIME_GRID")
    return raw,t,q,vel,nonfoot,com,support


def metrics_per_phase(clearances, t, impact, vel, limits, nonfoot, com, support, cfg):
    cuts=windows(t,impact,cfg)
    answer={}
    for window, m in cuts.items():
        joint_ratio=np.abs(vel[m])/limits[None,:]
        pair={}
        for name, c in clearances.items():
            cwin=c[m]
            pair[name]={
                "minimum_capsule_surface_clearance_m":float(cwin.min()),
                "first_minimum_time_s":float(t[np.flatnonzero(m)[np.argmin(cwin)]]),
                "possible_overlap_frame_fraction":float(np.mean(cwin<=0)),
                "under_20mm_alert_frame_fraction":float(np.mean(cwin<=0.02)),
                "certified_cylinder_noncontact_frame_fraction":float(np.mean(cwin>0)),
            }
        answer[window]={
            "frames":int(m.sum()),
            "pair_geometry":pair,
            "original29_qdot_peak_limit_ratio":float(joint_ratio.max()),
            "nonfoot_net_force_peak_N":float(nonfoot[m].max()),
            "COM_xy_delta_m":float(np.linalg.norm(com[m][-1,:2]-com[m][0,:2])),
            "left_ground_net_force_proxy_integral_Ns":float(support[m,0].sum()*.005),
            "right_ground_net_force_proxy_integral_Ns":float(support[m,1].sum()*.005),
        }
    return answer


def run(cfg, *, data_root=DATA_ROOT, keep_arrays=False):
    if head(ROOT)!=cfg["parent_sha"] and head(ROOT)!=FROZEN_SHA:
        # A post-analysis commit is allowed; frozen config is independently SHA locked.
        if not subprocess.run(["git","-C",str(ROOT),"cat-file","-e",cfg["parent_sha"]],
                              capture_output=True).returncode==0:
            raise ValueError("PARENT_COMMIT_NOT_IN_HISTORY")
    src125=data_root/"kinetickick-wt-v125"
    src116=data_root/"kinetickick-wt-v116"
    if head(src125)!=cfg["source_v125_commit"] or head(src116)!=cfg["source_urdf_commit"]:
        raise ValueError("SOURCE_GIT_HEAD_DRIFT")
    audit=src125/"reports/v125_original_g1_timeline_sha.audit.json"
    require_source(audit,cfg["source_v125_audit_sha256"])
    source=json.loads(audit.read_text())
    urdf=src116/"configs/paid_original/paid_main_resolved.urdf"
    require_source(urdf,cfg["source_urdf_sha256"])
    tree=parse_tree(urdf)
    for p in cfg["focus_collision_pairs"]:
        if any(tree["links"][a][0]["type"]!="cylinder" for a in p):
            raise ValueError("NONCYLINDER_COLLISION_PAIR")
    prior={(r["version"],r["scene"],r["control"]):r
           for r in source["per_episode"]}
    source_cfg=json.loads((src125/"configs/v125_frozen_surrogates_and_recovery.json").read_text())
    index={x["version"]:x for x in source_cfg["robot_sources"]}
    episodes=[]; same={}; parity=[]; curves={}; frame_total=0
    for version in cfg["source_episodes"]:
        c=index[version]; folder=data_root/("kinetickick-wt-v"+str(version))
        if head(folder)!=c["commit"]:
            raise ValueError("SOURCE_BRANCH_SHA_DRIFT")
        master=folder/"reports"/("v"+str(version)+"_original_g1_batch.audit.json")
        require_source(master,c["master_sha256"])
        manifest=json.loads(master.read_text())
        for item in manifest["runs"]:
            if item.get("certified_original_G1_cuda_200Hz") is not True:
                raise ValueError("NOT_REAL_PHYSX")
            fp=folder/"reports"/("v"+str(version)+"_"+item["name"]+".json")
            raw,t,q,vel,nonfoot,com,support=load_episode(
                fp,item["raw_sha256"],G1_JOINT_ORDER)
            key=(version,item["scene"],item["control"])
            matched=prior.get(key)
            if not matched:
                raise ValueError("NO_INDEPENDENT_SOURCE_RECORD")
            impact=float(matched["impact_s"])
            clearance=clearance_for_episode(tree,q,cfg["focus_collision_pairs"])
            if any(x.shape!=(2800,) or not np.isfinite(x).all() for x in clearance.values()):
                raise ValueError("BAD_GEOMETRY_OUTPUT")
            limits=np.asarray(raw["joint_velocity_limits"],dtype=float).reshape(-1)
            if limits.shape!=(29,) or np.any(limits<=0):
                raise ValueError("ORIGINAL_QDOT_LIMITS_INVALID")
            peak=float((np.abs(vel)/limits[None,:]).max())
            historical=float(item["hard_gates"]["original_29_qdot_max_ratio"])
            if abs(peak-historical)>2e-5:
                raise ValueError("REAL_QDOT_LIMIT_MISMATCH")
            summary=metrics_per_phase(clearance,t,impact,vel,limits,nonfoot,com,
                                      support,cfg)
            row={
                "version":version,"scene":item["scene"],"control":item["control"],
                "observer_ON_repeat":bool(item["observer"]),
                "seed":item["seed"],"raw_sha256":item["raw_sha256"],
                "original_strict_pass":bool(matched["strict_hard_gate_pass"]),
                "original_failed_gates":matched["failed_gates"],
                "impact_s":impact,"physical_frames":2800,
                "max_original29_qdot_ratio":peak,
                "geometry_origin":"SHA256_URDF_AND_29DOF_RECONSTRUCTION",
                "native_physx_pair_certified":False,
                "contact_not_excluded_is_not_true_contact":True,
                "per_phase":summary,
                "full14s_min_clearance_by_pair_m":{
                    k:float(v.min()) for k,v in clearance.items()},
                "full14s_nonfoot_net_force_peak_N":float(nonfoot.max()),
            }
            if not item["observer"]:
                if key in same:
                    raise ValueError("DUPLICATE_INDEPENDENT_EPISODE")
                same[key] = {k:v.copy() for k,v in clearance.items()}
                curves[key] = {"t":t,"clearance":clearance,"impact":impact,
                               "nonfoot":nonfoot}
            else:
                if key not in same:
                    raise ValueError("MISSING_OBSERVER_OFF")
                exact=all(np.array_equal(same[key][k],v) for k,v in clearance.items())
                if not exact:
                    raise ValueError("OFF_ON_RECONSTRUCTED_GEOMETRY_CHANGED")
                parity.append({"case":"/".join(map(str,key)),
                               "full2800frames_4pair_exact":exact})
            episodes.append(row)
            frame_total+=2800
    if (len(episodes)!=cfg["expected_episode_count"] or
            sum(not x["observer_ON_repeat"] for x in episodes)!=cfg["expected_independent_episode_count"]
            or len(parity)!=cfg["expected_observer_repeat_count"]
            or frame_total!=cfg["expected_traced_frames"]):
        raise ValueError("FROZEN_29_EPISODE_COHORT_DRIFT")
    independent=[x for x in episodes if not x["observer_ON_repeat"]]
    pair_keys=list(curves[next(iter(curves))]["clearance"])
    aggregates={}
    for pair in pair_keys:
        vals=np.array([x["full14s_min_clearance_by_pair_m"][pair] for x in independent])
        aggregates[pair]={
            "independent_episodes_with_nonpositive_capsule_lower_bound":int(np.sum(vals<=0)),
            "independent_episodes_all_2800_frames_positive":int(np.sum(vals>0)),
            "minimum_full14s_capsule_bound_m":float(vals.min()),
            "maximum_episode_minimum_bound_m":float(vals.max()),
        }
    return {
        "version":"v0.1.127","frozen_protocol_sha":FROZEN_SHA,
        "historical_true_physx_episodes_reaudited":len(episodes),
        "historical_actual_200hz_frames":frame_total,
        "new_physics_rollouts":0,
        "strict_pass_original_independent":sum(x["original_strict_pass"] for x in independent),
        "independent_conditions":len(independent),
        "observer_off_on_repeats_excluded":len(parity),
        "observer_geometry_exact_parity":all(x["full2800frames_4pair_exact"] for x in parity),
        "four_pair_full14s_geometry_bounds":aggregates,
        "geometry_absence_or_possible_proximity_only":True,
        "native_physx_actor_collider_manifold_status":"UNKNOWN",
        "not_true_GRF_or_contact_kinetics":True,
        "new_human_policy_trained":False,
        "physical_motor_command_issued":False,
        "episodes":episodes,"observer_parity":parity
    },curves


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result,_=run(frozen())
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("episodes","observer_parity")},indent=2,allow_nan=False))


if __name__=="__main__":
    main()
