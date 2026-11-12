#!/usr/bin/env python3
"""Source-qualified all-body URDF outer-capsule broadphase (CPU, NEVER contact truth).

Every original cylinder fits inside a same-axis capsule; sphere is a zero-length
capsule. A strictly positive lower bound excludes collision of the original
URDF primitives *at the observed time*. A nonpositive bound does not prove a
contact. Neither statement attests the Isaac Sim runtime USD shapes.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

import v127_collider_proximity as kinematics
import v128_v54_event_geometry as historical

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v132_frozen_fullbody_selfcollision_audit.json"
PROTOCOL_SHA = "c7150e1c362fb58756d183704f5b38ecb60df092"
BASE_SHA = "7f3c0e7880ba68743787d29224c1ea679ef70e04"


def frozen() -> dict:
    raw = subprocess.check_output([
        "git", "-C", str(ROOT), "show",
        PROTOCOL_SHA + ":configs/v132_frozen_fullbody_selfcollision_audit.json"])
    if PROTOCOL.read_bytes() != raw:
        raise ValueError("FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(raw)
    if (cfg["version"] != "v0.1.132" or cfg["parent_sha"] != BASE_SHA or
        cfg["observation_dt_s"] != .005 or cfg["original_robot_dof"] != 29 or
        cfg["min_kinematic_graph_edges_for_evaluated_pair"] != 3 or
        not all(cfg[k] for k in (
          "observed_clearance_not_native_physx_pair",
          "never_infer_collision_from_nonpositive_capsule_gap",
          "never_infer_no_collision_from_excluded_pairs",
          "never_infer_native_usd_shapes_from_urdf",
          "no_motor_actuation", "no_gpu_when_foreign_job",
          "no_old_evidence_mutation"))):
        raise ValueError("FROZEN_ASSUMPTION_DRIFT")
    return cfg


def adjacency(tree: dict) -> dict[str, set[str]]:
    graph = {name:set() for name in tree["links"]}
    for edge in tree["joints"]:
        graph[edge["parent"]].add(edge["child"])
        graph[edge["child"]].add(edge["parent"])
    return graph


def distances(graph: dict[str, set[str]], root: str) -> dict[str,int]:
    q = deque([(root,0)])
    seen = {}
    while q:
        name, d = q.popleft()
        if name in seen:
            continue
        seen[name] = d
        for other in sorted(graph[name]):
            if other not in seen:
                q.append((other,d+1))
    if len(seen)!=len(graph):
        raise ValueError("DISCONNECTED_G1_KINEMATIC_TREE")
    return seen


def eligible_pairs(tree: dict, min_edges: int) -> tuple[list[tuple[str,str]],int]:
    if min_edges<1:
        raise ValueError("INVALID_TOPOLOGY_EXCLUSION")
    shapes = sorted(n for n,s in tree["links"].items() if s)
    graph = adjacency(tree)
    pairs = []
    excluded=0
    for i,a in enumerate(shapes):
        step = distances(graph,a)
        for b in shapes[i+1:]:
            if step[b]>=min_edges:
                pairs.append((a,b))
            else:
                excluded+=1
    if len(pairs)+excluded!=len(shapes)*(len(shapes)-1)//2:
        raise AssertionError("PAIR_ACCOUNTING_BROKEN")
    return pairs,excluded


def sphere_radii(urdf: Path) -> dict[str,float]:
    xml=ET.parse(urdf).getroot()
    result={}
    for link in xml.findall("link"):
        name=link.get("name")
        for c in link.findall("collision"):
            sphere=c.find("geometry/sphere")
            if sphere is not None:
                if name in result:
                    raise ValueError("MULTIPLE_SPHERES_UNSUPPORTED")
                try:
                    radius=float(sphere.get("radius"))
                except (TypeError,ValueError) as exc:
                    raise ValueError("BAD_URDF_SPHERE") from exc
                if not math.isfinite(radius) or radius<=0:
                    raise ValueError("BAD_URDF_SPHERE")
                result[name]=radius
    return result


def primitive_capsules(tree, poses, spheres:dict[str,float]):
    """Return every original URDF collider as an OUTER capsule on its parent link.

    There can be multiple collision shapes attached to one link (the foot has
    seven cylinders). All must be included before a link-pair lower bound is
    certified. The return value maps link names to lists of capsules.
    """
    result={}
    for name,shapes in tree["links"].items():
        if not shapes:
            continue
        if name not in poses:
            raise ValueError("MISSING_FK_LINK")
        R,p=poses[name]
        individual=[]
        for shape in shapes:
            ctr=p+np.einsum("nij,j->ni",R,shape["xyz"])
            if shape["type"]=="cylinder":
                radius=shape["radius"]
                length=shape["length"]
                if radius is None or length is None or radius<=0 or length<=0:
                    raise ValueError("BAD_CYLINDER_GEOMETRY")
                axis=np.einsum("nij,j->ni",R,shape["R"][:,2])
                half=.5*length*axis
                individual.append((ctr-half,ctr+half,radius))
            elif shape["type"]=="sphere":
                radius=spheres.get(name)
                if radius is None or not math.isfinite(radius) or radius<=0:
                    raise ValueError("BAD_URDF_SPHERE")
                individual.append((ctr,ctr,radius))
            else:
                raise ValueError("UNSUPPORTED_COLLIDER_NOT_CLAIMED_SAFE:"+shape["type"])
        result[name]=individual
    return result


def closest_point_to_segment(points, start, end) -> np.ndarray:
    points,start,end=(np.asarray(x,dtype=float) for x in (points,start,end))
    if (points.ndim!=2 or points.shape[1]!=3 or
        start.shape!=points.shape or end.shape!=points.shape or
        not all(np.isfinite(x).all() for x in (points,start,end))):
        raise ValueError("BAD_POINT_SEGMENT")
    direction=end-start
    denom=np.sum(direction*direction,axis=1)
    t=np.clip(np.divide(np.sum((points-start)*direction,axis=1),
                        denom,out=np.zeros_like(denom),
                        where=denom>1e-15),0,1)
    return np.linalg.norm(points-(start+t[:,None]*direction),axis=1)


def conservative_gap(a,b) -> np.ndarray:
    p0,p1,ra=a
    q0,q1,rb=b
    for x in (p0,p1,q0,q1):
        if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
            raise ValueError("BAD_CAPSULE_ENDPOINTS")
    if not (p0.shape==p1.shape==q0.shape==q1.shape):
        raise ValueError("CAPSULE_LENGTH_MISMATCH")
    if not all(math.isfinite(x) and x>0 for x in (ra,rb)):
        raise ValueError("NONPOSITIVE_CAPSULE_RADIUS")
    ad=np.linalg.norm(p1-p0,axis=1)
    bd=np.linalg.norm(q1-q0,axis=1)
    if np.all(ad<=1e-14) and np.all(bd<=1e-14):
        gap=np.linalg.norm(p0-q0,axis=1)
    elif np.all(ad<=1e-14):
        gap=closest_point_to_segment(p0,q0,q1)
    elif np.all(bd<=1e-14):
        gap=closest_point_to_segment(q0,p0,p1)
    elif np.all(ad>1e-10) and np.all(bd>1e-10):
        gap=kinematics.segment_distance(p0,p1,q0,q1)
    else:
        raise ValueError("INCONSISTENT_OR_DEGENERATE_SHAPE")
    out=gap-ra-rb
    if not np.isfinite(out).all():
        raise ValueError("NONFINITE_BOUND")
    return out


def risk_label(min_gap_m:float) -> str:
    if not math.isfinite(min_gap_m):
        raise ValueError("NONFINITE_GEOMETRY_GAP")
    if min_gap_m<=0:
        return "OVERAPPROXIMATION_OVERLAP_UNKNOWN_NOT_NATIVE_CONTACT"
    if min_gap_m<=.02:
        return "POSITIVE_GAP_WITHIN_20MM_WARNING"
    if min_gap_m<=.05:
        return "POSITIVE_GAP_WITHIN_50MM_WATCH"
    return "URDF_PRIMITIVES_DISJOINT_AT_OBSERVED_SAMPLE_ONLY"


def audit_case(cfg, old_cfg, tree, spheres, pairs, spec):
    name=spec["case"]
    rawpath=Path(old_cfg["data_root"])/("v054_"+name+"_contact_diagnostics.json")
    historical.require_sha(rawpath,old_cfg["input_sha256"][rawpath.name])
    raw=json.loads(rawpath.read_text())
    values=historical.validate_rows(raw,spec["expected_trace_rows"],cfg["observation_dt_s"])
    t,q=values["t"],values["q"]
    if len(t)!=cfg["reference_steps_per_case"][name]:
        raise ValueError("FROZEN_PHYSICS_COUNT_CHANGED")
    poses=kinematics.fk_targets(tree,q,set(n for p in pairs for n in p))
    shapes=primitive_capsules(tree,poses,spheres)
    event_start,event_end=cfg["attribution_window_s"]
    pre=(t>=event_start-1e-6)&(t<=event_end+1e-6)
    if pre.sum()!=13:
        raise ValueError("MISSING_TRUE_5MS_EVENT_CONTEXT")
    at=np.flatnonzero(np.abs(t-5.135)<1e-5)
    if len(at)!=1:
        raise ValueError("MISSING_5_135_REAL_PHYSICS_STEP")
    event_s=old_cfg["true_nonfoot_event_t_s"]
    if name=="cmu_10_05" and (np.abs(t-event_s)<1e-5).any():
        raise ValueError("SYNTHETICALLY_INSERTED_5_145_STATE")
    pairs_out=[]
    near_curves={}
    for a,b in pairs:
        # Union of all child shapes: minimum of all per-shape valid lower bounds.
        # Dropping a single foot or torso shape would produce false exclusions.
        gap=np.minimum.reduce([conservative_gap(sa,sb)
               for sa in shapes[a] for sb in shapes[b]])
        if gap.shape!=t.shape:
            raise ValueError("INCOMPLETE_PAIR_TRACE")
        idx=int(np.argmin(gap))
        event_idx=int(np.argmin(gap[pre]))
        event_j=np.flatnonzero(pre)[event_idx]
        w=float(gap[event_j])
        near_curves[a+"__"+b]=gap[pre].tolist()
        row=dict(pair=[a,b],whole_min_lower_bound_m=float(gap[idx]),
                 whole_min_at_real_time_s=float(t[idx]),
                 near_event_min_lower_bound_m=w,
                 near_event_min_at_real_time_s=float(t[event_j]),
                 at_5_135_s_lower_bound_m=float(gap[at[0]]),
                 num_20mm_alert_physics_steps=int((gap<=.02).sum()),
                 num_50mm_watch_physics_steps=int((gap<=.05).sum()),
                 near_event_risk=risk_label(w))
        pairs_out.append(row)
    sorted_near=sorted(pairs_out,key=lambda r:(r["near_event_min_lower_bound_m"],r["pair"]))
    sorted_full=sorted(pairs_out,key=lambda r:(r["whole_min_lower_bound_m"],r["pair"]))
    target=["left_hip_roll_link","left_wrist_yaw_link"]
    wrist_hip=next((r for r in pairs_out if r["pair"]==target),None)
    if wrist_hip is None:
        raise ValueError("MISSING_PREVIOUSLY_HYPOTHESIZED_PAIR")
    top=[{**r,"rank":i+1} for i,r in enumerate(sorted_near[:15])]
    plot_pairs=[target, ["left_shoulder_yaw_link","torso_link"]]
    curves={a+"__"+b:near_curves[a+"__"+b] for a,b in plot_pairs}
    timeline=[float(x) for x in t[pre]]
    if len(timeline)!=13 or any(len(curve)!=13 for curve in curves.values()):
        raise ValueError("BROKEN_MEASURED_EVENT_CURVES")
    return dict(
      case=name,source_sha256=old_cfg["input_sha256"][rawpath.name],
      physical_states=len(t),min_time_s=float(t[0]),max_time_s=float(t[-1]),
      all_evaluated_pair_count=len(pairs),
      entire_horizon_pairs_with_nonpositive_capsule_bound=sum(
          r["whole_min_lower_bound_m"]<=0 for r in pairs_out),
      near_event_pairs_20mm_or_less=sum(r["near_event_min_lower_bound_m"]<=.02 for r in pairs_out),
      near_event_pairs_nonpositive_capsule_bound=sum(
          r["near_event_min_lower_bound_m"]<=0 for r in pairs_out),
      near_event_pairs_watch_50mm_or_less=sum(r["near_event_min_lower_bound_m"]<=.05 for r in pairs_out),
      near_event_top_15=top,
      sampled_event_window=dict(time_s=timeline,
                    selected_URDF_lower_bounds_m=curves,
                    sensor_event_s=float(event_s),
                    sensor_event_unobserved_for_failed_case=name=="cmu_10_05"),
      whole_horizon_top_10=sorted_full[:10],
      wrist_hip=wrist_hip,
      wrist_hip_near_event_rank=next(
          i+1 for i,r in enumerate(sorted_near) if r["pair"]==target),
      measured_nonfoot_force_at_5_135_N=float(values["nonfoot"][at[0]]),
      measured_original29_max_speed_limit_ratio=float(
          (np.abs(values["dq"])/values["limits"]).max()),
      COM_xyz_at_5_135_s=[float(x) for x in values["com"][at[0]]],
      foot_support_N_at_5_135_s=[float(x) for x in values["support"][at[0]]],
      no_real_joint_sample_at_sensor_5_145=name=="cmu_10_05",
      no_native_PhysX_pair_witness=True,
      excluded_topologically_adjacent_pairs_not_declared_safe=True)


def evaluate():
    cfg=frozen()
    old=historical.frozen_protocol()
    if old["version"]!="v0.1.128":
        raise ValueError("SOURCE_PROTOCOL_NOT_V128")
    # Independently revalidate original v52/v54 OFF/ON physical-field parity.
    original=historical.run(old)
    if not original["original_read_only_instrumentation_all_common_physics_fields_identical"]:
        raise ValueError("ORIGINAL_OFF_ON_INSTRUMENTATION_CHANGED_PHYSICS")
    urdf=Path(old["original_G1_urdf"])
    historical.require_sha(urdf,old["original_G1_urdf_sha256"])
    tree=kinematics.parse_tree(urdf)
    shapes={}
    for ss in tree["links"].values():
        for s in ss:
            shapes[s["type"]]=shapes.get(s["type"],0)+1
    if shapes!=cfg["original_urdf_collision_shapes_expected"]:
        raise ValueError("CHANGED_OR_UNSUPPORTED_ORIGINAL_URDF_SHAPES")
    spheres=sphere_radii(urdf)
    pairs,excluded=eligible_pairs(tree,cfg["min_kinematic_graph_edges_for_evaluated_pair"])
    cases={s["case"]:audit_case(cfg,old,tree,spheres,pairs,s)
           for s in old["original_v54_cases"]}
    actual=sum(c["physical_states"] for c in cases.values())
    if actual!=6628:
        raise ValueError("ORIGINAL_PHYSICS_DATA_LENGTH_DRIFT")
    return dict(version="v0.1.132",freeze_commit=PROTOCOL_SHA,
        source_v128_protocol_commit=historical.FROZEN_SHA,
        source_base_sha=BASE_SHA, source_urdf_sha256=old["original_G1_urdf_sha256"],
        original_OFF_ON_real_physics_parity=True,
        real_200hz_states_revalidated=actual,
        new_physical_episodes=0,new_motor_commands=0,
        original_29dof=True,geometric_shapes=shapes,
        evaluated_nonadjacent_URDF_link_pairs=len(pairs),
        excluded_near_kinematic_URDF_pairs=excluded,
        excluded_pairs_not_qualified_as_safe=True,
        exact_runtime_USD_shape_and_contact_manifold_unknown=True,
        nonpositive_outer_capsule_gap_is_not_contact=True,
        cannot_interpolate_unobserved_sensor_event=True,
        cases=cases)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("WRITE_ONCE_OUTPUT_ALREADY_EXISTS")
    result=evaluate()
    with args.output.open("x") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"output":str(args.output),
           "frames":result["real_200hz_states_revalidated"],
           "pair_count":result["evaluated_nonadjacent_URDF_link_pairs"],
           "excluded":result["excluded_near_kinematic_URDF_pairs"],
           "cases":{name:{"near_event_20mm_pairs":r["near_event_pairs_20mm_or_less"],
                           "near_event_nonpositive":r["near_event_pairs_nonpositive_capsule_bound"],
                           "wrist_hip_rank":r["wrist_hip_near_event_rank"],
                           "wrist_hip_min_mm":r["wrist_hip"]["near_event_min_lower_bound_m"]*1000,
                           "top5":r["near_event_top_15"][:5]}
                        for name,r in result["cases"].items()}},indent=2))

if __name__=="__main__":
    main()
