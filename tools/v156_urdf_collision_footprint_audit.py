#!/usr/bin/env python3
"""Source-qualified URDF ankle collision cylinder envelope vs v155 toy feet.

Bounding box is a conservative OUTER envelope of URDF collision primitives
in ankle-link coordinates, NOT a certified sole support region, contact patch,
PhysX collider material or ground-reaction pressure polygon.
"""
from __future__ import annotations
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from v156_real_urdf_pin_model import frozen, urdf_path, FEET

def rpy_rot(rpy):
    roll,pitch,yaw=[float(x) for x in rpy]
    cr,sr=math.cos(roll),math.sin(roll)
    cp,sp=math.cos(pitch),math.sin(pitch)
    cy,sy=math.cos(yaw),math.sin(yaw)
    return np.array([
        [cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
        [-sp,cp*sr,cp*cr]
    ],dtype=float)

def foot_collision_bbox(urdf,link):
    root=ET.parse(urdf).getroot()
    elt=root.find(f"./link[@name='{link}']")
    if elt is None:raise ValueError("V156_MISSING_ANKLE_COLLISION_LINK")
    colliders=elt.findall("./collision")
    if not colliders:
        raise ValueError("V156_NO_SOLE_COLLIDERS")
    lo=np.full(3,np.inf);hi=np.full(3,-np.inf);items=[]
    for c in colliders:
        o=c.find("origin")
        if o is None:raise ValueError("V156_COLLIDER_MISSING_ORIGIN")
        xyz=np.array([float(x) for x in o.attrib.get("xyz","0 0 0").split()])
        rpy=np.array([float(x) for x in o.attrib.get("rpy","0 0 0").split()])
        cylinder=c.find("./geometry/cylinder")
        if cylinder is None:
            raise ValueError("V156_UNACCOUNTED_NONCYLINDER_GEOMETRY")
        radius=float(cylinder.attrib["radius"]);length=float(cylinder.attrib["length"])
        if xyz.shape!=(3,) or rpy.shape!=(3,) or radius<=0 or length<=0:
            raise ValueError("V156_BAD_COLLIDER_MEASURE")
        axis=rpy_rot(rpy)[:,2]
        half=np.abs(axis)*(length/2)+radius*np.sqrt(np.maximum(0,1-axis**2))
        lo=np.minimum(lo,xyz-half);hi=np.maximum(hi,xyz+half)
        items.append({"center_xyz_m":xyz.tolist(),"axis_link":axis.tolist(),
                      "radius_m":radius,"length_m":length,"bbox_halfwidth_m":half.tolist()})
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo<hi)):
        raise ValueError("V156_NONFINITE_URDF_ENVELOPE")
    return dict(link=link,collider_count=len(items),enclosing_min_xyz_m=lo.tolist(),
                enclosing_max_xyz_m=hi.tolist(),sources=items,
                outer_bbox_not_true_support_polygon=True,
                no_physx_ground_calibration=True)

def footprint_audit():
    cfg=frozen()
    urdf=urdf_path(cfg)
    feet={link:foot_collision_bbox(urdf,link) for link in FEET}
    fixture=json.loads((Path(__file__).resolve().parents[1]/
                        "configs/v155_frozen_patched_stance_wrench.json").read_text())
    L=float(fixture["fixture_half_length_m"]);W=float(fixture["fixture_half_width_m"])
    points=[(-L,-W),(-L,W),(L,-W),(L,W)]
    for item in feet.values():
        mins=item["enclosing_min_xyz_m"];maxs=item["enclosing_max_xyz_m"]
        item["v155_unmeasured_corner_count_outside_urdf_collision_outer_bbox"] = sum(
            not (mins[0]<=x<=maxs[0] and mins[1]<=y<=maxs[1])
            for x,y in points)
        item["urdf_collision_xy_outer_bbox_m"]=[mins[:2],maxs[:2]]
        item["v155_toy_corners_xy_m"]=[list(p) for p in points]
        item["v155_corner_z_at_link_origin_m"]=0.
        item["observed_urdf_cylinder_lowest_z_m"]=mins[2]
        item["cannot_infer_true_cop_from_bounding_box"]=True
    return {
        "source_urdf_sha256":cfg["source_urdf_sha256"],
        "v155_toy_half_length_m":L,"v155_toy_half_width_m":W,
        "feet":feet,
        "real_physx_episode_count":0,
        "foot_contact_sensor_is_not_ground_unique_grf":True,
        "collision_outer_bounds_are_not_physical_ground_support_truth":True,
    }
