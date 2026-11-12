#!/usr/bin/env python3
"""Read-only filtered PhysX tensor G1 partner diagnostic, no controller changes.

Outputs are *tensor-view* evidence limited to configured rigid-body pair paths;
do NOT relabel it as native actor/collider/contact manifold evidence. A zero
report in an uncalibrated G1 tensor filter is UNKNOWN, not a noncollision proof.
"""
from __future__ import annotations
import numpy as np
from pxr import UsdPhysics
import omni.usd

PAIRS=(("left_wrist_yaw_link","left_hip_roll_link"),
       ("right_wrist_yaw_link","right_hip_roll_link"))
class G1PairDiagnostic:
 def __init__(self,env):
    stage=omni.usd.get_context().get_stage()
    candidates={}
    for prim in stage.Traverse():
        p=str(prim.GetPath())
        if "/Robot/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        for pair in PAIRS:
            for body in pair:
                if p.endswith("/"+body):candidates.setdefault(body,[]).append(p)
    self.results={}
    self.views={}
    phys=env.scene["contact_forces"]._physics_sim_view
    for a,b in PAIRS:
        label=a+"__"+b
        row={"observer_type":"READ_ONLY_FILTERED_TENSOR_VIEW",
             "sensor_link":a,"partner_link":b,"native_actor_collider_manifold":False,
             "actual_contact_pair_certified":False,"physical_states_mutated":False,
             "samples":0,"nonzero_force_frames":0,"observed_data_count":0,
             "peak_force_N":0.,"sample_contact_points":[],"status":"UNKNOWN"}
        pa,pb=candidates.get(a,[]),candidates.get(b,[])
        row["sensor_paths_found"]=pa;row["filter_paths_found"]=pb
        if len(pa)!=1 or len(pb)!=1:
            row["status"]="FAIL_CLOSED_AMBIGUOUS_OR_MISSING_RIGID_PATH"
        else:
            try:
                view=phys.create_rigid_contact_view(pa[0],[pb[0]],max_contact_data_count=128)
                row["view_sensor_count"]=int(view.sensor_count)
                row["view_filter_count"]=int(view.filter_count)
                row["resolved_sensor_paths"]=[str(s) for s in view.sensor_paths]
                row["resolved_filter_paths"]=[str(s) for s in view.filter_paths]
                if row["view_sensor_count"]!=1 or row["view_filter_count"]!=1:
                    row["status"]="FAIL_CLOSED_VIEW_NOT_ONE_TO_ONE"
                else:
                    row["status"]="READING_BUT_NOT_NATIVE_MANIFOLD"
                    self.views[label]=view
            except Exception as e:
                row["status"]="FAIL_CLOSED_VIEW_CREATION_ERROR"
                row["error"]=repr(e)[:240]
        self.results[label]=row
 def capture(self,dt):
    if dt<=0:return
    for label,view in list(self.views.items()):
        row=self.results[label];row["samples"]+=1
        try:
            matrix=view.get_contact_force_matrix(dt)
            f=float(matrix.abs().max().item())
            if not np.isfinite(f):raise ValueError("nonfinite")
            row["peak_force_N"]=max(f,row["peak_force_N"])
            if f>1e-5:row["nonzero_force_frames"]+=1
            forces,positions,normals,separations,counts,starts=view.get_contact_data(dt)
            k=int(counts.reshape(-1)[0].item())
            start=int(starts.reshape(-1)[0].item())
            row["observed_data_count"]+=k
            if k>0 and len(row["sample_contact_points"])<20:
                for i in range(start,min(start+k,start+2)):
                    p=np.asarray(positions[i].cpu().tolist(),float).reshape(-1)
                    n=np.asarray(normals[i].cpu().tolist(),float).reshape(-1)
                    magnitude=float(np.linalg.norm(np.asarray(forces[i].cpu().tolist(),float)))
                    if p.size>=3 and n.size>=3 and np.isfinite([*p[:3],*n[:3],magnitude]).all():
                        row["sample_contact_points"].append({
                           "post_step_physics_time_s":round(float(row["samples"])*.005,6),
                           "point_w":p[:3].tolist(),"normal_w":n[:3].tolist(),
                           "force_magnitude_N":magnitude,"origin":"tensor_view_not_native"})
        except Exception as e:
            row["status"]="READBACK_ERROR_UNCERTIFIED"
            row["error"]=repr(e)[:240]
            del self.views[label]
            break
 def result(self):
    return {"pairs":self.results,"original_G1_controller_modified":False,
            "native_actor0_actor1_collider0_collider1_MANIFOLD_certified":False,
            "zero_tensor_data_means":"UNKNOWN_UNLESS_POSITIVE_CONTROL_CERTIFIED"}
