#!/usr/bin/env python3
"""Filtered original-PhysX rigid-pair contact diagnostic; no USD/model mutation.

A PhysX rigid contact *view* is created, not a PhysxContactReportAPI;
both sides of the pair are queried independently. The view may influence
collision reporting, so ONLY exact off/tensor state parity establishes that
the *tested* recording has not been changed.

An empty tensor report is UNKNOWN. A filtered contact with finite geometry
supports a rigid-body-level pair, NOT a native collider/path manifold.
"""
from __future__ import annotations

import math

PAIR=("left_wrist_yaw_link", "left_hip_roll_link")
POSITIVE=("right_ankle_roll_link", "SoccerBall")
PHYSICS_DT=.005
MAX_SAVED=48

def tensor_contacts(force, position, normal, separation, counts, starts, cap=MAX_SAVED):
    """Convert one observed filter pair without using cross-pair force symmetry."""
    import numpy as np
    def arr(x):return np.asarray(x.detach().cpu() if hasattr(x,"detach") else x)
    f,p,n,s,k,b=map(arr,(force,position,normal,separation,counts,starts))
    if k.size!=1 or b.size!=1:raise ValueError("PAIR_EXPECTS_ONE_ORIGINAL_G1_ENV_AND_ONE_FILTER")
    count=int(k.reshape(-1)[0]);start=int(b.reshape(-1)[0])
    if count<0 or start<0 or start+count>len(f):
        raise ValueError("INVALID_PHYSX_CONTACT_SEGMENT")
    result=[]
    for idx in range(start,start+min(count,cap)):
        point=[float(x) for x in p[idx].reshape(-1)]
        direction=[float(x) for x in n[idx].reshape(-1)]
        gap=float(s[idx].reshape(-1)[0])
        impulse=float(f[idx].reshape(-1)[0])
        if len(point)!=3 or len(direction)!=3 or not all(map(math.isfinite,point+direction+[gap,impulse])):
            raise ValueError("NONFINITE_OR_INVALID_PHYSX_CONTACT_GEOMETRY")
        result.append(dict(point_w=point,normal_w=direction,
                           separation_m=gap,impulse_magnitude=impulse))
    return count,result

def direct_tensor_pair_contact(e):
    """Require single filtered view and complete, finite and positive contact."""
    if not isinstance(e,dict) or e.get("target_pair")!=list(PAIR):return False
    if e.get("direction") not in ("wrist_to_hip","hip_to_wrist"):return False
    for c in e.get("contacts",()):
        if not isinstance(c,dict):continue
        p=c.get("point_w");n=c.get("normal_w");i=c.get("impulse_magnitude")
        if not isinstance(p,list) or not isinstance(n,list) or len(p)!=3 or len(n)!=3:
            continue
        try:
            if i>0 and all(math.isfinite(float(v)) for v in p+n+[i]) and                     sum(float(v)**2 for v in n)>.01:
                return True
        except (ValueError,TypeError):pass
    return False

class OriginalG1TensorSelfPair:
    """Existing original G1 PhysX tensor backend, max 1 env, read-only sampled."""
    def __init__(self,env):
        from pxr import UsdPhysics
        import omni.usd
        self.env=env;self.errors=[];self.calls=0
        self.history={key:[] for key in ("wrist_to_hip","hip_to_wrist","foot_to_ball")}
        self.counts={key:0 for key in self.history}
        self.max_contact_magnitude={key:0. for key in self.history}
        self.sampled_frames={key:0 for key in self.history}
        self.overflow={key:0 for key in self.history}
        self.pair=PAIR
        self.stage_colliders={}
        stage=omni.usd.get_context().get_stage()
        for target in PAIR:
            prefix="/World/envs/env_0/Robot/"+target
            self.stage_colliders[target]=[
                str(prim.GetPath()) for prim in stage.Traverse()
                if (str(prim.GetPath())==prefix or str(prim.GetPath()).startswith(prefix+"/"))
                and prim.HasAPI(UsdPhysics.CollisionAPI)]
        scene=env.scene
        view=scene["contact_forces"]._physics_sim_view
        self.views={
            "wrist_to_hip":view.create_rigid_contact_view(
                "/World/envs/env_*/Robot/"+PAIR[0],
                ["/World/envs/env_*/Robot/"+PAIR[1]],max_contact_data_count=64),
            "hip_to_wrist":view.create_rigid_contact_view(
                "/World/envs/env_*/Robot/"+PAIR[1],
                ["/World/envs/env_*/Robot/"+PAIR[0]],max_contact_data_count=64),
            "foot_to_ball":view.create_rigid_contact_view(
                "/World/envs/env_*/Robot/"+POSITIVE[0],
                ["/World/envs/env_*/SoccerBall"],max_contact_data_count=64),
        }
        self.shape={key:{"body_count":int(v.count),"filter_count":int(v.filter_count)}
                    for key,v in self.views.items()}
        for key,st in self.shape.items():
            if st!={"body_count":1,"filter_count":1}:
                raise RuntimeError("PHYSX_PAIR_FILTER_NOT_SINGLE_ORIGINAL_G1:"+key+":"+str(st))

    def capture(self,dt):
        if dt<=0:return
        if abs(float(dt)-PHYSICS_DT)>1e-6:
            raise RuntimeError("ORIGINAL_G1_200HZ_REQUIRED")
        self.calls+=1
        t=float(self.env.sim.current_time)
        for label,view in self.views.items():
            within=(1.35<=t<=1.90) if label=="foot_to_ball" else (4.70<=t<=5.50)
            if not within:continue
            self.sampled_frames[label]+=1
            try:
                count,contacts=tensor_contacts(*view.get_contact_data(dt))
                self.counts[label]+=count
                self.overflow[label]+=max(0,count-MAX_SAVED)
                for c in contacts:
                    self.max_contact_magnitude[label]=max(
                        self.max_contact_magnitude[label],c["impulse_magnitude"])
                if count and len(self.history[label])<MAX_SAVED:
                    self.history[label].append(dict(t_s=t,count=count,contacts=contacts,
                        direction=label,target_pair=list(PAIR) if label!="foot_to_ball"
                        else list(POSITIVE)))
            except Exception as exc:
                if len(self.errors)<8:
                    self.errors.append(label+":"+type(exc).__name__+":"+str(exc)[:200])

    def result(self):
        evidence=[row for label in ("wrist_to_hip","hip_to_wrist")
                  for row in self.history[label] if direct_tensor_pair_contact(row)]
        positive=any(any(c["impulse_magnitude"]>0 for c in row.get("contacts",[]))
                     for row in self.history["foot_to_ball"])
        return dict(version="v0.1.81",mode="original_G1_filtered_PhysX_tensor_contacts",
            physically_read_only_requests=True,creates_filtered_contact_views=True,
            modifies_USD_or_controller=False,creates_native_contact_report_api=False,
            parent_collision_prim_inventory=self.stage_colliders,
            view_shapes=self.shape,physics_dt_s=PHYSICS_DT,physics_frames_seen=self.calls,
            sampled_frames=self.sampled_frames,raw_contact_counts=self.counts,
            sampled_max_reported_impulse=self.max_contact_magnitude,
            recorded_contact_events=self.history,overflow=self.overflow,errors=self.errors,
            foot_ball_positive_control_observed=positive,
            direct_filtered_rigid_pair_witness=bool(evidence and not self.errors),
            native_collider_pair_manifold_certified=False,
            scientific_conclusion=("FILTERED_RIGID_BODY_PAIR_SUPPORTED_NOT_NATIVE_COLLIDER_PATH"
                if evidence and not self.errors else "UNKNOWN_NO_POSITIVE_DIRECT_TENSOR_PAIR"),
            limitation="Tensor view filtering names are not the native actor0/1 collider0/1; if empty, cannot prove no self-collision.")
