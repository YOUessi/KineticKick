#!/usr/bin/env python3
"""Isolated, PhysX-tensor-proven contact positive control, NOT a G1 episode."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

def xyz(v):
    if v is None:
        return None
    try:
        result = [float(v[i]) for i in range(3)]
        return result if all(math.isfinite(x) for x in result) else None
    except (ValueError, TypeError, IndexError, AttributeError):
        return None

def pair_has_paths(event):
    a, c = event.get("actors", ()), event.get("colliders", ())
    names = ("KickProbeBall", "Ground")
    return (len(a) == 2 and len(c) == 2
            and all(any(n in p for p in a) for n in names)
            and all(any(n in p for p in c) for n in names))

def validated_point(e):
    if not pair_has_paths(e):
        return False
    for p in e.get("points", ()):
        normal, pos = p.get("normal"), p.get("position")
        impulse, scalar = p.get("impulse_w"), p.get("impulse_scalar")
        norm_ok = normal is not None and sum(x*x for x in normal) > .1
        imp_ok = (impulse is not None and sum(x*x for x in impulse) > 1e-8 or
                  isinstance(scalar, (int, float)) and math.isfinite(scalar) and scalar > 1e-8)
        if pos is not None and norm_ok and imp_ok:
            return True
    return False

def main(args):
    from isaaclab.app import AppLauncher
    print("V088_APP_ENTER", flush=True)
    app = AppLauncher(headless=True).app
    sim = None; body = None; view = None; subscription = None
    result = {"version":"v0.1.88","device":args.device,"mode":args.mode,"dt_s":.005,
              "original_G1_trial":False,"original_G1_scene_modified":False,"errors":[]}
    try:
        import omni.usd, omni.physx
        import isaaclab.sim as su
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
        sim = su.SimulationContext(su.SimulationCfg(dt=.005,device=args.device,gravity=(0,0,-9.81)))
        ground_cfg = su.GroundPlaneCfg()
        ground_cfg.func("/World/Ground",ground_cfg)
        body = RigidObject(RigidObjectCfg(
            prim_path="/World/KickProbeBall",
            spawn=su.SphereCfg(radius=.2,activate_contact_sensors=True,
                rigid_props=su.RigidBodyPropertiesCfg(),
                collision_props=su.CollisionPropertiesCfg(),
                mass_props=su.MassPropertiesCfg(mass=1.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.85))))
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/KickProbeBall")
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(0.)
        if args.mode == "authored_empty":
            api.CreateReportPairsRel().SetTargets([])
        rel=api.GetReportPairsRel()
        ground_paths=[str(p.GetPath()) for p in stage.Traverse()
                      if str(p.GetPath()).startswith("/World/Ground/") and p.HasAPI(UsdPhysics.CollisionAPI)]
        result["usd_inventory"] = {
            "ground_collision_paths":ground_paths,
            "ball_contact_api":bool(prim.HasAPI(PhysxSchema.PhysxContactReportAPI)),
            "ball_rigid_body":bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            "reportPairs_authored":bool(rel and rel.IsValid() and rel.IsAuthored()),
            "reportPairs_targets":[str(x) for x in rel.GetTargets()] if rel and rel.IsValid() else None}
        if not ground_paths:
            raise RuntimeError("GROUND_COLLIDER_MISSING")
        counts={"callback_calls":0,"callback_headers":0,"poll_calls":0,"poll_headers":0,"candidate_events":0}
        events=[]; current_step=[0]
        def ingest(headers, points, source):
            for h in headers:
                try:
                    paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k)))
                           for k in ("actor0","actor1","collider0","collider1")]
                    n=int(h.num_contact_data); off=int(h.contact_data_offset)
                    pp=[]
                    for pt in points[off:off+min(n,16)]:
                        raw=getattr(pt,"impulse",None)
                        scalar=float(raw) if isinstance(raw,(int,float)) else None
                        pp.append({"position":xyz(getattr(pt,"position",None)),
                                   "normal":xyz(getattr(pt,"normal",None)),
                                   "impulse_w":xyz(raw),"impulse_scalar":scalar})
                    item={"step":current_step[0],"source":source,"actors":paths[:2],
                          "colliders":paths[2:],"points":pp,"num_contact_data":n}
                    if pair_has_paths(item):
                        counts["candidate_events"]+=1
                        if len(events)<24:events.append(item)
                except Exception as ex:
                    if len(result["errors"])<8:result["errors"].append("header:"+repr(ex)[:220])
        def cb(headers, points):
            counts["callback_calls"]+=1;counts["callback_headers"]+=len(headers)
            ingest(headers,points,"callback")
        subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(cb)
        print("V088_USD_READY",json.dumps(result["usd_inventory"]),flush=True)
        sim.reset()
        body.write_root_state_to_sim(body.data.default_root_state.clone())
        body.reset();body.update(0.)
        print("V088_SIM_RESET",flush=True)
        view=body._physics_sim_view.create_rigid_contact_view(
            "/World/KickProbeBall",ground_paths,max_contact_data_count=128)
        z=[]; speeds=[]; peaks=[]; ncontacts=0; capacity=False
        dt=.005
        for i in range(args.steps):
            current_step[0]=i+1
            body.write_data_to_sim()
            sim.step(render=False);body.update(dt)
            z_now=float(body.data.root_pos_w[0,2].item())
            vz_now=float(body.data.root_lin_vel_w[0,2].item())
            if not math.isfinite(z_now) or not math.isfinite(vz_now):
                raise RuntimeError("NONFINITE_TENSOR_POSE")
            if i in (0,9,19,39,59,79,99,119,149,179):
                z.append([i+1,z_now]);speeds.append([i+1,vz_now])
            forces=view.get_contact_force_matrix(dt)
            mag=float(forces.abs().max().item())
            if not math.isfinite(mag):
                raise RuntimeError("NONFINITE_TENSOR_FORCE")
            peaks.append(mag)
            if mag>1e-3:ncontacts+=1
            native=omni.physx.get_physx_simulation_interface()
            headers,points=native.get_contact_report()
            counts["poll_calls"]+=1; counts["poll_headers"]+=len(headers)
            ingest(headers,points,"poll")
            if i%60==0:print("V088_STEP",i+1,round(z_now,4),round(mag,4),flush=True)
        physical = bool(z[0][1]>.6 and min(r[1] for r in z)<.35
                        and max(peaks)>10 and ncontacts>0)
        native_positive=[e for e in events if validated_point(e)]
        result.update(steps=args.steps,tensor_z_samples=z,tensor_vz_samples=speeds,
            force_max_N=max(peaks),contact_steps=ncontacts,
            physically_certified=physical,counts=counts,native_events=events,
            native_positive_events=native_positive,
            native_pair_calibrated=bool(physical and native_positive and not result["errors"]),
            original_G1_wrist_hip_manifold_certified=False,
            negative_semantics="No native stream does not imply absence of physical contact")
        with args.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
        print("V088_PHYSICS_RESULT",json.dumps({k:result[k] for k in
             ("physically_certified","force_max_N","contact_steps","counts",
              "native_pair_calibrated","errors")}),flush=True)
    finally:
        print("V088_CLEANUP_ENTER",flush=True)
        if subscription is not None: del subscription
        if view is not None: del view
        if body is not None: del body
        if sim is not None:
            try:sim.clear_all_callbacks()
            except Exception as exc:print("V088_CLEAR_CALLBACK_ERROR",repr(exc),flush=True)
            try:sim.clear_instance()
            except Exception as exc:print("V088_CLEAR_INSTANCE_ERROR",repr(exc),flush=True)
        print("V088_APP_CLOSE_ENTER",flush=True)
        app.close()
        print("V088_APP_CLOSED",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--device",required=True,choices=["cpu","cuda:0"])
    p.add_argument("--mode",required=True,choices=["wildcard","authored_empty"])
    p.add_argument("--steps",type=int,default=180)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.steps!=180:p.error("FROZEN_STEPS_180")
    if a.output.exists():p.error("WRITE_ONCE_EXISTS")
    main(a)
