#!/usr/bin/env python3
"""v92 physically qualified dynamic-ball vs kinematic-box PhysX contact probe.

Never loads or modifies original G1 USD. Off/on pairs have identical mechanics.
A witnessed tensor contact is NOT equivalent to native collider manifold truth.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def vector(x, n=3):
    if x is None:
        return None
    try:
        result=[float(x[i]) for i in range(n)]
        return result if all(math.isfinite(v) for v in result) else None
    except (TypeError, ValueError, IndexError, AttributeError):
        return None

def native_positive(event):
    actors=event.get("actors", [])
    colliders=event.get("colliders", [])
    if not (len(actors)==len(colliders)==2):
        return False
    for paths in (actors, colliders):
        if not all(any(name in path for path in paths)
                   for name in ("ProbeBall", "ProbeBox")):
            return False
    return any(p.get("position") is not None and
               p.get("normal") is not None and
               sum(x*x for x in p["normal"])>.1 and
               ((p.get("impulse") is not None and sum(x*x for x in p["impulse"])>1e-8)
                or (isinstance(p.get("impulse_scalar"), (float,int))
                    and p["impulse_scalar"]>1e-8))
               for p in event.get("points",[]))

def main(args):
    from isaaclab.app import AppLauncher
    print("V095_APP_ENTER",flush=True)
    app=AppLauncher(headless=True).app
    simulation=None;ball=None;box=None;view=None;sub=None
    evidence=dict(version="v0.1.95",device=args.device,report_mode=args.report_mode,
                  capture=args.capture,subscribe_phase=args.subscribe_phase,
                  layout=args.layout,steps_requested=args.steps,original_G1=False,
                  original_controller_modified=False,dt_s=.005,errors=[])
    try:
        import omni.usd,omni.physx
        import isaaclab.sim as su
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from pxr import UsdPhysics,PhysxSchema,PhysicsSchemaTools
        simulation=su.SimulationContext(su.SimulationCfg(
            dt=.005,device=args.device,gravity=(0.,0.,-9.81)))
        ball=RigidObject(RigidObjectCfg(
            prim_path="/World/ProbeBall",
            spawn=su.SphereCfg(radius=.18,activate_contact_sensors=False,
                rigid_props=su.RigidBodyPropertiesCfg(),
                collision_props=su.CollisionPropertiesCfg(),
                mass_props=su.MassPropertiesCfg(mass=1.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.95))))
        offset=5. if args.layout=="separated" else 0.
        box=RigidObject(RigidObjectCfg(
            prim_path="/World/ProbeBox",
            spawn=su.CuboidCfg(size=(1.0,1.0,.40),activate_contact_sensors=False,
                rigid_props=su.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,disable_gravity=True),
                collision_props=su.CollisionPropertiesCfg(),
                mass_props=su.MassPropertiesCfg(mass=1000.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(offset,0.,0.))))
        stage=omni.usd.get_context().get_stage()
        bprim=stage.GetPrimAtPath("/World/ProbeBall")
        fprim=stage.GetPrimAtPath("/World/ProbeBox")
        if args.report_mode=="on":
            for prim in (bprim,fprim):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        all_colliders={name:[str(prim.GetPath()) for prim in stage.Traverse()
           if (str(prim.GetPath())=="/World/"+name or str(prim.GetPath()).startswith("/World/"+name+"/"))
           and prim.HasAPI(UsdPhysics.CollisionAPI)] for name in ("ProbeBall","ProbeBox")}
        evidence["stage"]={"all_colliders":all_colliders,
            "ball_rigid":bprim.HasAPI(UsdPhysics.RigidBodyAPI),
            "box_rigid":fprim.HasAPI(UsdPhysics.RigidBodyAPI),
            "ball_collision":bprim.HasAPI(UsdPhysics.CollisionAPI),
            "box_collision":fprim.HasAPI(UsdPhysics.CollisionAPI),
            "ball_report":bprim.HasAPI(PhysxSchema.PhysxContactReportAPI),
            "box_report":fprim.HasAPI(PhysxSchema.PhysxContactReportAPI),
            "expected_pair_paths":[str(bprim.GetPath()),str(fprim.GetPath())]}
        native=omni.physx.get_physx_simulation_interface()
        counts={"callback_calls":0,"callback_headers":0,"poll_calls":0,
                "poll_headers":0,"native_pair_events":0}
        events=[];step=[0]
        def ingest(headers,points,origin):
            for h in headers:
                try:
                    paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k)))
                        for k in ("actor0","actor1","collider0","collider1")]
                    off=int(h.contact_data_offset);count=int(h.num_contact_data)
                    arr=[]
                    for p in points[off:off+min(count,10)]:
                        impulse=getattr(p,"impulse",None)
                        arr.append(dict(position=vector(getattr(p,"position",None)),
                          normal=vector(getattr(p,"normal",None)),
                          impulse=vector(impulse),
                          impulse_scalar=float(impulse) if isinstance(impulse,(int,float)) else None))
                    evt={"step":step[0],"via":origin,"actors":paths[:2],
                         "colliders":paths[2:],"points":arr,"contact_count":count}
                    if any("ProbeBall" in v for v in paths) and any("ProbeBox" in v for v in paths):
                        counts["native_pair_events"]+=1
                        if len(events)<32:events.append(evt)
                except Exception as exc:
                    if len(evidence["errors"])<8:evidence["errors"].append("native_parse:"+repr(exc)[:200])
        def callback(headers,points):
            counts["callback_calls"]+=1
            counts["callback_headers"]+=len(headers)
            ingest(headers,points,"callback")
        if args.capture in ("native","both") and args.subscribe_phase=="pre_reset":
            sub=native.subscribe_contact_report_events(callback)
        print("V095_USD",json.dumps(evidence["stage"]),flush=True)
        simulation.reset()
        ball.write_root_state_to_sim(ball.data.default_root_state.clone())
        ball.reset();box.reset();ball.update(0.);box.update(0.)
        print("V095_RESET_OK",flush=True)
        if args.capture in ("native","both") and args.subscribe_phase=="post_reset":
            sub=native.subscribe_contact_report_events(callback)
            print("V095_SUBSCRIBED_AFTER_RESET",flush=True)
        print("V095_PRE_VIEW",args.capture,flush=True)
        if args.capture in ("tensor","both"):
            view=ball._physics_sim_view.create_rigid_contact_view(
                "/World/ProbeBall",["/World/ProbeBox"],max_contact_data_count=128)
            evidence["contact_view_shape"]={
                "bodies":int(view.sensor_count),"filters":int(view.filter_count),
                "sensor_paths":[str(x) for x in view.sensor_paths],
                "filter_paths":[str(x) for x in view.filter_paths]}
        else:
            evidence["contact_view_shape"]=None
        print("V095_VIEW_READY",json.dumps(evidence["contact_view_shape"]),flush=True)
        trace=[];positive=[];peak_matrix=0.;nonzero_steps=0;counts_sum=0;net_peak_N=0.
        for i in range(args.steps):
            step[0]=i+1
            ball.write_data_to_sim();box.write_data_to_sim()
            simulation.step(render=False);ball.update(.005);box.update(.005)
            z=float(ball.data.root_pos_w[0,2].item())
            vz=float(ball.data.root_lin_vel_w[0,2].item())
            boxpos=[float(x) for x in box.data.root_pos_w[0].cpu().tolist()]
            if not math.isfinite(z+vz):raise RuntimeError("NONFINITE_ACTUAL_RUNTIME_POSE")
            trace.append([i+1,round(z,9),round(vz,9)])
            if view is not None:
              try:
                net=view.get_net_contact_forces(.005)
                net_peak_N=max(net_peak_N,float(net.abs().max().item()))
                fm=view.get_contact_force_matrix(.005)
                force=float(fm.abs().max().item())
                peak_matrix=max(peak_matrix,force)
                if force>.01:nonzero_steps+=1
                forces,positions,normals,separations,k,starts=view.get_contact_data(.005)
                k0=int(k.reshape(-1)[0].item())
                begin=int(starts.reshape(-1)[0].item())
                counts_sum+=k0
                if k0>0 and len(positive)<20:
                    for idx in range(begin,min(begin+k0,begin+3)):
                        p=vector(positions[idx]);n=vector(normals[idx])
                        imp=float(sum(float(v)**2 for v in forces[idx].reshape(-1).tolist())**.5)
                        gap=float(separations[idx].reshape(-1)[0].item())
                        if p and n and math.isfinite(imp) and math.isfinite(gap):
                            positive.append(dict(step=i+1,point_w=p,normal_w=n,
                                                 force_magnitude_N=imp,separation_m=gap))
              except Exception as exc:
                if len(evidence["errors"])<8:evidence["errors"].append("tensor:"+repr(exc)[:230])
            if args.capture in ("native","both"):
                hs,ps=native.get_contact_report()
                counts["poll_calls"]+=1;counts["poll_headers"]+=len(hs)
                ingest(hs,ps,"poll")
            if (i+1)%60==0:print("V095_STEP",i+1,round(z,5),round(peak_matrix,3),flush=True)
        # Actual runtime pose is the authoritative evidence of contact/noncontact,
        # tensor positive requires an additional finite nonzero contact observation.
        physical=(min(t[1] for t in trace)<.44 and max(t[1] for t in trace)>.9
                  and abs(trace[-1][1]-.38)<.08)
        separated=(trace[-1][1]<-.5 and max(t[1] for t in trace)>.9
                   and not positive and nonzero_steps==0)
        evidence.update(actual_runtime_frames=len(trace),runtime_z_vz=trace,
            box_position_w=boxpos,tensor_peak_force_N=peak_matrix,
            tensor_net_component_peak_N=net_peak_N,
            tensor_nonzero_force_steps=nonzero_steps,tensor_contact_counts=counts_sum,
            tensor_contact_points=positive,native_counts=counts,native_events=events,
            observed_collision_dynamics=physical if args.layout=="colliding" else False,
            observed_noncollision_dynamics=separated if args.layout=="separated" else False,
            tensor_pair_witness=any(p["force_magnitude_N"]>1e-8 for p in positive),
            native_pair_witness=any(native_positive(e) for e in events),
            original_G1_collision_manifold_certified=False)
        with args.output.open("x") as f:json.dump(evidence,f,indent=2);f.write("\n")
        print("V095_RESULT",json.dumps({k:evidence[k] for k in (
            "actual_runtime_frames","observed_collision_dynamics","observed_noncollision_dynamics",
            "tensor_peak_force_N","tensor_net_component_peak_N","tensor_contact_counts","native_pair_witness",
            "native_counts","errors")}),flush=True)
    except BaseException as exc:
        import traceback
        evidence["exception"]=repr(exc)
        print("V095_EXCEPTION",repr(exc),traceback.format_exc(),flush=True)
        if not args.output.exists():
            with args.output.open("x") as f:json.dump(evidence,f,indent=2);f.write("\n")
        raise
    finally:
        print("V095_CLEANUP_ENTER",flush=True)
        if sub is not None:del sub
        if view is not None:del view
        if box is not None:del box
        if ball is not None:del ball
        if simulation is not None:
            try:simulation.clear_all_callbacks()
            except Exception as exc:print("V095_CLEAR_CALLBACKS_ERROR",repr(exc),flush=True)
            try:simulation.clear_instance()
            except Exception as exc:print("V095_CLEAR_INSTANCE_ERROR",repr(exc),flush=True)
        print("V095_APP_CLOSE_ENTER",flush=True)
        app.close()
        print("V095_APP_CLOSED",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--subscribe-phase",choices=("pre_reset","post_reset"),default="pre_reset")
    p.add_argument("--device",required=True,choices=("cpu","cuda:0"))
    p.add_argument("--report-mode",required=True,choices=("off","on"))
    p.add_argument("--layout",required=True,choices=("colliding","separated"))
    p.add_argument("--capture",required=True,choices=("pose","native","tensor","both"))
    p.add_argument("--steps",type=int,default=180)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.steps!=180 or a.output.exists():p.error("FROZEN_180_STEPS_WRITE_ONCE")
    if (a.report_mode=="off" and a.capture!="pose") or (a.report_mode=="on" and a.capture=="pose"):
        p.error("INVALID_OFF_ON_REPORTER_CONTRACT")
    main(a)
