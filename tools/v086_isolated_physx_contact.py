#!/usr/bin/env python3
"""Isolated two-body PhysX contact reporter qualification; NOT an original G1 trial.

Create two fresh USD rigid actors BEFORE SimulationContext.reset; keep the original
G1 scene, its controller, assets, and raw traces entirely untouched.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path


def vec(v):
    if v is None: return None
    try: return [float(v[i]) for i in range(3)]
    except (TypeError, ValueError, IndexError, AttributeError): return None


def valid_pair(event):
    a=event.get("actors", [])
    b=event.get("colliders", [])
    expected={"KickProbeBall", "KickProbeFloor"}
    return (len(a)==2 and len(b)==2 and
            all(any(name in path for path in a) for name in expected) and
            all(any(name in path for path in b) for name in expected))


def positive(event):
    if not valid_pair(event): return False
    for p in event.get("points", []):
        if (p.get("position") is not None and p.get("normal") is not None
            and sum(x*x for x in p["normal"])>0.01
            and p.get("impulse") is not None
            and sum(x*x for x in p["impulse"])>1e-8):
            return True
    return False


def run(args):
    from isaaclab.app import AppLauncher
    print("V086_APP_ENTER", flush=True)
    app=AppLauncher(headless=True).app
    print("V086_APP_READY", flush=True)
    result={"mode":args.mode,"device":args.device,"dt":0.005,
            "original_G1_trial":False,"original_USD_modified":False,
            "calibrated_native_pair":False,"errors":[]}
    try:
        import omni.usd, omni.physx
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
        from isaaclab.sim import SimulationCfg, SimulationContext
        stage=omni.usd.get_context().get_stage()
        sim=SimulationContext(SimulationCfg(dt=0.005,device=args.device,
                                            gravity=(0.0,0.0,-9.81)))
        # Entirely new bodies, no reuse or mutation of G1 USD.
        sphere=UsdGeom.Sphere.Define(stage,"/World/KickProbeBall")
        sphere.CreateRadiusAttr(0.20)
        sphere.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,0.85))
        ball=sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(ball)
        rb=UsdPhysics.RigidBodyAPI.Apply(ball)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(ball).CreateMassAttr().Set(1.0)
        floor=UsdGeom.Cube.Define(stage,"/World/KickProbeFloor")
        floor.CreateSizeAttr(1.0)
        floor.AddScaleOp().Set(Gf.Vec3f(4.0,4.0,0.20))
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,-0.10))
        ground=floor.GetPrim()
        UsdPhysics.CollisionAPI.Apply(ground)
        api=PhysxSchema.PhysxContactReportAPI.Apply(ball)
        api.CreateThresholdAttr().Set(0.0)
        if args.mode=="authored_empty":
            api.CreateReportPairsRel().SetTargets([])
        rel=api.GetReportPairsRel()
        result["usd_inspection"]={
            "ball_api":bool(ball.HasAPI(PhysxSchema.PhysxContactReportAPI)),
            "ball_rb":bool(ball.HasAPI(UsdPhysics.RigidBodyAPI)),
            "floor_collision":bool(ground.HasAPI(UsdPhysics.CollisionAPI)),
            "threshold":float(api.GetThresholdAttr().Get()),
            "reportPairs_valid":bool(rel and rel.IsValid()),
            "reportPairs_authored":bool(rel and rel.IsValid() and rel.IsAuthored()),
            "reportPairs_targets":[str(x) for x in rel.GetTargets()] if rel and rel.IsValid() else None}
        interface=omni.physx.get_physx_simulation_interface()
        counts={"callback_calls":0,"callback_headers":0,"poll_calls":0,
                "poll_headers":0,"candidate_count":0}
        events=[]
        def ingest(headers,contacts,source,step):
            for h in headers:
                fields=[]
                try:
                    fields=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,n)))
                            for n in ("actor0","actor1","collider0","collider1")]
                    off=int(getattr(h,"contact_data_offset"))
                    size=int(getattr(h,"num_contact_data"))
                    points=[]
                    for p in contacts[off:off+min(size,12)]:
                        points.append({"position":vec(getattr(p,"position",None)),
                                       "normal":vec(getattr(p,"normal",None)),
                                       "impulse":vec(getattr(p,"impulse",None))})
                    item={"source":source,"step":step,"actors":fields[:2],
                          "colliders":fields[2:],"points":points,"num_contact_data":size}
                    if valid_pair(item):
                        counts["candidate_count"]+=1
                        if len(events)<24:events.append(item)
                except Exception as e:
                    if len(result["errors"])<8: result["errors"].append(repr(e)[:250])
        def callback(headers,contacts):
            counts["callback_calls"]+=1
            counts["callback_headers"]+=len(headers)
            ingest(headers,contacts,"callback",steps[0])
        steps=[0]
        sub=interface.subscribe_contact_report_events(callback)
        print("V086_USD_READY",json.dumps(result["usd_inspection"]),flush=True)
        sim.reset()
        print("V086_SIM_RESET",flush=True)
        for i in range(args.steps):
            steps[0]=i+1
            sim.step()
            try:
                headers,contacts=interface.get_contact_report()
                counts["poll_calls"]+=1
                counts["poll_headers"]+=len(headers)
                ingest(headers,contacts,"poll",i+1)
            except Exception as e:
                if len(result["errors"])<8: result["errors"].append("poll:"+repr(e)[:220])
            if i%80==0:print("V086_STEP",i+1,flush=True)
        result["steps"]=args.steps
        result["counts"]=counts
        result["positive_events"]=[e for e in events if positive(e)]
        result["contact_events"]=events
        result["calibrated_native_pair"]=len(result["positive_events"])>0
        result["status"]="positive_witness" if result["calibrated_native_pair"] else "uncalibrated_reporter"
        result["reporter_can_rule_out_g1_self_collision"]=False
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
        print("V086_PHYSICS_RESULT",json.dumps({"status":result["status"],"counts":counts,
                         "errors":result["errors"]}),flush=True)
        del sub
        try:sim.stop()
        except Exception as e:print("V086_STOP_ERROR",repr(e),flush=True)
        try:SimulationContext.clear_instance()
        except Exception as e:print("V086_CLEAR_ERROR",repr(e),flush=True)
        print("V086_PRE_CLOSE",flush=True)
    finally:
        app.close()
        print("V086_APP_CLOSED",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    p.add_argument("--mode",choices=("wildcard","authored_empty"),default="wildcard")
    p.add_argument("--steps",type=int,default=180)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.steps<100 or args.steps>400: p.error("Only 100-400 steps permitted")
    if args.output.exists():p.error("WRITE_ONCE_OUTPUT_EXISTS")
    run(args)
