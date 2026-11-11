#!/usr/bin/env python3
"""Disposable 200-Hz native PhysX toy contact positive/negative case.

No G1 source, weights, collider, policy or old evidence is ever modified.
The reporter API is applied only to disposable toy prims before sim.reset().
"""
import argparse,json
from pathlib import Path

def run(out:Path,control:str,sim_device:str="cuda:0"):
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    try:
        import omni.physx, omni.usd
        from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        import isaaclab.sim as sim_utils
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=sim_device))
        stage=omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage,"/World/Probe")
        for label,z,kin in (("A",.3,True),("B",1.3,False)):
            cfg=sim_utils.CuboidCfg(
                size=(.5,.5,.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,kinematic_enabled=kin,disable_gravity=kin),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                activate_contact_sensors=True)
            cfg.func("/World/Probe/"+label,cfg,translation=(
                3. if control=="separated" and label=="B" else 0.,0.,z))
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        contact_sensor=ContactSensor(ContactSensorCfg(
            prim_path="/World/Probe/B",update_period=0.0,
            filter_prim_paths_expr=["/World/Probe/A"],history_length=0))
        reporter=omni.physx.get_physx_simulation_interface()
        event_rows=[];errors=[]; polls=0;callbacks=0
        filtered_peak_N=0.;net_peak_N=0.;body_positions=[];force_errors=[]
        def decode(i):
            try:return str(PhysicsSchemaTools.intToSdfPath(i))
            except Exception as exc:return "UNDECODABLE_"+type(exc).__name__
        def cb(headers,data):
            nonlocal callbacks
            callbacks+=1
            for h in headers:
                paths=[decode(getattr(h,f,None)) for f in ("actor0","actor1","collider0","collider1")]
                if not any("/Probe/" in x for x in paths):continue
                offset=int(getattr(h,"contact_data_offset",0))
                count=int(getattr(h,"num_contact_data",0))
                details=[]
                for c in data[offset:offset+count]:
                    details.append({"position":str(getattr(c,"position",None)),
                        "normal":str(getattr(c,"normal",None)),
                        "impulse":str(getattr(c,"impulse",None)),
                        "separation":str(getattr(c,"separation",None))})
                event_rows.append({"paths":paths,"count":count,"contacts":details[:4]})
        sub=reporter.subscribe_contact_report_events(cb)
        sim.reset()
        for k in range(300):
            sim.step(render=False)
            try:
                contact_sensor.update(.005)
                data=contact_sensor.data
                filtered_peak_N=max(filtered_peak_N,float(data.force_matrix_w[0,0,0].norm()))
                net_peak_N=max(net_peak_N,float(data.net_forces_w[0,0].norm()))
                if k%40==0:
                    body_positions.append(dict(step=k,pose_w=data.pos_w.tolist() if data.pos_w is not None else None))
            except BaseException as exc:
                if len(force_errors)<5:force_errors.append(type(exc).__name__+":"+str(exc)[:150])
            try:
                headers,raw=reporter.get_contact_report()
                polls+=len(headers)
            except Exception as e:
                errors.append(type(e).__name__+":"+str(e)[:150]);break
        pair=any("/Probe/A" in " ".join(e["paths"]) and "/Probe/B" in " ".join(e["paths"]) for e in event_rows)
        result={"schema":"v056 isolation and signed negative","case":control,"sim_dt_s":.005,
                "callbacks":callbacks,"polled_headers":polls,
                "raw_events":event_rows[:20],"total_probe_events":len(event_rows),
                "native_pair_proven":pair,"filtered_pair_peak_N":filtered_peak_N,
                "net_body_peak_N":net_peak_N,"contact_tensor_errors":force_errors,
                "sensor_body_count":contact_sensor.body_physx_view.count,
                "sensor_filter_count":contact_sensor.contact_physx_view.filter_count,
                "errors":errors,"not_G1_evidence":True}
        with out.open("x") as f:json.dump(result,f,indent=2)
        print("ISOLATED_CONTACT_REPORT",json.dumps({k:v for k,v in result.items() if k!="raw_events"}),flush=True)
        sub=None
    finally:app.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--control",choices=("collide","separated"),default="collide")
    p.add_argument("--sim-device",choices=("cuda:0","cpu"),default="cuda:0")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    run(a.output,a.control,a.sim_device)
