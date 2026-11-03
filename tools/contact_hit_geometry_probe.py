"""G1 contact-point probe: report actual foot-ball collision path/position/normal."""
from __future__ import annotations
import json
from pathlib import Path
from foot_ball_contact_probe import scene, run_trial, DT, app
from pxr import PhysicsSchemaTools
import omni.physx
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sim import SimulationCfg, SimulationContext

try:
    sim=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor=scene()
    sim.reset()
    events=[]
    idx=robot.body_names.index("right_ankle_roll_link")
    def callback(headers,data):
        for h in headers:
            try:
                collider0=str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                collider1=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            except Exception as e:
                print("CONTACT_PATH_ERR",repr(e),flush=True)
                continue
            if not ("SoccerBall" in collider0 or "SoccerBall" in collider1): continue
            for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                c=data[i]
                pt=[float(v) for v in c.position]
                normal=[float(v) for v in c.normal]
                x=torch.tensor(pt,device=robot.device,dtype=torch.float32)-robot.data.body_pos_w[0,idx]
                local=quat_apply_inverse(robot.data.body_quat_w[0,idx].unsqueeze(0),x.unsqueeze(0))[0]
                record={"collider0":collider0,"collider1":collider1,
                        "world_contact":pt,"normal":normal,"local_contact":local.cpu().tolist(),
                        "impulse_N_s":float(c.impulse)}
                events.append(record)
                if len(events)<=16:print("CONTACT_POINT",json.dumps(record),flush=True)
    listener=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(callback)
    for m in ["instep","inside","outside"]:
        res=run_trial(sim,robot,ball,sensor,m,.38)
        print("TRIAL",m,json.dumps(res),flush=True)
    del listener
    Path("/home/you/projects/kinetickick/reports/contact_geometry_sample.json").write_text(json.dumps(events,indent=2))
    print("CONTACT_EVENTS",len(events),flush=True)
finally:
    app.close()
