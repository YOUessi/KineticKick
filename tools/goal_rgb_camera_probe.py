"""Real rendered RGB camera probe for soccer-goal detection.

Goal goalposts are bright blue. We detect raw RGB pixels, without USD goal
coordinates during inference (scene truth is used AFTERWARD to assess error).
A pure color marker detector is not generic object detection.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=True).app
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import isaaclab.sim as sim
from isaaclab.sensors.camera import Camera,CameraCfg
from isaaclab.sim import SimulationCfg,SimulationContext
from soccer_world import create_goal,move_goal,locate_goal_in_usd,GOAL_WIDTH

def frame_goal_rgb(rgb,K):
    arr=np.asarray(rgb)[:,:,:3].astype(np.int16)
    r,g,b=[arr[:,:,i] for i in range(3)]
    # Distinct blue training goal with pale scenery and grass.
    mask=(b>115)&(b>1.35*g)&(b>1.75*r)&(g<155)
    xx,yy=np.where(mask)[1],np.where(mask)[0]
    if len(xx)<30:return None
    u0=float(np.percentile(xx,5));u1=float(np.percentile(xx,95))
    v0=float(np.percentile(yy,5));v1=float(np.percentile(yy,95))
    cx=float(K[0,2]);fx=float(K[0,0])
    y_bearing=-np.arctan2((u0+u1)*.5-cx,fx)
    # Known frame width provides depth scale; perspective distortion not corrected.
    depth_est=fx*GOAL_WIDTH/max(1.,u1-u0)
    y_est=depth_est*np.tan(y_bearing)
    return {"pixel_bbox":[u0,v0,u1,v1],"pixel_count":int(mask.sum()),
            "horizontal_bearing_deg":float(np.degrees(y_bearing)),
            "estimated_depth_m":float(depth_est),"estimated_lateral_m":float(y_est)}

try:
    ctx=SimulationContext(SimulationCfg(dt=.02))
    ground=sim.GroundPlaneCfg(color=(.25,.37,.24))
    ground.func("/World/Ground",ground)
    light=sim.DomeLightCfg(intensity=2400,color=(.95,.96,1.))
    light.func("/World/Sky",light)
    goal=create_goal(initial_y=0)
    cfg=CameraCfg(prim_path="/World/RobotFrontCamera",update_period=0.0,
        height=384,width=640,data_types=["rgb"],
        spawn=sim.PinholeCameraCfg(focal_length=18.0,horizontal_aperture=20.955,clipping_range=(.05,100)))
    cam=Camera(cfg)
    ctx.reset()
    cam.set_world_poses_from_view(
        eyes=torch.tensor([[0.,0.,1.22]],device=ctx.device),
        targets=torch.tensor([[5.0,0.,.78]],device=ctx.device))
    outputs=[]
    folder=Path("/home/you/projects/kinetickick/media")
    folder.mkdir(exist_ok=True,parents=True)
    for gy in [-.5,0.,+.35]:
        move_goal(goal,gy)
        for j in range(10):
            ctx.step()
            cam.update(.02)
        arr=cam.data.output["rgb"][0].cpu().numpy()
        K=cam.data.intrinsic_matrices[0].cpu().numpy()
        result=frame_goal_rgb(arr,K)
        filename=f"camera_goal_y{gy:+.2f}.png"
        Image.fromarray(arr[:,:,:3]).save(folder/filename)
        print("VISION_FRAME",json.dumps({"set_goal_y":gy,"detected":result,
              "ground_truth_not_used_by_detector":locate_goal_in_usd()}),flush=True)
        outputs.append({"set_goal_y":gy,"rgb_detector":result})
    Path("/home/you/projects/kinetickick/reports/goal_rgb_camera_probe.json").write_text(json.dumps(outputs,indent=2))
    print("GOAL_RGB_PROBE_DONE",flush=True)
finally:app.close()
