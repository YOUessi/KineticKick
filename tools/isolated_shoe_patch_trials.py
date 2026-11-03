"""Physics-only isolation test: can right-foot side SHOE patches hit a ball?

The bare original foot collider and every non-target boot patch are disabled.
If ball moves, the named region must have caused the collision (on this link).
This is a kinematic rig with root fixed, not natural G1 soccer.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json
from pathlib import Path
import torch
from pxr import UsdPhysics
import omni.usd
import isaaclab.sim as sim
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.sim import SimulationCfg,SimulationContext
from soccer_world import attach_training_football_shoe
DT=1/120

def make():
    sim.GroundPlaneCfg().func("/World/Ground",sim.GroundPlaneCfg())
    cfg=G1_CFG.copy()
    cfg.spawn.articulation_props.fix_root_link=True
    cfg.prim_path="/World/UnitreeG1"
    robot=Articulation(cfg)
    patches=attach_training_football_shoe()
    stage=omni.usd.get_context().get_stage()
    original=stage.GetPrimAtPath("/World/UnitreeG1/right_ankle_roll_link/Cube")
    assert original.IsValid()
    UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(False)
    ball=RigidObject(RigidObjectCfg(
      prim_path="/World/Football",
      spawn=sim.SphereCfg(radius=.11,
        collision_props=sim.CollisionPropertiesCfg(),
        rigid_props=sim.RigidBodyPropertiesCfg(),
        mass_props=sim.MassPropertiesCfg(mass=.43),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.,.73,.15))),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(.2,.07,.113)),
    ))
    sensor=ContactSensor(ContactSensorCfg(
        prim_path="/World/UnitreeG1/right_ankle_roll_link",
        update_period=0.,history_length=2,filter_prim_paths_expr=["/World/Football"]))
    return robot,ball,sensor,stage,patches

def trial(ctx,robot,ball,sensor,stage,patches,mode,enabled):
    # In this lab we must isolate a single named collider; disable all others.
    for name,path in patches.items():
        p=stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(bool(enabled and name==mode))
    ctx.reset()
    ids={n:i for i,n in enumerate(robot.joint_names)}
    lim=robot.data.soft_joint_pos_limits[0]
    zero=torch.zeros_like(robot.data.default_joint_pos)
    q0=robot.data.default_joint_pos.clone()
    def write(q,name,value):
        i=ids[name]
        q[:,i]=max(float(lim[i,0])+.005,min(float(lim[i,1])-.005,value))
    is_med=mode=="medial"
    source=-.25 if is_med else .25
    dest=.25 if is_med else -.25
    by=.07 if is_med else -.34
    bx=.20
    def posture(roll):
        q=q0.clone()
        for name,value in [
            ("right_hip_pitch_joint",-.55),("right_knee_joint",.55),
            ("right_hip_roll_joint",roll),("right_hip_yaw_joint",0.),
            ("right_ankle_pitch_joint",0.),("right_ankle_roll_joint",0.),
        ]:write(q,name,value)
        return q
    robot.write_joint_state_to_sim(posture(source),zero)
    robot.reset()
    x=ball.data.default_root_state.clone()
    x[:,:3]=torch.tensor([bx,by,.113],device=ball.device)
    x[:,7:]=0.0
    ball.write_root_pose_to_sim(x[:,:7])
    ball.write_root_velocity_to_sim(x[:,7:])
    ball.reset()
    impulses=0
    peak=0.
    prev=[]
    for n in range(235):
        t=n*DT
        a=max(0.,min(1.,(t-.35)/.38))
        a=a*a*(3-2*a)
        q=posture(source+(dest-source)*a)
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        ctx.step()
        robot.update(DT)
        ball.update(DT)
        sensor.update(DT)
        f=sensor.data.force_matrix_w
        mag=float(torch.linalg.vector_norm(f).item()) if f is not None and f.numel() else 0.
        peak=max(peak,mag)
        if mag>2.0:impulses+=1
    xyz=ball.data.root_pos_w[0].cpu().tolist()
    result={
      "mode":mode,"selected_patch_enabled":bool(enabled),
      "ball_start":[bx,by,.113],"ball_end":xyz,
      "ball_horizontal_displacement_m":round(((xyz[0]-bx)**2+(xyz[1]-by)**2)**.5,4),
      "peak_ball_contact_force_N":round(peak,3),
      "contact_steps":impulses,
      "note":"Root fixed and other right-foot colliders disabled; one trial is NOT an athletic skill."
    }
    print("TRIAL",json.dumps(result),flush=True)
    return result

def main():
    ctx=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor,stage,patches=make()
    ctx.reset()
    outcomes=[]
    for mode in ("medial","lateral"):
        for enabled in (False,True):
            outcomes.append(trial(ctx,robot,ball,sensor,stage,patches,mode,enabled))
    Path("/home/you/projects/kinetickick/reports/isolated_shoe_patch_trials.json").write_text(json.dumps(outcomes,indent=2))
    print("DONE_ISOLATED_PATCH_TRIALS",len(outcomes),flush=True)

if __name__=="__main__":
    try:main()
    finally:app.close()
