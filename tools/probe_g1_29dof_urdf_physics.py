"""Minimal 29DoF G1 URDF-to-PhysX articulation import validation on Tang.

No training, no reference animation, no soccer; this is a required physical
asset gate for human-coordinated RL with the real 29-axis waist/arms.
Reuse PAiD's published actuator model with the locally downloaded, verified
G1 rev1.0 URDF. Does not modify the existing 37DoF Isaac Lab environment.
"""
from __future__ import annotations
import json,sys,traceback,math,types
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
import torch
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
ROOT=Path("/home/you/projects/kinetickick")
sys.path.insert(0,str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"))
# Tang has a sparse PAiD repo checkout lacking soccer/assets/__init__.py.
# Supply only the ASSET_DIR symbol; URDF asset path is overwritten below.
asset_module=types.ModuleType("soccer.assets")
asset_module.ASSET_DIR=str(ROOT/"data/assets/g1_29dof_lafan1")
sys.modules["soccer.assets"]=asset_module
from soccer.robots.g1 import G1_CYLINDER_CFG
URDF=ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
OUT=ROOT/"reports/g1_29dof_physics_import_probe.json"

def main():
    assert URDF.is_file(),URDF
    cfg=G1_CYLINDER_CFG.copy()
    cfg.prim_path="/World/PhysicsG1_29"
    cfg.spawn.asset_path=str(URDF)
    cfg.spawn.fix_base=False
    cfg.spawn.make_instanceable=False
    # PAiD's original robot's no-joint-drive URDF servo configuration.
    # Use it, but NEVER assume an unloaded 29DoF reference is physical.
    print("G1_29_URDF_SOURCE",str(URDF),flush=True)
    print("G1_29_URDF_IMPORT_START","base_fixed",cfg.spawn.fix_base,
          "drive_type",type(cfg.spawn.joint_drive).__name__,flush=True)
    ctx=SimulationContext(sim.SimulationCfg(dt=1./120,render_interval=4,device="cuda:0"))
    ground=sim.GroundPlaneCfg()
    ground.func("/World/SupportGround",ground)
    light=sim.DomeLightCfg(intensity=1500.)
    light.func("/World/Light",light)
    robot=Articulation(cfg)
    print("G1_29_ARTICULATION_OBJECT_CREATED",flush=True)
    ctx.reset()
    print("G1_29_ARTICULATION_PHYSICS_INITIALIZED","ready",robot.is_initialized,flush=True)
    names=list(robot.joint_names)
    links=list(robot.body_names)
    if len(names)!=29:
        raise AssertionError(f"expected 29 active joints; got {len(names)}: {names}")
    needed=["waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
            "right_ankle_roll_joint","left_ankle_roll_joint"]
    assert all(x in names for x in needed), needed
    init_q=robot.data.default_joint_pos.clone()
    results=[]
    for t in range(240):
        robot.set_joint_position_target(init_q)
        robot.write_data_to_sim()
        ctx.step()
        robot.update(ctx.get_physics_dt())
        if t%40==0:
            p=robot.data.root_pos_w[0].detach().cpu().tolist()
            q=robot.data.joint_pos[0].detach().cpu().tolist()
            results.append({"step":t,"root_xyz":[round(float(v),4) for v in p],
                            "knee_r":round(float(q[names.index("right_knee_joint")]),4)})
            print("G1_29_ROOT_PHYSICS_STEP",t,results[-1],flush=True)
    final=robot.data.root_pos_w[0].detach().cpu().tolist()
    result={
       "URDF":str(URDF),"base_fixed":False,"physical_articulation":True,
       "joint_count":len(names),"body_count":len(links),
       "joints":names,
       "waist_3axis_validated":all(x in names for x in needed[:3]),
       "pose_control_with_gravity_steps":240,
       "root_xyz_final":final,
       "default_pose_test_minroot_m":min(q["root_xyz"][2] for q in results),
       "fell_or_collapse_detected":any(q["root_xyz"][2]<.45 for q in results),
       "tested_steps":results,
       "not_trained_to_walk":True,
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("G1_29_URDF_PHYSICS_PROBE_COMPLETED",json.dumps(result),flush=True)

if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("G1_29_URDF_PHYSICS_PROBE_FAILED",flush=True)
        raise
    else:
        app.close()
