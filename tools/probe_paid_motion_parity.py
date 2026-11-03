"""Smoke the *actual PAiD 29DoF motion tracking environment* in Tang Isaac Lab.

Uses official TeleHuman HumanoidSoccer tracking task + local 29DoF G1 URDF
from our verified LAFAN asset. No global pip edits; isolated sys.path shim
supplies missing soccer.assets module in the sparse repository checkout.
Only 1 env / 40 untrained zero actions: expected fall, not a learned skill.
"""
from __future__ import annotations
import pathlib,sys,types,json,traceback
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
import torch,gymnasium as gym
import isaaclab.sim as sim
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
ROOT=pathlib.Path("/home/you/projects/kinetickick")
PACKAGE=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
sys.path.insert(0,str(PACKAGE))
assets=types.ModuleType("soccer.assets")
assets.ASSET_DIR=str(ROOT/"data/assets/g1_29dof_lafan1")
sys.modules["soccer.assets"]=assets

def main():
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=1
    cfg.sim.device="cuda:0"
    cfg.seed=51
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.observations.policy.enable_corruption=False
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    print("PAID_MOTION_TRACKING_CREATE",str(MOTION),"URDF",str(URDF),flush=True)
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg))
    raw=env.unwrapped
    robot=raw.scene["robot"]
    motion=raw.command_manager.get_term("motion")
    print("PAID_MOTION_TRACKING_INITIALIZED","robot_joints",len(robot.joint_names),
        "links",len(robot.body_names),"motion_type",type(motion).__name__,
        "motion_length_steps",motion.motion.time_step_total,
        "root_fixed",cfg.scene.robot.spawn.fix_base,
        flush=True)
    assert len(robot.joint_names)==29
    obs,_=env.get_observations()
    reference=motion.body_pos_w.detach().cpu()
    actual=motion.robot_body_pos_w.detach().cpu()
    import numpy as np
    e=torch.linalg.norm(actual-reference,dim=-1)[0].tolist()
    dz=(actual[0,:,2]-reference[0,:,2]).tolist()
    labels=list(cfg.commands.motion.body_names)
    mismatch=[{"body":b,"absolute_xyz_error_m":round(float(v),4),
               "z_error_m":round(float(w),4)} for b,v,w in zip(labels,e,dz)]
    print("PAID_REFERENCE_PHYSICAL_MISMATCH",json.dumps(mismatch),flush=True)
    observations=[]
    for t in range(40):
        action=torch.zeros((1,29),device=obs.device,dtype=obs.dtype)
        obs,reward,done,extras=env.step(action)
        if t%10==0:
            pos=robot.data.root_pos_w[0].detach().cpu().tolist()
            observations.append([t,[round(x,4) for x in pos],float(reward[0])])
            print("PAID_TRACKING_STEP_UNTRAINED",observations[-1],flush=True)
        if bool(done[0]):
            print("PAID_TRACKING_UNTRAINED_RESET",t,flush=True)
            break
    out={"initial_reference_mismatch":mismatch,
         "max_initial_abs_error_m":round(max(e),4),
         "PAiD_env_initialized":True,
         "URDF_joint_count":len(robot.joint_names),
         "reference_motion_frames":motion.motion.time_step_total,
         "actor_trained":False,"root_fixed":False,
         "observations":observations}
    path=ROOT/"reports/paid_29dof_motion_physics_parity.json"
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("PAID_TRACKING_TASK_SMOKE_DONE",json.dumps(out),flush=True)
    env.close()
if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("PAID_TRACKING_SMOKE_FAILED",flush=True)
        raise
    else:
        app.close()
