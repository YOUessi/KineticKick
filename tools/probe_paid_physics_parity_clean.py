#!/usr/bin/env python3
"""Gate A: same-frame physical-body parity, with reset noise disabled.

The robot is put at exact NPZ root + 29-joint pose and velocity, then advanced
only one PhysX tick. Unlike earlier smoke tests, this is NOT a stochastic reset
pose compared against an unperturbed motion reference.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, traceback, types

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
import gymnasium as gym
import isaaclab.sim as sim

ROOT=pathlib.Path("/home/you/projects/kinetickick")
PACKAGE=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
ASSET_ROOT=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking/soccer/assets"
URDFS={
    "paid":ROOT/"configs/paid_original/paid_main_resolved.urdf",
    "local":ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf",
}
sys.path.insert(0,str(PACKAGE))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(ASSET_ROOT)
sys.modules["soccer.assets"]=shim

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--model",choices=("paid","local"),default="paid")
    a=pa.parse_args()
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=1
    cfg.sim.device="cuda:0"
    cfg.seed=23
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDFS[a.model])
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.observations.policy.enable_corruption=False
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key]=(0.0,0.0)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key]=(0.0,0.0)
    cfg.commands.motion.joint_position_range=(0.0,0.0)
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    print("MODEL_PARITY_START",a.model,str(URDFS[a.model]),flush=True)
    env=gym.make("Tracking-Flat-G1-v0",cfg=cfg)
    raw=env.unwrapped
    env.reset()
    robot=raw.scene["robot"]
    motion=raw.command_manager.get_term("motion")
    assert len(robot.joint_names)==29 and len(robot.body_names)==30
    original_names=list(robot.body_names)
    print("PHYSX_JOINT_NAMES",json.dumps(robot.joint_names),flush=True)
    print("PHYSX_BODY_NAMES",json.dumps(original_names),flush=True)
    per_frame=[]
    for fr in (0,20,60,100,140,190,250):
        motion.time_steps[:]=fr
        q=motion.joint_pos.detach().clone()
        qvel=motion.joint_vel.detach().clone()
        root=torch.cat([motion.body_pos_w[:,0],motion.body_quat_w[:,0],
                        motion.body_lin_vel_w[:,0],motion.body_ang_vel_w[:,0]],dim=1)
        robot.write_joint_state_to_sim(q,qvel)
        robot.write_root_state_to_sim(root)
        raw.sim.step(render=False)
        raw.scene.update(raw.physics_dt)
        actual_xyz=robot.data.body_pos_w[0].detach().cpu()
        actual_q=robot.data.body_quat_w[0].detach().cpu()
        ref_xyz=motion.motion._body_pos_w[0,fr].detach().cpu()
        ref_q=motion.motion._body_quat_w[0,fr].detach().cpu()
        error=torch.linalg.vector_norm(actual_xyz-ref_xyz,dim=-1)
        alignment=torch.abs(torch.sum(actual_q*ref_q,dim=-1)).clamp(max=1.0)
        angles=2*torch.acos(alignment)*180/torch.pi
        result={"frame":fr,
                "mean_pos_error_m":round(float(error.mean()),6),
                "max_pos_error_m":round(float(error.max()),6),
                "mean_rot_error_deg":round(float(angles.mean()),4),
                "top_errors":[{"body":original_names[i],"m":round(float(error[i]),5)} for i in torch.argsort(error,descending=True)[:6].tolist()],
                "root_delta_m":round(float(torch.linalg.vector_norm(robot.data.root_pos_w[0].cpu()-root[0,:3].cpu())),6)}
        per_frame.append(result)
        print("PHYSX_PARITY_FRAME",json.dumps(result),flush=True)
    result={"model":a.model,"URDF":str(URDFS[a.model]),
      "URDF_sha256":hashlib.sha256(URDFS[a.model].read_bytes()).hexdigest(),
      "motions":str(MOTION),"same_frame":True,
      "reset_jitter_disabled":True,"physics_single_tick":True,"root_fixed":False,
      "joint_names":list(robot.joint_names),"body_names":original_names,
      "results":per_frame,
      "max_error_m":max(x["max_pos_error_m"] for x in per_frame),
      "mean_error_m":sum(x["mean_pos_error_m"] for x in per_frame)/len(per_frame)}
    output=ROOT/f"reports/paid_physics_parity_clean_{a.model}.json"
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("PHYSX_PARITY_SUMMARY",a.model,"mean",round(result["mean_error_m"],5),
          "max",round(result["max_error_m"],5),"saved",output,flush=True)
    env.close()

if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("PHYSX_PARITY_FAILED",flush=True)
        raise
    else:
        app.close()
