"""Headless benchmark of NVIDIA's official pretrained G1 flat-walking PPO.

Evaluation ONLY. Uses 37-DoF G1_MINIMAL_CFG with an UNFIXED root, its
original actuator model, and physically simulated terrain/contact.
Produces per-step actual root pose / velocities and fall/termination counts.
Does not yet couple a kicking motion to the policy.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from isaaclab.app import AppLauncher

app=AppLauncher(headless=True,enable_cameras=False).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

ROOT=Path("/home/you/projects/kinetickick")
CHECKPOINT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=240)
    ap.add_argument("--vx",type=float,default=0.45)
    ap.add_argument("--vy",type=float,default=0.)
    ap.add_argument("--omega",type=float,default=0.)
    ap.add_argument("--output",type=str,default="reports/pretrained_g1_walk_vx045.csv")
    args=ap.parse_args()
    assert CHECKPOINT.exists(), "G1 PPO model has not been downloaded"
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=1
    cfg.sim.device="cuda:0"
    cfg.seed=42
    cfg.commands.base_velocity.ranges.lin_vel_x=(args.vx,args.vx)
    cfg.commands.base_velocity.ranges.lin_vel_y=(args.vy,args.vy)
    cfg.commands.base_velocity.ranges.ang_vel_z=(args.omega,args.omega)
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.heading_command=False
    cfg.episode_length_s=40.
    policy_cfg=G1FlatPPORunnerCfg()
    task="Isaac-Velocity-Flat-G1-Play-v0"
    print("G1_BASELINE_CREATING_ENV",task,flush=True)
    raw=gym.make(task,cfg=cfg)
    env=RslRlVecEnvWrapper(raw,clip_actions=policy_cfg.clip_actions)
    robot=env.unwrapped.scene["robot"]
    runner=OnPolicyRunner(env,policy_cfg.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(CHECKPOINT))
    policy=runner.get_inference_policy(device="cuda:0")
    obs,_=env.get_observations()
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    initial=robot.data.root_pos_w[0].clone().detach().cpu().tolist()
    rows=[]
    resets=0
    heights=[]
    for step in range(args.steps):
        with torch.inference_mode():
            act=policy(obs)
            obs,rewards,dones,extras=env.step(act)
        pos=robot.data.root_pos_w[0].detach().cpu().tolist()
        worldvel=robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        grav=robot.data.projected_gravity_b[0].detach().cpu().tolist()
        reset=int(dones[0].item())
        resets+=reset
        heights.append(pos[2])
        rows.append([step,float(step*env.unwrapped.step_dt),
                     *[round(float(x),6) for x in pos],
                     *[round(float(x),6) for x in worldvel],
                     *[round(float(x),5) for x in grav],
                     reset])
        if step%40==0 or reset:
            print("G1_WALK_STEP",step,"pos",[round(v,3) for v in pos],
                  "world_vel",[round(v,3) for v in worldvel],
                  "projected_gravity",[round(v,3) for v in grav],
                  "done",reset,flush=True)
    with out.open("w",newline="") as file:
        wr=csv.writer(file);wr.writerow(
            ["step","t_s","root_x","root_y","root_z",
             "vel_x","vel_y","vel_z",
             "grav_bx","grav_by","grav_bz","episode_reset"])
        wr.writerows(rows)
    summary={"checkpoint":str(CHECKPOINT),"robot":"G1_MINIMAL_CFG 37DoF",
             "root_fixed":False,"desired_command_mps":[args.vx,args.vy],
             "command_omega_rad_s":args.omega,
             "sim_steps":args.steps,
             "duration_s":round(args.steps*env.unwrapped.step_dt,3),
             "root_initial_xyz":initial,
             "root_final_xyz":pos,
             "displacement_x_m":round(pos[0]-initial[0],4),
             "displacement_y_m":round(pos[1]-initial[1],4),
             "min_root_height_m":round(min(heights),4),
             "max_root_height_m":round(max(heights),4),
             "reset_count":resets,
             "estimated_fall_events_lower_bound":resets,
             "trajectory_file":str(out)}
    p=out.with_suffix(".json")
    p.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("G1_WALK_EVALUATION_DONE",json.dumps(summary),flush=True)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()
