#!/usr/bin/env python3
"""PAiD Stage I *curriculum variant* (NOT the unmodified baseline).

One reference clip, physical free-root original G1. Early training removes
random perturbations, relaxes end-effector termination, and adds survival/
termination rewards to mitigate the failure-at-step-8 reward shortcut.
Retain original baseline separately; evaluate both with the strict evaluator.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, traceback, time, types
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app

import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

ROOT=pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(UPSTREAM/"soccer/assets")
sys.modules["soccer.assets"]=shim

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--num-envs",type=int,default=96)
    pa.add_argument("--iters",type=int,default=320)
    pa.add_argument("--seed",type=int,default=84)
    pa.add_argument("--ee-threshold",type=float,default=0.40)
    pa.add_argument("--termination-weight",type=float,default=-200.)
    pa.add_argument("--alive-weight",type=float,default=2.)
    pa.add_argument("--resume-path",type=str,default="")
    a=pa.parse_args()
    assert 1<=a.num_envs<=192 and 1<=a.iters<=20000 and .25<=a.ee_threshold<=.7
    assert a.termination_weight<=0 and a.alive_weight>=0
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=a.num_envs
    cfg.scene.env_spacing=2.7
    cfg.sim.device="cuda:0"
    cfg.seed=a.seed
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    # Simplify only the stage-I introduction: no stochastic reset or external pushes.
    for key in cfg.commands.motion.pose_range: cfg.commands.motion.pose_range[key]=(0.,0.)
    for key in cfg.commands.motion.velocity_range: cfg.commands.motion.velocity_range[key]=(0.,0.)
    cfg.commands.motion.joint_position_range=(0.,0.)
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    # Marked curriculum changes, never overwrite the PAiD baseline.
    cfg.terminations.ee_body_pos.params["threshold"]=a.ee_threshold
    cfg.rewards.termination_penalty=RewTerm(func=isaac_mdp.is_terminated,weight=a.termination_weight)
    cfg.rewards.alive_bonus=RewTerm(func=isaac_mdp.is_alive,weight=a.alive_weight)
    agent=G1FlatPPORunnerCfg()
    agent.num_steps_per_env=24
    agent.save_interval=40
    agent.algorithm.learning_rate=3e-4
    agent.algorithm.num_learning_epochs=4
    agent.algorithm.num_mini_batches=4
    run_dir=ROOT/f"experiments/paid29_stage1_curriculum/seed{a.seed}_n{a.num_envs}_iter{a.iters}_ee{str(a.ee_threshold).replace('.','p')}"
    if (run_dir/"last_model.pt").exists() and not a.resume_path:
        raise FileExistsError(str(run_dir))
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        "purpose":"STAGED CURRICULUM PILOT (NOT ORIGINAL PAiD BASELINE)",
        "original_urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "motion_sha256":hashlib.sha256(MOTION.read_bytes()).hexdigest(),
        "root_fixed":False,"seed":a.seed,"num_envs":a.num_envs,
        "iterations":a.iters,"resume_path":a.resume_path or None,
        "perturbations":"disabled for Stage I","ee_z_termination_m":a.ee_threshold,
        "terminal_reward_weight":a.termination_weight,
        "survival_reward_weight":a.alive_weight,
        "eval_condition":"Strict 0.25m original PAiD threshold, on separate seeds",
        "notes":"No ball / no shot physics; tracks a retargeted reference clip, not human generated soccer policy.",
    }
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("CURRICULUM_START",json.dumps(manifest),flush=True)
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),clip_actions=agent.clip_actions)
    assert len(env.unwrapped.scene["robot"].joint_names)==29
    motion=env.unwrapped.command_manager.get_term("motion")
    # IsaacLab 2.1 computes terminations before the command update in step().
    # Ensure relative-body target initialized at frame 0, not all-zero cache.
    env.reset()
    motion.time_steps[:]=-1
    motion._update_command()
    assert int(motion.time_steps[0])==0
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(run_dir),device="cuda:0")
    if a.resume_path:
        assert pathlib.Path(a.resume_path).exists()
        runner.load(a.resume_path,load_optimizer=True)
    print("CURRICULUM_READY",env.get_observations()[0].shape,flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=a.iters,init_at_random_ep_len=False)
    finally:
        runner.save(str(run_dir/"last_model.pt"))
        print("CURRICULUM_CHECKPOINT",str(run_dir/"last_model.pt"),"seconds",round(time.monotonic()-t0,2),flush=True)
        env.close()
if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("CURRICULUM_FAILED",flush=True)
        raise
    else:
        app.close()
