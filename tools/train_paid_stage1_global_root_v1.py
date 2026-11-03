#!/usr/bin/env python3
"""Stage B controlled research branch: global-root goal-aware PAiD tracking.

Starts from a frozen 154->166 observation-migrated checkpoint. Adds explicit
robot-frame reference anchor pose (3+6) and base linear velocity (3) to actor.
Adds nonsaturating global planar root-trajectory reward, alongside upstream
PAiD reward and existing Stage I support/height curriculum.
Not the untouched upstream baseline and not target-conditioned kicking yet.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time, traceback, types

from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app

import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

ROOT=pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
MIGRATION=ROOT/"experiments/paid29_root_reference_init/model_154_to166.pt"
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(UPSTREAM/"soccer/assets")
sys.modules["soccer.assets"]=shim
from paid_stage1_contact_rewards import (torso_height_tracking_exp,
    lifted_foot_z_tracking_exp, support_contact_stability_exp)
from paid_stage1_root_rewards import (global_torso_planar_tracking_exp,
    global_torso_planar_huber, global_torso_linear_velocity_tracking)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--num-envs",type=int,default=96)
    pa.add_argument("--iters",type=int,default=240)
    pa.add_argument("--seed",type=int,default=85)
    pa.add_argument("--phase-min",type=int,default=55)
    pa.add_argument("--phase-max",type=int,default=125)
    pa.add_argument("--phase-zero-prob",type=float,default=0.65)
    pa.add_argument("--root-exp-weight",type=float,default=8.0)
    pa.add_argument("--root-huber-weight",type=float,default=-4.0)
    pa.add_argument("--root-velocity-weight",type=float,default=3.0)
    pa.add_argument("--init-checkpoint",type=str,default=str(MIGRATION))
    a=pa.parse_args()
    assert 1<=a.num_envs<=160 and 1<=a.iters<=10000
    assert 0<=a.phase_min<=a.phase_max<250 and 0<=a.phase_zero_prob<=1
    assert a.root_exp_weight>=0 and a.root_huber_weight<=0 and a.root_velocity_weight>=0
    checkpoint=pathlib.Path(a.init_checkpoint)
    assert checkpoint.is_file(),checkpoint
    import soccer.tasks
    from soccer.tasks.tracking import mdp
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=a.num_envs
    cfg.scene.env_spacing=3.0
    cfg.sim.device="cuda:0"
    cfg.seed=a.seed
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    for k in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[k]=(0.,0.)
    for k in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[k]=(0.,0.)
    cfg.commands.motion.joint_position_range=(0.,0.)
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    cfg.terminations.ee_body_pos.params["threshold"]=0.4
    # Append observations, preserving the ordering of the original 154 features.
    cfg.observations.policy.motion_anchor_pos_b=ObsTerm(
        func=mdp.motion_anchor_pos_b,params={"command_name":"motion"})
    cfg.observations.policy.motion_anchor_ori_b=ObsTerm(
        func=mdp.motion_anchor_ori_b,params={"command_name":"motion"})
    cfg.observations.policy.base_lin_vel=ObsTerm(func=isaac_mdp.base_lin_vel)
    cfg.observations.policy.enable_corruption=False
    # Repeat prior support/pelvis reward settings, new global reward explicitly marked.
    cfg.rewards.termination_penalty=RewTerm(func=isaac_mdp.is_terminated,weight=-200.0)
    cfg.rewards.alive_bonus=RewTerm(func=isaac_mdp.is_alive,weight=2.0)
    cfg.rewards.torso_height_tracking=RewTerm(
        func=torso_height_tracking_exp,weight=4.0,
        params={"command_name":"motion","std":0.11})
    cfg.rewards.lifted_foot_z_tracking=RewTerm(
        func=lifted_foot_z_tracking_exp,weight=3.0,
        params={"command_name":"motion","std":0.09,"min_lift":0.05})
    cfg.rewards.support_force_matching=RewTerm(
        func=support_contact_stability_exp,weight=1.0,
        params={"command_name":"motion","target_normal_force":35.0})
    cfg.rewards.global_planar_anchor_tracking=RewTerm(
        func=global_torso_planar_tracking_exp,weight=a.root_exp_weight,
        params={"command_name":"motion","std":0.8})
    cfg.rewards.global_planar_anchor_huber=RewTerm(
        func=global_torso_planar_huber,weight=a.root_huber_weight,
        params={"command_name":"motion","delta_m":0.25})
    cfg.rewards.global_planar_velocity_tracking=RewTerm(
        func=global_torso_linear_velocity_tracking,weight=a.root_velocity_weight,
        params={"command_name":"motion","std":1.2})
    runner_cfg=G1FlatPPORunnerCfg()
    runner_cfg.num_steps_per_env=24
    runner_cfg.save_interval=40
    runner_cfg.algorithm.learning_rate=2e-4
    runner_cfg.algorithm.num_learning_epochs=4
    runner_cfg.algorithm.num_mini_batches=4
    run_dir=ROOT/f"experiments/paid29_global_root_v1/seed{a.seed}_n{a.num_envs}_iter{a.iters}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Preserve existing experiment: {run_dir}")
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        "purpose":"GLOBAL WORLD ROOT TRAJECTORY EXPERIMENT, NOT ORIGINAL PAiD",
        "seed":a.seed,"num_envs":a.num_envs,"iters":a.iters,"root_fixed":False,
        "motion_sha256":hashlib.sha256(MOTION.read_bytes()).hexdigest(),
        "urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "checkpoint_154_to166":str(checkpoint),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_original_features":154,"appended_features":12,
        "observation_new":["relative reference anchor position3","reference orientation6",
                           "robot base linear velocity3"],
        "root_exp_weight":a.root_exp_weight,
        "root_huber_weight":a.root_huber_weight,
        "root_velocity_weight":a.root_velocity_weight,
        "phase": [a.phase_min,a.phase_max],"phase_zero_prob":a.phase_zero_prob,
        "notes":"No ball, not kicking accuracy. Evaluate global root path and strict survival independently."
    }
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2))
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),
                            clip_actions=runner_cfg.clip_actions)
    robot=env.unwrapped.scene["robot"]
    assert len(robot.joint_names)==29 and len(robot.body_names)==30
    motion=env.unwrapped.command_manager.get_term("motion")
    def sample_phase(self,env_ids):
        n=len(env_ids)
        nonzero=torch.randint(a.phase_min,a.phase_max+1,(n,),device=self.device)
        zero=torch.rand(n,device=self.device)<a.phase_zero_prob
        self.motion_idx[env_ids]=0
        self.motion_length[env_ids]=self.motion.file_lengths[0]
        self.time_steps[env_ids]=torch.where(zero,torch.zeros_like(nonzero),nonzero)
    motion._uniform_sampling=types.MethodType(sample_phase,motion)
    obs,_=env.reset()
    phase=motion.time_steps.clone()
    motion.time_steps[:]=phase-1
    motion._update_command()
    assert torch.equal(motion.time_steps,phase)
    assert obs.shape[1]==166,obs.shape
    print("ROOT_TRAJECTORY_ENV_READY",obs.shape,
          "critical_observations",str([x for x in env.unwrapped.observation_manager.active_terms.get("policy",[])
                                     if "anchor" in x or "base_lin_vel" in x]),flush=True)
    runner=OnPolicyRunner(env,runner_cfg.to_dict(),log_dir=str(run_dir),device="cuda:0")
    runner.load(str(checkpoint),load_optimizer=False)
    print("ROOT_TRAJECTORY_TRAIN_START",json.dumps(manifest),flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=a.iters,init_at_random_ep_len=False)
    finally:
        runner.save(str(run_dir/"last_model.pt"))
        print("ROOT_TRAJECTORY_CHECKPOINT",str(run_dir/"last_model.pt"),"sec",round(time.monotonic()-t0,2),flush=True)
        env.close()

if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("ROOT_TRAJECTORY_TRAIN_FAILED",flush=True)
        raise
    else:app.close()
