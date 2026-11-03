"""Train a FREE-ROOT G1 left-support/right-leg-lift policy, warm-starting gait PPO.

Goal is a learnable left support stance to precede a right-foot soccer strike.
This is a standalone 37DoF G1_MINIMAL support-policy pilot, NOT human-motion
imitation and NOT a goal-directed whole-body shooting controller.

The published NVIDIA gait checkpoint is left unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,time,copy,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app

import gymnasium as gym
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from single_leg_support_rewards import (
    left_leg_ground_support,swing_foot_height_reward,
    right_foot_unloaded_reward,single_stance_balanced_reward,
    left_foot_stance_slip_cost,root_drop_cost,moving_base_cost,
)

@configclass
class ShootingSupportRewards(G1Rewards):
    left_support_force=RewTerm(func=left_leg_ground_support,weight=1.6)
    right_foot_clearance=RewTerm(
        func=swing_foot_height_reward,weight=2.0,params={
           "target_z":.115,"sigma":.065})
    right_foot_unloaded=RewTerm(
        func=right_foot_unloaded_reward,weight=.6,params={"max_force_n":35.})
    simultaneous_balanced_support=RewTerm(
        func=single_stance_balanced_reward,weight=3.2,params={
           "target_z":.115,"sigma":.065})
    left_support_slip=RewTerm(
        func=left_foot_stance_slip_cost,weight=-.8,params={"ground_min_force_n":30.})
    root_drop=RewTerm(func=root_drop_cost,weight=-2.4,
                      params={"min_root_height_m":.58})
    base_motion=RewTerm(func=moving_base_cost,weight=-1.6)

ROOT=Path("/home/you/projects/kinetickick")
INITIAL=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
EXPERIMENTS=ROOT/"experiments/g1_single_leg_support"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=64)
    ap.add_argument("--iterations",type=int,default=220)
    ap.add_argument("--seed",type=int,default=61)
    ap.add_argument("--lr",type=float,default=0.00015)
    ap.add_argument("--target-height",type=float,default=.115)
    args=ap.parse_args()
    if not (1<=args.n<=128):raise ValueError("n must be 1..128")
    if not (1<=args.iterations<=750):raise ValueError("iterations 1..750")
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=args.n
    cfg.scene.env_spacing=2.65
    cfg.sim.device="cuda:0"
    cfg.seed=args.seed
    cfg.episode_length_s=12.
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    base_rewards=cfg.rewards
    s=ShootingSupportRewards()
    for key,value in vars(base_rewards).items():
        if key not in (
            "left_support_force","right_foot_clearance",
            "right_foot_unloaded","simultaneous_balanced_support",
            "left_support_slip","root_drop","base_motion"):
            setattr(s,key,copy.deepcopy(value))
    cfg.rewards=s
    assert cfg.rewards.undesired_contacts is None
    cfg.rewards.feet_air_time.weight=0.
    cfg.rewards.feet_slide.weight=-.8
    cfg.rewards.flat_orientation_l2.weight=-2.5
    cfg.rewards.lin_vel_z_l2.weight=-.50
    cfg.rewards.right_foot_clearance.params["target_z"]=args.target_height
    cfg.rewards.simultaneous_balanced_support.params["target_z"]=args.target_height

    cfg.observations.policy.enable_corruption=False  # pilot, reenable for robustness training
    ppo=G1FlatPPORunnerCfg()
    ppo.num_steps_per_env=24
    ppo.save_interval=20
    ppo.algorithm.learning_rate=args.lr
    ppo.algorithm.num_learning_epochs=4
    ppo.algorithm.num_mini_batches=4
    ppo.algorithm.entropy_coef=.004
    run=EXPERIMENTS/f"seed{args.seed}_n{args.n}_h{args.target_height:.3f}"
    run.mkdir(parents=True,exist_ok=True)
    manifest={
      "purpose":"single-leg support skill ONLY, no kicks/no natural-motion reference",
      "start_checkpoint":str(INITIAL),
      "official_checkpoint_sha256":hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
      "num_envs":args.n,"iterations":args.iterations,"seed":args.seed,
      "target_right_ankle_world_z_m":args.target_height,
      "base_root_fixed":False,
      "learning_rate":args.lr,
      "intended_usage":"whole-body policy transition gait->left support->kick residual->gait recovery",
    }
    (run/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("SUPPORT_RL_INIT",json.dumps(manifest),flush=True)
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-v0",cfg=cfg),
        clip_actions=ppo.clip_actions)
    print("SUPPORT_RL_REWARDS",env.unwrapped.reward_manager.active_terms,flush=True)
    runner=OnPolicyRunner(env,ppo.to_dict(),log_dir=str(run),device="cuda:0")
    runner.load(str(INITIAL),load_optimizer=False)
    runner.current_learning_iteration=0
    if hasattr(runner.alg,"learning_rate"):
        runner.alg.learning_rate=args.lr
    print("SUPPORT_RL_ACTOR_WARM_STARTED",type(runner.alg.policy).__name__,flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=args.iterations,
                     init_at_random_ep_len=False)
    finally:
        runner.save(str(run/"last_model.pt"))
        print("SUPPORT_RL_SAVED",run/"last_model.pt","elapsed_s",
              round(time.monotonic()-t0,1),flush=True)
        env.close()

if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    else:
        app.close()
