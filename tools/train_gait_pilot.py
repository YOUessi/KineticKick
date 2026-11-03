"""Small, interruptible pilot: G1 37DoF PPO warm-start + stance-specific knee reward.

This is NOT full motion-capture imitation nor an accomplished natural gait.
Existing official checkpoint is never changed. All trained outputs go to a new
Tang project directory. Evaluate on held-out seed/speeds before claims.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,faulthandler,copy
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
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg,G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

from humanlike_gait_rewards import (
    stance_knee_flexion_cost,excessive_airborne_cost,synthetic_reward_self_test
)

@configclass
class GaitNaturalnessPilotRewards(G1Rewards):
    # Negative weights convert these costs into reward penalties.
    stance_knee_flexion = RewTerm(
        func=stance_knee_flexion_cost,weight=-0.42,
        params={"target_max_rad":.50,"contact_threshold_n":15.,"min_command_speed":.12},
    )
    excessive_airborne = RewTerm(
        func=excessive_airborne_cost,weight=-.12,
        params={"contact_threshold_n":15.,"min_command_speed":.12},
    )

ROOT=Path("/home/you/projects/kinetickick")
OFFICIAL=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
TRAIN_DIR=ROOT/"experiments/gait_naturalness_pilot_20261021"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--num-envs",type=int,default=64)
    ap.add_argument("--iterations",type=int,default=16)
    ap.add_argument("--seed",type=int,default=40)
    ap.add_argument("--lr",type=float,default=1.5e-4)
    ap.add_argument("--variant",choices=("v1","v2"),default="v1")
    p=ap.parse_args()
    if not (1<=p.num_envs<=128):raise ValueError("num_envs 1..128")
    if not (1<=p.iterations<=500):raise ValueError("iterations 1..500")
    assert OFFICIAL.exists()
    assert synthetic_reward_self_test()
    TRAIN_DIR.mkdir(parents=True,exist_ok=True)
    cfg=G1FlatEnvCfg_PLAY()  # pilot starts from the verified single-env playback scene
    cfg.scene.num_envs=p.num_envs
    cfg.scene.env_spacing=2.7
    cfg.seed=p.seed
    cfg.sim.device="cuda:0"
    cfg.commands.base_velocity.debug_vis=False
    cfg.episode_length_s=16.
    original_reward_cfg=cfg.rewards
    natural_cfg=GaitNaturalnessPilotRewards()
    # Crucial: copying the subclass defaults directly overwrote G1FlatEnvCfg
    # post-init overrides (undesired_contacts=None, unsupported THIGH regex).
    # Preserve ALL stock G1 env reward changes and only append our two terms.
    for name,value in vars(original_reward_cfg).items():
        if name not in ("stance_knee_flexion","excessive_airborne"):
            setattr(natural_cfg,name,copy.deepcopy(value))
    cfg.rewards=natural_cfg
    assert cfg.rewards.undesired_contacts is None
    # Keep velocity tracking, falls, foot sliding, inertial costs intact.
    # Reduce incentive for rapid flight and relax the default arm-pose penalty
    # to allow future arm/leg timing learning.
    cfg.rewards.feet_air_time.weight=.45
    cfg.rewards.joint_deviation_arms.weight=-.065
    cfg.rewards.stance_knee_flexion.weight=-.42
    cfg.rewards.excessive_airborne.weight=-.12
    if p.variant=="v2":
        # V1 increased foot slip and trunk lean despite straighter knees.
        # Preserve the same stance reward, add corrective physics penalties.
        cfg.rewards.feet_slide.weight=-.65
        cfg.rewards.flat_orientation_l2.weight=-2.2
    cfg.commands.base_velocity.ranges.lin_vel_x=(.15,.75)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(-.3,.3)
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    ppo=G1FlatPPORunnerCfg()
    ppo.num_steps_per_env=24
    ppo.save_interval=8
    ppo.algorithm.learning_rate=p.lr
    ppo.algorithm.num_learning_epochs=4
    ppo.algorithm.num_mini_batches=4
    ppo.algorithm.entropy_coef=.004
    run_dir=TRAIN_DIR/f"{p.variant}_seed{p.seed}_env{p.num_envs}"
    run_dir.mkdir(exist_ok=True,parents=True)
    desc={
        "stage":"Reference-free biomechanics pilot, NOT human-motion imitation",
        "base_checkpoint":str(OFFICIAL),
        "original_sha256":hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
        "variant":p.variant,
        "num_envs":p.num_envs,"iterations":p.iterations,"seed":p.seed,"lr":p.lr,
        "reward_weights":{"stance_knee_flexion":-.42,"excessive_airborne":-.12,
                          "feet_air_time":.45,"arm_deviation":-.065,
                          "feet_slide":cfg.rewards.feet_slide.weight,
                          "flat_orientation_l2":cfg.rewards.flat_orientation_l2.weight},
        "baseline_gait":"reports/human_walk_baseline_v050.json",
        "requires_followup":["independent seed","equal baseline control budget",
             "held-out reference walking motion","human naturalness rater study"],
    }
    (run_dir/"manifest.json").write_text(json.dumps(desc,indent=2))
    print("GAIT_PILOT_SCENE_CREATE",json.dumps(desc),flush=True)
    faulthandler.dump_traceback_later(25,repeat=True)
    print("GAIT_PILOT_BEFORE_GYM_MAKE",flush=True)
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-v0",cfg=cfg),
        clip_actions=ppo.clip_actions)
    print("GAIT_PILOT_WRAPPED_ENV",env.num_envs,flush=True)
    print("GAIT_PILOT_REWARD_TERMS",env.unwrapped.reward_manager.active_terms,flush=True)
    runner=OnPolicyRunner(env,ppo.to_dict(),log_dir=str(run_dir),device="cuda:0")
    runner.load(str(OFFICIAL),load_optimizer=False)
    # Make the fine-tuning run its own 0-based iteration sequence.
    runner.current_learning_iteration=0
    if hasattr(runner.alg,"learning_rate"):
        runner.alg.learning_rate=p.lr
    print("GAIT_PILOT_WARM_START_LOADED","actor",type(runner.alg.policy).__name__,
          "environments",p.num_envs,flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=p.iterations,
                     init_at_random_ep_len=False)
    finally:
        runner.save(str(run_dir/"last_model.pt"))
        env.close()
        print("GAIT_PILOT_CHECKPOINT_SAVED",run_dir/"last_model.pt",
              "elapsed_wall_s",round(time.monotonic()-t0,1),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception:
        # Do not hide the actual exception behind Isaac Sim app.close(),
        # which can itself hang while its stop/render callback runs.
        import traceback
        traceback.print_exc()
        raise
    else:
        app.close()
