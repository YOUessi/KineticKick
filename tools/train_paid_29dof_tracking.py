"""Run PAiD G1 29DoF physically constrained Stage-1 motion tracking PPO.

Non-destructive KineticKick reproduction adapter for Tang's sparse PAiD repo.
No modification to Isaac Lab or RSL-RL packages, no W&B, no GUI.
Uses the task shipped by TeleHuman/HumanoidSoccer in Isaac Lab v2.1 family.
The retargeted soccer motion is KINEMATIC REFERENCE, policy must learn dynamic
tracking with free pelvis, contacts and torque limits.

First run --num-envs 8 --iters 4 to validate training, then scale conditionally.
"""
from __future__ import annotations
import sys,types,json,time,hashlib,traceback,argparse
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
ROOT=Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(ROOT/"data/assets/g1_29dof_lafan1")
sys.modules["soccer.assets"]=shim

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--num-envs",type=int,default=8)
    pa.add_argument("--iters",type=int,default=4)
    pa.add_argument("--seed",type=int,default=74)
    opts=pa.parse_args()
    if not 1<=opts.num_envs<=256:raise ValueError("num-envs must be in 1..256")
    if not 1<=opts.iters<=20000:raise ValueError("iters must be in 1..20000")
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=opts.num_envs
    cfg.scene.env_spacing=2.7
    cfg.sim.device="cuda:0"
    cfg.seed=opts.seed
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
    agent=G1FlatPPORunnerCfg()
    agent.num_steps_per_env=24
    agent.save_interval=20
    agent.algorithm.learning_rate=3e-4
    agent.algorithm.num_learning_epochs=4
    agent.algorithm.num_mini_batches=4
    run_dir=ROOT/f"experiments/paid29_physical_tracking/seed{opts.seed}_n{opts.num_envs}_iter{opts.iters}"
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest={
      "purpose":"UPSTREAM-PAiD PHYSICAL MOTION-TRACKING BASELINE, NOT OUR ALGORITHM",
      "robot":"Unitree G1 29DoF rev1_0",
      "root_fixed":False,
      "urdf":str(URDF),"urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
      "motion_file":str(MOTION),
      "motion_sha256":hashlib.sha256(MOTION.read_bytes()).hexdigest(),
      "upstream":"https://github.com/TeleHuman/HumanoidSoccer",
      "training_envs":opts.num_envs,"ppo_iters":opts.iters,
      "seed":opts.seed,
      "simulation_stack":"Tang Isaac Sim 4.5 / Isaac Lab 2.1",
      "do_not_claim_humanlike_shooting_until_eval":True
    }
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("PAID_29_TRACKING_TRAIN_INITIALIZING",json.dumps(manifest),flush=True)
    env=RslRlVecEnvWrapper(
        gym.make("Tracking-Flat-G1-v0",cfg=cfg),
        clip_actions=agent.clip_actions)
    robot=env.unwrapped.scene["robot"]
    assert len(robot.joint_names)==29
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(run_dir),device="cuda:0")
    print("PAID_29_TRACKING_TRAIN_READY","obs_shape",env.get_observations()[0].shape,
          "actions",29,flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=opts.iters,init_at_random_ep_len=False)
    finally:
        runner.save(str(run_dir/"last_model.pt"))
        print("PAID_29_TRACKING_TRAIN_SAVED",
              str(run_dir/"last_model.pt"),
              "elapsed_seconds",round(time.monotonic()-t0,2),flush=True)
        env.close()
if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("PAID_29_TRACKING_TRAIN_FAILED",flush=True)
        raise
    else:
        app.close()
