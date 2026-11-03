#!/usr/bin/env python3
"""KineticKick Gate B: PAiD Stage I baseline on the PAiD ORIGINAL 29DoF robot.

Preserves all upstream tracking rewards, randomization, and action semantics.
Free-base PhysX with the collision geometries from upstream main.urdf.
Use --resume-path to continue from a saved full RSL checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time, traceback, types
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app

import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

ROOT=pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
ASSET_ROOT=UPSTREAM/"soccer/assets"
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(ASSET_ROOT)
sys.modules["soccer.assets"]=shim

def parse():
    pa=argparse.ArgumentParser()
    pa.add_argument("--num-envs",type=int,default=64)
    pa.add_argument("--iters",type=int,default=160)
    pa.add_argument("--seed",type=int,default=74)
    pa.add_argument("--resume-path",type=str,default="")
    a=pa.parse_args()
    if not (1<=a.num_envs<=192 and 1<=a.iters<=20000):
        raise ValueError("Require 1-192 envs and 1-20000 iterations")
    return a

def main():
    args=parse()
    # Fail closed if the same-frame reference/model parity gate is not satisfied.
    audit=json.loads((ROOT/"reports/paid_reference_urdf_fk_audit.json").read_text())
    fk=audit["PAiD_original__isaac_runtime"]
    phys=json.loads((ROOT/"reports/paid_physics_parity_clean_paid.json").read_text())
    assert fk["matched_max_position_error_m"]<0.005
    assert phys["mean_error_m"]<0.005 and phys["max_error_m"]<0.03
    assert pathlib.Path(phys["URDF"]).resolve()==URDF.resolve()
    assert URDF.exists() and MOTION.exists()
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=args.num_envs
    cfg.scene.env_spacing=2.7
    cfg.sim.device="cuda:0"
    cfg.seed=args.seed
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    # All default PAiD train-time noise and material perturbations remain on.
    agent=G1FlatPPORunnerCfg()
    agent.num_steps_per_env=24
    agent.save_interval=20
    agent.algorithm.learning_rate=3e-4
    agent.algorithm.num_learning_epochs=4
    agent.algorithm.num_mini_batches=4
    run_dir=ROOT/f"experiments/paid29_stage1_original/seed{args.seed}_n{args.num_envs}_iter{args.iters}"
    if (run_dir/"last_model.pt").exists() and not args.resume_path:
        raise FileExistsError(f"Won't overwrite completed checkpoint: {run_dir}")
    run_dir.mkdir(parents=True,exist_ok=True)
    meta={
      "purpose":"UPSTREAM PAiD STAGE-I PHYSICS TRACKING BASELINE, NOT ORIGINAL ALGORITHM",
      "robot":"PAiD original G1 29DoF", "root_fixed":False,
      "urdf":str(URDF),"urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
      "motion":str(MOTION),"motion_sha256":hashlib.sha256(MOTION.read_bytes()).hexdigest(),
      "upstream":"https://github.com/TeleHuman/HumanoidSoccer",
      "iterations":args.iters, "num_envs":args.num_envs, "seed":args.seed,
      "resume_path":args.resume_path or None,
      "sim_stack":"Tang Isaac Sim 4.5 / Isaac Lab 2.1",
      "physical_policy":True,
      "mode":"baseline Stage I, no soccer outcome reward",
      "gate_A_parity_max_error_m":phys["max_error_m"],
      "license_note":"Source PAiD is CC BY-NC 4.0, do not publish original motion assets blindly.",
    }
    (run_dir/"manifest.json").write_text(json.dumps(meta,indent=2))
    print("PAID_STAGE1_START",json.dumps(meta),flush=True)
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),clip_actions=agent.clip_actions)
    robot=env.unwrapped.scene["robot"]
    assert len(robot.joint_names)==29 and len(robot.body_names)==30
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(run_dir),device="cuda:0")
    if args.resume_path:
        checkpoint=pathlib.Path(args.resume_path)
        assert checkpoint.exists(),checkpoint
        runner.load(str(checkpoint),load_optimizer=True)
        print("PAID_STAGE1_RESUMED",str(checkpoint),flush=True)
    print("PAID_STAGE1_READY",env.get_observations()[0].shape,flush=True)
    t0=time.monotonic()
    try:
        runner.learn(num_learning_iterations=args.iters,init_at_random_ep_len=False)
    finally:
        runner.save(str(run_dir/"last_model.pt"))
        print("PAID_STAGE1_CHECKPOINT",str(run_dir/"last_model.pt"),"seconds",round(time.monotonic()-t0,2),flush=True)
        env.close()
if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        print("PAID_STAGE1_FAILED",flush=True)
        raise
    else:
        app.close()
