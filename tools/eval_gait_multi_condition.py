"""Matched physical G1 gait evaluation: 3 controllers × multiple commands in ONE scene.

OFFICIAL, V1, V2 share task/physics, deterministic startup, and seeds.
Each actor controls independent clone with no fixed base and no visual marker.
This is not a human mocap benchmark or proof of human likeness.

Run:
  python tools/eval_gait_multi_condition.py --seed 42 --steps 390
"""
from __future__ import annotations
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app

import argparse,copy,csv,json,math
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

ROOT=Path("/home/you/projects/kinetickick")
WEIGHTS={
    "OFFICIAL":ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt",
    "V1":ROOT/"experiments/gait_naturalness_pilot_20261021/seed40_env64/last_model.pt",
    "V2":ROOT/"experiments/gait_naturalness_pilot_20261021/v2_seed40_env64/last_model.pt",
}
MODELS=list(WEIGHTS)
SPEEDS=(.25,.50,.80)
SIDES=("left","right")
JOINTS=(
 "left_knee_joint","right_knee_joint",
 "left_hip_pitch_joint","right_hip_pitch_joint",
 "left_shoulder_pitch_joint","right_shoulder_pitch_joint",
)
FOOT_BODIES=tuple(f"{s}_ankle_roll_link" for s in SIDES)
TOOL_NOTE="Foot LINK contact/velocity proxies, not measured sole point of contact."

def pct(vals,p):
    return None if not vals else float(np.percentile(vals,p))

def summarize(rows,model,speed,seed,dt):
    active=[r for r in rows if r["step"]*dt>=.5 and not r["reset"]]
    if not active:return {"model":model,"speed_command_mps":speed,"seed":seed,"usable":False}
    left=[r["left_contact_N"]>45 for r in active]
    right=[r["right_contact_N"]>45 for r in active]
    metrics={
      "model":model,"speed_command_mps":speed,"seed":seed,
      "usable":True,"steps":len(rows),"analysis_steps":len(active),
      "episode_resets":sum(r["reset"] for r in rows),
      "mean_root_vx_mps":float(np.mean([r["root_vx"] for r in active])),
      "mean_abs_velocity_error_mps":float(np.mean([abs(r["root_vx"]-speed) for r in active])),
      "min_root_height_m":min(r["root_z"] for r in active),
      "median_root_height_m":pct([r["root_z"] for r in active],50),
      "base_tilt_p95_deg":pct([r["base_tilt_deg"] for r in active],95),
      "torso_link_tilt_p95_deg":pct([r["torso_link_tilt_deg"] for r in active],95),
      "double_support_fraction_45N":float(np.mean([l and rr for l,rr in zip(left,right)])),
      "no_support_fraction_45N":float(np.mean([not l and not rr for l,rr in zip(left,right)])),
      "contact_metric_caveat":TOOL_NOTE,
    }
    for side,mask in (("left",left),("right",right)):
        q_stance=[abs(r[f"{side}_knee"]) for r,b in zip(active,mask) if b]
        q_swing=[abs(r[f"{side}_knee"]) for r,b in zip(active,mask) if not b]
        slip=[r[f"{side}_foot_link_xy_speed"] for r,b in zip(active,mask) if b]
        metrics[f"{side}_stance_knee_abs_median_rad"]=pct(q_stance,50)
        metrics[f"{side}_swing_knee_abs_median_rad"]=pct(q_swing,50)
        metrics[f"{side}_stance_foot_link_xy_speed_p90_mps"]=pct(slip,90)
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--steps",type=int,default=390)
    ap.add_argument("--out",type=str)
    p=ap.parse_args()
    if not (40<=p.steps<=1600):raise ValueError("steps must be between 40 and 1600")
    mapping=[(model,float(speed)) for model in MODELS for speed in SPEEDS]
    n=len(mapping)
    assert n==9

    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=n
    cfg.scene.env_spacing=4.
    cfg.seed=p.seed
    cfg.sim.device="cuda:0"
    cfg.episode_length_s=35.
    cfg.observations.policy.enable_corruption=False
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,.80)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(1000.,1000.)
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    # Identical joint/start pose; no initial yaw or velocity randomness.
    cfg.events.reset_base.params["pose_range"]={
       "x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
       "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
       "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_robot_joints.params["position_range"]=(1.,1.)
    cfg.events.reset_robot_joints.params["velocity_range"]=(0.,0.)
    # Fixed material for fair model comparisons (no per-clone friction randomization).
    cfg.events.physics_material=None

    print("MATCHED_GAIT_SCENE_CREATE","num_envs",n,"seed",p.seed,
          "steps",p.steps,"conditions",mapping,flush=True)
    agent=G1FlatPPORunnerCfg()
    env=RslRlVecEnvWrapper(
       gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
       clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(WEIGHTS["OFFICIAL"]),load_optimizer=False)
    policies={}
    for name,path in WEIGHTS.items():
        net=copy.deepcopy(runner.alg.policy).to("cuda:0").eval()
        net.load_state_dict(torch.load(str(path),map_location="cpu",weights_only=False)["model_state_dict"],
                            strict=True)
        policies[name]=net
    robot=env.unwrapped.scene["robot"]
    foot_sensor=env.unwrapped.scene["contact_forces"]
    cmd=env.unwrapped.command_manager.get_term("base_velocity")
    ji={j:i for i,j in enumerate(robot.joint_names)}
    bi={j:i for i,j in enumerate(robot.body_names)}
    ci={j:i for i,j in enumerate(foot_sensor.body_names)}
    assert all(k in ji for k in JOINTS)
    assert all(k in bi and k in ci for k in FOOT_BODIES)
    assert "torso_link" in bi
    # Warm-start observation commands set BEFORE requesting the observation.
    req=torch.tensor([[speed,0.,0.] for _,speed in mapping],
                     device=robot.device,dtype=torch.float)
    cmd.vel_command_b[:]=req
    cmd.is_standing_env[:]=False
    cmd.time_left[:]=10000.0
    # Exact paired initial-world-velocity perturbations across 3 policies.
    # Model/speed group positions are different clone origins, but each
    # matched speed receives the SAME physical perturbation per seed.
    rng=np.random.default_rng(p.seed)
    disturbances={str(speed):[
        round(float(rng.uniform(-.085,.085)),6),
        round(float(rng.uniform(-.045,.045)),6)] for speed in SPEEDS}
    v0=robot.data.root_state_w[:,7:].clone()
    for i,(name,speed) in enumerate(mapping):
        dv=disturbances[str(speed)]
        v0[i,0]=dv[0]
        v0[i,1]=dv[1]
    robot.write_root_velocity_to_sim(v0)
    obs,_=env.get_observations()
    print("MATCHED_GAIT_PAIRED_INITIAL_PERTURBATIONS",
          "seed",p.seed,disturbances,flush=True)
    rows={i:[] for i in range(n)}
    dt=env.unwrapped.step_dt
    all_indices={name:torch.tensor([i for i,(model,sp) in enumerate(mapping) if model==name],
                                device=robot.device,dtype=torch.long)
                 for name in MODELS}
    for k in range(p.steps):
        cmd.vel_command_b[:]=req
        cmd.is_standing_env[:]=False
        cmd.time_left[:]=10000.0
        with torch.inference_mode():
            act=torch.zeros((n,37),dtype=obs.dtype,device=robot.device)
            for name,index in all_indices.items():
                act[index]=policies[name].act_inference(obs[index])
            obs,rew,dones,extras=env.step(act)
            rootxyz=robot.data.root_pos_w[:,:3].detach().cpu().numpy()
            rootvel=robot.data.root_lin_vel_w[:,:3].detach().cpu().numpy()
            base_g=robot.data.projected_gravity_b[:,2].detach().cpu().numpy()
            # Torso LINK is not the floating BASE: track its actual orientation.
            torso_q=robot.data.body_quat_w[:,bi["torso_link"]].detach().cpu().numpy()
            joint=robot.data.joint_pos.detach().cpu().numpy()
            footv=robot.data.body_lin_vel_w[:,[bi[k] for k in FOOT_BODIES],:].detach().cpu().numpy()
            cf=foot_sensor.data.net_forces_w[:,[ci[k] for k in FOOT_BODIES],:].detach().cpu().numpy()
            done=dones.detach().cpu().numpy()
        for env_id,(name,speed) in enumerate(mapping):
            q=torso_q[env_id]
            torso_up=np.clip(1.-2.*(q[1]**2+q[2]**2),-1.,1.)
            base_up=np.clip(-base_g[env_id],-1.,1.)
            row={
                "step":k,"t_s":k*dt,"model":name,"speed_command_mps":speed,"eval_seed":p.seed,
                "root_x":float(rootxyz[env_id,0]),
                "root_z":float(rootxyz[env_id,2]),
                "root_vx":float(rootvel[env_id,0]),
                "base_tilt_deg":float(math.degrees(math.acos(base_up))),
                "torso_link_tilt_deg":float(math.degrees(math.acos(torso_up))),
                "reset":int(done[env_id]),
            }
            for side,ii in (("left",0),("right",1)):
                row[f"{side}_knee"]=float(joint[env_id,ji[f"{side}_knee_joint"]])
                row[f"{side}_contact_N"]=float(np.linalg.norm(cf[env_id,ii]))
                row[f"{side}_foot_link_xy_speed"]=float(np.linalg.norm(footv[env_id,ii,:2]))
            rows[env_id].append(row)
        if k%80==0:
            print("MATCHED_GAIT_STEP",k,
                  "root_vx",[round(float(v),3) for v in rootvel[:,0]],
                  "reset_count",sum(sum(r["reset"] for r in ar) for ar in rows.values()),
                  flush=True)

    outfile=ROOT/(p.out or f"reports/gait_matched_seed{p.seed}.csv")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    with outfile.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0][0].keys()))
        writer.writeheader()
        for i in range(n):writer.writerows(rows[i])
    metrics=[summarize(rows[i],name,speed,p.seed,dt) for i,(name,speed) in enumerate(mapping)]
    output={"evaluation_seed":p.seed,"steps_per_condition":p.steps,
            "sim_dt":dt,"same_scene":True,
            "same_initial_pose":True,
            "paired_initial_velocity_disturbances_mps":disturbances,
            "scene_friction_randomization_disabled":True,
            "models":[str(WEIGHTS[x]) for x in MODELS],
            "summary":metrics,"raw_csv":str(outfile)}
    outfile.with_suffix(".json").write_text(json.dumps(output,indent=2))
    for m in metrics:
        print("MATCHED_GAIT_RESULT",json.dumps(
            {k:v for k,v in m.items() if k in (
                "model","speed_command_mps","mean_abs_velocity_error_mps",
                "left_stance_knee_abs_median_rad","right_stance_knee_abs_median_rad",
                "base_tilt_p95_deg","torso_link_tilt_p95_deg",
                "left_stance_foot_link_xy_speed_p90_mps",
                "right_stance_foot_link_xy_speed_p90_mps",
                "episode_resets")}),flush=True)
    env.close()
    print("MATCHED_GAIT_DONE",str(outfile.with_suffix(".json")),flush=True)

if __name__=="__main__":
    try:main()
    finally:app.close()
