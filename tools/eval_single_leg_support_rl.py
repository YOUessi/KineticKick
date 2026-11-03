"""Independent free-root single-leg support check: official G1 vs trained support PPO.

8 parallel physical environments with paired initial velocity perturbations:
4 clones OFFICIAL PPO + 4 clones SUPPORT trained PPO. Under the same Isaac
Lab flat G1 37DoF robot with body unfixed, measure:
 - true left/right sensor load, right ankle z clearance
 - trunk/base tilt and height; stance foot link slip
 - consecutive seconds of left-loaded & right-unloaded & right-foot raised
 - falling/episode reset
No soccer kick is attempted here. Neither model sees an additional command
that says it should kick. This does not assess shooting success.
"""
from __future__ import annotations
import argparse,copy,csv,json,math
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
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
  "SUPPORT":ROOT/"experiments/g1_single_leg_support/seed61_n96_h0.115/last_model.pt",
}
def clamp(x,lo,hi):
    return min(hi,max(lo,x))
def summarize(records,dt):
    if not records:return {"error":"no records"}
    data=np.asarray([r["stable_left_support"] for r in records],dtype=bool)
    runs=[];start=None
    for i,flag in enumerate(data.tolist()+[False]):
        if flag and start is None:start=i
        if not flag and start is not None:
            runs.append((i-start)*dt);start=None
    good=[r for r in records if not r["reset"]]
    if not good:good=records
    return {
      "duration_s":round(len(records)*dt,3),
      "stable_single_leg_longest_consecutive_s":round(max(runs,default=0.),3),
      "fraction_stable_left_support":round(float(data.mean()),4),
      "right_foot_raised_fraction":round(float(np.mean([r["right_link_z"]>.13 for r in good])),4),
      "left_load_fraction":round(float(np.mean([r["left_contact_N"]>35 for r in good])),4),
      "right_foot_unloaded_fraction":round(float(np.mean([r["right_contact_N"]<40 for r in good])),4),
      "right_link_z_median":round(float(np.median([r["right_link_z"] for r in good])),4),
      "left_foot_slip_mps_p90":round(float(np.percentile([r["left_foot_xy_speed_mps"] for r in good],90)),4),
      "root_height_min_m":round(float(min(r["root_z"] for r in good)),4),
      "root_tilt_deg_p95":round(float(np.percentile([r["root_tilt_deg"] for r in good],95)),3),
      "fell_or_reset":any(r["reset"] for r in records),
      "notes":"G1 ankle-roll-link position/force is proxy for actual foot-patch sole contact.",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=340)
    ap.add_argument("--seed",type=int,default=71)
    ap.add_argument("--out",type=str,default="reports/support_rl_independent_eval.json")
    opts=ap.parse_args()
    models=["OFFICIAL"]*4+["SUPPORT"]*4
    n=len(models)
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=n
    cfg.scene.env_spacing=4.
    cfg.sim.device="cuda:0"
    cfg.seed=opts.seed
    cfg.episode_length_s=30.
    cfg.observations.policy.enable_corruption=False
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(1000.,1000.)
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    cfg.events.reset_base.params["pose_range"]={"x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
       "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
       "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_robot_joints.params["position_range"]=(1.,1.)
    cfg.events.reset_robot_joints.params["velocity_range"]=(0.,0.)
    cfg.events.physics_material=None
    ppo=G1FlatPPORunnerCfg()
    print("SUPPORT_EVAL_CREATE_ENV",n,"paired_initial_conditions",opts.seed,flush=True)
    env=RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
                            clip_actions=ppo.clip_actions)
    runner=OnPolicyRunner(env,ppo.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(WEIGHTS["OFFICIAL"]),load_optimizer=False)
    policies={}
    for name,path in WEIGHTS.items():
        net=copy.deepcopy(runner.alg.policy).to("cuda:0").eval()
        net.load_state_dict(torch.load(str(path),map_location="cpu",weights_only=False)["model_state_dict"],
                            strict=True)
        policies[name]=net
    raw=env.unwrapped
    robot=raw.scene["robot"]
    sensor=raw.scene["contact_forces"]
    force_indices=[sensor.body_names.index(s+"_ankle_roll_link") for s in ("left","right")]
    bone_indices=[robot.body_names.index(s+"_ankle_roll_link") for s in ("left","right")]
    terms=raw.command_manager.get_term("base_velocity")
    terms.vel_command_b[:]=0.
    terms.time_left[:]=10000.
    terms.is_standing_env[:]=False
    # Perturbations are paired for corresponding official/support env clone;
    # same seed, setup and model geometry.
    rng=np.random.default_rng(opts.seed)
    perturbations=rng.uniform(low=[-.045,-.035],high=[.045,.035],size=(4,2))
    initial=robot.data.root_state_w[:,7:].clone()
    for j in range(4):
        for i in (j,j+4):
            initial[i,0]=float(perturbations[j,0])
            initial[i,1]=float(perturbations[j,1])
    robot.write_root_velocity_to_sim(initial)
    obs,_=env.get_observations()
    indices={name:torch.tensor([i for i,m in enumerate(models) if m==name],
                              dtype=torch.long,device=robot.device) for name in WEIGHTS}
    results={i:[] for i in range(n)}
    started=[False]*n
    dt=raw.step_dt
    for step in range(opts.steps):
        terms.vel_command_b[:]=0.
        terms.time_left[:]=10000.
        terms.is_standing_env[:]=False
        with torch.inference_mode():
            actions=torch.zeros((n,37),dtype=obs.dtype,device=robot.device)
            for name,ind in indices.items():
                actions[ind]=policies[name].act_inference(obs[ind])
            obs,rew,done,info=env.step(actions)
        forces=torch.linalg.vector_norm(sensor.data.net_forces_w[:,force_indices,:],dim=-1).detach().cpu().numpy()
        worldpos=robot.data.root_pos_w.detach().cpu().numpy()
        gravity=robot.data.projected_gravity_b.detach().cpu().numpy()
        anklepos=robot.data.body_pos_w[:,bone_indices,:].detach().cpu().numpy()
        anklevel=robot.data.body_lin_vel_w[:,bone_indices,:].detach().cpu().numpy()
        origins=raw.scene.env_origins.detach().cpu().numpy()
        lin=robot.data.root_lin_vel_w.detach().cpu().numpy()
        dones=done.detach().cpu().numpy()
        for i in range(n):
            if started[i]:continue
            root_z=float(worldpos[i,2]-origins[i,2])
            tilt=math.degrees(math.acos(clamp(-gravity[i,2],-1.,1.)))
            lf,rf=map(float,forces[i,:2])
            rz=float(anklepos[i,1,2]-origins[i,2])
            footslip=float(np.linalg.norm(anklevel[i,0,:2]))
            speed=float(np.linalg.norm(lin[i,:2]))
            reset=bool(dones[i])
            stable=(lf>35 and rf<40 and rz>.13 and root_z>.53
                    and tilt<12 and footslip<.16 and speed<.28 and not reset)
            results[i].append({"step":step,"root_z":root_z,"root_tilt_deg":tilt,
                 "left_contact_N":lf,"right_contact_N":rf,
                 "right_link_z":rz,"left_foot_xy_speed_mps":footslip,
                 "stable_left_support":bool(stable),"reset":reset})
            if reset:started[i]=True
        if step%85==0:
            print("SUPPORT_EVAL_STEP",step,
                  "stable",[sum(q["stable_left_support"] for q in results[i]) for i in range(n)],
                  "resets",[any(q["reset"] for q in results[i]) for i in range(n)],
                  "right_z",[round(float(x[1,2]-origins[j,2]),3) for j,x in enumerate(anklepos)],
                  flush=True)

    entries=[]
    for i in range(n):
        result=summarize(results[i],dt)
        result.update({"env_i":i,"model":models[i],
                       "initial_root_velocity_xy":[round(float(v),5) for v in perturbations[i%4]]})
        entries.append(result)
        print("SUPPORT_EVAL_RESULT",json.dumps(result),flush=True)
    for name in ("OFFICIAL","SUPPORT"):
        vals=[v for v in entries if v["model"]==name]
        print("SUPPORT_EVAL_MODEL_SUMMARY",name,
              "stable_support_s",[v["stable_single_leg_longest_consecutive_s"] for v in vals],
              "root_resets",sum(v["fell_or_reset"] for v in vals),flush=True)
    output=ROOT/opts.out
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({
      "seed":opts.seed,"steps":opts.steps,"dt_s":dt,
      "robot":"NVIDIA G1_MINIMAL 37DOF, 100% free root",
      "reward_training_model_not_necessarily_safe":True,
      "not_a_soccer_kick":True,
      "paired_initial_root_velocity_xy":perturbations.tolist(),
      "trials":entries
    },indent=2,ensure_ascii=False))
    print("SUPPORT_EVAL_DONE",str(output),flush=True)
    env.close()
if __name__=="__main__":
    try:main()
    finally:app.close()
