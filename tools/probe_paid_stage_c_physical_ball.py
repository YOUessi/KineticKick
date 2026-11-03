#!/usr/bin/env python3
"""KineticKick Gate C probe: free-root learned G1 PPO versus ONE real rigid football.

No scripted kick, no external ball velocity, no frozen root, no action retargeting.
The frozen Stage I policy does not observe the ball; this is a controlled
contact-feasibility test and NOT a trained target-conditioned shooting policy.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import traceback
import types

parser=argparse.ArgumentParser()
parser.add_argument("--num-envs",type=int,default=12)
parser.add_argument("--seed",type=int,default=501)
parser.add_argument("--steps",type=int,default=310)
parser.add_argument("--output",type=str,default="")
parser.add_argument("--grid",choices=("coarse","fine"),default="coarse")
parser.add_argument("--root-policy",action="store_true")
parser.add_argument("--root-policy-v2",action="store_true")
parser.add_argument("--ball-x",type=float,default=None)
parser.add_argument("--ball-y",type=float,default=None)
parser.add_argument("--trace-csv",type=str,default="")
parser.add_argument("--trace-env",type=int,default=0)
args=parser.parse_args()
assert 1<=args.num_envs<=24
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True, enable_cameras=False).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

ROOT=pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
MODEL=ROOT/("experiments/paid29_global_root_v2/seed86_n96_iter200/last_model.pt" if args.root_policy_v2
          else "experiments/paid29_global_root_v1/seed85_n96_iter240/last_model.pt" if args.root_policy
          else "experiments/paid29_stage1_contact_phase/seed84_n96_iter200_phase108_148_contactv1/last_model.pt")
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(UPSTREAM/"soccer/assets")
sys.modules["soccer.assets"]=shim

def positions(grid):
    # World +X is across the robot; the forward axis in the reference is ~world -Y.
    if grid=="coarse":
        return [(x,y) for y in (-0.77,-0.93,-1.09,-1.25) for x in (0.04,0.14,0.24)]
    return [(x,y) for y in (-0.90,-1.00,-1.10,-1.20) for x in (-0.02,0.05,0.12,0.19,0.26,0.33)]

def main():
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    points=positions(args.grid)[:args.num_envs]
    if args.ball_x is not None or args.ball_y is not None:
        assert args.ball_x is not None and args.ball_y is not None and args.num_envs==1
        points=[(args.ball_x,args.ball_y)]
    assert len(points)==args.num_envs
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=args.num_envs
    cfg.scene.env_spacing=4.0
    cfg.sim.device="cuda:0"
    cfg.seed=args.seed
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.observations.policy.enable_corruption=False
    if args.root_policy or args.root_policy_v2:
        from soccer.tasks.tracking import mdp
        from isaaclab.envs import mdp as isaac_mdp
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        cfg.observations.policy.motion_anchor_pos_b=ObsTerm(func=mdp.motion_anchor_pos_b,params={"command_name":"motion"})
        cfg.observations.policy.motion_anchor_ori_b=ObsTerm(func=mdp.motion_anchor_ori_b,params={"command_name":"motion"})
        cfg.observations.policy.base_lin_vel=ObsTerm(func=isaac_mdp.base_lin_vel)
    for k in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[k]=(0.0,0.0)
    for k in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[k]=(0.0,0.0)
    cfg.commands.motion.joint_position_range=(0.0,0.0)
    cfg.terminations.ee_body_pos.params["threshold"]=0.25
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    cfg.scene.soccer_ball=RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim.SphereCfg(
            radius=0.11,
            collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=0.43),
            physics_material=sim.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.55, restitution=0.3),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.9,0.86,0.8))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.8,2.8,0.113)))
    cfg.scene.right_foot_ball_contact=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.0,
        history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        debug_vis=False)
    print("GATE_C_SCENE_CREATE","envs",args.num_envs,"points",points,flush=True)
    runner_cfg=G1FlatPPORunnerCfg()
    runner_cfg.num_steps_per_env=24
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),
                           clip_actions=runner_cfg.clip_actions)
    raw=env.unwrapped
    robot=raw.scene["robot"]
    motion=raw.command_manager.get_term("motion")
    ball=raw.scene["soccer_ball"]
    sensor=raw.scene["right_foot_ball_contact"]
    right_index=robot.body_names.index("right_ankle_roll_link")
    assert not cfg.scene.robot.spawn.fix_base
    assert len(raw.scene.rigid_objects)==1, list(raw.scene.rigid_objects)
    runner=OnPolicyRunner(env,runner_cfg.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(MODEL),load_optimizer=False)
    policy=runner.get_inference_policy(device="cuda:0")
    print("GATE_C_PHYSICAL_POLICY_READY","root_fixed",False,
          "ball_kg",0.43,"ball_radius_m",0.11,flush=True)
    obs,_=env.reset()
    # Critical: PAiD relative body targets remain uninitialized after reset.
    motion.time_steps[:]=-1
    motion._update_command()
    assert int(motion.time_steps.min())==0
    # Do not let the PAiD command generator teleport the real robot back to frame 0
    # when the motion finishes. Freeze the 259th reference during post-shot hold.
    native_update=motion._update_command
    def no_reference_teleport(self):
        cap=self.motion_length-1
        finished=self.time_steps>=cap
        if bool(finished.any()):
            self.time_steps[finished]=cap[finished]-1
        return native_update()
    motion._update_command=types.MethodType(no_reference_teleport,motion)
    print("GATE_C_REFERENCE_NO_TELEPORT",True,flush=True)
    ball_start=ball.data.default_root_state.clone()
    new_xyz=torch.tensor([[x,y,.113] for x,y in points],
                          dtype=ball_start.dtype,device=ball_start.device)
    ball_start[:,:3]=new_xyz+raw.scene.env_origins
    ball_start[:,7:]=0.0
    ball.write_root_pose_to_sim(ball_start[:,:7])
    ball.write_root_velocity_to_sim(ball_start[:,7:])
    ball.reset()
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    # Use per-environment arrays, no teleport or velocity writes during rollout.
    alive=torch.ones(args.num_envs,dtype=torch.bool,device=robot.device)
    episode_steps=np.full(args.num_envs,args.steps,dtype=np.int32)
    survived_reference=None
    first_contacts=[None]*args.num_envs
    max_contact_force=np.zeros(args.num_envs)
    max_ball_speed=np.zeros(args.num_envs)
    ball_speed_at_contact=np.zeros(args.num_envs)
    last_ball_xy=[None]*args.num_envs
    min_foot_ball_dist=np.full(args.num_envs,1e5)
    term_causes=[{} for _ in range(args.num_envs)]
    sensor_shape=None
    trace_rows=[]
    for t in range(args.steps):
        with torch.no_grad():
            action=policy(obs)
            obs,rew,done,_=env.step(action)
        just_done=done.to(torch.bool)&alive
        for idx in torch.nonzero(just_done,as_tuple=False).flatten().tolist():
            episode_steps[idx]=t+1
            term_causes[idx]={name:bool(raw.termination_manager.get_term(name)[idx])
                              for name in raw.termination_manager.active_terms}
        alive &= ~done.to(torch.bool)
        if t==259:
            survived_reference=alive.detach().cpu().tolist()
        bvel=ball.data.root_lin_vel_w.detach()
        bpos=ball.data.root_pos_w.detach()
        fpos=robot.data.body_pos_w[:,right_index,:].detach()
        ball_speed=torch.linalg.vector_norm(bvel[:,:2],dim=-1)
        distance=torch.linalg.vector_norm(bpos-fpos,dim=-1)
        if t in (0,59,79,89,99,109,119,129,139,149,159,179,199,219,239,259):
            root_local=(robot.data.root_pos_w[0]-raw.scene.env_origins[0]).cpu().tolist()
            foot_local=(fpos[0]-raw.scene.env_origins[0]).cpu().tolist()
            reference_index=motion.cfg.body_names.index("right_ankle_roll_link")
            foot_goal=(motion.body_pos_relative_w[0,reference_index]-raw.scene.env_origins[0]).detach().cpu().tolist()
            print("GATE_C_FOOT_TRACE",t+1,"root",[round(x,3) for x in root_local],
                  "right_foot",[round(x,3) for x in foot_local],
                  "ref_right_foot",[round(x,3) for x in foot_goal],
                  "ball_center_distance",round(float(distance[0]),3),flush=True)
        force_matrix=sensor.data.force_matrix_w
        if force_matrix is not None:
            sensor_shape=list(force_matrix.shape)
            force=torch.linalg.vector_norm(force_matrix.reshape(args.num_envs,-1,3),
                                           dim=-1).max(dim=-1).values
        else:
            force=torch.zeros_like(ball_speed)
        if args.trace_csv and bool(alive[args.trace_env]):
            ii=args.trace_env
            vball=bvel[ii].cpu().tolist()
            pball=(bpos[ii]-raw.scene.env_origins[ii]).cpu().tolist()
            pfoot=(fpos[ii]-raw.scene.env_origins[ii]).cpu().tolist()
            trace_rows.append({"step":t+1,"time_s":round((t+1)*float(raw.step_dt),4),
                "ball_x":round(pball[0],5),"ball_y":round(pball[1],5),"ball_z":round(pball[2],5),
                "ball_vx":round(vball[0],5),"ball_vy":round(vball[1],5),"ball_vz":round(vball[2],5),
                "ball_speed_xy":round(float(ball_speed[ii]),5),
                "right_foot_ball_force_N":round(float(force[ii]),5),
                "right_ankle_xyz":str([round(float(q),5) for q in pfoot]),
                "center_to_ankle_distance_m":round(float(distance[ii]),5)})
        for idx in torch.nonzero(alive,as_tuple=False).flatten().tolist():
            v=float(ball_speed[idx])
            f=float(force[idx])
            max_ball_speed[idx]=max(max_ball_speed[idx],v)
            max_contact_force[idx]=max(max_contact_force[idx],f)
            min_foot_ball_dist[idx]=min(min_foot_ball_dist[idx],float(distance[idx]))
            last_ball_xy[idx]=(bpos[idx,:2]-raw.scene.env_origins[idx,:2]).cpu().tolist()
            if f>2.0 and first_contacts[idx] is None:
                first_contacts[idx]=t+1
                ball_speed_at_contact[idx]=v
        if t in (0,29,79,99,114,124,139,159,199,259):
            print("GATE_C_STEP",t+1,"alive",int(alive.sum()),
                  "contact_envs",sum(c is not None for c in first_contacts),
                  "peak_ball_speed",round(float(max(max_ball_speed)),4),flush=True)
    if args.trace_csv and trace_rows:
        import csv
        p=pathlib.Path(args.trace_csv)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=list(trace_rows[0]))
            wr.writeheader()
            wr.writerows(trace_rows)
        print("GATE_C_CONTACT_TRACE",str(p),len(trace_rows),flush=True)
    records=[]
    for i,(x,y) in enumerate(points):
        bxy=last_ball_xy[i]
        dist_xy=[round(bxy[0]-x,5),round(bxy[1]-y,5)] if bxy is not None else None
        records.append({
          "env":i,"start_ball_xyz_m":[x,y,0.113],
          "ball_end_xy_world_local": [round(float(k),5) for k in bxy] if bxy else None,
          "ball_xy_displacement_m":dist_xy,
          "max_measured_ball_xy_speed_mps":round(float(max_ball_speed[i]),5),
          "first_filtered_right_foot_ball_force_step":first_contacts[i],
          "max_filtered_right_foot_ball_force_N":round(float(max_contact_force[i]),5),
          "ball_xy_speed_at_first_contact_mps":round(float(ball_speed_at_contact[i]),5),
          "min_center_to_right_ankle_link_m":round(float(min_foot_ball_dist[i]),5),
          "first_episode_steps":int(episode_steps[i]),
          "full_260_frames":bool(survived_reference[i]) if survived_reference is not None else None,
          "survived_to_final_step_including_hold":bool(alive[i]),
          "early_termination_causes":term_causes[i],
        })
    # Position arrays are logged as local coordinate differences, per cloned environment.
    report={"status":"STAGE_C_CONTACT_FEASIBILITY_UNTRAINED_SHOT",
            "checkpoint":str(MODEL),"reference":str(MOTION),
            "robot":"PAiD original G1, 29DoF, floating base",
            "ball_mass_kg":0.43,"radius_m":0.11,
            "policy_observes_ball":False,"root_reference_observation_enabled":bool(args.root_policy or args.root_policy_v2),
            "scripted_joint_actions":False,
            "direct_ball_velocity_writes_after_initialization":False,
            "filtered_sensor_force_matrix_shape":sensor_shape,
            "physics_dt_s":float(raw.physics_dt),"policy_dt_s":float(raw.step_dt),
            "total_complete":int(sum(bool(alive[i]) for i in range(args.num_envs))),
            "reference_survivors":int(sum(survived_reference)) if survived_reference is not None else None,
            "post_hold_steps":max(0,args.steps-260),
            "reference_teleport_prevented":True,
            "filtered_contact_count":sum(c is not None for c in first_contacts),
            "records":records,
            "limitations":["Ball placement preselected by grid, not policy-conditioned.",
                           "Foot-link distance is not exact surface contact.",
                           "Contact means measured filtered foot-ball force >2N; only this qualifies as verified robot-ball contact.",
                           "Motion is an existing reference with no approach-to-ball planner, trajectory target, or post-shot 1s recovery.",
                           "Only one soccer ball per independent scene; no extra shoe, fixed root or synthetic ball impulse."]}
    output=pathlib.Path(args.output or str(ROOT/"reports/gate_c_foot_ball_coarse.json"))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("GATE_C_REPORT",output,flush=True)
    print("GATE_C_RESULTS",json.dumps({
        "complete":report["total_complete"],
        "filtered_contact_count":report["filtered_contact_count"],
        "top_speed_mps":float(max(max_ball_speed)),
        "best_contact_trials":[r for r in records if r["first_filtered_right_foot_ball_force_step"] is not None]
    },ensure_ascii=False),flush=True)
    env.close()

if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("GATE_C_PROBE_FAILED",flush=True)
        raise
    else:
        app.close()
