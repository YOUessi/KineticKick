"""PhysX G1 free-root goal-conditioned APPROACH / SETTLE benchmark.

Uses NVIDIA's official pretrained flat G1 locomotion controller; goal planner
calculates approach pose and desired heading. This experiment does NOT kick the
ball yet. It tests whether the robot can reach a goal-aligned shot location
without falling, then become stationary before the striking phase.

For target ground-truth coordinates: USD/simulation oracle, not RGB perception.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from isaaclab.app import AppLauncher

app=AppLauncher(headless=True,enable_cameras=False).app

import gymnasium as gym
import torch
import isaaclab_tasks
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from goal_conditioned_planner import goal_conditioned_plan

ROOT=Path("/home/you/projects/kinetickick")
CKPT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"

def wrap_pi(x):
    return (x+math.pi)%(2*math.pi)-math.pi

def cap(v,limit):
    return max(-limit,min(limit,v))

def goal_command(pos_xy,heading,approach_xy,target_heading):
    """Robot-base (vx,vy,wz). Only locomotion ranges seen during PPO training."""
    dx=approach_xy[0]-pos_xy[0]
    dy=approach_xy[1]-pos_xy[1]
    dist=math.hypot(dx,dy)
    desired=(cap(1.1*dx,.52),cap(1.1*dy,.52))
    ch=math.cos(heading);sh=math.sin(heading)
    body_x=ch*desired[0]+sh*desired[1]
    body_y=-sh*desired[0]+ch*desired[1]
    body_x=max(0.0,min(.60,body_x))
    body_y=cap(body_y,.25)
    omega=cap(1.3*wrap_pi(target_heading-heading),.62)
    if dist<.24:
        body_x=0.0
        body_y=0.0
    return body_x,body_y,omega,dist

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--goal_y",type=float,default=0.0)
    parser.add_argument("--steps",type=int,default=460)
    parser.add_argument("--out",type=str,default="reports/goal_approach_center.csv")
    args=parser.parse_args()
    if args.steps<20 or args.steps>1200:raise ValueError("steps must be 20..1200")
    ball=(2.0,0.0)
    goal=(5.0,args.goal_y)

    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=1
    # Add a real dynamic football; it can move only through PhysX contact.
    # This happens before gym.make so the rigid body is properly initialized.
    cfg.scene.football=RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.SphereCfg(radius=.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.96,.78,.18))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0,0.0,.113)),
    )
    cfg.scene.right_foot_ball_sensor=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    cfg.sim.device="cuda:0"
    cfg.seed=42
    cfg.episode_length_s=40.0
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.heading=(0.0,0.0)
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.0,0.6)
    cfg.commands.base_velocity.ranges.lin_vel_y=(-.25,.25)
    cfg.commands.base_velocity.ranges.ang_vel_z=(-.62,.62)
    cfg.commands.base_velocity.rel_standing_envs=0.0
    cfg.commands.base_velocity.resampling_time_range=(50.0,50.0)
    cfg.events.reset_base.params["pose_range"]={
        "x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
        "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
        "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.observations.policy.enable_corruption=False
    agent=G1FlatPPORunnerCfg()

    print("APPROACH_CREATING_SIM",flush=True)
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
        clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(CKPT))
    inference=runner.get_inference_policy(device="cuda:0")
    robot=env.unwrapped.scene["robot"]
    soccer_ball=env.unwrapped.scene["football"]
    ball_contact=env.unwrapped.scene["right_foot_ball_sensor"]
    print("SOCCER_BALL_BODY_ACTIVE",soccer_ball.is_initialized,
          "ball_xyz",soccer_ball.data.root_pos_w[0].cpu().tolist(),flush=True)
    cmd=env.unwrapped.command_manager.get_term("base_velocity")
    sensor=env.unwrapped.scene["contact_forces"]
    left_i=sensor.body_names.index("left_ankle_roll_link")
    right_i=sensor.body_names.index("right_ankle_roll_link")
    obs,_=env.get_observations()
    pos=robot.data.root_pos_w[0].detach().cpu().tolist()
    plan=goal_conditioned_plan(ball_xy=ball,goal_xy=goal,
             robot_xy=pos[:2],surface="instep_top",target_ball_speed_mps=3.0)
    target=plan["approach_xy"]
    preplant_target=plan["preplant_base_xy"]
    support_foot_target=plan["support_foot_plant_xy"]
    foot_body_i=robot.body_names.index("left_ankle_roll_link")
    stage="approach"
    phase_switch_step=None
    best_foot_error=float("inf")
    support_ready_steps=0
    support_stable_at=None
    heading_target=plan["shot_heading_rad"]
    stable_steps=0
    phase="approach"
    resets=0
    min_root_height=999.
    max_ball_contact_force=0.
    rows=[]
    previous=None
    stop_since=None
    step_dt=env.unwrapped.step_dt
    print("APPROACH_PLAN",json.dumps({"target":target,
           "heading_deg":plan["shot_heading_deg"],
           "ball":ball,"goal":goal}),flush=True)

    for step in range(args.steps):
        pos=robot.data.root_pos_w[0].detach().cpu().tolist()
        heading=float(robot.data.heading_w[0].item())
        vx,vy,wz,dist=goal_command(pos[:2],heading,target,heading_target)
        yaw_error=wrap_pi(heading_target-heading)
        if stage=="approach" and dist<.24 and abs(yaw_error)<.25:
            stage="preplant"
            target=preplant_target
            phase_switch_step=step
            print("GAIT_SWITCH_TO_PREPLANT",step,"target",target,flush=True)
        vx,vy,wz,dist=goal_command(pos[:2],heading,target,heading_target)
        if stage=="preplant" and dist<.24 and abs(yaw_error)<.25:
            phase="preplant_hold"
            cmd.vel_command_b[0]=torch.tensor([0.,0.,0.],device=robot.device)
        else:
            phase=stage
            stable_steps=0
            cmd.vel_command_b[0]=torch.tensor([vx,vy,wz],device=robot.device)
        cmd.is_standing_env[:]=False
        cmd.time_left[:]=1000.0
        with torch.inference_mode():
            act=inference(obs)
            obs,reward,done,extras=env.step(act)
        contact_force=sensor.data.net_forces_w[0]
        lc=float(torch.linalg.vector_norm(contact_force[left_i]).item())
        rc=float(torch.linalg.vector_norm(contact_force[right_i]).item())
        pf=ball_contact.data.force_matrix_w
        ball_foot_force=float(torch.linalg.vector_norm(pf).item()) if pf is not None and pf.numel() else 0.0
        max_ball_contact_force=max(max_ball_contact_force,ball_foot_force)
        pg=robot.data.projected_gravity_b[0].detach().cpu().tolist()
        xyz=robot.data.root_pos_w[0].detach().cpu().tolist()
        root_vel=robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        upright=pg[2]<-.90
        foot_xyz=robot.data.body_pos_w[0,foot_body_i].detach().cpu().tolist()
        foot_vel=robot.data.body_lin_vel_w[0,foot_body_i].detach().cpu().tolist()
        foot_error=math.hypot(foot_xyz[0]-support_foot_target[0],foot_xyz[1]-support_foot_target[1])
        best_foot_error=min(best_foot_error,foot_error)
        foot_slip=math.hypot(foot_vel[0],foot_vel[1])
        h=xyz[2]
        min_root_height=min(h,min_root_height)
        reset=int(done[0].item())
        resets+=reset
        if reset:
            phase="fall_or_task_reset"
            stable_steps=0
        elif phase=="preplant_hold" and upright and h>.48 and (lc>5 or rc>5):
            stable_steps+=1
        else:
            stable_steps=0
        if stable_steps>=25 and stop_since is None:
            stop_since=step
            print("PREPLANT_BASE_ZONE_STABLE",step,flush=True)
        # True support-foot placement needs left foot specifically to be grounded,
        # close to its target, and nearly stationary. A generic walking policy
        # might not be capable of selecting the desired contact timing.
        if phase=="preplant_hold" and upright and lc>30.0 and foot_error<.20 and foot_slip<.08:
            support_ready_steps+=1
        else:
            support_ready_steps=0
        if support_ready_steps>=15 and support_stable_at is None:
            support_stable_at=step
            print("SUPPORT_LEFT_FOOT_PLANT_VERIFIED",step,flush=True)
        rows.append({
            "step":step,"t_s":round(step*step_dt,3),
            "root_x":round(xyz[0],5),"root_y":round(xyz[1],5),
            "root_z":round(h,5),"heading_deg":round(math.degrees(float(robot.data.heading_w[0])),3),
            "approach_error_m":round(dist,5),
            "heading_error_deg":round(math.degrees(yaw_error),3),
            "cmd_vx":round(vx,4),"cmd_vy":round(vy,4),"cmd_omega":round(wz,4),
            "root_world_vx":round(root_vel[0],4),
            "root_world_vy":round(root_vel[1],4),
            "left_ground_force_n":round(lc,2),"right_ground_force_n":round(rc,2),
            "left_support_foot_error_m":round(foot_error,4),
            "left_support_foot_slip_mps":round(foot_slip,4),
            "projected_gravity_z":round(pg[2],5),
            "ball_foot_contact_N":round(ball_foot_force,3),
            "ball_root_x":round(float(soccer_ball.data.root_pos_w[0,0]),4),
            "ball_root_y":round(float(soccer_ball.data.root_pos_w[0,1]),4),
            "phase":phase,"reset":reset})
        if step%50==0 or reset:
            print("APPROACH_STEP",step,phase,"xyz",[round(x,3) for x in xyz],
                  "dist",round(dist,3),"heading_err",round(math.degrees(yaw_error),2),
                  "contacts",round(lc),round(rc),"left_target_error",round(foot_error,3),
                  "left_slip",round(foot_slip,3),"done",reset,flush=True)
    out=ROOT/args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary={
        "no_fixed_root":True,
        "training_source":"NVIDIA G1 flat pretrained PPO; NOT trained for soccer",
        "goal":goal,"ball":ball,"target_approach_xy":plan["approach_xy"],
        "target_preplant_base_xy":preplant_target,
        "target_support_left_foot_xy":support_foot_target,
        "approach_to_preplant_switch_step":phase_switch_step,
        "min_support_left_foot_target_error_m":round(best_foot_error,4),
        "support_left_foot_stationary_verified_step":support_stable_at,
        "shot_heading_deg":plan["shot_heading_deg"],
        "steps":args.steps,"duration_s":round(args.steps*step_dt,2),
        "min_root_height_m":round(min_root_height,3),
        "resets":resets,
        "dynamic_soccer_ball_present":True,
        "max_foot_ball_contact_force_N":round(max_ball_contact_force,4),
        "ball_final_xyz":[round(float(x),3) for x in soccer_ball.data.root_pos_w[0].cpu().tolist()],
        "stable_plant_zone_first_step":stop_since,
        "final_root_xyz":[round(x,3) for x in xyz],
        "final_distance_to_approach_xy_m":round(math.hypot(xyz[0]-target[0],xyz[1]-target[1]),3),
        "final_heading_error_deg":round(math.degrees(wrap_pi(heading_target-float(robot.data.heading_w[0]))),2),
        "note":"Goal-conditioned approach + preplant base + ACTUAL left support foot placement test; NO kick policy. support_ready_steps verifies consecutive foot error/contact/slip, not only root proximity. Stage truth is USD, not RGB.",
    }
    out.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("GOAL_GUIDED_PREPLANT_DONE",json.dumps(summary,ensure_ascii=False),flush=True)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()
