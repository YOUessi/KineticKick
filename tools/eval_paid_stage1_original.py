#!/usr/bin/env python3
"""Frozen, physically simulated Stage-I PAiD policy evaluation (NO soccer ball).

Measures first-episode survival and tracking, distinct from rendered/kinematic replay.
A 260-step completion is only counted if no early episode reset occurred.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, traceback, types
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
import numpy as np
import torch
import gymnasium as gym
import isaaclab.sim as sim
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
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--num-envs",type=int,default=8)
    p.add_argument("--seed",type=int,default=501)
    p.add_argument("--frames",type=int,default=260)
    p.add_argument("--post-hold-steps",type=int,default=0)
    p.add_argument("--freeze-end",action="store_true")
    p.add_argument("--ee-threshold",type=float,default=0.25)
    p.add_argument("--perturb-level",choices=("none","mild","moderate"),default="none")
    p.add_argument("--root-observations",action="store_true")
    p.add_argument("--trace-csv",type=str,default="")
    p.add_argument("--output",type=str,required=True)
    opts=p.parse_args()
    assert pathlib.Path(opts.checkpoint).is_file()
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=opts.num_envs
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
    # Frozen controls; perturbation levels sample independent reset states by env.
    levels={"none":(0.0,0.0,0.0,0.0),"mild":(0.012,0.035,0.08,0.03),
            "moderate":(0.035,0.07,0.15,0.06)}
    trans,angular,vel,joint_jitter=levels[opts.perturb_level]
    for k in cfg.commands.motion.pose_range:
        span = trans if k in ("x","y","z") else angular
        cfg.commands.motion.pose_range[k]=(-span,span)
    for k in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[k]=(-vel,vel)
    cfg.commands.motion.joint_position_range=(-joint_jitter,joint_jitter)
    if opts.root_observations:
        from soccer.tasks.tracking import mdp
        from isaaclab.envs import mdp as isaac_mdp
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        cfg.observations.policy.motion_anchor_pos_b=ObsTerm(func=mdp.motion_anchor_pos_b,params={"command_name":"motion"})
        cfg.observations.policy.motion_anchor_ori_b=ObsTerm(func=mdp.motion_anchor_ori_b,params={"command_name":"motion"})
        cfg.observations.policy.base_lin_vel=ObsTerm(func=isaac_mdp.base_lin_vel)
    cfg.terminations.ee_body_pos.params["threshold"]=opts.ee_threshold
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    agent=G1FlatPPORunnerCfg()
    agent.num_steps_per_env=24
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),clip_actions=agent.clip_actions)
    raw=env.unwrapped
    robot=raw.scene["robot"]
    motion=raw.command_manager.get_term("motion")
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device="cuda:0")
    runner.load(opts.checkpoint,load_optimizer=False)
    policy=runner.get_inference_policy(device="cuda:0")
    print("PAID_EVAL_READY",opts.checkpoint,"num_envs",opts.num_envs,flush=True)
    # Explicit reset loads PAiD's motion pose; command-relative body arrays may
    # remain zero until the next command-manager update. Initialize them here.
    obs,_=env.reset()
    def show_init(phase):
        with torch.no_grad():
            diff=(motion.body_pos_relative_w-motion.robot_body_pos_w).abs()[0,:,-1]
            return {"phase":phase,"worst_abs_z_error_m":round(float(diff.max()),5),
                    "wrist_ref_z_m":round(float(motion.body_pos_relative_w[0,10,2]),5),
                    "wrist_actual_z_m":round(float(motion.robot_body_pos_w[0,10,2]),5),
                    "reference_time_step":int(motion.time_steps[0])}
    print("PAID_EVAL_COMMAND_INIT",json.dumps(show_init("before")),flush=True)
    # _update_command increments phase first. Start from -1 to synchronize frame 0.
    motion.time_steps[:]=-1
    motion._update_command()
    assert int(motion.time_steps[0])==0
    print("PAID_EVAL_COMMAND_INIT",json.dumps(show_init("after")),flush=True)
    if opts.freeze_end or opts.post_hold_steps:
        assert opts.frames==260, "Frozen-end protocol requires full 260-frame reference"
        native_update=motion._update_command
        def update_without_reference_wrap(self):
            # The stock MotionCommand resets/teleports the physical root when index == 260.
            # Freeze reference index to 259 instead; do NOT write root state or joint targets.
            cap=self.motion_length-1
            at_end=self.time_steps>=cap
            if bool(at_end.any()):
                self.time_steps[at_end]=cap[at_end]-1
            return native_update()
        motion._update_command=types.MethodType(update_without_reference_wrap,motion)
        print("PAID_EVAL_REFERENCE_NO_TELEPORT",True,flush=True)
    total_steps=opts.frames+opts.post_hold_steps
    done_early=torch.zeros(opts.num_envs,dtype=torch.bool,device="cuda:0")
    first_done=np.full(opts.num_envs,total_steps,dtype=np.int32)
    reference_alive_mask=None
    root_at_reference_end=None
    termination_causes={}
    root_heights, tracking_errors, foot_velocities=[],[],[]
    global_root_xy_errors=[]
    initial_root_xyz=robot.data.root_pos_w.detach().clone()
    final_live_root_xyz=initial_root_xyz.clone()
    initial_root_xy_cpu=robot.data.root_pos_w[:,:2].detach().cpu().numpy().copy()
    last_valid_root_xy_cpu=initial_root_xy_cpu.copy()
    support_slip_velocities=[]
    idx=[robot.body_names.index(z) for z in ["left_ankle_roll_link","right_ankle_roll_link"]]
    sensor=raw.scene["contact_forces"]
    sensor_idx=[sensor.body_names.index(z) for z in ["left_ankle_roll_link","right_ankle_roll_link"]]
    trace_rows=[]
    ref_foot_idx=[motion.cfg.body_names.index(z) for z in ["left_ankle_roll_link","right_ankle_roll_link"]]
    try:
        with torch.no_grad():
            for i in range(total_steps):
                if opts.trace_csv and not bool(done_early[0]):
                    target_z=float(motion.anchor_pos_w[0,2])
                    actual_z=float(motion.robot_anchor_pos_w[0,2])
                    ref_foot_z=motion.body_pos_relative_w[0,ref_foot_idx,2].cpu().tolist()
                    physical_foot_z=robot.data.body_pos_w[0,idx,2].cpu().tolist()
                    force_z=sensor.data.net_forces_w[0,sensor_idx,2].cpu().tolist()
                    ref_toe=motion.body_pos_relative_w[0,ref_foot_idx,2]
                    rob_toe=motion.robot_body_pos_w[0,ref_foot_idx,2]
                    pelvis_ref=motion.body_pos_w[0,0,:].detach().cpu().tolist()
                    pelvis_actual=robot.data.root_pos_w[0,:].detach().cpu().tolist()
                    trace_rows.append({"step":i,"reference_frame":int(motion.time_steps[0]),
                        "root_ref_x_m":round(pelvis_ref[0],5),"root_ref_y_m":round(pelvis_ref[1],5),
                        "root_actual_x_m":round(pelvis_actual[0],5),"root_actual_y_m":round(pelvis_actual[1],5),
                        "root_world_xy_error_m":round(float(torch.linalg.vector_norm(torch.tensor(pelvis_ref[:2])-torch.tensor(pelvis_actual[:2]))),5),
                        "torso_target_z_m":round(target_z,5),"torso_actual_z_m":round(actual_z,5),
                        "torso_abs_z_error_m":round(abs(target_z-actual_z),5),
                        "root_z_m":round(float(robot.data.root_pos_w[0,2]),5),
                        "left_ref_foot_z_m":round(ref_foot_z[0],5),"left_real_foot_z_m":round(physical_foot_z[0],5),
                        "right_ref_foot_z_m":round(ref_foot_z[1],5),"right_real_foot_z_m":round(physical_foot_z[1],5),
                        "max_foot_abs_z_error_m":round(float(torch.max(torch.abs(ref_toe-rob_toe))),5),
                        "left_foot_force_z_N":round(force_z[0],3),"right_foot_force_z_N":round(force_z[1],3)})
                action=policy(obs)
                obs,reward,done,extras=env.step(action)
                just_done=done.to(torch.bool)&(~done_early)
                first_done[just_done.cpu().numpy()]=i+1
                if just_done.any():
                    new_causes={name:int(raw.termination_manager.get_term(name)[just_done].sum().item())
                        for name in raw.termination_manager.active_terms}
                    for name,count in new_causes.items():
                        termination_causes[name]=termination_causes.get(name,0)+count
                    print("PAID_EVAL_TERMINATION",i+1,json.dumps(new_causes),flush=True)
                done_early|=done.to(torch.bool)
                live=~done_early
                if i==opts.frames-1:
                    reference_alive_mask=live.detach().cpu().numpy().copy()
                    root_at_reference_end=robot.data.root_pos_w[:,:2].detach().cpu().numpy().copy()
                if live.any():
                    final_live_root_xyz[live]=robot.data.root_pos_w[live].detach()
                    last_valid_root_xy_cpu[live.detach().cpu().numpy()]=robot.data.root_pos_w[live,:2].detach().cpu().numpy()
                    root_xy_err=torch.linalg.vector_norm(
                        robot.data.root_pos_w[live,:2]-motion.body_pos_w[live,0,:2],dim=-1)
                    global_root_xy_errors.extend(root_xy_err.cpu().tolist())
                    root_heights.extend(robot.data.root_pos_w[live,2].cpu().tolist())
                    pose_error=torch.linalg.vector_norm(motion.robot_body_pos_w[live]-motion.body_pos_relative_w[live],dim=-1)
                    tracking_errors.extend(pose_error.mean(dim=-1).cpu().tolist())
                    foot_speed=torch.linalg.vector_norm(robot.data.body_lin_vel_w[live][:,idx,:2],dim=-1)
                    foot_velocities.extend(foot_speed.mean(dim=-1).cpu().tolist())
                    foot_force_z=sensor.data.net_forces_w[live][:,sensor_idx,2]
                    in_support=foot_force_z>30.0
                    support_slip_velocities.extend(foot_speed[in_support].cpu().tolist())
                if i in (0,4,19,49,99,199,opts.frames-1):
                    print("PAID_EVAL_STEP",i+1,"survivors",int(live.sum()),
                          "root_height",round(float(robot.data.root_pos_w[:,2].mean()),3),flush=True)
    finally:
        env.close()
    if opts.trace_csv and trace_rows:
        import csv
        trace_path=pathlib.Path(opts.trace_csv)
        trace_path.parent.mkdir(parents=True,exist_ok=True)
        with trace_path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=trace_rows[0].keys())
            writer.writeheader()
            writer.writerows(trace_rows)
        print("PAID_EVAL_TRACE_CSV",str(trace_path),len(trace_rows),flush=True)
    def stats(v):
        return {"n":len(v),"mean":round(float(np.mean(v)),6) if v else None,
                "p95":round(float(np.percentile(v,95)),6) if v else None}
    result={
        "checkpoint":opts.checkpoint,
        "robot":"PAiD original 29DoF", "root_fixed":False,
        "clip":str(MOTION), "clip_frames":opts.frames,
        "post_hold_steps":opts.post_hold_steps,
        "prevent_reference_wrap_teleport":bool(opts.freeze_end or opts.post_hold_steps),
        "ee_z_termination_threshold_m":opts.ee_threshold,
        "seed":opts.seed, "n_envs":opts.num_envs,
        "reset_perturb_level":opts.perturb_level,
        "root_observations_enabled":bool(opts.root_observations),
        "unique_test_conditions":1 if opts.perturb_level=="none" else opts.num_envs,
        "caution":"Unperturbed parallel environments are identical; perturbed environments sample different seeded reset states but are not independent source motion clips.",
        "first_episode_steps":first_done.tolist(),
        "first_episode_termination_causes":termination_causes,
        "survivor_count":int((~done_early).sum().item()),
        "completion_rate":round(float((~done_early).float().mean().item()),4),
        "reference_completion_rate":round(float(np.mean(reference_alive_mask)),4) if reference_alive_mask is not None else None,
        "post_hold_success_rate":round(float((~done_early).float().mean().item()),4) if opts.post_hold_steps else None,
        "root_at_reference_end_world_xy_m":root_at_reference_end.round(5).tolist() if root_at_reference_end is not None else None,
        "mean_first_episode_steps":round(float(first_done.mean()),2),
        "root_height_m":stats(root_heights),
        "absolute_reference_root_xy_tracking_error_m":stats(global_root_xy_errors),
        "reference_end_minus_start_root_xy_m":(motion.motion._body_pos_w[0,-1,0,:2]-motion.motion._body_pos_w[0,0,0,:2]).detach().cpu().tolist(),
        "actual_end_minus_start_root_xy_m":(last_valid_root_xy_cpu-initial_root_xy_cpu).round(5).tolist(),
        "body_tracking_error_m":stats(tracking_errors),
        "ankle_horizontal_velocity_mps_proxy":stats(foot_velocities),
        "support_foot_slip_proxy_mps_Fz_gt_30N":stats(support_slip_velocities),
        "notes":["No ball was spawned; this is Stage I, not a successful kick.",
                 "Force-gated ankle rigid-body speed is only a support-slip proxy, not contact-point slip.",
                 "Unperturbed eval only; robustness requires additional seeds/disturbances."]
    }
    out=pathlib.Path(opts.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("PAID_EVAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        print("PAID_EVAL_FAILED",flush=True)
        raise
    else:
        app.close()
