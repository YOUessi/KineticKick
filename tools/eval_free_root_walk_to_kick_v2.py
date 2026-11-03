"""KineticKick free-root walk-to-ball + conditioned kick feasibility search.

REAL unfixed G1 (37DOF minimal collision) + actual PhysX 0.43kg football,
NVIDIA pretrained G1 gait, goal-directed approach, support-load guard, kick
residual, follow-through and recovery. 12 parallel trials, each with one ball.
The kick is a *scripted residual*, NOT learned from human videos. Reports
whether a COMPLETE dynamically stable goal-conditioned run-up-and-kick exists.
Does not invent success if robot falls or hits ball before strike.

Never set football linear velocity except once at initial spawn to 0.
"""
from __future__ import annotations
import argparse,json,math,collections
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False).app
import gymnasium as gym
import torch
import isaaclab_tasks
import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

ROOT=Path("/home/you/projects/kinetickick")
CKPT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"

def smooth(x):
    x=max(0.,min(1.,x));return x*x*(3-2*x)

def clamp(x,a,b):
    return max(a,min(x,b))

# x-position and right-foot swing residual strength: 4 x 3 = 12 simulations.
# x is absolute ball center in local env coordinates, G1 root initially x=0.
BALL_X=(1.22,1.30,1.38,1.46)
AMPLITUDES=(.30,.50,.75)
CASES=[{"ball_x":bx,"ball_y":-.11,"magnitude":amp,
        "name":f"bx{bx:.2f}_a{amp:.2f}"}
       for bx in BALL_X for amp in AMPLITUDES]
REQUIRED_JOINTS=[
    "right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint",
    "right_hip_roll_joint","left_hip_roll_joint",
    "left_shoulder_roll_joint","right_shoulder_roll_joint",
]
PHASES=("approach","settle","preload","strike","follow_through","recover","complete","abort")
PHASE_DURATIONS={"preload":.46,"strike":.32,"follow_through":.34,"recover":.80}

def desired_residual(phase,elapsed,strength,hip_shift=0.08):
    """Controlled joint position offsets relative to learned gait commands.
    These are a preliminary motor pattern (not copied human video angles).
    """
    m=strength
    if phase=="preload":
        w=smooth(elapsed/PHASE_DURATIONS["preload"])
        swing=(.28*w,.22*w,.25*w)
        lean=w
    elif phase=="strike":
        w=smooth(elapsed/PHASE_DURATIONS["strike"])
        swing=(.28-1.06*w,.22-.48*w,.25+.04*w)
        lean=1.
    elif phase=="follow_through":
        w=smooth(elapsed/PHASE_DURATIONS["follow_through"])
        swing=(-.78+.24*w,-.26+.20*w,.29-.04*w)
        lean=1.-.35*w
    elif phase=="recover":
        w=1.-smooth(elapsed/PHASE_DURATIONS["recover"])
        swing=(-.54*w,-.06*w,.25*w)
        lean=.65*w
    else:
        return {}
    # NOTE: Hip roll support shift direction sign must be empirically tested;
    # effects are logged and rejected if torso tilt/slip/foot contact fails.
    return {
        "right_hip_pitch_joint":m*swing[0],
        "right_knee_joint":m*swing[1],
        "right_ankle_pitch_joint":m*swing[2],
        "right_hip_roll_joint":m*(-.09)*lean,
        "left_hip_roll_joint":m*hip_shift*lean,
        "left_shoulder_roll_joint":m*(-.10)*lean,
        "right_shoulder_roll_joint":m*(+.10)*lean,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=550)
    ap.add_argument("--output",type=str,default="reports/shot_free_root_multitrial_v1.json")
    ap.add_argument("--n",type=int,default=12)
    ap.add_argument("--step_log_every",type=int,default=60)
    a=ap.parse_args()
    cases=CASES[:a.n]
    n=len(cases)
    if not (1<=n<=12):raise ValueError("n=1..12")
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=n;cfg.scene.env_spacing=5.
    cfg.sim.device="cuda:0";cfg.seed=42
    cfg.episode_length_s=30.
    cfg.observations.policy.enable_corruption=False
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,.55)
    cfg.commands.base_velocity.ranges.lin_vel_y=(-.20,.20)
    cfg.commands.base_velocity.ranges.ang_vel_z=(-.50,.50)
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.resampling_time_range=(100.,100.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.events.reset_base.params["pose_range"]={"x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
        "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
        "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.scene.football=RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/SoccerBall",
      spawn=sim.SphereCfg(
          radius=.11,rigid_props=sim.RigidBodyPropertiesCfg(),
          collision_props=sim.CollisionPropertiesCfg(),
          mass_props=sim.MassPropertiesCfg(mass=.43)),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(1.32,-.11,.113)))
    cfg.scene.foot_ball_contact=ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
      update_period=0.,history_length=2,
      filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"])

    runner_cfg=G1FlatPPORunnerCfg()
    print("FREE_ROOT_MULTITRIAL_SCENE_INIT",n,"trials",cases,flush=True)
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
        clip_actions=runner_cfg.clip_actions)
    runner=OnPolicyRunner(env,runner_cfg.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(CKPT),load_optimizer=False)
    policy=runner.get_inference_policy(device="cuda:0")
    raw=env.unwrapped
    robot=raw.scene["robot"]
    ball=raw.scene["football"]
    cmd=raw.command_manager.get_term("base_velocity")
    sensor=raw.scene["contact_forces"]
    hit=raw.scene["foot_ball_contact"]
    action_term=raw.action_manager.get_term("joint_pos")
    aids={name:i for i,name in enumerate(action_term._joint_names)}
    assert all(name in aids for name in REQUIRED_JOINTS)
    assert abs(action_term._scale-.5)<1e-6
    l_idx=sensor.body_names.index("left_ankle_roll_link")
    r_idx=sensor.body_names.index("right_ankle_roll_link")
    l_bidx=robot.body_names.index("left_ankle_roll_link")
    r_bidx=robot.body_names.index("right_ankle_roll_link")
    roots=raw.scene.env_origins

    # Place once during initialization, each ball at a local x offset matching
    # its case. No repositioning or speed injection during any action episode.
    bs=ball.data.default_root_state.clone()
    for i,c in enumerate(cases):
        bs[i,:3]=roots[i]+torch.tensor([c["ball_x"],c["ball_y"],.113],device=robot.device)
    ball.write_root_pose_to_sim(bs[:,:7])
    ball.write_root_velocity_to_sim(torch.zeros_like(bs[:,7:]))
    obs,_=env.get_observations()
    dt=raw.step_dt
    goal_local_x=4.1
    state=[dict(c,stage="approach",elapsed=0.,settle_steps=0,
                touched=False,first_contact=None,first_kick_contact=None,
                max_ball_horizontal_mps=0.,max_foot_ball_force_n=0.,
                after_hit_ball_x=None,initial_robot_x=None,
                reached_ready_step=None,
                abort_reason=None,initial_hit=False,
                supports_preload_s=0.,min_root_height_m=10.,
                max_root_tilt_deg=0.,max_left_foot_slip_mps=0.,
                swing_command_peak_rad=0.,physics_reset_count=0,
                strike_started_step=None,contact_phase=None,
                restabilized=False)
           for c in cases]
    stable_end=raw.episode_length_buf.cpu().tolist()
    print("FREE_ROOT_MULTITRIAL_INITIALIZED","dt",dt,
          "root_z",robot.data.root_pos_w[:,2].detach().cpu().tolist(),
          "ball_local_x",[(float(bs[i,0]-roots[i,0])) for i in range(n)],
          "foot_action_ids",{q:aids[q] for q in REQUIRED_JOINTS},flush=True)

    for step in range(a.steps):
        # Batched observations/physics; state feedback reads BEFORE command.
        root_local=robot.data.root_pos_w[:,:3]-roots
        rootxy=root_local[:,:2].detach().cpu().tolist()
        height=root_local[:,2].detach().cpu().tolist()
        yaw=robot.data.heading_w.detach().cpu().tolist()
        gv=robot.data.projected_gravity_b.detach().cpu().tolist()
        contacts=sensor.data.net_forces_w.detach().cpu()
        l_force=contacts[:,l_idx,:].norm(dim=-1).tolist()
        r_force=contacts[:,r_idx,:].norm(dim=-1).tolist()
        lv=robot.data.body_lin_vel_w[:,l_bidx,:2].norm(dim=-1).detach().cpu().tolist()
        rpos=(robot.data.body_pos_w[:,r_bidx,:3]-roots).detach().cpu().tolist()
        speeds=ball.data.root_lin_vel_w.detach().cpu().tolist()
        ballxyz=(ball.data.root_pos_w[:,:3]-roots).detach().cpu().tolist()
        rollouts=[]
        cmd_tensor=torch.zeros((n,3),device=robot.device)
        for i,(case,s) in enumerate(zip(cases,state)):
            xpos,ypos=rootxy[i]
            if s["initial_robot_x"] is None:s["initial_robot_x"]=xpos
            tilt=math.degrees(math.acos(clamp(-gv[i][2],-1.,1.)))
            s["min_root_height_m"]=min(s["min_root_height_m"],height[i])
            s["max_root_tilt_deg"]=max(s["max_root_tilt_deg"],tilt)
            if l_force[i]>30.:
                s["max_left_foot_slip_mps"]=max(s["max_left_foot_slip_mps"],lv[i])
            if s["stage"]=="approach":
                target=case["ball_x"]-.35
                dx=target-xpos
                vx=clamp(1.35*dx,0.,.55)
                vy=clamp(-.5*ypos,-.13,.13)
                omega=clamp(-1.0*yaw[i],-.35,.35)
                if abs(dx)<.085:
                    s["stage"]="settle"
                    s["elapsed"]=0.
                    s["reached_ready_step"]=step
                    vx=vy=omega=0.
                    print("FREE_ROOT_RUNUP_REACHED",i,case["name"],
                          "step",step,"root_xy",[round(v,3) for v in rootxy[i]],
                          "ball_rel_x",round(ballxyz[i][0]-xpos,3),flush=True)
                cmd_tensor[i]=torch.tensor([vx,vy,omega],device=robot.device)
            elif s["stage"]=="settle":
                target=case["ball_x"]-.35
                correction=clamp(1.2*(target-xpos),-.23,.23)
                cmd_tensor[i]=torch.tensor([correction,clamp(-.5*ypos,-.08,.08),clamp(-.8*yaw[i],-.2,.2)],device=robot.device)
            else:
                cmd_tensor[i]=0.0
            if s["stage"]=="settle":
                healthy=(height[i]>.53 and tilt<15. and l_force[i]>35.
                         and lv[i]<.35)
                if healthy:s["settle_steps"]+=1
                else:s["settle_steps"]=0
                # Diagnostic release after 4 consecutive physically loaded and
                # upright steps. The full strict 13-frame support criterion is
                # still a *separate* requirement for certification.
                if s["settle_steps"]>=4:
                    s["stage"]="preload";s["elapsed"]=0.
                    print("FREE_ROOT_BALANCE_GATE_PASSED",i,case["name"],
                          "step",step,"left_force",round(l_force[i],1),
                          "left_foot_speed",round(lv[i],3),
                          "root_tilt_deg",round(tilt,2),flush=True)
            elif s["stage"]=="preload":
                if l_force[i]>35.:
                    s["supports_preload_s"]+=dt
            if s["stage"] in ("preload","strike","follow_through","recover"):
                if height[i]<.49 or tilt>30.:
                    s["abort_reason"]="unsafe_stance"
                    s["stage"]="abort"
                    print("FREE_ROOT_SAFETY_ABORT",i,case["name"],
                          "step",step,"z",round(height[i],3),
                          "tilt",round(tilt,1),flush=True)
        cmd.vel_command_b[:]=cmd_tensor
        cmd.is_standing_env[:]=False
        cmd.time_left[:]=1000.
        with torch.inference_mode():
            inferred=policy(obs)
        act=inferred.clone() # writable torch Tensor outside inference mode

        for i,(case,s) in enumerate(zip(cases,state)):
            ph=s["stage"]
            if ph in PHASE_DURATIONS:
                delta=desired_residual(ph,s["elapsed"],case["magnitude"])
                for joint,offset in delta.items():
                    act[i,aids[joint]]+=offset/.5
                    s["swing_command_peak_rad"]=max(s["swing_command_peak_rad"],abs(offset))
        with torch.inference_mode():
            obs,_,dones,_=env.step(act)
        new_ball_v=ball.data.root_lin_vel_w.detach().cpu().tolist()
        fb=hit.data.force_matrix_w
        fb_force=fb.reshape(n,-1).norm(dim=-1).detach().cpu().tolist() if fb is not None else [0.]*n
        new_bxyz=(ball.data.root_pos_w[:,:3]-roots).detach().cpu().tolist()
        root_now=(robot.data.root_pos_w[:,:3]-roots).detach().cpu().tolist()
        done=dones.detach().cpu().tolist()
        for i,(case,s) in enumerate(zip(cases,state)):
            spd=math.hypot(new_ball_v[i][0],new_ball_v[i][1])
            s["max_ball_horizontal_mps"]=max(s["max_ball_horizontal_mps"],spd)
            s["max_foot_ball_force_n"]=max(s["max_foot_ball_force_n"],fb_force[i])
            if bool(done[i]):
                s["physics_reset_count"]+=1
                if s["abort_reason"] is None:s["abort_reason"]="physx_episode_terminated"
                s["stage"]="abort"
                print("FREE_ROOT_FAILED_RESET",i,case["name"],"step",step,flush=True)
            if fb_force[i]>2. and not s["touched"]:
                s["touched"]=True
                s["first_contact"]={
                    "step":step,"time_s":round(step*dt,4),
                    "phase":s["stage"],"force_N":round(fb_force[i],3),
                    "ball_velocity_xyz":[round(x,4) for x in new_ball_v[i]],
                    "root_z":round(root_now[i][2],4)}
                if s["stage"] in ("strike","follow_through"):
                    s["first_kick_contact"]=dict(s["first_contact"])
                else:s["initial_hit"]=True
                print("FREE_ROOT_BALL_CONTACT",i,case["name"],
                      json.dumps(s["first_contact"]),flush=True)
            stage=s["stage"]
            if stage in PHASE_DURATIONS:
                s["elapsed"]+=dt
                if s["elapsed"]>=PHASE_DURATIONS[stage]:
                    next_stage={
                       "preload":"strike","strike":"follow_through",
                       "follow_through":"recover","recover":"complete"
                    }[stage]
                    s["stage"]=next_stage;s["elapsed"]=0.
                    if next_stage=="strike":s["strike_started_step"]=step
                    print("FREE_ROOT_STAGE",i,case["name"],stage,"->",next_stage,
                          "step",step,flush=True)
            if s["stage"]=="complete":
                z=root_now[i][2]
                healthy=z>.53 and not bool(done[i])
                s["restabilized"]=s["restabilized"] or healthy
                if s["after_hit_ball_x"] is None:
                    s["after_hit_ball_x"]=new_bxyz[i][0]
        if step%a.step_log_every==0:
            print("FREE_ROOT_SUPPORT_DIAGNOSTIC",step,
                  "forceL",[round(x,1) for x in l_force],
                  "footSpeedL",[round(x,2) for x in lv],
                  "tilt",[round(math.degrees(math.acos(clamp(-x[2],-1.,1.))),1) for x in gv],
                  "gate_streak",[s["settle_steps"] for s in state],flush=True)
            print("FREE_ROOT_MULTITRIAL_STEP",step,
                  "stages",[x["stage"] for x in state],
                  "root_local_x",[round(v[0],2) for v in rootxy],
                  "ball_speed",[round(x["max_ball_horizontal_mps"],2) for x in state],
                  "done",[x["physics_reset_count"] for x in state],
                  flush=True)

    outcomes=[]
    for s in state:
        s["kick_success"]=bool(
          s["first_kick_contact"] and s["physics_reset_count"]==0
          and s["min_root_height_m"]>.49 and s["restabilized"]
          and s["max_ball_horizontal_mps"]>.40 and s["abort_reason"] is None
        )
        s["certified_stable_support_phase"]=False
        s["diagnostic_only_support_gate_4_frames"]=True
        s["ball_flight_heading_supported"]=False
        s["note"]="Scripted residual gait integration; NOT a learned human-like shooting controller."
        outcomes.append(s)
    report={
      "simulator":"Isaac Lab G1_MINIMAL 37DoF free root, real gravity, real feet + football",
      "scripted_kick_residual":True,"learned_football_shot":False,
      "pretrained_gait_only":True,"football_speed_injected":False,
      "headless_goal_location_relative_x":goal_local_x,
      "seconds_simulated":round(a.steps*dt,3),
      "cases":outcomes,
      "success_cases":[s["name"] for s in outcomes if s["kick_success"]]
    }
    path=ROOT/a.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    for s in outcomes:
        print("FREE_ROOT_TRIAL_RESULT",s["name"],
              "walk_arrived",s["reached_ready_step"] is not None,
              "hit_phase",s["first_contact"]["phase"] if s["first_contact"] else None,
              "ball_speed",round(s["max_ball_horizontal_mps"],3),
              "min_root_z",round(s["min_root_height_m"],3),
              "reset",s["physics_reset_count"],
              "stage",s["stage"],
              "success",s["kick_success"],
              flush=True)
    print("FREE_ROOT_MULTITRIAL_DONE","saved",path,
          "total_success",len(report["success_cases"]),flush=True)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()
