"""Tang real-time G1 walking -> goal-aligned pre-kick setup.

This is a GENUINE pretrained 37-DoF G1 policy with unfixed robot base,
a real dynamic 0.43 kg PhysX soccer ball, and an Isaac USD goal model.

IMPORTANT RESEARCH LIMITATION:
  * This demo does NOT yet perform the kick or run human-video motion priors.
  * Goal location is USD scene truth, NOT camera object detection.
  * A planted-foot target marker is a PLAN, NOT achieved contact.
  * Balancing/approach commands come from official pre-trained locomotion policy.
"""
from __future__ import annotations
import csv,math,time,json
from pathlib import Path
from isaaclab.app import AppLauncher

app=AppLauncher(headless=False,enable_cameras=False).app

import gymnasium as gym
import torch
import omni.ui as ui
import omni.usd
from pxr import Gf,UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab_tasks
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from goal_conditioned_planner import goal_conditioned_plan
from soccer_world import create_goal,move_goal,locate_goal_in_usd

ROOT=Path("/home/you/projects/kinetickick")
CKPT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
BALL=(2.0,0.0)
GOAL_X=5.0

def wrap_pi(x):
    return (x+math.pi)%(2*math.pi)-math.pi

def clamp(x,a,b):
    return max(a,min(b,x))

def approach_command(xy,theta,target_xy,target_yaw):
    dx,dy=target_xy[0]-xy[0],target_xy[1]-xy[1]
    d=math.hypot(dx,dy)
    desired_x=clamp(1.1*dx,-.52,.52)
    desired_y=clamp(1.1*dy,-.52,.52)
    c,s=math.cos(theta),math.sin(theta)
    vx=clamp(c*desired_x+s*desired_y,0.,.6)
    vy=clamp(-s*desired_x+c*desired_y,-.25,.25)
    wz=clamp(1.3*wrap_pi(target_yaw-theta),-.62,.62)
    if d<.24:
        vx=0.;vy=0.
    return vx,vy,wz,d

def create_training_stage():
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=1
    cfg.sim.device="cuda:0"
    cfg.seed=42
    cfg.episode_length_s=7.6  # short replay loop: auto-truncation by Gym/RL environment
    cfg.observations.policy.enable_corruption=False
    cfg.events.reset_base.params["pose_range"]={
        "x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
        "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
        "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,.6)
    cfg.commands.base_velocity.ranges.lin_vel_y=(-.25,.25)
    cfg.commands.base_velocity.ranges.ang_vel_z=(-.62,.62)
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(50.,50.)
    cfg.scene.football=RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.SphereCfg(radius=.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            visual_material=sim_utils.PreviewSurfaceCfg(
              diffuse_color=(.97,.80,.18),roughness=.58)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.,0.,.113)),
    )
    cfg.scene.right_foot_ball_sensor=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    agent=G1FlatPPORunnerCfg()
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
        clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(CKPT))
    policy=runner.get_inference_policy(device="cuda:0")
    return env,policy

class Panel:
    def __init__(self,state):
        self.win=ui.Window("KineticKick | MOVING G1 AUTO LOOP",width=500,height=385,visible=True)
        self.win.position_x=925
        self.win.position_y=66
        with self.win.frame:
            with ui.VStack(spacing=7):
                ui.Label("UNFIXED G1 | PRETRAINED RL WALKING + PHYSX BALL",height=25,
                    style={"font_size":16})
                with ui.HStack(height=30,spacing=5):
                    ui.Button("GOAL LEFT",clicked_fn=lambda:self.goal(state,-.6))
                    ui.Button("GOAL CENTER",clicked_fn=lambda:self.goal(state,0.))
                    ui.Button("GOAL RIGHT",clicked_fn=lambda:self.goal(state,.6))
                with ui.HStack(height=30,spacing=5):
                    ui.Button("RESTART WALK",clicked_fn=lambda:self.restart(state))
                    ui.Button("RUN APPROACH",clicked_fn=lambda:self.pause(state,False))
                    ui.Button("HOLD POSITION",clicked_fn=lambda:self.pause(state,True))
                self.status=ui.Label("AUTO: WALK -> APPROACH -> RESET -> REPEAT",height=23)
                self.state=ui.Label("PHASE: launching...",height=22)
                self.pose=ui.Label("G1 root XYZ / yaw: --",height=20)
                self.ball=ui.Label("Ball XYZ (dynamic): --",height=20)
                self.distance=ui.Label("Approach position error: --",height=20)
                self.contacts=ui.Label("Feet ground forces L/R: --",height=20)
                self.goalstatus=ui.Label("Goal yaw and selected plan: --",height=20)
                self.support=ui.Label("Left support foot target error: --",height=20)
                self.no_kick=ui.Label("NO KICK POLICY YET - run-up/plant baseline ONLY",height=20)
    @staticmethod
    def goal(s,y):
        s["goal_y"]=y
        s["reset"]=True
    @staticmethod
    def restart(s):
        s["reset"]=True
    @staticmethod
    def pause(s,yes):
        s["hold"]=bool(yes)


def create_plan_markers():
    stage=omni.usd.get_context().get_stage()
    root=UsdGeom.Xform.Define(stage,"/World/ShotPlanner")
    markers={}
    for name,color,size in (
        ("Approach",(.18,.93,.25),.095),
        ("SupportFoot",(.94,.34,.16),.060),
    ):
        path=f"/World/ShotPlanner/{name}"
        marker=UsdGeom.Sphere.Define(stage,path)
        marker.CreateRadiusAttr(size)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf=UsdGeom.Xformable(marker)
        xf.AddTranslateOp()
        markers[name]=marker
    return markers

def set_marker(marker,pos):
    xf=UsdGeom.Xformable(marker)
    op=xf.GetOrderedXformOps()[0]
    op.Set(Gf.Vec3d(float(pos[0]),float(pos[1]),.055))

def main():
    env,policy=create_training_stage()
    raw=env.unwrapped
    robot=raw.scene["robot"]
    football=raw.scene["football"]
    # Isaac Lab G1 flat scene has a nonzero local env-origin Z in this setup.
    # Automatic resets add it to *both* default robot and football heights;
    # without correction the second episode starts with a floating ball.
    env_origin_before=raw.scene.env_origins.detach().cpu().tolist()
    default_robot_xyz=robot.data.default_root_state[0,:3].detach().cpu().tolist()
    default_ball_xyz=football.data.default_root_state[0,:3].detach().cpu().tolist()
    print("ENV_RESET_SCHEMA_BEFORE",env_origin_before,default_robot_xyz,default_ball_xyz,flush=True)
    if float(raw.scene.env_origins[:,2].abs().max())>1.0e-4:
        raw.scene.env_origins[:,2]=0.0
        print("ENV_ORIGIN_HEIGHT_CORRECTED_FOR_REPEATED_RESET",
              raw.scene.env_origins.detach().cpu().tolist(),flush=True)
    ball_contacts=raw.scene["right_foot_ball_sensor"]
    contact=raw.scene["contact_forces"]
    left_id=contact.body_names.index("left_ankle_roll_link")
    right_id=contact.body_names.index("right_ankle_roll_link")
    foot_body_id=robot.body_names.index("left_ankle_roll_link")
    cmd=raw.command_manager.get_term("base_velocity")
    goal=create_goal(goal_x=GOAL_X,initial_y=.6)
    markers=create_plan_markers()
    raw.sim.set_camera_view(eye=[3.25,-4.05,2.07],target=[1.65,-.05,.69])

    state={"goal_y":.6,"reset":True,"replan":False,"hold":False,"stage":"approach","episode":0}
    panel=Panel(state)
    obs,_=env.get_observations()
    plan=None
    iteration=0
    falls=0
    foot_ball_peak=0.
    contact_status="NOT STRUCK"
    records=[]
    out=ROOT/"reports/live_free_g1_goal_approach.csv"
    out.parent.mkdir(exist_ok=True,parents=True)
    while app.is_running():
        if state["reset"] and plan is not None:
            # NEVER call env.reset() again from a live Isaac Sim GUI render loop.
            # The previous version froze here after 580 steps. Instead, ask
            # the RL environment to terminate on its next normal env.step().
            raw.episode_length_buf[:] = raw.max_episode_length - 1
            state["reset"]=False
            print("LIVE_RESTART_REQUEST_SCHEDULED",iteration,flush=True)
        elif state["reset"] or state["replan"]:
            move_goal(goal,state["goal_y"])
            # After a normal RSL-RL environment reset, policy(obs) already
            # contains the reset observations. Do not call env.reset() here.
            ball_xyz=football.data.root_pos_w[0].cpu().tolist()
            root=robot.data.root_pos_w[0].cpu().tolist()
            goal_xyz=locate_goal_in_usd()
            plan=goal_conditioned_plan(ball_xy=ball_xyz[:2],
                goal_xy=goal_xyz[:2],robot_xy=root[:2],
                foot="right",surface="instep_top")
            set_marker(markers["Approach"],plan["approach_xy"])
            set_marker(markers["SupportFoot"],plan["support_foot_plant_xy"])
            cmd.is_standing_env[:]=False
            cmd.time_left[:]=1000.
            print("LIVE_WALK_RESET","goal",goal_xyz,
                  "ball",ball_xyz,"plan",plan["approach_xy"],
                  "yaw",plan["shot_heading_deg"],flush=True)
            iteration=0
            state["episode"]+=1
            state["stage"]="approach"
            support_ready_steps=0
            support_verified=False
            foot_ball_peak=0.
            contact_status="NOT STRUCK"
            state["reset"]=False
            state["replan"]=False
        pos=robot.data.root_pos_w[0].cpu().tolist()
        yaw=float(robot.data.heading_w[0])
        target_xy=(plan["approach_xy"] if state["stage"]=="approach" else plan["preplant_base_xy"])
        vx,vy,wz,d=approach_command(pos[:2],yaw,target_xy,plan["shot_heading_rad"])
        yaw_error=math.degrees(wrap_pi(plan["shot_heading_rad"]-yaw))
        if state["stage"]=="approach" and d<.24 and abs(yaw_error)<15.:
            state["stage"]="preplant"
            target_xy=plan["preplant_base_xy"]
            vx,vy,wz,d=approach_command(pos[:2],yaw,target_xy,plan["shot_heading_rad"])
            print("LIVE_APPROACH_TO_PREPLANT",iteration,"target",target_xy,flush=True)
        if state["stage"]=="preplant" and .08<=d<.24:
            dx=target_xy[0]-pos[0];dy=target_xy[1]-pos[1]
            wx=max(-.28,min(.28,1.2*dx));wy=max(-.28,min(.28,1.2*dy))
            c=math.cos(yaw);s=math.sin(yaw)
            vx=max(0.,min(.32,c*wx+s*wy))
            vy=max(-.18,min(.18,-s*wx+c*wy))
        phase=("HOLD (manual)" if state["hold"] else
               ("APPROACH" if state["stage"]=="approach" else
                ("PREPLANT POSITIONING" if d>.08 else "PREPLANT HOLD / FOOT NOT VERIFIED")))
        if state["hold"] or (state["stage"]=="preplant" and d<.08):
            vx,vy,wz=0.,0.,0.
        cmd.vel_command_b[0]=torch.tensor([vx,vy,wz],device="cuda:0")
        cmd.is_standing_env[:]=False
        cmd.time_left[:]=1000.
        with torch.inference_mode():
            action=policy(obs)
            obs,rew,done,info=env.step(action)
        resets=int(done[0].item())
        falls+=resets
        fs=contact.data.net_forces_w[0]
        fl=float(torch.linalg.vector_norm(fs[left_id]))
        fr=float(torch.linalg.vector_norm(fs[right_id]))
        pf=ball_contacts.data.force_matrix_w
        force=float(torch.linalg.vector_norm(pf)) if pf is not None and pf.numel() else 0.
        foot_ball_peak=max(force,foot_ball_peak)
        if force>2.:
            contact_status=f"UNPLANNED foot-ball contact {force:.1f}N"
        b=football.data.root_pos_w[0].cpu().tolist()
        xyz=robot.data.root_pos_w[0].cpu().tolist()
        upright=float(robot.data.projected_gravity_b[0,2])<-.90
        lfxyz=robot.data.body_pos_w[0,foot_body_id].cpu().tolist()
        lfvel=robot.data.body_lin_vel_w[0,foot_body_id].cpu().tolist()
        lerr=math.hypot(lfxyz[0]-plan["support_foot_plant_xy"][0],
                        lfxyz[1]-plan["support_foot_plant_xy"][1])
        lslip=math.hypot(lfvel[0],lfvel[1])
        # No actual shot until the support foot is genuinely stable.
        if state["stage"]=="preplant" and d<.08 and upright and fl>30 and lerr<.2 and lslip<.08:
            support_ready_steps+=1
        else:
            support_ready_steps=0
        if support_ready_steps>=15 and not support_verified:
            support_verified=True
            print("LIVE_SUPPORT_PLANTED_VERIFIED",iteration,flush=True)
        if support_verified:
            phase="SUPPORT PLANTED / NO KICK POLICY YET"
        if resets:
            phase="AUTO REPLAY - PHYSX RL EPISODE RESET"
            state["replan"]=True
            print("LIVE_NORMAL_ENV_RESET",iteration,"episode",state["episode"],
                  "env_origins",raw.scene.env_origins.detach().cpu().tolist(),
                  "default_ball",football.data.default_root_state[0,:3].detach().cpu().tolist(),
                  "ball_now",football.data.root_pos_w[0].detach().cpu().tolist(),
                  "robot_default",robot.data.default_root_state[0,:3].detach().cpu().tolist(),
                  "robot_now",robot.data.root_pos_w[0].detach().cpu().tolist(),flush=True)
        if iteration%16==0 or resets:
            remaining=max(0.0,(raw.max_episode_length-int(raw.episode_length_buf[0].item()))*raw.step_dt)
            panel.status.text=f"AUTO MOVING LOOP | Episode {state['episode']} | replay in ~{remaining:.1f}s"
            panel.state.text=f"PHASE: {phase}"
            panel.pose.text=f"Root: ({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) yaw {math.degrees(yaw):.1f} deg"
            panel.ball.text=f"Physics ball: ({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})"
            panel.distance.text=f"Approach error {d:.2f}m | Heading error {yaw_error:.1f} deg"
            panel.contacts.text=f"Ground contact L/R: {fl:.0f}/{fr:.0f} N | upright={upright}"
            panel.goalstatus.text=f"Goal Y={state['goal_y']:+.2f} | policy resets={falls} | {contact_status}"
            panel.support.text=f"Planned LEFT foot error {lerr:.2f}m | slip {lslip:.2f}m/s | ready={support_verified}"
            print("LIVE_G1_APPROACH",iteration,phase,"xyz",
                  [round(v,3) for v in xyz],"dist",round(d,3),
                  "ground",round(fl),round(fr),"support_error_m",round(lerr,3),flush=True)
        if iteration%12==0:
            records.append([iteration,round(iteration*raw.step_dt,3),
               round(xyz[0],4),round(xyz[1],4),round(xyz[2],4),
               round(math.degrees(yaw),2),round(d,3),
               round(yaw_error,2),round(fl,1),round(fr,1),
               round(b[0],4),round(b[1],4),
               round(lerr,4),round(lslip,4),int(support_verified),
               state["goal_y"],phase])
        if iteration>0 and iteration%220==0:
            with out.open("w",newline="") as f:
                wr=csv.writer(f)
                wr.writerow(["step","t_s","root_x","root_y","root_z",
                    "heading_deg","approach_error_m","yaw_error_deg",
                    "left_foot_force_n","right_foot_force_n",
                    "ball_x","ball_y","left_support_foot_error_m",
                    "left_support_foot_slip_mps","support_verified",
                    "goal_y","phase"])
                wr.writerows(records)
        iteration+=1
        # Only the built-in RL auto-reset may replay the episode. In particular,
        # do not call env.reset() or manually teleport root mid-render.
        time.sleep(.015)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()
