"""KineticKick: purposeful long-distance G1 approach and return demonstration.

ONE real 0.43 kg soccer ball, free-base pretrained G1 walking PPO, one goal.
This demo makes measurable forward progress from a distant starting point.
ROUTE: RUN-UP -> FINAL STEPS -> READY -> TURN AROUND -> WALK BACK -> TURN -> RUN-UP.
No lateral-in-place lane loop, no environment reset, no teleport, no fixed base.
IMPORTANT: this is locomotion / shot setup, NOT a learned kicking controller.
"""
from __future__ import annotations
import math,time,csv
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=False,enable_cameras=False).app
import gymnasium as gym
import torch
import omni.ui as ui
import omni.usd
import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from goal_conditioned_planner import goal_conditioned_plan
from soccer_world import create_goal,move_goal,locate_goal_in_usd
from classic_soccer_visual import attach_classic_soccer_visual
from soccer_field_visual import create_field_visuals,decorate_goal_net

ROOT=Path("/home/you/projects/kinetickick")
CHECKPOINT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
START_XY=(-1.6,0.0)
BALL_XY=(2.0,0.0)
GOAL_X=5.0
DT=.02
RUNUP_MAX_SPEED=.62
SLOW_MAX_SPEED=.35

def angle_diff(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi

def saturate(v,lo,hi):
    return max(lo,min(hi,v))

def navigate(xy,heading,target,heading_goal,speed_limit=.58,stop_radius=.11):
    diff=(target[0]-xy[0],target[1]-xy[1])
    dist=math.hypot(*diff)
    err=angle_diff(heading_goal,heading)
    ch,sh=math.cos(heading),math.sin(heading)
    # World target velocities transformed into body frame.
    wx=saturate(1.05*diff[0],-.60,.60)
    wy=saturate(1.05*diff[1],-.60,.60)
    forward=saturate(ch*wx+sh*wy,0.,speed_limit)
    lateral=saturate(-sh*wx+ch*wy,-.24,.24)
    turn=saturate(1.5*err,-.8,.8)
    # Avoid moonwalking sideways when body yaw does not match target path.
    if abs(err)>.50:
        forward=0.0
        lateral=0.0
    if dist<stop_radius:
        forward=0.0
        lateral=0.0
    return forward,lateral,turn,dist,err

def build_sim():
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=1
    cfg.seed=42
    cfg.sim.device="cuda:0"
    cfg.episode_length_s=3600.0   # bypass known bad episodic reset in this GUI
    cfg.observations.policy.enable_corruption=False
    cfg.events.reset_base.params["pose_range"]={
        "x":(START_XY[0],START_XY[0]),"y":(START_XY[1],START_XY[1]),
        "yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
        "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
        "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,.65)
    cfg.commands.base_velocity.ranges.lin_vel_y=(-.3,.3)
    cfg.commands.base_velocity.ranges.ang_vel_z=(-.8,.8)
    # This is the EXACT source of the green/blue arrows near the G1 head.
    # The official locomotion task enables goal/current velocity debug vis,
    # positioned at root+0.5m. Disable visual markers, not the head mesh.
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(3600.,3600.)
    cfg.scene.football=RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/SoccerBall",
      spawn=sim.SphereCfg(radius=.11,
          collision_props=sim.CollisionPropertiesCfg(),
          rigid_props=sim.RigidBodyPropertiesCfg(),
          mass_props=sim.MassPropertiesCfg(mass=.43),
          visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.02,.02,.02))),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(BALL_XY[0],BALL_XY[1],.113)),
    )
    cfg.scene.foot_ball_contact=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    runner_cfg=G1FlatPPORunnerCfg()
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
        clip_actions=runner_cfg.clip_actions)
    runner=OnPolicyRunner(env,runner_cfg.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(CHECKPOINT))
    policy=runner.get_inference_policy(device="cuda:0")
    return env,policy

class Panel:
    def __init__(self,state):
        self.win=ui.Window("KineticKick | ORIGINAL G1 / NO ARROWS",width=495,height=355,visible=True)
        self.win.position_x=925
        self.win.position_y=64
        with self.win.frame:
            with ui.VStack(spacing=7):
                ui.Label("G1 RUN-UP: LONG FORWARD WALK | REAL 3D SOCCER BALL",
                         height=25,style={"font_size":16})
                with ui.HStack(height=30):
                    ui.Button("AUTO ROUTE",clicked_fn=lambda:self.auto(state))
                    ui.Button("GO TO BALL",clicked_fn=lambda:self.approach(state))
                    ui.Button("WALK BACK",clicked_fn=lambda:self.back(state))
                    ui.Button("HOLD",clicked_fn=lambda:self.hold(state))
                with ui.HStack(height=30):
                    ui.Button("GOAL LEFT",clicked_fn=lambda:self.goal(state,-.6))
                    ui.Button("GOAL CENTER",clicked_fn=lambda:self.goal(state,0.))
                    ui.Button("GOAL RIGHT",clicked_fn=lambda:self.goal(state,.6))
                with ui.HStack(height=30):
                    ui.Button("ROBOT + BALL",clicked_fn=lambda:self.camera(state,True))
                    ui.Button("FULL PITCH",clicked_fn=lambda:self.camera(state,False))
                self.phase=ui.Label("PHASE: initializing")
                self.dist=ui.Label("RUN-UP distance: 0.00 m")
                self.speed=ui.Label("ACTUAL speed: 0.00 m/s | command 0.00 m/s")
                self.pos=ui.Label("ROOT: x/y -- | target x/y --")
                self.ball=ui.Label("SOCCER BALL: physical rigid body; target --")
                self.result=ui.Label("Support foot and shot policy are NOT trained yet")
                ui.Label("No teleport and no fake kicking. Return is ordinary walking.")
    @staticmethod
    def auto(s):
        s["auto"]=True
        s["hold"]=False
        s["phase"]="RETURN_TURN" if s["phase"]=="READY" else s["phase"]
    @staticmethod
    def approach(s):
        s["auto"]=False;s["hold"]=False;s["phase"]="RUN_UP"
    @staticmethod
    def back(s):
        s["auto"]=False;s["hold"]=False;s["phase"]="RETURN_TURN"
    @staticmethod
    def hold(s):
        s["hold"]=not s["hold"]
    @staticmethod
    def goal(s,y):
        s["goal_y"]=y;s["replan"]=True
    @staticmethod
    def camera(s,close):
        s["close"]=close;s["camera_pending"]=True

def main():
    env,policy=build_sim()
    raw=env.unwrapped
    cmd_debug=raw.command_manager.get_term("base_velocity")
    print("G1_HEAD_ORIGINAL_INTACT",True,
          "velocity_debug_vis",cmd_debug.cfg.debug_vis,
          "goal_marker_exists",hasattr(cmd_debug,"goal_vel_visualizer"),flush=True)
    assert cmd_debug.cfg.debug_vis is False
    robot=raw.scene["robot"]
    ball=raw.scene["football"]
    command=raw.command_manager.get_term("base_velocity")
    feet=raw.scene["contact_forces"]
    foot_ball=raw.scene["foot_ball_contact"]
    goal=create_goal(goal_x=GOAL_X,initial_y=.5)
    create_field_visuals()
    decorate_goal_net()
    visual=attach_classic_soccer_visual("/World/envs/env_0/SoccerBall")
    print("SCENE_ONE_SOCCER_BALL",list(raw.scene.rigid_objects),visual["pentagons"],
          visual["hexagons"],flush=True)
    # Do not alter G1 official visual, joint topology, or collision.
    for name in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
        panel=ui.Workspace.get_window(name)
        if panel is not None:panel.visible=False
    state={"phase":"RUN_UP","auto":True,"hold":False,"goal_y":.5,
           "replan":False,"close":False,"camera_pending":True}
    panel=Panel(state)
    obs,_=env.get_observations()
    plan=None
    step=0
    phase_tick=0
    lap=0
    total_travel=0.0
    max_fwd_travel=0.0
    best_goal_distance=1e9
    reset_count=0
    prev_xy=None
    records=[]
    out=ROOT/"reports/live_g1_actual_runup.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    def replan():
        nonlocal plan
        b=ball.data.root_pos_w[0].detach().cpu().tolist()
        r=robot.data.root_pos_w[0].detach().cpu().tolist()
        g=locate_goal_in_usd()
        plan=goal_conditioned_plan(ball_xy=b[:2],goal_xy=g[:2],robot_xy=r[:2],
               foot="right",surface="instep_top")
        print("RUNUP_PLAN_GOAL",g,"ball",b,"from",r,
              "shot_yaw_deg",plan["shot_heading_deg"],flush=True)
    replan()
    # Start far enough away that real movement is unambiguous in the viewport.
    init=robot.data.root_pos_w[0].detach().cpu().tolist()
    print("RUNUP_START_ROOT",init,"start_override_requested",START_XY,flush=True)
    def camera():
        if state["close"]:
            raw.sim.set_camera_view(eye=[1.1,-5.15,2.00],target=[.25,-.04,.60])
        else:
            raw.sim.set_camera_view(eye=[2.35,-7.65,2.8],target=[.6,0.,.65])
        print("RUNUP_CAMERA_MODE",("CLOSE" if state["close"] else "WIDE"),flush=True)
    camera()
    while app.is_running():
        if state["camera_pending"]:
            camera()
            state["camera_pending"]=False
        if state["replan"]:
            move_goal(goal,state["goal_y"])
            replan()
            state["replan"]=False
        pos=robot.data.root_pos_w[0].detach().cpu().tolist()
        xy=(float(pos[0]),float(pos[1]))
        heading=float(robot.data.heading_w[0].item())
        v_world=robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        actual_speed=math.hypot(v_world[0],v_world[1])
        if prev_xy is not None:
            total_travel+=min(.3,math.dist(xy,prev_xy))
        prev_xy=xy
        if plan is None:
            raise RuntimeError("No valid goal-aligned approach plan")

        phase=state["phase"]
        target=None
        if phase=="RUN_UP":
            target=plan["approach_xy"]
            target_heading=plan["shot_heading_rad"]
            vmax=RUNUP_MAX_SPEED
            tol=.22
        elif phase=="FINAL_STEPS":
            target=plan["preplant_base_xy"]
            target_heading=plan["shot_heading_rad"]
            vmax=SLOW_MAX_SPEED
            tol=.16
        elif phase=="READY":
            target=plan["preplant_base_xy"]
            target_heading=plan["shot_heading_rad"]
            vmax=0.0
            tol=.2
        elif phase in ("RETURN_TURN","WALK_BACK"):
            target=START_XY
            target_heading=math.atan2(START_XY[1]-xy[1],START_XY[0]-xy[0])
            vmax=(0.0 if phase=="RETURN_TURN" else .55)
            tol=.20
        elif phase=="FACE_GOAL":
            target=START_XY
            target_heading=plan["shot_heading_rad"]
            vmax=0.0
            tol=.24
        else:
            raise RuntimeError("Unknown controller phase: "+str(phase))

        vx,vy,wz,dist,heading_err=navigate(
              xy,heading,target,target_heading,speed_limit=vmax,stop_radius=tol)
        if state["hold"]:
            vx,vy,wz=0.,0.,0.
        else:
            if phase=="RUN_UP" and dist<.22 and abs(heading_err)<.35:
                state["phase"]="FINAL_STEPS";phase_tick=step
                print("RUNUP_LONG_APPROACH_DONE","step",step,"xy",xy,
                      "displacement_x_m",round(xy[0]-init[0],3),flush=True)
            elif phase=="FINAL_STEPS" and dist<.17 and abs(heading_err)<.35:
                state["phase"]="READY";phase_tick=step
                max_fwd_travel=max(max_fwd_travel,xy[0]-init[0])
                best_goal_distance=min(best_goal_distance,dist)
                print("RUNUP_ARRIVED_PRE_SHOT_ZONE","step",step,"xy",xy,
                      "travel_m",round(xy[0]-init[0],3),
                      "note","NO_KICK_POLICY",flush=True)
            elif phase=="READY" and state["auto"] and (step-phase_tick)*DT>1.65:
                state["phase"]="RETURN_TURN";phase_tick=step
                print("RUNUP_WALK_BACK_START",step,flush=True)
            elif phase=="RETURN_TURN" and abs(heading_err)<.25:
                state["phase"]="WALK_BACK";phase_tick=step
                print("RUNUP_RETURN_FACING_START",step,flush=True)
            elif phase=="WALK_BACK" and dist<.20:
                state["phase"]="FACE_GOAL";phase_tick=step
                print("RUNUP_RETURNED_TO_START_PHYSICALLY",step,"xy",xy,flush=True)
            elif phase=="FACE_GOAL" and abs(heading_err)<.22:
                state["phase"]="RUN_UP";phase_tick=step;lap+=1
                replan()
                print("RUNUP_NEXT_LAP",lap,"step",step,flush=True)

        command.vel_command_b[0]=torch.tensor([vx,vy,wz],device=robot.device)
        command.is_standing_env[:]=False
        command.time_left[:]=1e6
        with torch.inference_mode():
            act=policy(obs)
            obs,_,dones,_=env.step(act)
        if bool(dones[0]):
            reset_count+=1
            print("RUNUP_UNEXPECTED_RL_RESET_ABORT",step,flush=True)
            break
        f=foot_ball.data.force_matrix_w
        foot_to_ball=float(torch.linalg.vector_norm(f)) if f is not None and f.numel() else 0.0
        if step%20==0:
            print("RUNUP_TELEMETRY","step",step,"phase",state["phase"],
                 "root_xy",[round(z,3) for z in xy],
                 "v_actual_mps",round(actual_speed,3),
                 "cmd_body_xy",[round(vx,3),round(vy,3)],
                 "target_xy",[round(z,3) for z in target],
                 "distance_to_target_m",round(dist,3),
                 "total_travel_m",round(total_travel,3),
                 "foot_ball_force_N",round(foot_to_ball,2),
                 "lap",lap,flush=True)
            panel.phase.text=f"PHASE: {state['phase']} | lap {lap} | {('HOLD' if state['hold'] else 'ACTIVE')}"
            panel.dist.text=f"FORWARD displacement {xy[0]-init[0]:+.2f}m | path length {total_travel:.2f}m"
            panel.speed.text=f"ACTUAL speed {actual_speed:.2f}m/s | COMMAND {vx:.2f}m/s"
            panel.pos.text=f"Root ({xy[0]:.2f},{xy[1]:.2f}) -> target ({target[0]:.2f},{target[1]:.2f})"
            panel.ball.text=f"Goal Y={state['goal_y']:+.2f}m | physical football x={BALL_XY[0]:.1f}m"
            panel.result.text=f"Distance to target {dist:.2f}m | resets {reset_count} | NO KICK"
            records.append([step,round(step*DT,3),state["phase"],
                            round(xy[0],4),round(xy[1],4),round(actual_speed,4),
                            round(vx,4),round(vy,4),round(dist,4),
                            round(total_travel,4),lap,round(foot_to_ball,3)])
        if step>0 and step%360==0:
            with out.open("w",newline="") as fp:
                wr=csv.writer(fp)
                wr.writerow(["step","time_s","phase","root_x","root_y","actual_speed",
                             "cmd_forward","cmd_lateral","target_error",
                             "travel_distance","lap","foot_ball_force"])
                wr.writerows(records)
        step+=1
        time.sleep(.009)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()
