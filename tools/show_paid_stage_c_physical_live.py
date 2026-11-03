#!/usr/bin/env python3
"""G1 Stage C LIVE: physically simulated foot-ball contact using frozen global-root V2 PPO.

Live GUI shows one 0.43kg ball, free-root 29DoF G1, real PhysX contact force and
outgoing ball speed. Not a ball-conditioned policy, not calibrated shooting.
No hand-controlled joints, no ball impulse after start, no teleport at frame260.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, traceback, types

p=argparse.ArgumentParser()
p.add_argument("--headless",action="store_true")
p.add_argument("--max-steps",type=int,default=0)
p.add_argument("--ball-x",type=float,default=0.04)
p.add_argument("--ball-y",type=float,default=-0.77)
p.add_argument("--camera",choices=("FOLLOW","SIDE","WIDE"),default="FOLLOW")
cli=p.parse_args()
from isaaclab.app import AppLauncher
app=AppLauncher(headless=cli.headless,enable_cameras=False).app

import gymnasium as gym
import torch
import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
if not cli.headless:
    import omni.ui as ui
else:
    ui=None

ROOT=pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM=ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
MODEL=ROOT/"experiments/paid29_global_root_v2/seed86_n96_iter200/last_model.pt"
sys.path.insert(0,str(UPSTREAM))
shim=types.ModuleType("soccer.assets")
shim.ASSET_DIR=str(UPSTREAM/"soccer/assets")
sys.modules["soccer.assets"]=shim

class Panel:
    def __init__(self,state):
        self.win=ui.Window("KineticKick | G1 REAL SOCCER CONTACT",width=600,height=405,visible=True)
        self.win.position_x=865
        self.win.position_y=65
        with self.win.frame:
            with ui.VStack(spacing=7):
                ui.Label("G1 29DoF + REAL FOOTBALL PHYSX IMPACT",height=25,style={"font_size":16})
                ui.Label("Learned global-root PPO | free base | ground/contact dynamics")
                ui.Label("Frozen motion tracking — BALL POSITION NOT OBSERVED BY POLICY",
                         style={"color":0xFF57B2FF})
                with ui.HStack(height=29):
                    ui.Button("BALL NEAR",clicked_fn=lambda:self.ballpos(state,.04,-.77))
                    ui.Button("BALL FAR",clicked_fn=lambda:self.ballpos(state,.04,-.93))
                    ui.Button("RESTART",clicked_fn=lambda:self.restart(state))
                with ui.HStack(height=29):
                    ui.Button("PLAY / PAUSE",clicked_fn=lambda:self.pause(state))
                    ui.Button("SLOW 0.5x",clicked_fn=lambda:self.pace(state,"SLOW"))
                    ui.Button("NORMAL",clicked_fn=lambda:self.pace(state,"NORMAL"))
                with ui.HStack(height=29):
                    ui.Button("FOLLOW",clicked_fn=lambda:self.camera(state,"FOLLOW"))
                    ui.Button("SIDE",clicked_fn=lambda:self.camera(state,"SIDE"))
                    ui.Button("FULL PITCH",clicked_fn=lambda:self.camera(state,"WIDE"))
                self.phase=ui.Label("PHYSICAL POLICY: starting",height=24)
                self.ball=ui.Label("BALL xyz / speed: awaiting simulation",height=24)
                self.contact=ui.Label("Right foot-ball filtered force: 0 N",height=24)
                self.pose=ui.Label("G1 root / target tracking error: --",height=24)
                self.result=ui.Label("Measured contact / recovery: --",height=24)
                ui.Label("No shoe; no fixed root; no scripted shot or injected ball velocity.")
                ui.Label("Goal is visual only; no goal-accuracy claim. No ball-conditioned RL yet.")
    @staticmethod
    def ballpos(s,x,y):
        s["ball_xy"]=(x,y)
        s["restart"]=True
        print("LIVE_C_BALL_POSITION_REQUEST",s["ball_xy"],flush=True)
    @staticmethod
    def restart(s):s["restart"]=True
    @staticmethod
    def pause(s):s["paused"]=not s["paused"]
    @staticmethod
    def pace(s,value):s["pace"]=value
    @staticmethod
    def camera(s,value):s["camera"]=value;s["camera_update"]=True

def add_visual_goal():
    """Simple noncolliding target goal toward world -Y (illustration only)."""
    from pxr import Gf,UsdGeom
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    base="/World/NoCollisionSoccerGoal"
    UsdGeom.Xform.Define(stage,base)
    def beam(name,pos,size):
        cube=UsdGeom.Cube.Define(stage,base+"/"+name)
        cube.CreateSizeAttr(1.)
        t=UsdGeom.Xformable(cube)
        t.AddTranslateOp().Set(Gf.Vec3d(*pos))
        t.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.91,.94,.95)])
    beam("left_post",(-.8,-3.7,.55),(.06,.06,1.1))
    beam("right_post",(.8,-3.7,.55),(.06,.06,1.1))
    beam("crossbar",(0.,-3.7,1.1),(1.66,.06,.06))
    print("LIVE_C_GOAL_VISUAL_ONLY",True,flush=True)

def main():
    import soccer.tasks
    from soccer.tasks.tracking import mdp
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from isaaclab.envs import mdp as isaac_mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from soccer_field_visual import create_field_visuals
    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=1
    cfg.scene.env_spacing=4.
    cfg.sim.device="cuda:0"
    cfg.seed=501
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.observations.policy.enable_corruption=False
    cfg.observations.policy.motion_anchor_pos_b=ObsTerm(func=mdp.motion_anchor_pos_b,params={"command_name":"motion"})
    cfg.observations.policy.motion_anchor_ori_b=ObsTerm(func=mdp.motion_anchor_ori_b,params={"command_name":"motion"})
    cfg.observations.policy.base_lin_vel=ObsTerm(func=isaac_mdp.base_lin_vel)
    for key in cfg.commands.motion.pose_range:cfg.commands.motion.pose_range[key]=(0.,0.)
    for key in cfg.commands.motion.velocity_range:cfg.commands.motion.velocity_range[key]=(0.,0.)
    cfg.commands.motion.joint_position_range=(0.,0.)
    cfg.terminations.ee_body_pos.params["threshold"]=.25
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None
    cfg.scene.soccer_ball=RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim.SphereCfg(
            radius=.11,
            collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=.43),
            physics_material=sim.RigidBodyMaterialCfg(
                static_friction=.7,dynamic_friction=.55,restitution=.3),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.89,.86,.76))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.8,2.8,.113)))
    cfg.scene.right_foot_ball_contact=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        debug_vis=False)
    ac=G1FlatPPORunnerCfg()
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),clip_actions=ac.clip_actions)
    raw=env.unwrapped
    robot=raw.scene["robot"]
    ball=raw.scene["soccer_ball"]
    foot_ball=raw.scene["right_foot_ball_contact"]
    motion=raw.command_manager.get_term("motion")
    assert len(robot.joint_names)==29 and not cfg.scene.robot.spawn.fix_base
    assert len(raw.scene.rigid_objects)==1
    print("LIVE_C_PHYSICS_READY","robot_DoF",29,"floating",True,"ball_count",1,flush=True)
    create_field_visuals()
    add_visual_goal()
    if not cli.headless:
        for name in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
            pane=ui.Workspace.get_window(name)
            if pane is not None:pane.visible=False
    # Keep reference on final frame, no robot reposition/reset at frame 260.
    native_update=motion._update_command
    def no_reference_teleport(self):
        cap=self.motion_length-1
        at_end=self.time_steps>=cap
        if bool(at_end.any()):
            self.time_steps[at_end]=cap[at_end]-1
        return native_update()
    motion._update_command=types.MethodType(no_reference_teleport,motion)
    runner=OnPolicyRunner(env,ac.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(MODEL),load_optimizer=False)
    policy=runner.get_inference_policy(device="cuda:0")
    print("LIVE_C_POLICY_LOADED",str(MODEL),flush=True)
    state={"ball_xy":(cli.ball_x,cli.ball_y),"restart":True,"paused":False,
           "pace":"NORMAL","camera":cli.camera,"camera_update":True,
           "episode":0,"hit":False,"peak_force":0.,"max_speed":0.,
           "phase":"init","resets":0}
    panel=None if cli.headless else Panel(state)
    health=ROOT/"reports/g1_paid_stage_c_live_physics_health.json"
    obs=None
    count=0
    steps=0
    while app.is_running():
        if state["restart"]:
            obs,_=env.reset()
            motion.time_steps[:]=-1
            motion._update_command()
            assert int(motion.time_steps[0])==0
            ball_state=ball.data.default_root_state.clone()
            ball_state[:,:3]=torch.tensor([[state["ball_xy"][0],state["ball_xy"][1],.113]],
                  device=ball.device,dtype=ball_state.dtype)+raw.scene.env_origins
            ball_state[:,7:]=0.
            ball.write_root_pose_to_sim(ball_state[:,:7])
            ball.write_root_velocity_to_sim(ball_state[:,7:])
            ball.reset()
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            state["restart"]=False
            state["episode"]+=1
            state["hit"]=False
            state["peak_force"]=0.
            state["max_speed"]=0.
            state["phase"]="MOTION"
            count=0
            state["camera_update"]=True
            print("LIVE_C_EPISODE_START",state["episode"],"ball",state["ball_xy"],flush=True)
        if state["camera_update"] or steps%18==0:
            pos=robot.data.root_pos_w[0].detach().cpu().tolist()
            if state["camera"]=="WIDE":
                eye=[3.2,-1.7,2.2];target=[.1,-1.6,.45]
            elif state["camera"]=="SIDE":
                eye=[2.3,pos[1]+.15,1.3];target=[pos[0],pos[1],.5]
            else:
                eye=[pos[0]+2.1,pos[1]+2.5,1.5];target=[pos[0],pos[1],.62]
            raw.sim.set_camera_view(eye=eye,target=target)
            state["camera_update"]=False
        if state["paused"]:
            app.update()
            time.sleep(.03)
            if cli.max_steps and steps>=cli.max_steps:break
            continue
        if count>=310:
            result={"episode":state["episode"],"reference_complete":True,"survived_recovery_s":1.,
                    "right_foot_contact":state["hit"],"max_filtered_force_N":state["peak_force"],
                    "max_ball_speed_xy_mps":state["max_speed"],"ball_xy_start":state["ball_xy"],
                    "policy_ball_conditioned":False}
            health.write_text(json.dumps(result,indent=2))
            print("LIVE_C_PHYSICAL_SEGMENT_COMPLETE",json.dumps(result),flush=True)
            if panel is not None:
                panel.result.text="Full 260+50 physical steps | "+("Contact" if state["hit"] else "No contact")
                app.update()
                time.sleep(.50)
            state["restart"]=True
            continue
        anchor_error=float(torch.linalg.vector_norm(
            motion.anchor_pos_w[0,:2]-motion.robot_anchor_pos_w[0,:2]))
        # torch.no_grad, NOT inference_mode (manager reset buffers must remain editable).
        with torch.no_grad():
            action=policy(obs)
            obs,_,done,_=env.step(action)
        count+=1
        steps+=1
        fm=foot_ball.data.force_matrix_w
        force=float(torch.linalg.vector_norm(fm)) if fm is not None and fm.numel() else 0.
        v=ball.data.root_lin_vel_w[0,:2]
        speed=float(torch.linalg.vector_norm(v))
        state["peak_force"]=max(state["peak_force"],force)
        state["max_speed"]=max(state["max_speed"],speed)
        if force>2.0 and not state["hit"]:
            state["hit"]=True
            print("LIVE_C_FOOT_BALL_CONTACT",state["episode"],"step",count,
                  "force_N",round(force,3),"ball_xy_speed_mps",round(speed,4),flush=True)
        if count%10==0 or bool(done[0]):
            root=robot.data.root_pos_w[0].detach().cpu().tolist()
            b=ball.data.root_pos_w[0].detach().cpu().tolist()
            print("LIVE_C_TELEMETRY",state["episode"],"step",count,
                  "root",list(map(lambda a:round(a,3),root)),
                  "ball",list(map(lambda a:round(a,3),b)),
                  "speed_mps",round(speed,4),"F_N",round(force,2),
                  "world_xy_err",round(anchor_error,3),flush=True)
            if panel is not None:
                panel.phase.text=f"PHASE: {'REFERENCE' if count<=260 else '1s RECOVERY'} | step {count}/310 | episode {state['episode']}"
                panel.ball.text=f"Ball ({b[0]:+.2f}, {b[1]:+.2f})m | horizontal {speed:.2f}m/s | peak {state['max_speed']:.2f}"
                panel.contact.text=f"RIGHT-FOOT CONTACT: {'YES' if state['hit'] else 'NO'} | force {force:.1f}N | peak {state['peak_force']:.1f}N"
                panel.pose.text=f"G1 root z {root[2]:.2f}m | global path error {anchor_error:.2f}m"
                panel.result.text="Measured impact + follow-through; no goal targeting" if state["hit"] else "No confirmed foot impact yet"
        if bool(done[0]):
            causes={k:bool(raw.termination_manager.get_term(k)[0]) for k in raw.termination_manager.active_terms}
            result={"episode":state["episode"],"ended_step":count,"right_foot_contact":state["hit"],
                    "max_filtered_force_N":state["peak_force"],"max_ball_speed_xy_mps":state["max_speed"],
                    "causes":causes,"recovery_complete":False}
            health.write_text(json.dumps(result,indent=2))
            print("LIVE_C_PHYSICAL_EARLY_RESET",json.dumps(result),flush=True)
            if panel is not None:
                panel.result.text="Terminated "+str(count)+" | "+",".join(k for k,v in causes.items() if v)
                app.update();time.sleep(.25)
            state["restart"]=True
        if cli.max_steps and steps>=cli.max_steps:break
        if not cli.headless:time.sleep(.027 if state["pace"]=="SLOW" else .006)
    print("LIVE_C_VIEWER_FINISHED",steps,flush=True)
    env.close()

if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        print("LIVE_C_VIEWER_FAILED",flush=True)
        raise
    else:app.close()
