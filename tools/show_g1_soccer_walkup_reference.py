"""KineticKick Stage 3A — Soccer walk-up on Tang Isaac Sim, A -> READY.

One physical black/white soccer ball at x=4.8, a practice goal at x=8.8,
a real-looking marked pitch and the already validated LAFAN1 29DoF G1
human-derived STRAIGHT motion reference.

Read this carefully:
  - The ball, goal frame and ground have physics.
  - G1 is kinematic *reference playback*, not an RL-driven or dynamically
    balanced robot. Its motion is NOT an autonomous walk-to-ball policy.
  - Each source clip keeps ALL its measured time and 29 joint angles; we only
    translate the complete clip along world X so it ends at the SAME READY
    position x=4.2, 0.6m behind the stationary ball at x=4.8.
  - Different walking speeds start at different A markers; no resampling/
    trajectory warping/hardcoded shot after approaching.
  - Reaching READY stops. Replay requires a button; no random looping.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--headless",action="store_true")
parser.add_argument("--frames",type=int,default=0)
args=parser.parse_args()

from isaaclab.app import AppLauncher
app=AppLauncher(headless=args.headless,enable_cameras=not args.headless).app

import numpy as np
import isaaclab.sim as sim
from isaaclab.sim import SimulationCfg,SimulationContext
if not args.headless:
    import omni.ui as ui
else:
    ui=None
from g1_29dof_reference_usd import G1HumanReferenceVisual
from soccer_walkup_scene import (
    create_soccer_walkup_scene,move_start_line,
    BALL_XY,READY_XY,GOAL_XY
)

ROOT=Path("/home/you/projects/kinetickick")
MANIFEST=json.loads((ROOT/"reports/straight_gait_reference_manifest.json").read_text())
CLIPS=MANIFEST["clips"]
AVAILABLE=("WALK SLOW","WALK NORMAL","WALK BRISK","RUN STRAIGHT")

class Panel:
    def __init__(self,state):
        self.window=ui.Window("KineticKick | SOCCER WALK-UP A -> BALL",width=525,height=442,visible=True)
        self.window.position_x=930
        self.window.position_y=63
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("G1 NATURAL WALK REFERENCE -> SOCCER READY",height=25,
                         style={"font_size":17})
                ui.Label("1 PHYSX FOOTBALL + PITCH + GOAL | NO KICK / NOT RL")
                with ui.HStack(height=32,spacing=5):
                    ui.Button("SLOW WALK",clicked_fn=lambda:self.choose(state,"WALK SLOW"))
                    ui.Button("NORMAL WALK",clicked_fn=lambda:self.choose(state,"WALK NORMAL"))
                    ui.Button("BRISK WALK",clicked_fn=lambda:self.choose(state,"WALK BRISK"))
                    ui.Button("RUN",clicked_fn=lambda:self.choose(state,"RUN STRAIGHT"))
                with ui.HStack(height=31,spacing=5):
                    ui.Button("REPLAY TO BALL",clicked_fn=lambda:self.restart(state))
                    ui.Button("PAUSE",clicked_fn=lambda:self.pause(state,True))
                    ui.Button("CONTINUE",clicked_fn=lambda:self.pause(state,False))
                with ui.HStack(height=30,spacing=5):
                    ui.Button("FOLLOW G1",clicked_fn=lambda:self.camera(state,"follow"))
                    ui.Button("FULL PITCH",clicked_fn=lambda:self.camera(state,"wide"))
                    ui.Button("SLOW 0.5X",clicked_fn=lambda:self.rate(state,.5))
                    ui.Button("NORMAL 1X",clicked_fn=lambda:self.rate(state,1.))
                self.phase=ui.Label("STATUS: loading...",height=24)
                self.path=ui.Label("Route: A (...) -> READY(4.20,0.00)",height=21)
                self.dist=ui.Label("Distance to football: --",height=22)
                self.ready=ui.Label("Distance to ready position: --",height=22)
                self.progress=ui.Label("Progress: --",height=22)
                self.ball=ui.Label("Football: x4.80 / one physics rigid body",height=22)
                self.goal=ui.Label("Goal: x8.80 straight ahead",height=22)
                ui.Label("At READY it freezes. Replay ONLY when you click.")
                ui.Label("Reference animation: no balance, no autonomous goal policy.")
    @staticmethod
    def choose(s,name):
        print("SOCCER_WALKUP_UI_MODE_CLICK",name,flush=True)
        s["mode"]=name;s["reload"]=True
    @staticmethod
    def restart(s):
        print("SOCCER_WALKUP_UI_REPLAY_CLICK",s["mode"],flush=True)
        s["reload"]=True
    @staticmethod
    def pause(s,yes):
        s["paused"]=bool(yes)
        s["last_clock"]=time.monotonic()
    @staticmethod
    def camera(s,which):
        s["camera_mode"]=which;s["force_camera"]=True
    @staticmethod
    def rate(s,speed):
        s["rate"]=float(speed)
        s["last_clock"]=time.monotonic()


def main():
    ctx=SimulationContext(SimulationCfg(dt=1./60,render_interval=1))
    ground=sim.GroundPlaneCfg(color=(.05,.22,.09))
    ground.func("/World/WalkupPhysicalGround",ground)
    sky=sim.DomeLightCfg(intensity=2600,color=(.91,.94,1.))
    sky.func("/World/WalkupSky",sky)
    ball,start_op,scene_info=create_soccer_walkup_scene()
    g1=G1HumanReferenceVisual(max_triangles_per_visual=1000000)
    ctx.reset()
    if not ball.is_initialized:
        raise RuntimeError("Single physical football failed to initialize")
    print("SOCCER_WALKUP_PHYSICS_CONFIRMED",
          "football_initialized",ball.is_initialized,
          "ball_root",ball.data.root_pos_w[0].detach().cpu().tolist(),
          "g1_reference_has_physics",False,flush=True)
    if not args.headless:
        for name in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
            pane=ui.Workspace.get_window(name)
            if pane is not None:pane.visible=False

    state={"mode":"WALK NORMAL","reload":True,"paused":False,"completed":False,
           "rate":1.,"camera_mode":"wide","force_camera":True,
           "last_clock":time.monotonic()}
    panel=None if args.headless else Panel(state)
    frames=None
    last_idx=-1
    start_x=0.
    final_pos=[0.,0.,.75]
    target=READY_XY
    source_duration=0.
    source_speed=0.
    t=0.
    last_root_xy=None
    frame_count=0
    last_reported=-1
    while app.is_running():
        if state["reload"]:
            metadata=CLIPS[state["mode"]]
            frames=np.load(metadata["file"])["frames"]
            if frames.ndim!=2 or frames.shape[1]!=36:
                raise ValueError("LAFAN1 29DoF source must have 36 channels")
            original_travel=float(metadata["goal_world_xy"][0])
            start_x=READY_XY[0]-original_travel
            move_start_line(start_op,start_x)
            # World-space rigid translation only; the source 29 joint positions,
            # root rotations and original 30FPS timing are unchanged.
            t=0.
            source_duration=float(metadata["play_duration_s"])
            last_idx=-1
            source_speed=0.
            last_root_xy=None
            state["completed"]=False
            state["paused"]=False
            state["reload"]=False
            state["force_camera"]=True
            state["last_clock"]=time.monotonic()
            last_reported=-1
            print("SOCCER_WALKUP_START","mode",state["mode"],
                "start_x",round(start_x,4),
                "ready_x",READY_XY[0],
                "ball_x",BALL_XY[0],
                "goal_x",GOAL_XY[0],
                "source_clip_unchanged",True,flush=True)

        now=time.monotonic()
        elapsed=max(0.,now-state["last_clock"])
        state["last_clock"]=now
        if not state["paused"] and not state["completed"]:
            t=min(source_duration,t+elapsed*state["rate"])
        idx=min(len(frames)-1,int(t*30))
        if idx!=last_idx:
            row=frames[idx]
            xyz=g1.update(row,origin_xy=(start_x,0.),ref_start_xy=(0.,0.))
            final_pos=[float(x) for x in xyz]
            if last_root_xy is not None:
                delta=float(np.linalg.norm(np.array(final_pos[:2])-last_root_xy))
                source_speed=delta*30
            last_root_xy=np.array(final_pos[:2])
            last_idx=idx
            if idx//30 != last_reported:
                last_reported=idx//30
                print("SOCCER_WALKUP_PROGRESS",
                      "mode",state["mode"],"frame",idx,
                      "root_xyz",[round(x,3) for x in final_pos],
                      "ready_distance_m",round(math.dist(final_pos[:2],READY_XY),3),
                      "ball_distance_m",round(math.dist(final_pos[:2],BALL_XY),3),
                      flush=True)
        ready_dist=math.dist(final_pos[:2],READY_XY)
        ball_dist=math.dist(final_pos[:2],BALL_XY)
        if idx>=len(frames)-1 and not state["completed"]:
            state["completed"]=True
            state["paused"]=True
            print("SOCCER_WALKUP_READY_REACHED",
                "mode",state["mode"],"root_xy",[round(x,4) for x in final_pos[:2]],
                "ready_error_m",round(ready_dist,5),
                "ball_distance_m",round(ball_dist,4),
                "auto_loop",False,"physical_kick",False,flush=True)

        if not args.headless and (frame_count%12==0 or state["force_camera"]):
            if state["camera_mode"]=="follow":
                ctx.set_camera_view(
                    eye=[final_pos[0]+1.75,final_pos[1]-3.10,1.74],
                    target=[final_pos[0],final_pos[1],.66])
            else:
                ctx.set_camera_view(eye=[4.70,-11.6,4.0],
                                    target=[4.55,0.,.75])
            state["force_camera"]=False
        if panel and frame_count%12==0:
            status=("READY (STOPPED) - CLICK REPLAY" if state["completed"] else
                     ("PAUSED" if state["paused"] else "WALKING TOWARD SOCCER BALL"))
            completion=max(0.,min(1.,(final_pos[0]-start_x)/(READY_XY[0]-start_x)))
            panel.phase.text="PHASE: "+status
            panel.path.text=f"A({start_x:+.2f},0.00) -> READY({READY_XY[0]:.2f},0.00)"
            panel.dist.text=f"Distance to ball: {ball_dist:.2f} m (target >= 0.60 m)"
            panel.ready.text=f"Ready error: {ready_dist:.3f} m"
            panel.progress.text=f"Forward progress {completion*100:.0f}% | t={t:.2f}/{source_duration:.2f}s"
            panel.ball.text=f"ONE soccer ball: ({BALL_XY[0]:.2f},0.00), 0.43 kg PhysX"
            panel.goal.text=f"Goal: ({GOAL_XY[0]:.2f},0.00) | no shot controller"
        ctx.step()
        ball.update(ctx.get_physics_dt())
        frame_count+=1
        if frame_count%600==0:
            print("SOCCER_WALKUP_GUI_HEARTBEAT",frame_count,
                  "mode",state["mode"],"completed",state["completed"],
                  "ball_xyz",[round(float(v),4) for v in ball.data.root_pos_w[0]],
                  flush=True)
        if args.frames and frame_count>=args.frames:
            break
        if not args.headless:
            time.sleep(.005)

    status={
        "render_ticks":frame_count,
        "mode":state["mode"],
        "reached_ready":state["completed"],
        "final_reference_xy":final_pos[:2],
        "ready_xy":READY_XY,"ball_xy":BALL_XY,"goal_xy":GOAL_XY,
        "ball_physics_initialized":bool(ball.is_initialized),
        "ball_final_xyz":ball.data.root_pos_w[0].detach().cpu().tolist(),
        "one_ball":scene_info["football_count"]==1,
        "no_autoplay_restarting":True,
        "reference_is_kinematic_not_policy":True
    }
    outfile=ROOT/"reports/soccer_walkup_viewer_health.json"
    outfile.write_text(json.dumps(status,indent=2))
    print("SOCCER_WALKUP_VIEWER_EXIT",status,flush=True)

if __name__=="__main__":
    try:main()
    finally:app.close()
