"""Straight, PURPOSEFUL A -> B human-derived gait playback on Tang Isaac Sim.

This program only renders G1 29DoF kinematics from true LAFAN1 retargeted
motion clips. No free-root physics policy, no G1 motor controller, and no goal-
conditioned RL. A is fixed x=0; B is chosen by the source segment's displacement.
At B it STOPS and freezes (no teleport, no circling). REPLAY is explicit and
visibly restarts from A, because clips are finite motion capture, not a policy.
"""
from __future__ import annotations
import argparse,json,time
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
if not args.headless:import omni.ui as ui
else:ui=None
from g1_29dof_reference_usd import G1HumanReferenceVisual
from gait_route_visual import create_straight_lane,update_route_finish

BASE=Path("/home/you/projects/kinetickick")
MANIFEST=json.loads((BASE/"reports/straight_gait_reference_manifest.json").read_text())
INFO=MANIFEST["clips"]
MODES=("WALK SLOW","WALK NORMAL","WALK BRISK","RUN STRAIGHT")


class Panel:
    def __init__(self,state):
        self.window=ui.Window("KineticKick | STRAIGHT HUMAN WALK A -> B",
            width=495,height=405,visible=True)
        self.window.position_x=925
        self.window.position_y=62
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("STRAIGHT HUMAN WALKING REFERENCE | TARGET A -> B",
                         height=27,style={"font_size":16})
                ui.Label("Kinematic LAFAN1 -> G1 29DoF; NOT learned steering/balance")
                with ui.HStack(height=29):
                    ui.Button("SLOW WALK",clicked_fn=lambda:self.choose(state,"WALK SLOW"))
                    ui.Button("NORMAL WALK",clicked_fn=lambda:self.choose(state,"WALK NORMAL"))
                    ui.Button("BRISK WALK",clicked_fn=lambda:self.choose(state,"WALK BRISK"))
                    ui.Button("RUN",clicked_fn=lambda:self.choose(state,"RUN STRAIGHT"))
                with ui.HStack(height=29):
                    ui.Button("REPLAY A -> B",clicked_fn=lambda:self.restart(state))
                    ui.Button("PAUSE",clicked_fn=lambda:self.pause(state,True))
                    ui.Button("CONTINUE",clicked_fn=lambda:self.pause(state,False))
                with ui.HStack(height=29):
                    ui.Button("FOLLOW CAMERA",clicked_fn=lambda:self.camera(state,"follow"))
                    ui.Button("WHOLE ROUTE",clicked_fn=lambda:self.camera(state,"wide"))
                    ui.Button("0.5x",clicked_fn=lambda:self.rate(state,.5))
                    ui.Button("1.0x",clicked_fn=lambda:self.rate(state,1.))
                self.phase=ui.Label("GOAL: A -> B",height=22,style={"font_size":16})
                self.goal=ui.Label("TARGET B: x=-- m",height=22)
                self.progress=ui.Label("FORWARD PROGRESS: -- / -- m",height=23)
                self.line=ui.Label("OFF-ROUTE LATERAL: -- m",height=22)
                self.speed=ui.Label("REFERENCE SPEED: -- m/s",height=22)
                self.note=ui.Label("Arrive at B -> STOP. No automatic random walking.")
                ui.Label("Yellow finish line = target, not a second ball")
    @staticmethod
    def choose(state,name):
        state["mode"]=name
        state["reload"]=True
    @staticmethod
    def restart(state):
        state["reload"]=True
    @staticmethod
    def pause(state,yes):
        state["paused"]=bool(yes)
        state["last_time"]=time.monotonic()
    @staticmethod
    def rate(state,rate):
        state["rate"]=float(rate)
        state["last_time"]=time.monotonic()
    @staticmethod
    def camera(state,name):
        state["camera_mode"]=name
        state["force_camera"]=True


def main():
    ctx=SimulationContext(SimulationCfg(dt=1./60,render_interval=1))
    grass=sim.GroundPlaneCfg(color=(.07,.28,.12))
    grass.func("/World/GaitReferencePhysicsGround",grass)
    light=sim.DomeLightCfg(intensity=2500,color=(.92,.94,1.0))
    light.func("/World/GaitReferenceSky",light)
    reference=G1HumanReferenceVisual(max_triangles_per_visual=1000000)
    finish_marker=create_straight_lane()
    ctx.reset()
    if not args.headless:
        for name in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
            pane=ui.Workspace.get_window(name)
            if pane is not None:pane.visible=False

    state={
       "mode":"WALK SLOW",
       "reload":True,
       "paused":False,
       "completed":False,
       "rate":1.0,
       "camera_mode":"wide",
       "force_camera":True,
       "last_time":time.monotonic()
    }
    panel=None if args.headless else Panel(state)
    data=None
    meta=None
    prev_idx=-1
    k=0
    current_t=0.
    goal_x=0.
    pose=[0.,0.,.75]
    prev_xy=None
    speed=0.
    log_next=None
    while app.is_running():
        if state["reload"]:
            entry=INFO[state["mode"]]
            data=np.load(entry["file"])["frames"]
            meta=entry
            assert data.shape[1]==36
            goal_x=float(meta["goal_world_xy"][0])
            update_route_finish(finish_marker,goal_x)
            current_t=0.
            prev_idx=-1
            prev_xy=None
            state["paused"]=False
            state["completed"]=False
            state["reload"]=False
            state["force_camera"]=True
            state["last_time"]=time.monotonic()
            print("STRAIGHT_REFERENCE_DEMO_START",state["mode"],
                  "goal_x_m",goal_x,"duration_s",meta["play_duration_s"],
                  "source_reference_only",True,flush=True)
        now=time.monotonic()
        dt=now-state["last_time"]
        state["last_time"]=now
        if not state["paused"] and not state["completed"]:
            current_t=min(float(meta["play_duration_s"]),current_t+dt*state["rate"])
        idx=min(len(data)-1,int(current_t*30))
        if idx!=prev_idx:
            xyz=reference.update(data[idx])
            pose=[float(v) for v in xyz]
            if prev_xy is not None:
                step_length=float(np.linalg.norm(np.asarray(pose[:2])-prev_xy))
                # Numerical differentiated speed in reference timeline.
                speed=step_length*30
            prev_xy=np.asarray(pose[:2],dtype=float)
            prev_idx=idx
            if k%24==0:
                print("STRAIGHT_REFERENCE_PROGRESS",state["mode"],"frame",idx,
                      "root",np.round(pose,3).tolist(),
                      "target_m",round(goal_x,3),
                      "remaining_m",round(goal_x-pose[0],3),flush=True)
        if idx>=len(data)-1 and not state["completed"]:
            state["completed"]=True
            state["paused"]=True
            print("STRAIGHT_REFERENCE_GOAL_REACHED",state["mode"],
                  "goal_x",round(goal_x,3),"actual_x",round(pose[0],3),
                  "lateral_abs",round(abs(pose[1]),3),
                  "auto_replay",False,flush=True)
        if not args.headless and (k%8==0 or state["force_camera"]):
            if state["camera_mode"]=="follow":
                ctx.set_camera_view(
                    eye=[pose[0]+1.9,-3.1,1.6],
                    target=[pose[0],pose[1],.72])
            else:
                cx=.5*goal_x
                ctx.set_camera_view(
                    eye=[cx+2.45,-6.7,2.3],
                    target=[cx,0.,.68])
            state["force_camera"]=False
        if panel and k%10==0:
            phase="ARRIVED B - CLICK REPLAY" if state["completed"] else (
                "PAUSED" if state["paused"] else "WALKING A -> B")
            panel.phase.text="STATE: "+phase
            panel.goal.text=f"TARGET B: x={goal_x:.2f}m | t={current_t:.2f}s"
            panel.progress.text=f"PROGRESS {pose[0]:.2f} / {goal_x:.2f}m | {100*min(1,max(0,pose[0]/goal_x)):.0f}%"
            panel.line.text=f"CROSS-TRACK DEVIATION: {pose[1]:+.3f}m"
            panel.speed.text=f"REFERENCE INSTANT SPEED: {speed:.2f}m/s"
            panel.note.text=f"Source straightness: {meta['straightness_ratio']:.3f} | one-shot"
        ctx.step()
        k+=1
        if args.frames and k>=args.frames:break
        if not args.headless:time.sleep(.006)
    health={
        "frames":k,"mode":state["mode"],"completed":state["completed"],
        "final_root_xy":pose[:2],"target_x":goal_x,
        "reference_only_no_dynamic_control":True,
        "replay_loops_automatically":False
    }
    output=BASE/"reports/gait_straight_route_viewer_health.json"
    output.write_text(json.dumps(health,indent=2))
    print("STRAIGHT_GAIT_VIEWER_EXIT",health,flush=True)

if __name__=="__main__":
    try:main()
    finally:app.close()
