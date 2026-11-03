"""Tang Isaac Sim human-derived G1 29DoF retargeted WALK REFERENCE.

Strictly KINEMATIC motion replay, not controlled free-standing robot motion.
Requires no extra packages, no WandB, and no physics collision/force claims.
Use for visual research reference and 29DoF asset validation only.
"""
from __future__ import annotations
import argparse,sys,time,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--headless",action="store_true")
parser.add_argument("--frames",type=int,default=0)
cli=parser.parse_args()
from isaaclab.app import AppLauncher
app=AppLauncher(headless=cli.headless,enable_cameras=not cli.headless).app

import numpy as np
if not cli.headless:
    import omni.ui as ui
else:
    ui=None
import isaaclab.sim as sim
from isaaclab.sim import SimulationCfg,SimulationContext
from g1_29dof_reference_usd import G1HumanReferenceVisual
from g1_retarget_kinematics import DATA
from soccer_field_visual import create_field_visuals

FILES={
    "WALK-1":"walk1_subject1_selected8s.npz",
    "WALK-2":"walk2_subject3_selected8s.npz",
    "WALK-3":"walk3_subject5_selected8s.npz",
    "RUN-1":"run1_subject2_selected8s.npz",
}
ROOT=Path("/home/you/projects/kinetickick")

class Panel:
    def __init__(self,state):
        self.win=ui.Window("KineticKick | LAFAN1 HUMAN-GAIT REFERENCE",width=485,height=335,visible=True)
        self.win.position_x=925
        self.win.position_y=62
        with self.win.frame:
            with ui.VStack(spacing=7):
                ui.Label("HUMAN-DERIVED G1 29DoF WALK REFERENCE - NOT RL",
                         height=27,style={"font_size":16})
                ui.Label("Render-only retargeted motion; gravity & balance NOT tested")
                with ui.HStack(height=30):
                    ui.Button("WALK-1",clicked_fn=lambda:self.motion(state,"WALK-1"))
                    ui.Button("WALK-2",clicked_fn=lambda:self.motion(state,"WALK-2"))
                    ui.Button("WALK-3",clicked_fn=lambda:self.motion(state,"WALK-3"))
                    ui.Button("RUN-1",clicked_fn=lambda:self.motion(state,"RUN-1"))
                with ui.HStack(height=30):
                    ui.Button("PLAY",clicked_fn=lambda:self.pause(state,False))
                    ui.Button("PAUSE",clicked_fn=lambda:self.pause(state,True))
                    ui.Button("SLOW 0.5x",clicked_fn=lambda:self.rate(state,.5))
                    ui.Button("NORMAL 1.0x",clicked_fn=lambda:self.rate(state,1.))
                self.title=ui.Label("MOTION: WALK-1",height=22)
                self.timer=ui.Label("REFERENCE CLOCK: 0.0s / 8.0s",height=22)
                self.xyz=ui.Label("G1 root pose: pending...",height=22)
                ui.Label("Source: LAFAN1 → Unitree G1 29DoF IK retarget (30 FPS)")
                ui.Label("A successful visual replay is NOT stable physical walking.")
    @staticmethod
    def motion(s,m):
        s["motion"]=m
        s["change"]=True
    @staticmethod
    def pause(s,val):
        if bool(val)!=s["pause"]:
            s["pause"]=bool(val)
            s["pause_changed"]=True
    @staticmethod
    def rate(s,val):
        s["rate"]=float(val)
        s["rate_changed"]=True

def main():
    ctx=SimulationContext(SimulationCfg(dt=1./60,render_interval=1))
    ground=sim.GroundPlaneCfg(color=(.07,.28,.12))
    ground.func("/World/PhysicalReferenceGround",ground)
    daylight=sim.DomeLightCfg(intensity=2400,color=(.93,.94,1.))
    daylight.func("/World/ReferenceSky",daylight)
    create_field_visuals()
    reference=G1HumanReferenceVisual(max_triangles_per_visual=1000000)
    ctx.reset()
    if not cli.headless:
        for name in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
            pane=ui.Workspace.get_window(name)
            if pane is not None:pane.visible=False
    state={"motion":"WALK-1","change":True,"pause":False,
           "pause_changed":False,"rate":1.0,"rate_changed":False}
    panel=None if cli.headless else Panel(state)
    frames=None
    start_time=time.monotonic()
    reference_start_xy=None
    prev_frame=-1
    k=0
    t_progress=0.
    duration=8.
    current_clock=0.
    current_pose=None
    while app.is_running():
        if state["change"]:
            arr=np.load(DATA/FILES[state["motion"]])
            frames=arr["frames"]
            rate_hz=int(arr["fps"])
            if frames.shape[1]!=36 or rate_hz!=30:
                raise RuntimeError("Invalid retarget reference")
            state["change"]=False
            start_time=time.monotonic()
            current_clock=0.
            prev_frame=-1
            duration=len(frames)/rate_hz
            reference_start_xy=frames[0,:2].copy()
            print("G1_29_REFERENCE_CLIP_ACTIVE",state["motion"],
                  "frames",len(frames),"seconds",round(duration,2),flush=True)
        now=time.monotonic()
        if state["pause_changed"] or state["rate_changed"]:
            start_time=now-current_clock/state["rate"]
            state["pause_changed"]=False
            state["rate_changed"]=False
        if not state["pause"]:
            current_clock=((now-start_time)*state["rate"])%duration
        else:
            start_time=now-current_clock/state["rate"]
        fidx=int(current_clock*rate_hz)%len(frames)
        if fidx!=prev_frame:
            xyz=reference.update(frames[fidx],origin_xy=(-.5,0.),
                                  ref_start_xy=reference_start_xy)
            prev_frame=fidx
            current_pose=xyz
            if not cli.headless and k%3==0:
                ctx.set_camera_view(
                    eye=[float(xyz[0]+1.58),float(xyz[1]-2.95),1.50],
                    target=[float(xyz[0]),float(xyz[1]),.72])
            if k%60==0:
                print("G1_29_REFERENCE_FRAME","k",k,
                      "motion",state["motion"],"index",fidx,
                      "root",np.round(xyz,3).tolist(),flush=True)
        if panel is not None and k%10==0 and current_pose is not None:
            panel.title.text=f"MOTION {state['motion']} | speed {state['rate']:.1f}x | {('PAUSED' if state['pause'] else 'PLAYING')}"
            panel.timer.text=f"Human-derived motion clock {current_clock:.2f}s / {duration:.2f}s"
            panel.xyz.text=f"Reference root xyz: {current_pose[0]:.2f}, {current_pose[1]:.2f}, {current_pose[2]:.2f}"
        ctx.step()
        k+=1
        if cli.frames and k>=cli.frames:break
        if not cli.headless:time.sleep(.012)
    out=ROOT/"reports/g1_29dof_reference_viewer_health.json"
    out.write_text(json.dumps({"total_render_frames":k,
       "model":"Unitree G1 29DoF rev1_0",
       "reference_only":True,"physics_robot_policy_present":False,
       "collision_meshes_added":0,
       "fps_source":30,
       "last_motion":state["motion"]},indent=2))
    print("G1_29_REFERENCE_VIEWER_EXIT_OK",out,flush=True)

if __name__=="__main__":
    try:main()
    finally:app.close()
