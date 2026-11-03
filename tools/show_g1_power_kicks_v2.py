"""KineticKick V2: **VISIBLE DIFFERENT KICKS + ADJUSTABLE POWER**, Tang Isaac Lab GUI.

Verified in headless calibration with original G1 37-DoF foot:
- FRONT: actually toe/sole-edge-like strike, NOT authentic instep.
- INSIDE: approx medial side of original bare rigid rectangular foot.
- OUTSIDE: approx lateral side of original bare rigid rectangular foot.
- Power uses controlled JOINT MOVEMENT TIMING. Football velocity comes only
  from collisions, never assigned artificially after being struck.

This experiment uses FIXED ROOT to isolate kicking contacts; not natural walking,
a learned football policy, or an RGB camera goal recognizer.
"""
from __future__ import annotations

from isaaclab.app import AppLauncher
app=AppLauncher(headless=False,enable_cameras=False).app

import json,math,time,csv
from pathlib import Path
from pxr import Gf,UsdGeom
import omni.usd
import omni.ui as ui
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.utils.math import quat_apply_inverse,quat_apply
from soccer_world import create_goal,locate_goal_in_usd,classify_ball_crossing

ROOT=Path("/home/you/projects/kinetickick")
CONFIG=json.loads((ROOT/"configs/kick_modes_v1.json").read_text())
MODES=CONFIG["modes"]
POWERS=CONFIG["power_duration_seconds"]
MODENAMES=("instep_top","medial","lateral","front")
POWER_ORDER=("soft","medium","strong")
DT=1./120.
GOAL_X=2.20
FOOT_C=[.035917,0.,-.025170]
FOOT_HALF=[.203109/2,.065469/2,.018508/2]

def smooth(x):
    v=max(0.,min(1.,float(x)))
    return v*v*(3-2*v)

def infer_face(ball_local):
    d=[float(ball_local[i])-FOOT_C[i] for i in range(3)]
    gap=[d[i]-max(-FOOT_HALF[i],min(FOOT_HALF[i],d[i])) for i in range(3)]
    axis=max(range(3),key=lambda i:abs(gap[i]))
    if axis==0:return "TOE / FRONT" if gap[axis]>0 else "HEEL"
    if axis==1:return "INNER / MEDIAL" if gap[axis]>0 else "OUTER / LATERAL"
    return "TOP" if gap[axis]>0 else "SOLE / LOWER EDGE"

def create_stage():
    ground=sim_utils.GroundPlaneCfg(color=(.21,.39,.26))
    ground.func("/World/GrassGround",ground)
    skylight=sim_utils.DomeLightCfg(intensity=2300,color=(.91,.93,1.))
    skylight.func("/World/Skylight",skylight)
    sun=sim_utils.DistantLightCfg(intensity=1100,color=(.98,.94,.84))
    sun.func("/World/Sun",sun)
    botcfg=G1_CFG.copy()
    botcfg.prim_path="/World/G1"
    botcfg.spawn.articulation_props.fix_root_link=True
    robot=Articulation(botcfg)
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/Football",
        spawn=sim_utils.SphereCfg(
            radius=.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(.96,.81,.17),roughness=.53)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.4,.2,.113)),
    ))
    goal=create_goal(goal_x=GOAL_X,initial_y=0.)
    footsensor=ContactSensor(ContactSensorCfg(
        prim_path="/World/G1/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["/World/Football"]))
    return robot,ball,footsensor,goal


def add_display_only_contact_dots(stage):
    """Tiny colored spheres as target-region VISUAL GUIDES (no collision)."""
    specs={
        "instep_top":([.075,0.,-.006],(.72,.30,1.)),
        "front":([.148,0.,-.020],(.32,.63,1.)),
        "medial":([.05,.047,-.025],(.2,.95,.34)),
        "lateral":([.05,-.047,-.025],(1.,.36,.16)),
    }
    out={}
    for name,(coord,color) in specs.items():
        path="/World/G1/right_ankle_roll_link/ContactGuide_"+name
        marker=UsdGeom.Sphere.Define(stage,path)
        marker.CreateRadiusAttr(.017)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf=UsdGeom.Xformable(marker)
        xf.AddTranslateOp().Set(Gf.Vec3d(*coord))
        # Crucial: DO NOT ADD COLLISION API. This is NOT a soccer shoe.
        out[name]=marker
    return out

def set_guide_visibility(markers,current,enabled):
    for mode,geo in markers.items():
        highlighted="front" if current=="knee_only" else current
        if enabled and mode==highlighted:
            geo.MakeVisible()
        else:
            geo.MakeInvisible()

def set_goal_y(goal,y):
    xf=UsdGeom.Xformable(goal)
    for op in xf.GetOrderedXformOps():
        if str(op.GetOpName())=="xformOp:translate":
            op.Set(Gf.Vec3d(GOAL_X,float(y),0.))
            return
    raise RuntimeError("Missing goal transform")

class LabPanel:
    def __init__(self,state):
        self.win=ui.Window("KineticKick V3 | TRUE INSTEP + 3 KICKS",width=495,height=560,visible=True)
        self.win.position_x=910
        self.win.position_y=63
        with self.win.frame:
            with ui.VStack(spacing=7):
                ui.Label("TRUE TOP-FOOT INSTEP + OTHER KICKS | ORIGINAL G1",
                         height=25,style={"font_size":17})
                ui.Label("Only visual contact dots; NO added shoe or colliders",height=20)
                with ui.HStack(height=32,spacing=5):
                    ui.Button("INSTEP (TOP FOOT)",clicked_fn=lambda:self.set_kind(state,"instep_top"))
                    ui.Button("FRONT (TOE)",clicked_fn=lambda:self.set_kind(state,"front"))
                    ui.Button("INNER SIDE",clicked_fn=lambda:self.set_kind(state,"medial"))
                    ui.Button("OUTER SIDE",clicked_fn=lambda:self.set_kind(state,"lateral"))
                with ui.HStack(height=31,spacing=5):
                    ui.Button("THIGH -> SHANK (INSTEP)",clicked_fn=lambda:self.set_kind(state,"instep_top"))
                    ui.Button("KNEE ONLY (EXPERIMENTAL)",clicked_fn=lambda:self.set_kind(state,"knee_only"))
                ui.Label("POWER / joint swing speed (not a ball velocity override)",height=20)
                with ui.HStack(height=31,spacing=5):
                    ui.Button("LOW",clicked_fn=lambda:self.set_power(state,"soft"))
                    ui.Button("MED",clicked_fn=lambda:self.set_power(state,"medium"))
                    ui.Button("HIGH",clicked_fn=lambda:self.set_power(state,"strong"))
                with ui.HStack(height=31,spacing=5):
                    ui.Button("CYCLE 4 KICKS",clicked_fn=lambda:self.set_demo(state,"methods"))
                    ui.Button("CYCLE 3 POWERS",clicked_fn=lambda:self.set_demo(state,"power_ramp"))
                    ui.Button("MANUAL",clicked_fn=lambda:self.set_demo(state,"manual"))
                with ui.HStack(height=31,spacing=5):
                    ui.Button("CLOSE FOOT",clicked_fn=lambda:self.set_view(state,"close"))
                    ui.Button("WHOLE FIELD",clicked_fn=lambda:self.set_view(state,"wide"))
                    ui.Button("SLOW",clicked_fn=lambda:self.set_pace(state,True))
                    ui.Button("NORMAL",clicked_fn=lambda:self.set_pace(state,False))
                with ui.HStack(height=30,spacing=5):
                    ui.Button("SHOW DOT",clicked_fn=lambda:self.set_marker(state,True))
                    ui.Button("HIDE DOT",clicked_fn=lambda:self.set_marker(state,False))
                    ui.Button("REPLAY",clicked_fn=lambda:self.restart(state))
                self.mode=ui.Label("MODE: loading...",height=23,style={"font_size":16})
                self.phase=ui.Label("PHASE: initializing...",height=20)
                self.contact=ui.Label("CONTACT: not yet",height=20)
                self.force=ui.Label("FORCE peak: -- N",height=20)
                self.velocity=ui.Label("BALL horiz peak: -- m/s",height=20)
                self.goal=ui.Label("GOAL: --",height=20)
                ui.Label("Fixed root. No human-motion RL policy; face is geometric estimate.")
    @staticmethod
    def set_kind(s,mode):
        s["mode"]=mode
        s["demo"]="manual"
        s["reset"]=True
    @staticmethod
    def set_power(s,power):
        s["power"]=power
        s["reset"]=True
    @staticmethod
    def set_demo(s,demo):
        s["demo"]=demo
        if demo=="methods":s["mode_idx"]=0
        if demo=="power_ramp":s["power_idx"]=0
        s["reset"]=True
    @staticmethod
    def set_view(s,view):
        s["view"]=view
        s["refresh_view"]=True
    @staticmethod
    def set_pace(s,slow):
        s["slow"]=bool(slow)
    @staticmethod
    def set_marker(s,enabled):
        s["marker"]=bool(enabled)
        s["refresh_marker"]=True
    @staticmethod
    def restart(s):
        s["reset"]=True

def main():
    sim=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor,goal=create_stage()
    # first reset creates articulated links
    sim.reset()
    stage=omni.usd.get_context().get_stage()
    assert not stage.GetPrimAtPath("/World/G1/right_ankle_roll_link/KineticShoe").IsValid()
    assert stage.GetPrimAtPath("/World/G1/right_ankle_roll_link/Cube").IsValid()
    markers=add_display_only_contact_dots(stage)
    joint_map={j:i for i,j in enumerate(robot.joint_names)}
    foot_i=robot.body_names.index("right_ankle_roll_link")
    limits=robot.data.soft_joint_pos_limits[0]
    default=robot.data.default_joint_pos.clone()
    zero=torch.zeros_like(default)

    state={"mode":"instep_top","power":"medium","demo":"manual",
           "mode_idx":0,"power_idx":0,"reset":True,"view":"close",
           "refresh_view":True,"marker":True,"refresh_marker":True,
           "slow":True}
    panel=LabPanel(state)

    def write_q(q,name,val):
        i=joint_map[name]
        q[:,i]=max(float(limits[i,0])+.008,min(float(limits[i,1])-.008,float(val)))

    def joint_targets(kind,power,t):
        p=MODES[kind]
        duration=POWERS[power]
        hp=smooth((t-float(p["hip_delay"]))/duration)
        kn=smooth((t-float(p["knee_delay"]))/duration)
        q=default.clone()
        for name,value in [
            ("right_hip_yaw_joint",p["yaw"]),
            ("right_ankle_pitch_joint",p["pitch"]),
            ("right_ankle_roll_joint",p["roll"]),
            ("right_hip_pitch_joint",p["hip_start"]+(p["hip_end"]-p["hip_start"])*hp),
            ("right_knee_joint",p["knee_start"]+(p["knee_end"]-p["knee_start"])*kn),
        ]:
            write_q(q,name,value)
        # Tiny counter-swing of arms. Does not substitute for balance.
        if kind in ("medial","lateral"):
            write_q(q,"left_shoulder_pitch_joint",
                    float(default[0,joint_map["left_shoulder_pitch_joint"]])+.12*hp)
            write_q(q,"right_shoulder_pitch_joint",
                    float(default[0,joint_map["right_shoulder_pitch_joint"]])-.12*hp)
        return q

    def update_camera():
        if state["view"]=="close":
            # Full lower body + football closeup, goal is out of frame.
            sim.set_camera_view(eye=[2.17,-2.78,1.59],
                                target=[.19,-.10,.56])
        else:
            sim.set_camera_view(eye=[3.15,-3.8,1.85],
                                target=[1.05,-.06,.46])
        print("CAMERA_VIEW",state["view"],flush=True)

    report_path=ROOT/"reports/live_power_kicks.csv"
    report_path.parent.mkdir(exist_ok=True,parents=True)
    headers=["shot_id","mode","power","face_est","impact_time_s",
             "impact_force_N","peak_force_N","peak_horizontal_ball_speed_mps",
             "initial_ball_xyz","crossing","crossing_xyz"]
    if not report_path.exists():
        with report_path.open("w",newline="") as f:
            csv.writer(f).writerow(headers)
    def write_report(record):
        with report_path.open("a",newline="") as f:
            csv.writer(f).writerow([record.get(k,"") for k in headers])

    frame=500
    shot_id=0
    shot_duration_s=2.15
    nframes=int(shot_duration_s/DT)
    impact=None
    face="NOT YET"
    max_force=0.
    max_speed=0.
    goal_crossing=None
    crossing_xyz=None
    previous=None
    kind="medial"
    power="medium"
    while app.is_running():
        if state["refresh_view"]:
            update_camera()
            state["refresh_view"]=False
        if state["refresh_marker"]:
            set_guide_visibility(markers,state["mode"],state["marker"])
            state["refresh_marker"]=False
        if state["reset"] or frame>=nframes:
            if shot_id and impact is not None:
                # Recording at end of a shot gives measured end-to-end outcome.
                write_report({
                    "shot_id":shot_id,"mode":kind,"power":power,
                    "face_est":face,"impact_time_s":impact["time_s"],
                    "impact_force_N":impact["first_force_N"],
                    "peak_force_N":round(max_force,2),
                    "peak_horizontal_ball_speed_mps":round(max_speed,3),
                    "initial_ball_xyz":json.dumps(MODES[kind]["ball_start"]),
                    "crossing":("PENDING" if goal_crossing is None else
                                ("INSIDE" if goal_crossing else "OUTSIDE")),
                    "crossing_xyz":json.dumps(crossing_xyz)
                })
                print("SHOT_RESULT",shot_id,kind,power,face,
                      "peak_ball_speed",round(max_speed,3),
                      "peak_force",round(max_force,1),
                      "goal",goal_crossing,flush=True)
            elif shot_id:
                print("SHOT_MISSED_CONTACT",shot_id,kind,power,flush=True)

            state["reset"]=False
            if state["demo"]=="methods":
                kind=MODENAMES[state["mode_idx"]%len(MODENAMES)]
                state["mode_idx"]+=1
                power=state["power"]
            elif state["demo"]=="power_ramp":
                kind=state["mode"]
                power=POWER_ORDER[state["power_idx"]%len(POWER_ORDER)]
                state["power_idx"]+=1
            else:
                kind=state["mode"]
                power=state["power"]
            state["mode"]=kind
            set_guide_visibility(markers,kind,state["marker"])
            ball_init=MODES[kind]["ball_start"]
            pose=joint_targets(kind,power,0.)
            robot.write_joint_state_to_sim(pose,zero)
            robot.reset()
            root=ball.data.default_root_state.clone()
            root[:,:3]=torch.tensor(ball_init,device=ball.device)
            root[:,7:]=0.
            ball.write_root_pose_to_sim(root[:,:7])
            ball.write_root_velocity_to_sim(root[:,7:])
            ball.reset()
            sensor.reset()
            previous=None
            goal_crossing=None
            crossing_xyz=None
            max_force=0.
            max_speed=0.
            face="NOT YET"
            impact=None
            frame=0
            shot_id+=1
            print("SHOT_START",shot_id,kind,power,"ball",ball_init,
                  "goal",locate_goal_in_usd(),
                  "mode_display",MODES[kind]["label"],flush=True)
        t=frame*DT
        robot.set_joint_position_target(joint_targets(kind,power,t))
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(DT)
        ball.update(DT)
        sensor.update(DT)

        position=[float(v) for v in ball.data.root_pos_w[0]]
        velocity=[float(v) for v in ball.data.root_lin_vel_w[0]]
        horizontal_speed=math.hypot(velocity[0],velocity[1])
        max_speed=max(max_speed,horizontal_speed)
        mat=sensor.data.force_matrix_w
        force=float(torch.linalg.vector_norm(mat).item()) if mat is not None and mat.numel()>0 else 0.
        max_force=max(max_force,force)
        if force>2. and impact is None:
            pos=robot.data.body_pos_w[0,foot_i]
            quat=robot.data.body_quat_w[0,foot_i]
            local=quat_apply_inverse(quat.unsqueeze(0),
                  (ball.data.root_pos_w[0]-pos).unsqueeze(0))[0]
            face=infer_face(local)
            top_normal=quat_apply(quat.unsqueeze(0),
                 torch.tensor([[0.,0.,1.]],device=quat.device))[0]
            top_forward=float(top_normal[0])
            within_top_interior=bool(-.053<float(local[0])<.125 and
                                     abs(float(local[1]))<.018)
            if kind=="instep_top":
                if face=="TOP" and top_forward>.60 and within_top_interior:
                    face="INSTEP / DORSAL TOP (GEOMETRY)"
                else:
                    face="INSTEP CLAIM FAILED - "+face
            bearing=math.degrees(math.atan2(velocity[1],velocity[0]))
            impact={"time_s":round(t,4),"first_force_N":round(force,2),
                    "face_est":face,"top_forward_component":round(top_forward,3),
                    "bearing_deg":round(bearing,1),
                    "vxyz":[round(float(x),3) for x in velocity]}
            print("PHYSICAL_BALL_HIT",shot_id,kind,power,
                  json.dumps(impact),flush=True)

        if goal_crossing is None:
            result=classify_ball_crossing(previous,position,locate_goal_in_usd())
            if result is not None:
                goal_crossing=result
                crossing_xyz=position
                print("GOAL_CROSSING",shot_id,kind,power,result,position,flush=True)
        previous=position

        if frame%10==0:
            if kind=="knee_only":
                phase="KNEE-ONLY EXPERIMENT: POWER MAY BE NON-MONOTONIC"
            elif t<.31:phase="PREPARE: foot orientation"
            elif t<.74:phase="STRIKE: hip/knee leg chain"
            else:phase="FOLLOW-THROUGH / BALL FLIGHT"
            panel.mode.text=f"NOW: {MODES[kind]['label']} | {power.upper()}"
            panel.phase.text=f"PHASE: {phase}"
            panel.contact.text=f"Impact: {face} ({'YES' if impact is not None else 'waiting'})"
            panel.force.text=f"Measured foot-ball FORCE peak: {max_force:.1f} N"
            panel.velocity.text=f"Measured BALL speed peak: {max_speed:.2f} m/s"
            panel.goal.text=("Goal crossing: " + ("pending" if goal_crossing is None else
                ("INSIDE goal width" if goal_crossing else "OUTSIDE goal width")))
        frame+=1
        time.sleep(.022 if state["slow"] else .002)

if __name__=="__main__":
    try:main()
    finally:app.close()
