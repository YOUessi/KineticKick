"""G1 BARE FOOT soccer-lab visualization — forward medial-foot passing.

No added shoe, no alternative foot colliders. The original G1 right-foot Cube
collider stays enabled. An anchored-base contact rig compares proximal-to-distal
hip+knee swing and predominantly knee-driven control.

Goal coordinates come from the USD stage (NOT an RGB goal detector).
Auto yaw choice is based on previously measured bare-foot physics outcomes,
not a learned policy. A contact face is estimated from the sphere center and
G1 collision cuboid; no actual PhysX per-point contact location is claimed.
"""
from __future__ import annotations

from isaaclab.app import AppLauncher
app = AppLauncher(headless=False, enable_cameras=False).app

import math
import time
import torch
import omni.ui as ui
import omni.usd
from pxr import Gf, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.utils.math import quat_apply_inverse
from soccer_world import create_goal, locate_goal_in_usd, classify_ball_crossing

DT=1.0/120.0
GOAL_X=2.20
PASS_START=(0.40,0.20,.113)
STRIKE_ANGLES={
    -1.30:(1.951,0.452),
    -1.60:(1.908,0.268),
    -1.80:(1.631,0.108),
}
BARE_FOOT_COLLIDER_CENTER=(.035917,0.,-.025170)
BARE_FOOT_HALF_EXTENTS=(.203109/2,.065469/2,.018508/2)

def smooth(x):
    x=max(0.,min(1.,x))
    return x*x*(3.-2.*x)

def make_scene():
    ground=sim_utils.GroundPlaneCfg(color=(.22,.38,.25))
    ground.func("/World/Grass",ground)
    light=sim_utils.DomeLightCfg(intensity=2300,color=(.9,.94,1.))
    light.func("/World/Light",light)
    config=G1_CFG.copy()
    config.prim_path="/World/UnitreeG1"
    config.spawn.articulation_props.fix_root_link=True
    robot=Articulation(config)
    goal=create_goal(goal_x=GOAL_X,initial_y=.25)
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/Football",
        spawn=sim_utils.SphereCfg(
            radius=.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.97,.79,.21),roughness=.55),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PASS_START),
    ))
    contact=ContactSensor(ContactSensorCfg(
        prim_path="/World/UnitreeG1/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["/World/Football"],
    ))
    return robot,goal,ball,contact

def set_goal_y(goal,y):
    x=UsdGeom.Xformable(goal)
    for op in x.GetOrderedXformOps():
        if str(op.GetOpName())=="xformOp:translate":
            op.Set(Gf.Vec3d(GOAL_X,float(y),0.))
            return
    raise RuntimeError("Goal root has no translation attribute")

def infer_cube_face(ball_local):
    # The G1 foot is a box. This is approximate closest-face inference based
    # on the sphere center, NOT actual PhysX contact points on a curved arch.
    delta=[float(ball_local[i])-BARE_FOOT_COLLIDER_CENTER[i] for i in range(3)]
    gap=[]
    for i in range(3):
        h=BARE_FOOT_HALF_EXTENTS[i]
        gap.append(delta[i]-max(-h,min(+h,delta[i])))
    a=max(range(3),key=lambda i:abs(gap[i]))
    if a==0:return "TOE" if gap[0]>0 else "HEEL"
    if a==1:return "MEDIAL SIDE" if gap[1]>0 else "LATERAL SIDE"
    return "TOP" if gap[2]>0 else "SOLE"

def select_yaw(goal_y):
    candidates={}
    for yaw,(vx,vy) in STRIKE_ANGLES.items():
        y_pred=PASS_START[1]+(GOAL_X-PASS_START[0])*vy/vx
        candidates[yaw]=float(y_pred)
    chosen=min(candidates,key=lambda yaw:abs(candidates[yaw]-goal_y))
    return chosen,candidates

class Panel:
    def __init__(self,s):
        self.win=ui.Window("KineticKick | ORIGINAL G1 FOOT",width=450,height=362,visible=True)
        self.win.position_x=945
        self.win.position_y=65
        with self.win.frame:
            with ui.VStack(spacing=6):
                ui.Label("BARE G1 FOOT + REAL PHYSICAL FOOTBALL",height=24,
                         style={"font_size":17})
                ui.Label("No football shoe, no extra collision geometry",height=21)
                with ui.HStack(height=27,spacing=4):
                    ui.Button("AUTO from goal",clicked_fn=lambda:self.auto(s))
                    ui.Button("Yaw -1.3",clicked_fn=lambda:self.yaw(s,-1.30))
                    ui.Button("Yaw -1.6",clicked_fn=lambda:self.yaw(s,-1.60))
                    ui.Button("Yaw -1.8",clicked_fn=lambda:self.yaw(s,-1.80))
                with ui.HStack(height=27,spacing=5):
                    ui.Button("Hip then knee",clicked_fn=lambda:self.chain(s,"hip_knee"))
                    ui.Button("Knee only",clicked_fn=lambda:self.chain(s,"knee_only"))
                    ui.Button("Instep control",clicked_fn=lambda:self.instep(s))
                ui.Label("Move research goal: location from stage, NOT camera",height=20)
                with ui.HStack(height=27,spacing=5):
                    ui.Button("Goal Y=0.25",clicked_fn=lambda:self.goal(s,.25))
                    ui.Button("Goal Y=0.45",clicked_fn=lambda:self.goal(s,.45))
                    ui.Button("Goal Y=0.65",clicked_fn=lambda:self.goal(s,.65))
                self.pose=ui.Label("Preparing foot pose")
                self.foot=ui.Label("Foot nearest face: awaiting first contact")
                self.speed=ui.Label("Ball forward x: -- m/s | lateral y: -- m/s")
                self.power=ui.Label("Filtered foot-ball force: -- N")
                self.goalstatus=ui.Label("Goal plane: pending")
                ui.Label("Root is FIXED. This is NOT a learned natural soccer policy.")
    @staticmethod
    def auto(s):
        s["auto"]=True;s["mode"]="inside";s["reset"]=True
    @staticmethod
    def yaw(s,val):
        s["auto"]=False;s["mode"]="inside";s["yaw"]=val;s["reset"]=True
    @staticmethod
    def chain(s,val):
        s["style"]=val;s["reset"]=True
    @staticmethod
    def instep(s):
        s["auto"]=False;s["mode"]="instep";s["reset"]=True
    @staticmethod
    def goal(s,val):
        s["goal_y"]=val;s["reset"]=True

def main():
    sim=SimulationContext(SimulationCfg(dt=DT))
    sim.set_camera_view(eye=[2.9,-3.65,1.72],target=[1.05,.05,.48])
    robot,goal,ball,sensor=make_scene()
    sim.reset()
    stage=omni.usd.get_context().get_stage()
    assert not stage.GetPrimAtPath("/World/UnitreeG1/right_ankle_roll_link/KineticShoe").IsValid(), \
           "Shoe should not exist in bare-foot stage"
    print("BARE_FOOT_CONFIRMED no KineticShoe prim, original collider intact",flush=True)
    s={"auto":True,"mode":"inside","yaw":-1.80,"style":"hip_knee",
       "goal_y":.25,"reset":True}
    panel=Panel(s)
    ids={n:i for i,n in enumerate(robot.joint_names)}
    fi=robot.body_names.index("right_ankle_roll_link")
    base=robot.data.default_joint_pos.clone()
    zero=torch.zeros_like(base)
    limits=robot.data.soft_joint_pos_limits[0]
    def put(q,name,value):
        i=ids[name]
        q[:,i]=max(float(limits[i,0])+.006,min(float(limits[i,1])-.006,float(value)))
    def joint_target(t):
        q=base.clone()
        a=smooth((t-.32)/.24)
        if s["mode"]=="instep":
            yaw=0.
            pitch=-.48
            roll=0.
            hip=.35+(-.87-.35)*smooth((t-.30)/.24)
            knee=.55+(.43-.55)*smooth((t-.37)/.21)
        else:
            yaw=s["yaw"]
            pitch=-.20
            roll=-.23
            if s["style"]=="hip_knee":
                hip=.20+(-.57-.20)*a
                knee=.36+(.15-.36)*smooth((t-.40)/.18)
            else:
                hip=-.48
                knee=1.06+(.13-1.06)*smooth((t-.33)/.25)
        for name,value in [
            ("right_hip_yaw_joint",yaw),("right_hip_pitch_joint",hip),
            ("right_knee_joint",knee),("right_ankle_pitch_joint",pitch),
            ("right_ankle_roll_joint",roll)]:
            put(q,name,value)
        return q
    frame=0
    previous=None
    crossed=None
    hit=False
    peak_force=0.
    face="unverified"
    impact_velocity=None
    max_frames=int(2.9/DT)
    shot_counter=0
    while app.is_running():
        if s["reset"] or frame>=max_frames:
            s["reset"]=False
            set_goal_y(goal,s["goal_y"])
            goal_pos=locate_goal_in_usd()
            if s["auto"] and s["mode"]=="inside":
                s["yaw"],pred=select_yaw(goal_pos[1])
                print("GOAL_LOCATED_USD auto yaw",s["yaw"],"candidates",pred,flush=True)
            q=joint_target(0)
            robot.write_joint_state_to_sim(q,zero)
            robot.reset()
            bp=ball.data.default_root_state.clone()
            spawn=PASS_START if s["mode"]=="inside" else (.38,-.12,.113)
            bp[:,:3]=torch.tensor(spawn,device=ball.device)
            bp[:,7:]=0.
            ball.write_root_pose_to_sim(bp[:,:7]);ball.write_root_velocity_to_sim(bp[:,7:])
            ball.reset();sensor.reset()
            previous=None;crossed=None;hit=False
            peak_force=0;face="unverified";impact_velocity=None
            frame=0;shot_counter+=1
            print("SHOT_START",shot_counter,s["mode"],s["yaw"],s["style"],
                  "goal",goal_pos,"ball",spawn,flush=True)
        t=frame*DT
        robot.set_joint_position_target(joint_target(t))
        robot.write_data_to_sim();ball.write_data_to_sim()
        sim.step();robot.update(DT);ball.update(DT);sensor.update(DT)
        xyz=[float(v) for v in ball.data.root_pos_w[0]]
        vel=[float(v) for v in ball.data.root_lin_vel_w[0]]
        force_data=sensor.data.force_matrix_w
        force=float(torch.linalg.vector_norm(force_data).item()) if force_data is not None and force_data.numel() else 0.
        peak_force=max(peak_force,force)
        if force>2.0 and not hit:
            hit=True
            fp=robot.data.body_pos_w[0,fi]
            fq=robot.data.body_quat_w[0,fi]
            center=ball.data.root_pos_w[0]
            local=quat_apply_inverse(fq.unsqueeze(0),(center-fp).unsqueeze(0))[0]
            face=infer_cube_face(local)
            impact_velocity=vel
            bearing=math.degrees(math.atan2(vel[1],vel[0]))
            print("ACTUAL_BARE_FOOT_IMPACT",s["mode"],s["style"],
                  "yaw",s["yaw"],"face_est",face,
                  "ball_local",[round(float(v),3) for v in local],
                  "vxyz",[round(x,3) for x in vel],
                  "bearing_deg",round(bearing,2),
                  "force_n",round(force,2),flush=True)
        if crossed is None:
            crossed=classify_ball_crossing(previous,xyz,locate_goal_in_usd())
            if crossed is not None:
                print("GOAL_CROSSING", "INSIDE" if crossed else "OUTSIDE",
                      "ball",xyz,"goal",locate_goal_in_usd(),flush=True)
        previous=xyz
        if frame%12==0:
            panel.pose.text=f"MODE: {s['mode']} | yaw {s['yaw']:.2f} | {s['style']}"
            panel.foot.text="Estimated G1 foot face: "+face
            panel.speed.text=f"Ball forward vx: {vel[0]:.2f} | sideways vy: {vel[1]:.2f} m/s"
            panel.power.text=f"Filtered foot-ball force peak: {peak_force:.1f} N"
            panel.goalstatus.text=("Goal crossing: "+("PENDING" if crossed is None else ("INSIDE" if crossed else "OUTSIDE")))
        frame+=1
        time.sleep(.007)

if __name__=="__main__":
    try:main()
    finally:app.close()
