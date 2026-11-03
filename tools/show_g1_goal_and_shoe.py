"""Live KineticKick soccer mechanics demo on Tang.

This is a physics test rig, not a trained soccer or goal-recognition policy:
- G1 built-in 37-DOF anchored root.
- A *prototype* 5-region, rigid football-shoe collider added to right foot.
- Movable 1.6 m wide goal placed in real IsaacLab stage.
- Goal center located from USD stage; it is NOT camera/image recognition.
- Switch hip-driven whole-leg swing / knee-only extension.
- Sensor-measured foot-to-ball force and physically simulated ball speed.
- Contact collider / anatomical shoe region NOT verified yet.
"""
from __future__ import annotations
import math,time
from isaaclab.app import AppLauncher
app=AppLauncher(headless=False,enable_cameras=False).app

import torch
import omni.ui as ui
import omni.physx
import omni.usd
from pxr import PhysicsSchemaTools, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
from soccer_world import (
    attach_training_football_shoe,create_goal,move_goal,locate_goal_in_usd,
    choose_pose_by_goal,classify_ball_crossing,GOAL_X,GOAL_WIDTH,GOAL_HEIGHT
)

DT=1/120
PRESETS={
 "instep":dict(title="Instep-oriented",yaw=0.0,pitch=-.48,roll=0.,x=.38,y=-.12),
 "inside":dict(title="Medial-oriented",yaw=.48,pitch=0.,roll=.12,x=.38,y=-.27),
 "outside":dict(title="Lateral-oriented",yaw=-.48,pitch=.08,roll=-.12,x=.38,y=.037),
 "medial_side":dict(title="Medial PATCH contact (side sweep)",yaw=0.,pitch=0.,roll=0.,x=.20,y=.07,side_patch="medial"),
 "lateral_side":dict(title="Lateral PATCH contact (side sweep)",yaw=0.,pitch=0.,roll=0.,x=.20,y=-.34,side_patch="lateral"),
}
def smooth(x):
    x=min(1.0,max(0.0,x))
    return x*x*(3-2*x)

def make_scene():
    sim_utils.GroundPlaneCfg(color=(.24,.39,.26)).func(
        "/World/FootballPitch",sim_utils.GroundPlaneCfg(color=(.24,.39,.26)))
    sun=sim_utils.DistantLightCfg(intensity=1200,color=(1.,.96,.86))
    sun.func("/World/SunLight",sun)
    sky=sim_utils.DomeLightCfg(intensity=2400,color=(.91,.93,1.))
    sky.func("/World/SkyLight",sky)
    g1=G1_CFG.copy()
    g1.prim_path="/World/UnitreeG1"
    g1.spawn.articulation_props.fix_root_link=True
    robot=Articulation(g1)
    shoe_patches=attach_training_football_shoe()
    goal=create_goal()
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/Football",
        spawn=sim_utils.SphereCfg(
            radius=.11,rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.97,.82,.27),roughness=.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.38,-.12,.113)),
    ))
    sensor=ContactSensor(ContactSensorCfg(
        prim_path="/World/UnitreeG1/right_ankle_roll_link",
        update_period=0.,history_length=2,
        filter_prim_paths_expr=["/World/Football"],
    ))
    return robot,ball,sensor,goal,shoe_patches


class Controls:
    def __init__(self,state):
        self.win=ui.Window("KineticKick | Soccer Science Lab",width=455,height=415,visible=True)
        self.win.position_x=950
        self.win.position_y=70
        with self.win.frame:
            with ui.VStack(spacing=6):
                ui.Label("G1 + Soccer Goal + Contact Shoe | FIXED BASE",height=22,
                         style={"font_size":17})
                ui.Label("Boot prototype is a rigid collision shell; not yet a validated shoe.")
                with ui.HStack(height=32,spacing=5):
                    ui.Button("Auto select pose",clicked_fn=lambda:self.auto(state))
                    ui.Button("Instep",clicked_fn=lambda:self.set_mode(state,"instep"))
                    ui.Button("Medial",clicked_fn=lambda:self.set_mode(state,"inside"))
                    ui.Button("Lateral",clicked_fn=lambda:self.set_mode(state,"outside"))
                with ui.HStack(height=28,spacing=4):
                    ui.Button("MEDIAL PATCH sweep",clicked_fn=lambda:self.set_mode(state,"medial_side"))
                    ui.Button("LATERAL PATCH sweep",clicked_fn=lambda:self.set_mode(state,"lateral_side"))
                with ui.HStack(height=32,spacing=5):
                    ui.Button("Hip + knee",clicked_fn=lambda:self.set_swing(state,"proximal"))
                    ui.Button("Knee only",clicked_fn=lambda:self.set_swing(state,"knee_only"))
                ui.Label("Move real goal: selection reads its USD world position")
                with ui.HStack(height=29,spacing=5):
                    ui.Button("Goal left",clicked_fn=lambda:self.move(state,-.50))
                    ui.Button("Goal center",clicked_fn=lambda:self.move(state,0.0))
                    ui.Button("Goal right",clicked_fn=lambda:self.move(state,+.35))
                self.goal_label=ui.Label("Goal: scanning USD stage...")
                self.pose_label=ui.Label("Contact pose...")
                self.ball_label=ui.Label("Ball speed: 0.00 m/s")
                self.force_label=ui.Label("Foot-ball contact: waiting")
                self.goal_result=ui.Label("Goal-plane crossing: waiting")
                self.patch_label=ui.Label("Actual contact collider: awaiting PhysX data")
                ui.Label("Scene locator != camera vision. Contact region not certified.")
    @staticmethod
    def auto(state):
        state["auto"]=True
        state["reset"]=True
    @staticmethod
    def set_mode(state,mode):
        state["auto"]=False
        state["pose"]=mode
        state["reset"]=True
    @staticmethod
    def set_swing(state,swing):
        state["swing"]=swing
        state["reset"]=True
    @staticmethod
    def move(state,y):
        state["goal_y"]=y
        state["reset"]=True

def main():
    sim=SimulationContext(SimulationCfg(dt=DT))
    sim.set_camera_view(eye=[3.4,-4.8,2.2],target=[1.42,-.05,.46])
    robot,ball,sensor,goal,shoe=make_scene()
    sim.reset()
    state={"auto":False,"pose":"medial_side","swing":"proximal","goal_y":0.0,"reset":True,
           "measured_cross_y":{"proximal":{},"knee_only":{}}}
    panel=Controls(state)
    joints={n:i for i,n in enumerate(robot.joint_names)}
    limits=robot.data.soft_joint_pos_limits[0]
    q0=robot.data.default_joint_pos.clone()
    qvel=torch.zeros_like(q0)
    def put(q,name,v):
        ix=joints[name]
        q[:,ix]=max(float(limits[ix,0])+.006,min(float(limits[ix,1])-.006,float(v)))
    def pose(ref,style,phase):
        q=q0.clone()
        if ref.get("side_patch"):
            # True SIDE impact: hip roll carries the foot sideways into the ball.
            # The toe, instep, heel and original foot colliders will be disabled.
            begin=-.25 if ref["side_patch"]=="medial" else .25
            end=.25 if ref["side_patch"]=="medial" else -.25
            a=smooth((phase-.35)/.38)
            for j,v in [("right_hip_pitch_joint",-.55),("right_knee_joint",.55),
                        ("right_hip_roll_joint",begin+(end-begin)*a),
                        ("right_hip_yaw_joint",0.),("right_ankle_pitch_joint",0.),
                        ("right_ankle_roll_joint",0.)]:
                put(q,j,v)
            return q
        for j,v in [("right_hip_yaw_joint",ref["yaw"]),
                    ("right_ankle_pitch_joint",ref["pitch"]),
                    ("right_ankle_roll_joint",ref["roll"])]:
            put(q,j,v)
        if style=="proximal":
            a=smooth((phase-.28)/.27)
            # Hip starts the swing; shank extends as the thigh moves.
            hip=.35+(-.87-.35)*a
            knee=.55+(.15-.55)*smooth((phase-.41)/.20)
            put(q,"right_hip_pitch_joint",hip)
            put(q,"right_knee_joint",knee)
            put(q,"left_shoulder_pitch_joint",float(q0[0,joints["left_shoulder_pitch_joint"]]) + .12*a)
            put(q,"right_shoulder_pitch_joint",float(q0[0,joints["right_shoulder_pitch_joint"]]) - .12*a)
            return q
        a=smooth((phase-.33)/.25)
        # Isolated knee extension; thigh hip is held nearly fixed.
        put(q,"right_hip_pitch_joint",-.55)
        put(q,"right_knee_joint",1.10+(.11-1.10)*a)
        return q

    frame=0
    shot_duration=2.1
    previous=None
    crossing=None
    was_hit=False
    peak_force=0
    peak_speed=0
    contact_region="unverified"
    physx_poll_logged=False
    shot=PRESETS["medial_side"]
    last_isolation=None
    while app.is_running():
        if state["reset"] or frame*DT>=shot_duration:
            # Finish prior trial; update style-specific physical shot predictor.
            if frame>0 and crossing is None and frame*DT>=shot_duration:
                state["measured_cross_y"][state["swing"]].setdefault(state["pose"],None)
            state["reset"]=False
            move_goal(goal,state["goal_y"])
            target=locate_goal_in_usd()
            if state["auto"]:
                outcomes=state["measured_cross_y"][state["swing"]]
                unmeasured=[k for k in ("instep","inside","outside") if k not in outcomes]
                if unmeasured:
                    chosen=unmeasured[0]
                    print("AUTO_CALIBRATING_PHYSICS",chosen,"GOAL_FROM_USD",target,flush=True)
                else:
                    valid={k:v for k,v in outcomes.items() if v is not None}
                    chosen=min(valid,key=lambda k:abs(valid[k]-target[1])) if valid else "instep"
                    print("AUTO_SELECT_BY_MEASURED_CROSSING",chosen,"GOAL",target,"MODEL",outcomes,flush=True)
                state["pose"]=chosen
            name=state["pose"]
            shot=PRESETS[name]
            # Isolate the selected collision patch for a conclusive side-contact test.
            # Switch back to all shoe regions for ordinary forward aiming shots.
            target_patch=shot.get("side_patch")
            if target_patch!=last_isolation:
                stage=omni.usd.get_context().get_stage()
                original=stage.GetPrimAtPath("/World/UnitreeG1/right_ankle_roll_link/Cube")
                UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(target_patch is None)
                for part,path in shoe.items():
                    prim=stage.GetPrimAtPath(path)
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(target_patch is None or part==target_patch)
                sim.reset()
                last_isolation=target_patch
                print("SHOE_COLLIDER_ISOLATION",target_patch or "all_enabled",flush=True)
            q=pose(shot,state["swing"],0)
            robot.write_joint_state_to_sim(q,qvel)
            robot.reset()
            bs=ball.data.default_root_state.clone()
            bs[:,:3]=torch.tensor([shot["x"],shot["y"],.113],device=ball.device)
            bs[:,7:]=0.0
            ball.write_root_pose_to_sim(bs[:,:7])
            ball.write_root_velocity_to_sim(bs[:,7:])
            ball.reset()
            sensor.reset()
            frame=0
            crossing=None
            previous=None
            was_hit=False
            peak_force=0.
            peak_speed=0.
            contact_region="unverified"
            print("SHOT_START",name,state["swing"],"goal_xyz",target,
                  "ball_xyz",(shot["x"],shot["y"],.113),flush=True)
        t=frame*DT
        q=pose(shot,state["swing"],t)
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(DT)
        ball.update(DT)
        sensor.update(DT)
        bxyz=[float(x) for x in ball.data.root_pos_w[0]]
        speed=float(torch.linalg.vector_norm(ball.data.root_lin_vel_w[0]).item())
        force=0.
        mat=sensor.data.force_matrix_w
        if mat is not None and mat.numel():
            force=float(torch.linalg.vector_norm(mat).item())
        peak_force=max(force,peak_force)
        peak_speed=max(speed,peak_speed)
        # Ask PhysX which named collision primitive was involved, not merely
        # which whole foot rigid link touched the ball. Names are shoe patches.
        if .35<t<.85:
            try:
                physx=omni.physx.get_physx_simulation_interface()
                headers,contacts=physx.get_contact_report()
                if not physx_poll_logged and .48<t<.52:
                    print("PHYSX_REPORT_POLL",len(headers),flush=True)
                    physx_poll_logged=True
                for header in headers:
                    ca=str(PhysicsSchemaTools.intToSdfPath(header.collider0))
                    cb=str(PhysicsSchemaTools.intToSdfPath(header.collider1))
                    if "Football" not in ca+" "+cb:continue
                    other=cb if "Football" in ca else ca
                    label=other.split("/Patch_")[-1] if "/Patch_" in other else "original_foot_mesh"
                    if label!=contact_region:
                        contact_region=label
                        print("PHYSX_CONTACT_COLLIDER",label,ca,cb,flush=True)
            except Exception as ex:
                if contact_region=="unverified":
                    print("PHYSX_CONTACT_REPORT_ERROR",type(ex).__name__,str(ex)[:180],flush=True)
                    contact_region="report_unavailable"
        if force>2.0 and not was_hit:
            was_hit=True
            print("ACTUAL_PHYSICAL_CONTACT",shot["title"],state["swing"],
                  "t",round(t,3),"force",round(force,2),
                  "speed",round(speed,3),flush=True)
        if crossing is None:
            crossing=classify_ball_crossing(previous,bxyz,locate_goal_in_usd())
            if crossing is not None:
                # Interpolated ball crossing is a measured post-contact outcome,
                # not a preprogrammed prediction from a different shoe geometry.
                fraction=(GOAL_X-previous[0])/(bxyz[0]-previous[0])
                cross_y=previous[1]+fraction*(bxyz[1]-previous[1])
                state["measured_cross_y"][state["swing"]][state["pose"]]=cross_y
                print("GOAL_PLANE_CROSSING","INSIDE" if crossing else "OUTSIDE",
                      "cross_y",round(cross_y,3),
                      "ball",bxyz,"goal",locate_goal_in_usd(),flush=True)
        previous=bxyz
        if frame%12==0:
            xyz=locate_goal_in_usd()
            panel.goal_label.text=f"Goal center from USD: x={xyz[0]:.2f}, y={xyz[1]:.2f}"
            panel.pose_label.text=f"Pose: {shot['title']} | swing: {state['swing']}"
            panel.ball_label.text=f"Ball speed: {speed:.2f} m/s | peak {peak_speed:.2f}"
            panel.force_label.text=f"Filtered foot-ball force peak: {peak_force:.1f} N"
            panel.goal_result.text=("Side contact test: goal aiming disabled" if shot.get("side_patch") else
                "Goal plane: "+("PENDING" if crossing is None else ("INSIDE (center)" if crossing else "OUTSIDE")))
            panel.patch_label.text="Collider: "+contact_region
        frame+=1
        time.sleep(.005)

if __name__=="__main__":
    try:
        main()
    finally:
        app.close()
