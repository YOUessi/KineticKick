"""Tang live Isaac Lab G1 foot-to-ball physical-contact demonstrator.

Three scripted foot-orientation PRESETS; physically simulated 0.43 kg ball,
filtered foot-to-ball contact sensors, ball displacement/speed, and live controls.
This is not a learned control policy and the root is fixed to isolate contact.
Pose names are approximations, NOT verified anatomically precise strike surfaces.
"""
from __future__ import annotations
import math
import time
from isaaclab.app import AppLauncher

simulation_app=AppLauncher(headless=False, enable_cameras=False).app

import torch
import omni.ui as ui
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

DT=1.0/120
N=120
PRESETS=[
    dict(key="instep",label="Instep-oriented / pointed foot",yaw=0.0,pitch=-0.48,roll=0.0,y=-0.12,x=0.38),
    dict(key="inside",label="Medial-side orientation",yaw=0.48,pitch=0.0,roll=0.12,y=-0.270,x=0.38),
    dict(key="outside",label="Lateral-side orientation",yaw=-0.48,pitch=0.08,roll=-0.12,y=0.037,x=0.38),
]
STATUS = {
  "AUTO":"自动依次展示：正脚背 → 内侧方向 → 外侧方向",
  "MANUAL":"手动选择：持续重复选定模式",
}
def ease(t):
    t=max(0.0,min(1.0,t))
    return t*t*(3-2*t)


def create_world():
    ground=sim_utils.GroundPlaneCfg(color=(0.21,0.39,0.27))
    ground.func("/World/FootballPitch",ground)
    dome=sim_utils.DomeLightCfg(intensity=2600.0,color=(.94,.94,1.0))
    dome.func("/World/Lighting",dome)
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/UnitreeG1"
    # Safety: base fixed to separate *contact* from untrained balance.
    cfg.spawn.articulation_props.fix_root_link=True
    robot=Articulation(cfg)
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/Football",
        spawn=sim_utils.SphereCfg(
            radius=.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95,.78,.23),roughness=.55),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.38,-.12,.113)),
    ))
    foot_sensor=ContactSensor(ContactSensorCfg(
       prim_path="/World/UnitreeG1/right_ankle_roll_link",
       update_period=0.0,
       history_length=2,
       filter_prim_paths_expr=["/World/Football"],
    ))
    return robot,ball,foot_sensor


class Panel:
    def __init__(self,settings):
        self.window=ui.Window("KineticKick | G1 FOOT CONTACT",width=410,height=250,visible=True)
        self.window.position_x=1000
        self.window.position_y=70
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("G1 + PHYSICAL BALL | FIXED ROOT",height=24,style={"font_size":18})
                ui.Label("Pose preset != verified anatomical strike surface",height=22)
                with ui.HStack(height=30,spacing=5):
                    ui.Button("Instep",clicked_fn=lambda:self.select(settings,"instep"))
                    ui.Button("Medial pose",clicked_fn=lambda:self.select(settings,"inside"))
                    ui.Button("Lateral pose",clicked_fn=lambda:self.select(settings,"outside"))
                ui.Button("Auto cycle all three presets",height=29,clicked_fn=lambda:self.select(settings,None))
                self.mode_label=ui.Label("正在初始化...",height=22)
                self.speed_label=ui.Label("Ball speed 0 m/s",height=22)
                self.hit_label=ui.Label("Foot-ball contact: WAITING",height=22)
                self.note=ui.Label("Scripted contact only: no trained soccer skill",height=22)
    @staticmethod
    def select(settings,mode):
        settings["manual"]=mode
        settings["switch"]=True
        print("UI_SELECTED", mode or "AUTO",flush=True)


def main():
    sim=SimulationContext(SimulationCfg(dt=DT))
    sim.set_camera_view(eye=[2.65,-3.15,1.51],target=[.45,-.12,.64])
    robot,ball,foot_sensor=create_world()
    sim.reset()
    print("KINETICKICK_FOOT_LAB_READY",len(robot.joint_names),len(robot.body_names),flush=True)
    settings={"manual":None,"switch":False}
    try: panel=Panel(settings)
    except Exception as e:
        print("PANEL_ERROR",repr(e),flush=True)
        panel=None
    joints={v:i for i,v in enumerate(robot.joint_names)}
    lim=robot.data.soft_joint_pos_limits[0]
    base=robot.data.default_joint_pos
    zero=torch.zeros_like(base)
    def write(q,name,v):
        k=joints[name]
        q[:,k]=max(float(lim[k,0])+.005,min(float(lim[k,1])-.005,v))
    def posture(p,hip=.35,knee=.55):
        q=base.clone()
        for k,v in [("right_hip_yaw_joint",p["yaw"]),
                    ("right_hip_pitch_joint",hip),
                    ("right_knee_joint",knee),
                    ("right_ankle_pitch_joint",p["pitch"]),
                    ("right_ankle_roll_joint",p["roll"])]:
            write(q,k,v)
        return q
    mode_index=0
    frames=N
    last_hit=False
    peak_foot_ball_force=0.
    peak_ball_speed=0.
    p=PRESETS[0]
    start_q=posture(p)
    while simulation_app.is_running():
        if frames>=N or settings["switch"]:
            settings["switch"]=False
            if settings["manual"] is None:
                p=PRESETS[mode_index % len(PRESETS)]
                mode_index+=1
            else:
                p=next(x for x in PRESETS if x["key"]==settings["manual"])
            start_q=posture(p)
            robot.write_joint_state_to_sim(start_q,zero)
            robot.reset()
            root=ball.data.default_root_state.clone()
            root[:,:3]=torch.tensor([p["x"],p["y"],.113],device=ball.device)
            root[:,7:]=0.0
            ball.write_root_pose_to_sim(root[:,:7])
            ball.write_root_velocity_to_sim(root[:,7:])
            ball.reset()
            foot_sensor.reset()
            frames=0
            last_hit=False
            peak_foot_ball_force=0.
            peak_ball_speed=0.
            print("MODE_START",p["key"],"ball",p["x"],p["y"],flush=True)
        t=frames*DT
        a=ease((t-.30)/.24)
        target=posture(p,hip=.35+(-.87-.35)*a,knee=.55+(.43-.55)*a)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(DT)
        ball.update(DT)
        foot_sensor.update(DT)
        speed=float(torch.linalg.vector_norm(ball.data.root_lin_vel_w[0]).item())
        peak_ball_speed=max(peak_ball_speed,speed)
        force=0.0
        mat=foot_sensor.data.force_matrix_w
        if mat is not None and mat.numel():
            force=float(torch.linalg.vector_norm(mat).item())
        peak_foot_ball_force=max(peak_foot_ball_force,force)
        if force>2.0 and not last_hit:
            last_hit=True
            print("PHYSICAL_FOOT_BALL_IMPACT",p["key"],"t_s",round(t,3),
                "force_N",round(force,1),"ball_speed_mps",round(speed,2),flush=True)
        if panel and frames%9==0:
            panel.mode_label.text="Preset: "+p["label"]
            panel.speed_label.text=f"Ball speed: {speed:.2f} m/s | peak {peak_ball_speed:.2f}"
            panel.hit_label.text=f"Foot-ball contact: {'YES' if last_hit else 'NO'} | peak force {peak_foot_ball_force:.1f} N"
        frames+=1
        time.sleep(.006)

if __name__=="__main__":
    try: main()
    finally: simulation_app.close()
