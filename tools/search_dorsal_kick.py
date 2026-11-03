"""Calibrate visible soccer KICK MODES and power on the ORIGINAL G1 foot.

No shoe, no added colliders, no RL. Scripted fixed-base physics only.
The contact face is an approximate nearest face from the sphere center in the
foot link coordinate system. The contact sensor filters only Football.
"""
from __future__ import annotations
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json,math,argparse
from pathlib import Path
from isaaclab.utils.math import quat_apply
import torch
from isaaclab.utils.math import quat_apply_inverse
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

DT=1/120.
FOOT_CENTER=[.035917,0.,-.025170]
FOOT_HALF=[.203109/2,.065469/2,.018508/2]

# Timing is an actuator SETPOINT RATE; actual force is emergent physical force.
POWER_DURATIONS={"soft":.43,"medium":.27,"strong":.16}

def smooth(x):
    x=max(0.,min(1.,float(x)))
    return x*x*(3.-2.*x)

def face(local):
    d=[float(local[i])-FOOT_CENTER[i] for i in range(3)]
    gap=[d[i]-max(-FOOT_HALF[i],min(FOOT_HALF[i],d[i])) for i in range(3)]
    a=max(range(3),key=lambda i:abs(gap[i]))
    if a==0:return "toe" if gap[a]>0 else "heel"
    if a==1:return "medial" if gap[a]>0 else "lateral"
    return "top" if gap[a]>0 else "sole"

def create():
    sim_utils.GroundPlaneCfg().func("/World/Ground",sim_utils.GroundPlaneCfg())
    config=G1_CFG.copy()
    config.prim_path="/World/UnitreeG1"
    config.spawn.articulation_props.fix_root_link=True
    robot=Articulation(config)
    ball=RigidObject(RigidObjectCfg(
      prim_path="/World/Football",
      spawn=sim_utils.SphereCfg(
          radius=.11,
          rigid_props=sim_utils.RigidBodyPropertiesCfg(),
          collision_props=sim_utils.CollisionPropertiesCfg(),
          mass_props=sim_utils.MassPropertiesCfg(mass=.43)),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(.38,-.12,.113))
    ))
    sensor=ContactSensor(ContactSensorCfg(
        prim_path="/World/UnitreeG1/right_ankle_roll_link",
        update_period=0.,history_length=2,filter_prim_paths_expr=["/World/Football"]))
    return robot,ball,sensor

def trial(sim,robot,ball,sensor,kind,power,by,bx=.4,yaw=None,pitch_override=None):
    ids={name:i for i,name in enumerate(robot.joint_names)}
    fbody=robot.body_names.index("right_ankle_roll_link")
    lim=robot.data.soft_joint_pos_limits[0]
    base=robot.data.default_joint_pos.clone()
    zero=torch.zeros_like(base)
    duration=POWER_DURATIONS[power]
    def put(q,name,val):
        i=ids[name]
        q[:,i]=max(float(lim[i,0])+.008,min(float(lim[i,1])-.008,val))
    def target(t):
        q=base.clone()
        a=smooth((t-.32)/duration)
        if kind=="instep":
            tyaw=0.
            ankle_pitch=-.48 if pitch_override is None else pitch_override
            ankle_roll=0.
            hip=.20+(-1.05-.20)*a
            knee=1.20+(.80-1.20)*smooth((t-.42)/duration)
        elif kind=="medial":
            tyaw=-1.8 if yaw is None else yaw
            ankle_pitch=-.20
            ankle_roll=-.23
            hip=.20+(-.57-.20)*a
            knee=.36+(.15-.36)*smooth((t-.40)/duration)
        elif kind=="lateral":
            tyaw=+1.45 if yaw is None else yaw
            ankle_pitch=-.20
            ankle_roll=.23
            hip=.20+(-.57-.20)*a
            knee=.36+(.15-.36)*smooth((t-.40)/duration)
        else:raise ValueError(kind)
        for name,v in [
            ("right_hip_yaw_joint",tyaw),("right_hip_pitch_joint",hip),
            ("right_knee_joint",knee),("right_ankle_pitch_joint",ankle_pitch),
            ("right_ankle_roll_joint",ankle_roll)
        ]:put(q,name,v)
        return q
    q=target(0.)
    robot.write_joint_state_to_sim(q,zero);robot.reset()
    root=ball.data.default_root_state.clone()
    root[:,:3]=torch.tensor([bx,by,.113],device=ball.device)
    root[:,7:]=0.
    ball.write_root_pose_to_sim(root[:,:7])
    ball.write_root_velocity_to_sim(root[:,7:])
    ball.reset();sensor.reset()
    impact=None
    maxforce=0.
    maxspeed=0.
    maxtheta=0.
    for j in range(210):
        t=j*DT
        robot.set_joint_position_target(target(t))
        robot.write_data_to_sim();ball.write_data_to_sim()
        sim.step();robot.update(DT);ball.update(DT);sensor.update(DT)
        vel=ball.data.root_lin_vel_w[0]
        v_h=float(torch.linalg.vector_norm(vel[:2]).item())
        cf=sensor.data.force_matrix_w
        force=float(torch.linalg.vector_norm(cf).item()) if cf is not None and cf.numel() else 0.
        maxforce=max(force,maxforce)
        maxspeed=max(v_h,maxspeed)
        if impact is None and force>2.0:
            fpos=robot.data.body_pos_w[0,fbody]
            fquat=robot.data.body_quat_w[0,fbody]
            local=quat_apply_inverse(fquat.unsqueeze(0),(ball.data.root_pos_w[0]-fpos).unsqueeze(0))[0]
            vv=vel.cpu().numpy()
            top_normal_world=quat_apply(fquat.unsqueeze(0),torch.tensor([[0.0,0.0,1.0]],device=fquat.device))[0]
            ll=[float(q) for q in local]
            cubemin=[-.065637,-.032735,-.034424]
            cubemax=[.137472,.032735,-.015916]
            face_inside_x=cubemin[0]+.012<ll[0]<cubemax[0]-.012
            face_inside_y=abs(ll[1])<.018
            top_gap=ll[2]-cubemax[2]
            top_confident=bool(face_inside_x and face_inside_y and top_gap>.025 and abs(float(top_normal_world[0]))>.15)
            impact={"time_s":round(t,4),
                    "face_est":face(local),"top_patch_confident":top_confident,
                    "top_gap_m":round(top_gap,4),
                    "top_normal_world":[round(float(x),3) for x in top_normal_world],
                    "ball_center_in_foot_frame":[round(float(x),3) for x in local],
                    "ball_v_mps":[round(float(x),3) for x in vv],
                    "bearing_deg":round(math.degrees(math.atan2(float(vv[1]),float(vv[0]))),2)}
    end=ball.data.root_pos_w[0].cpu().tolist()
    res={"kind":kind,"power":power,"swing_duration_s":duration,
         "yaw":yaw,"initial_ball":[bx,by,.113],"impact":impact,
         "peak_contact_force_N":round(maxforce,1),
         "peak_horizontal_ball_speed_mps":round(maxspeed,3),
         "end_ball_xyz":[round(float(x),3) for x in end],
         "delta_xy":[round(end[0]-bx,3),round(end[1]-by,3)]}
    print("RESULT",json.dumps(res),flush=True)
    return res

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",choices=["lateral","power","instep"],default="lateral")
    args=parser.parse_args()
    sim=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor=create()
    sim.reset()
    tests=[]
    for pitch in [.45,.55,.64]:
        for bx in [.15,.18,.21,.24,.27,.30]:
            tests.append(("instep","medium",-.12,bx,0.0,pitch))
    arr=[]
    for test in tests:
        arr.append(trial(sim,robot,ball,sensor,*test))
    path=Path("/home/you/projects/kinetickick/reports/dorsal_kick_search.json")
    path.write_text(json.dumps(arr,indent=2))
    print("CALIBRATION_COMPLETE",str(path),len(arr),flush=True)
if __name__=="__main__":
    try:main()
    finally:app.close()