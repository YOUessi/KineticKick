"""Physical ball contact diagnostic. Root-fixed G1; scripted swing; no learned control.
Runs in Isaac Lab headless, logs measured ball/root motion and contact sensor.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from isaaclab.app import AppLauncher

app=AppLauncher(headless=True).app
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

DT=1/120
MODES={
    "instep":(0.0, -0.48, 0.0, -0.12),
    "inside":(0.48, 0.0, 0.12, -0.270),
    "outside":(-0.48, 0.08, -0.12, 0.037),
}
def scene():
    ground=sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    config=G1_CFG.copy()
    config.spawn.articulation_props.fix_root_link=True
    config.prim_path="/World/UnitreeG1"
    robot=Articulation(config)
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.7,.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.46,-0.12,0.113)),
    ))
    sensor=ContactSensor(ContactSensorCfg(
        prim_path="/World/UnitreeG1/right_ankle_roll_link",
        update_period=0.0,
        history_length=2,
        filter_prim_paths_expr=["/World/SoccerBall"],
    ))
    return robot,ball,sensor

def ease(x):
    x=max(0.0,min(1.0,x))
    return x*x*(3-2*x)
def run_trial(sim,robot,ball,sensor, mode, bx):
    yaw,pitch,roll,by=MODES[mode]
    joints={n:i for i,n in enumerate(robot.joint_names)}
    foot_index=robot.body_names.index("right_ankle_roll_link")
    base=robot.data.default_joint_pos
    lim=robot.data.soft_joint_pos_limits
    zero=torch.zeros_like(base)
    q_start=base.clone()
    def override(q,name,angle):
        k=joints[name]
        q[:,k]=torch.clamp(torch.tensor(angle,device=q.device),lim[0,k,0]+.005,lim[0,k,1]-.005)
    override(q_start,"right_hip_pitch_joint", .35)
    override(q_start,"right_knee_joint", .55)
    override(q_start,"right_hip_yaw_joint", yaw)
    override(q_start,"right_ankle_pitch_joint", pitch)
    override(q_start,"right_ankle_roll_joint", roll)
    robot.write_joint_state_to_sim(q_start,zero)
    robot.reset()
    init=ball.data.default_root_state.clone()
    init[:, :3]=torch.tensor([bx,by,.113], device=ball.device)
    init[:, 7:]=0.0
    ball.write_root_pose_to_sim(init[:,:7])
    ball.write_root_velocity_to_sim(init[:,7:])
    ball.reset()
    max_speed=0.0
    foot_initial=None
    hit_count=0
    max_contact_force=0.0
    minimum_center_distance=999.0
    ball_final=None
    for n in range(290):
        t=n*DT
        q=q_start.clone()
        alpha=ease((t-0.30)/0.24)
        override(q,"right_hip_pitch_joint", .35 + (-.87-.35)*alpha)
        override(q,"right_knee_joint", .55 + (.43-.55)*alpha)
        # Small whole-body kinetic follow-through; not natural locomotion.
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(DT)
        ball.update(DT)
        sensor.update(DT)
        p=ball.data.root_pos_w[0].detach()
        v=ball.data.root_lin_vel_w[0].detach()
        fp=robot.data.body_pos_w[0,foot_index].detach()
        if foot_initial is None:
            foot_initial=fp.cpu().tolist()
        dist=torch.linalg.vector_norm(p-fp).item()
        minimum_center_distance=min(minimum_center_distance,dist)
        sp=torch.linalg.vector_norm(v).item()
        max_speed=max(max_speed,sp)
        # Filtered foot--ball force. Unfiltered force also includes foot--ground contact.
        if sensor.data.force_matrix_w is not None and sensor.data.force_matrix_w.numel():
            cf=float(torch.linalg.vector_norm(sensor.data.force_matrix_w).item())
            if cf>2.0: hit_count+=1
            max_contact_force=max(max_contact_force,cf)
        if n==289:
            ball_final=p.cpu().tolist()
    return {
      "mode":mode,"ball_start":[bx,by,.113],"ball_end":ball_final,
      "ball_dx":round(ball_final[0]-bx,4),"ball_dy":round(ball_final[1]-by,4),
      "ball_max_speed_m_s":round(max_speed,4),
      "max_filtered_foot_ball_force_n":round(max_contact_force,3),
      "contact_steps":hit_count,
      "min_center_to_ankle_link_m":round(minimum_center_distance,4),
      "foot_link_initial":foot_initial,
      "note":"Root-fixed scripted joint swing; contact force filtered to soccer ball; pose presets do not establish anatomical contact surface."
    }

def main():
    sim=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor=scene()
    sim.reset()
    outcomes=[]
    for mode in ["instep","inside","outside"]:
        for bx in [.31,.38,.44]:
            try:
                result=run_trial(sim,robot,ball,sensor,mode,bx)
                outcomes.append(result)
                print("TRIAL",json.dumps(result),flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                break
    Path("/home/you/projects/kinetickick/reports/contact_probe.json").write_text(json.dumps(outcomes,indent=2))
    print("CONTACT_PROBE_DONE", len(outcomes),flush=True)
if __name__=="__main__":
    try: main()
    finally: app.close()
