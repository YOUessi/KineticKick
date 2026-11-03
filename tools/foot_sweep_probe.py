"""Probe built-in G1 foot forward workspace (fixed root) in Isaac Lab.

Kinematics ONLY: no soccer skill/training, no real foot/ball collisions tested.
Produces joint range and actual PhysX-simulated body-link coordinates.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

def main():
    sim=SimulationContext(SimulationCfg(dt=1/120))
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/Robot"
    cfg.spawn.articulation_props.fix_root_link=True
    robot=Articulation(cfg)
    sim.reset()
    idx={n:i for i,n in enumerate(robot.joint_names)}
    bidx={n:i for i,n in enumerate(robot.body_names)}
    foot_index=bidx["right_ankle_roll_link"]
    ankle=[n for n in robot.joint_names if "ankle" in n]
    lim=robot.data.soft_joint_pos_limits[0,:,:].detach().cpu().tolist()
    print("PITCH_ROLL_LIMITS",json.dumps({j:[round(x,3) for x in lim[idx[j]]] for j in ankle}),flush=True)
    base=robot.data.default_joint_pos.clone()
    zero=torch.zeros_like(base)
    poses={}
    for hp in [0.65,0.45,0.20,0.0,-0.25,-0.45,-0.65,-0.85]:
      for kd in [0.0,0.2,0.45]:
        q=base.clone()
        q[:,idx["right_hip_pitch_joint"]]=hp
        q[:,idx["right_knee_joint"]]=base[:,idx["right_knee_joint"]]+kd
        # dynamically clamp to joint limits
        low=robot.data.soft_joint_pos_limits[0,:,0]+0.01
        high=robot.data.soft_joint_pos_limits[0,:,1]-0.01
        q=torch.max(torch.min(q,high),low)
        robot.write_joint_state_to_sim(q,zero)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        pos=robot.data.body_pos_w[0,foot_index].cpu().tolist()
        poses[f"hip{hp:.2f}_knee{kd:.2f}"]=[round(x,3) for x in pos]
    print("SWEEP_POSES",json.dumps(poses),flush=True)
    print("FOOT_DEFAULT", robot.data.default_joint_pos[0,idx["right_hip_pitch_joint"]].item(),
         robot.data.default_joint_pos[0,idx["right_knee_joint"]].item(), flush=True)
    Path("/home/you/projects/kinetickick/reports/foot_sweep_probe.json").write_text(json.dumps({"ankle_limits":{j:lim[idx[j]] for j in ankle},"poses":poses},indent=2))
if __name__ == "__main__":
    try: main()
    finally: app.close()
