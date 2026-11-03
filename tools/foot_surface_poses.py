"""Inspect foot pose at candidate contact phases for three ankle/hip orientations.
No ball. Direct state resets mean this is geometry feasibility, not torque-limited swing.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import torch,json
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
try:
    ctx=SimulationContext(SimulationCfg(dt=1/120))
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/G1"
    cfg.spawn.articulation_props.fix_root_link=True
    r=Articulation(cfg)
    ctx.reset()
    ji={x:i for i,x in enumerate(r.joint_names)}
    bi=r.body_names.index("right_ankle_roll_link")
    lim=r.data.soft_joint_pos_limits[0]
    base=r.data.default_joint_pos
    for mode,yaw,ankle,roll in [("instep",0.,-.48,0.),("inside",.48,0.,.12),("outside",-.48,.08,-.12)]:
        for hp in [0.35,0.,-.35,-.55,-.75,-.87]:
            q=base.clone()
            for name,v in [("right_hip_pitch_joint",hp),("right_knee_joint",.5),
                           ("right_hip_yaw_joint",yaw),("right_ankle_pitch_joint",ankle),
                           ("right_ankle_roll_joint",roll)]:
                k=ji[name]
                q[:,k]=max(float(lim[k,0])+.005,min(float(lim[k,1])-.005,v))
            r.write_joint_state_to_sim(q,torch.zeros_like(q))
            r.write_data_to_sim()
            ctx.step()
            r.update(ctx.get_physics_dt())
            pos=r.data.body_pos_w[0,bi].cpu().numpy().round(3).tolist()
            print("POSE",mode,hp,pos,flush=True)
finally: app.close()
