"""Inspect bare G1 hip yaw and actual medial-facing surface direction.
No added football shoe, scene/locomotion not trained.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json
import torch
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

try:
    s=SimulationContext(SimulationCfg(dt=1./120))
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/G1"
    cfg.spawn.articulation_props.fix_root_link=True
    r=Articulation(cfg)
    s.reset()
    ids={name:i for i,name in enumerate(r.joint_names)}
    fi=r.body_names.index("right_ankle_roll_link")
    lim=r.data.soft_joint_pos_limits[0]
    zero=torch.zeros_like(r.data.default_joint_pos)
    jnames=["right_hip_yaw_joint","right_hip_roll_joint",
            "right_hip_pitch_joint","right_knee_joint",
            "right_ankle_pitch_joint","right_ankle_roll_joint"]
    print("JOINT_SOFT_LIMITS_RAD",json.dumps({name:[round(float(x),4) for x in lim[ids[name]]] for name in jnames}),flush=True)
    for hipyaw in [-.8,-1.0,-1.3,-1.6]:
        q=r.data.default_joint_pos.clone()
        for name,value in [
            ("right_hip_yaw_joint",hipyaw),
            ("right_hip_pitch_joint",-.35),
            ("right_knee_joint",.20),
            ("right_ankle_pitch_joint",-.20),
            ("right_ankle_roll_joint",-.23),
        ]:
            k=ids[name]
            q[:,k]=torch.clamp(torch.tensor(value,device=q.device),
                               lim[k,0]+.005, lim[k,1]-.005)
        r.write_joint_state_to_sim(q,zero)
        r.write_data_to_sim()
        s.step()
        r.update(s.get_physics_dt())
        pos=r.data.body_pos_w[0,fi]
        quat=r.data.body_quat_w[0,fi]
        m=quat_apply(quat.unsqueeze(0),torch.tensor([[0.,1.,0.]],device=quat.device))[0]
        toe=quat_apply(quat.unsqueeze(0),torch.tensor([[1.,0.,0.]],device=quat.device))[0]
        print("POSE",
            "requested_yaw",hipyaw,
            "clamped_yaw",round(float(q[0,ids["right_hip_yaw_joint"]]),4),
            "medial_normal_world",[round(float(v),3) for v in m],
            "toe_axis_world",[round(float(v),3) for v in toe],
            "foot_pos_world",[round(float(v),3) for v in pos],
            flush=True)
finally:
    app.close()
