"""Find feasible true side-contact ball placements for G1 shoe patches."""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import torch,json
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.utils.math import quat_apply
try:
    ctx=SimulationContext(SimulationCfg(dt=1/120))
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/UnitreeG1"
    cfg.spawn.articulation_props.fix_root_link=True
    r=Articulation(cfg)
    ctx.reset()
    ji={n:i for i,n in enumerate(r.joint_names)}
    fi=r.body_names.index("right_ankle_roll_link")
    lim=r.data.soft_joint_pos_limits[0]
    base=r.data.default_joint_pos
    z=torch.zeros_like(base)
    for hip_pitch in [-.25,-.4,-.55]:
      for knee in [.35,.55,.8]:
        for roll in [-.5,-.25,0.,.25,.5]:
            q=base.clone()
            for name,v in [("right_hip_pitch_joint",hip_pitch),
                           ("right_knee_joint",knee),
                           ("right_hip_roll_joint",roll),
                           ("right_hip_yaw_joint",0.),
                           ("right_ankle_pitch_joint",0.),
                           ("right_ankle_roll_joint",0.)]:
                i=ji[name]
                q[:,i]=max(float(lim[i,0])+.005,min(float(lim[i,1])-.005,v))
            r.write_joint_state_to_sim(q,z)
            r.write_data_to_sim()
            ctx.step()
            r.update(ctx.get_physics_dt())
            fp=r.data.body_pos_w[0,fi]
            fq=r.data.body_quat_w[0,fi]
            # shoe medial local side patch is +y at x=0.042
            pts=[]
            for sign in [-1,1]:
                local=torch.tensor([.04,sign*.057,-.026],device=fp.device)
                p=fp+quat_apply(fq.unsqueeze(0),local.unsqueeze(0))[0]
                pts.append([round(float(x),3) for x in p])
            print("SWEEP",hip_pitch,knee,roll,"foot",[round(float(x),3) for x in fp],
                  "patches",pts,flush=True)
finally:app.close()
