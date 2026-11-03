"""Search feasible bare-foot medial-facing contact postures on built-in G1.

No extra shoe geometry. Kinematic sampled poses only, not a trained controller.
Measured in world frame using the real simulated foot-link transform.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json
from pathlib import Path
import numpy as np
import torch
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG

try:
    ctx=SimulationContext(SimulationCfg(dt=1/120))
    c=G1_CFG.copy()
    c.prim_path="/World/G1"
    c.spawn.articulation_props.fix_root_link=True
    bot=Articulation(c)
    ctx.reset()
    ji={v:i for i,v in enumerate(bot.joint_names)}
    ri=bot.body_names.index("right_ankle_roll_link")
    lim=bot.data.soft_joint_pos_limits[0]
    zero=torch.zeros_like(bot.data.default_joint_pos)
    base=bot.data.default_joint_pos.clone()
    rng=np.random.default_rng(25)
    orig=np.array([.035917,0.,-.02517],dtype=float)
    ranked=[]
    for k in range(16):
        yaw=float(rng.uniform(-2.35,-.80))
        hiproll=float(rng.uniform(-.5,.08))
        hippitch=float(rng.uniform(-1.02,-.12))
        knee=float(rng.uniform(.22,1.10))
        ankpitch=float(rng.uniform(-.48,.55))
        ankroll=float(rng.uniform(-.22,.22))
        q=base.clone()
        candidate={
            "right_hip_yaw_joint":yaw,
            "right_hip_roll_joint":hiproll,
            "right_hip_pitch_joint":hippitch,
            "right_knee_joint":knee,
            "right_ankle_pitch_joint":ankpitch,
            "right_ankle_roll_joint":ankroll,
        }
        for n,value in candidate.items():
            i=ji[n]
            q[:,i]=torch.clamp(torch.tensor(value,device=q.device),lim[i,0]+.008,lim[i,1]-.008)
        bot.write_joint_state_to_sim(q,zero)
        bot.write_data_to_sim()
        ctx.step()
        bot.update(ctx.get_physics_dt())
        p=bot.data.body_pos_w[0,ri]
        quat=bot.data.body_quat_w[0,ri]
        normal=quat_apply(quat.unsqueeze(0),torch.tensor([[0.,1.,0.]],device=quat.device))[0]
        surf=quat_apply(quat.unsqueeze(0),torch.tensor([[orig[0], .032735, orig[2]]],device=quat.device))[0]+p
        xyz=surf.cpu().numpy()
        nn=normal.cpu().numpy()
        # require medial surface to point mostly in +x, not upwards.
        score=4*nn[0]-1.0*abs(nn[1])-2*abs(nn[2])-5*abs(xyz[2]-.11)-1.5*abs(xyz[0]-.28)-.8*abs(xyz[1]+.10)
        if xyz[2]<.015 or xyz[2]>.36:score-=3
        ranked.append((float(score),candidate,[round(float(v),3) for v in xyz],[round(float(v),3) for v in nn]))
        print('SCAN_PROGRESS',k,round(float(score),3),'center',[round(float(v),3) for v in xyz],'normal',[round(float(v),3) for v in nn],flush=True)
    ranked.sort(reverse=True,key=lambda x:x[0])
    for i,(score,q,xyz,norm) in enumerate(ranked[:20]):
        print("TOP",i,"score",round(score,3),"surface",xyz,"normal",norm,"joints",{n:round(v,3) for n,v in q.items()},flush=True)
    target=Path("/home/you/projects/kinetickick/reports/medial_forward_pose_search.json")
    target.write_text(json.dumps([{"score":x[0],"joint_targets":x[1],"medial_surface_xyz":x[2],"medial_outward_normal":x[3]} for x in ranked[:60]],indent=2))
    print("POSE_SCAN_DONE",len(ranked),flush=True)
finally:app.close()
