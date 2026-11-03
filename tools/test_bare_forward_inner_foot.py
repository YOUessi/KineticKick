"""G1 original foot only: forward inner-foot pass impact and orientation test.

No shoe, no joint rewiring; root fixed, scripted stroke. Reports first actual
foot/ball contact, approximate closest face of original cube, ball velocity.
"""
from __future__ import annotations
import json
from pathlib import Path
from foot_ball_contact_probe import scene, app, DT
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sim import SimulationCfg, SimulationContext

def smooth(x):
    x=max(0.,min(1.,x))
    return x*x*(3-2*x)

def face_from_ball_local(local):
    # G1 original right foot collision: rectangular prism, NOT an anatomical arch.
    center=torch.tensor([.035917,0.,-.025170],device=local.device)
    half=torch.tensor([.203109/2,.065469/2,.018508/2],device=local.device)
    d=local-center
    clamped=torch.clamp(d,min=-half,max=half)
    delta=(d-clamped).cpu().numpy()
    kinds=["toe (+x)","medial (+y)","top (+z)"]
    ax=int(abs(delta).argmax())
    if ax==0:return "toe" if delta[0]>0 else "heel"
    if ax==1:return "medial_side" if delta[1]>0 else "lateral_side"
    return "upper_surface" if delta[2]>0 else "sole"

def test(sim,robot,ball,sensor,yaw,bx,by,style="hip_knee"):
    ids={n:i for i,n in enumerate(robot.joint_names)}
    foot=robot.body_names.index("right_ankle_roll_link")
    base=robot.data.default_joint_pos.clone()
    limits=robot.data.soft_joint_pos_limits[0]
    zero=torch.zeros_like(base)
    def cfg(t):
        a=smooth((t-.32)/.24)
        q=base.clone()
        if style=="hip_knee":
            hip=.20+(-.57-.20)*a
            knee=.36+(.15-.36)*smooth((t-.40)/.18)
        else:
            hip=-.48
            knee=1.06+(.13-1.06)*smooth((t-.33)/.25)
        for name,val in [
          ("right_hip_yaw_joint",yaw),
          ("right_hip_pitch_joint",hip),
          ("right_knee_joint",knee),
          ("right_ankle_pitch_joint",-.20),
          ("right_ankle_roll_joint",-.23),
        ]:
            i=ids[name]
            q[:,i]=max(float(limits[i,0])+.005,min(float(limits[i,1])-.005,val))
        return q
    robot.write_joint_state_to_sim(cfg(0.),zero)
    robot.reset()
    start=ball.data.default_root_state.clone()
    start[:,:3]=torch.tensor([bx,by,.113],device=ball.device)
    start[:,7:]=0
    ball.write_root_pose_to_sim(start[:,:7])
    ball.write_root_velocity_to_sim(start[:,7:])
    ball.reset();sensor.reset()
    hit=None
    maxv=0.
    for i in range(190):
        t=i*DT
        robot.set_joint_position_target(cfg(t))
        robot.write_data_to_sim();ball.write_data_to_sim()
        sim.step();robot.update(DT);ball.update(DT);sensor.update(DT)
        speed=float(torch.linalg.vector_norm(ball.data.root_lin_vel_w[0]).item())
        maxv=max(maxv,speed)
        f=sensor.data.force_matrix_w
        force=float(torch.linalg.vector_norm(f).item()) if f is not None and f.numel()>0 else 0.
        if force>2. and hit is None:
            fp=robot.data.body_pos_w[0,foot]
            fq=robot.data.body_quat_w[0,foot]
            bp=ball.data.root_pos_w[0]
            local=quat_apply_inverse(fq[None,:],(bp-fp)[None,:])[0]
            hit={"contact_t_s":round(t,3),"force_N":round(force,1),
                 "ball_local_in_foot":[round(float(v),3) for v in local],
                 "approx_nearest_cube_face":face_from_ball_local(local),
                 "ball_vxyz":[round(float(v),3) for v in ball.data.root_lin_vel_w[0]]}
    end=ball.data.root_pos_w[0].cpu().tolist()
    result={"hip_yaw":yaw,"ball_start":[bx,by,.113],"style":style,"impact":hit,
            "dx":round(end[0]-bx,3),"dy":round(end[1]-by,3),
            "max_speed_mps":round(maxv,3)}
    print("TEST",json.dumps(result),flush=True)
    return result

try:
    sim=SimulationContext(SimulationCfg(dt=DT))
    robot,ball,sensor=scene()
    sim.reset()
    results=[]
    for yaw in [-.8,-1.0,-1.3]:
        for bx,by in [(.34,.17),(.40,.20),(.42,.25)]:
            results.append(test(sim,robot,ball,sensor,yaw,bx,by))
    p=Path("/home/you/projects/kinetickick/reports/bare_forward_inner_foot_tests.json")
    p.write_text(json.dumps(results,indent=2))
    print("BARE_INNER_FOOT_DONE",len(results),flush=True)
finally:
    app.close()
