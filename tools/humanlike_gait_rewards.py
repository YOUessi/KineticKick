"""Experimental human-inspired gait reward terms (for 37-DoF G1 pilot).

Neither function is a substitute for measured human motion/ground-contact events.
Both return a nonnegative cost per environment for negative RewardTermCfg weights.
Stance masking is REQUIRED: rigidly straightening the swing knee is incorrect.
"""
from __future__ import annotations
import torch


def _contacts_and_knees(env, contact_threshold_n):
    bot=env.scene["robot"]
    sensor=env.scene["contact_forces"]
    s_idx=[sensor.body_names.index(f"{side}_ankle_roll_link") for side in ("left","right")]
    j_idx=[bot.joint_names.index(f"{side}_knee_joint") for side in ("left","right")]
    f=sensor.data.net_forces_w[:,s_idx,:]
    force=torch.linalg.vector_norm(f,dim=-1)
    in_contact=force>contact_threshold_n
    q=bot.data.joint_pos[:,j_idx]
    return q,force,in_contact


def _moving(env, min_command_speed):
    cmd=env.command_manager.get_command("base_velocity")
    return (torch.linalg.vector_norm(cmd[:,:2],dim=-1)>min_command_speed).to(cmd.dtype)


def stance_knee_flexion_cost(env, target_max_rad:float=.55,
                            contact_threshold_n:float=15.,
                            min_command_speed:float=.12):
    q,_,contact=_contacts_and_knees(env,contact_threshold_n)
    moving=_moving(env,min_command_speed)
    excess=torch.nn.functional.relu(torch.abs(q)-target_max_rad)
    # Only penalize a knee when that FOOT supports ground reaction force.
    weighted=(excess.square()*contact.float()).sum(dim=-1)
    denom=contact.float().sum(dim=-1).clamp_min(1.0)
    return weighted/denom*moving


def excessive_airborne_cost(env, contact_threshold_n:float=15.,
                            min_command_speed:float=.12):
    _,_,contact=_contacts_and_knees(env,contact_threshold_n)
    neither=(~contact.any(dim=1)).float()
    return neither*_moving(env,min_command_speed)


def synthetic_reward_self_test():
    from types import SimpleNamespace
    class C:
        body_names=["left_ankle_roll_link","right_ankle_roll_link"]
        data=SimpleNamespace(net_forces_w=torch.tensor([
            [[0.,0.,100.],[0.,0.,0.]],  # only left support
            [[0.,0.,0.],[0.,0.,100.]],  # only right support
            [[0.,0.,0.],[0.,0.,0.]],    # both swing / airborne
            [[0.,0.,100.],[0.,0.,100.]],# double support
        ]))
    class R:
        joint_names=["left_knee_joint","right_knee_joint"]
        data=SimpleNamespace(joint_pos=torch.tensor([
            [1.0,1.4],
            [0.2,0.9],
            [1.0,1.0],
            [0.3,0.2],
        ]))
    class CM:
        def get_command(self,_):
            return torch.tensor([[.5,0.,0.],[.5,0.,0.],[.5,0.,0.],[0.,0.,0.]])
    env=SimpleNamespace(scene={"robot":R(),"contact_forces":C()},command_manager=CM())
    loss=stance_knee_flexion_cost(env,.50,15.,.12)
    air=excessive_airborne_cost(env,15.,.12)
    assert torch.allclose(loss,torch.tensor([.25,.16,0.,0.])),loss
    assert torch.allclose(air,torch.tensor([0.,0.,1.,0.])),air
    return True

if __name__=="__main__":
    print("HUMAN_GAIT_REWARD_SELF_TEST",synthetic_reward_self_test())
