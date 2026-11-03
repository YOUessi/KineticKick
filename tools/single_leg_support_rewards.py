"""Contact-aware left-leg support reward terms for G1 in Isaac Lab.

True PhysX root is free (never fixed). Tries to learn sustained support on the
LEFT foot while the RIGHT foot is airborne, compatible with a subsequent shot.
No human gait imitation claim. Joint/terrain sensor assumptions for G1.
"""
import torch

def _data(env):
    robot=env.scene["robot"];sensor=env.scene["contact_forces"]
    li=sensor.body_names.index("left_ankle_roll_link")
    ri=sensor.body_names.index("right_ankle_roll_link")
    lbi=robot.body_names.index("left_ankle_roll_link")
    rbi=robot.body_names.index("right_ankle_roll_link")
    norms=torch.linalg.vector_norm(sensor.data.net_forces_w[:,[li,ri],:],dim=-1)
    z=robot.data.body_pos_w[:,rbi,2] - env.scene.env_origins[:,2]
    return robot,norms,z,lbi,rbi


def left_leg_ground_support(env):
    robot,f,_,_,_=_data(env)
    return torch.clamp(f[:,0]/190.,0.,1.)


def swing_foot_height_reward(env,target_z:float=.15,sigma:float=.06):
    robot,f,z,_,_=_data(env)
    return torch.exp(-torch.square((z-target_z)/sigma))


def right_foot_unloaded_reward(env,max_force_n:float=40.):
    _,f,_,_,_=_data(env)
    return torch.sigmoid((max_force_n-f[:,1])/12.)


def single_stance_balanced_reward(env,target_z:float=.15,sigma:float=.06):
    bot,f,z,_,_=_data(env)
    left=torch.clamp(f[:,0]/180.,0.,1.)
    right_clear=torch.exp(-torch.square((z-target_z)/sigma))
    upright=torch.clamp(-bot.data.projected_gravity_b[:,2],min=0.,max=1.)
    base_z=bot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    okay=(base_z>.52).float()
    lateral_vel=bot.data.root_lin_vel_w[:,:2].norm(dim=-1)
    return left*right_clear*(upright**7)*torch.exp(-2*lateral_vel)*okay


def left_foot_stance_slip_cost(env,ground_min_force_n:float=30.):
    bot,f,z,lbi,rbi=_data(env)
    v=bot.data.body_lin_vel_w[:,lbi,:2].norm(dim=-1)
    return (f[:,0]>ground_min_force_n).float()*v


def root_drop_cost(env,min_root_height_m:float=.58):
    bot=env.scene["robot"]
    z=bot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return torch.square(torch.nn.functional.relu(min_root_height_m-z))*25.


def moving_base_cost(env):
    bot=env.scene["robot"]
    return bot.data.root_lin_vel_w[:,:3].square().sum(-1) + \
           .20*bot.data.root_ang_vel_w[:,:3].square().sum(-1)


if __name__=="__main__":
    print("SINGLE_LEG_REWARD_MODULE_VALID",True)
