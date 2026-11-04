"""Running feedback for original G1: capsule feet, actual ground force, 176-obs actor.

Import after Isaac AppLauncher in simulator processes. CPU policy/geometry checks
can import directly. No function applies external forces or writes physical state.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from rsl_rl.modules import ActorCriticRecurrent


class RunningBalanceActor(ActorCriticRecurrent):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, warmstart_path,
                 init_noise_std=.18, **kwargs):
        if (num_actor_obs, num_actions) != (176,29):
            raise ValueError('Running balance actor requires exactly 176 observations and 29 joints')
        super().__init__(num_actor_obs,num_critic_obs,num_actions,
            actor_hidden_dims=[128,64,32],critic_hidden_dims=[128,64,32],activation='elu',
            rnn_type='lstm',rnn_hidden_dim=128,rnn_num_layers=2,init_noise_std=init_noise_std,**kwargs)
        saved=torch.load(str(warmstart_path),map_location='cpu',weights_only=False)['model_state_dict']
        target=self.state_dict()
        for key in target:
            if key not in saved: raise ValueError('Missing warm-start parameter '+key)
            value=saved[key]
            if key=='memory_a.rnn.weight_ih_l0' and value.shape[1]==160:
                expanded=torch.zeros_like(target[key]);expanded[:,:160]=value;value=expanded
            if value.shape!=target[key].shape: raise ValueError('Warm-start shape mismatch '+key)
            target[key]=value
        self.load_state_dict(target)
        # New features: anchor position3, orientation6, base velocity3, force2, reference contact2.
        mean=saved['paid_obs_mean'];den=saved['paid_obs_denominator']
        if mean.shape[1]==160:
            mean=torch.cat([mean,torch.zeros(1,16)],-1)
            scale=torch.tensor([[1.]*9+[3.]*3+[1.]*4])
            den=torch.cat([den,scale],-1)
        if mean.shape!=(1,176) or den.shape!=(1,176):raise ValueError('Unexpected normalization')
        self.register_buffer('paid_obs_mean',mean)
        self.register_buffer('paid_obs_denominator',den)

    def normalize(self,x):return (x-self.paid_obs_mean)/self.paid_obs_denominator
    def act(self,observations,masks=None,hidden_states=None):
        return super().act(self.normalize(observations),masks,hidden_states)
    def act_inference(self,observations):return super().act_inference(self.normalize(observations))


def capsule_foot_local(urdf,side):
    """Endpoints/radii of the imported capsule/sphere collision primitives."""
    from scipy.spatial.transform import Rotation
    xml=ET.parse(urdf).getroot()
    link=next(x for x in xml.findall('link') if x.attrib['name']==side+'_ankle_roll_link')
    points=[];radii=[]
    for c in link.findall('collision'):
        origin=c.find('origin');a=origin.attrib if origin is not None else {}
        xyz=np.fromstring(a.get('xyz','0 0 0'),sep=' ')
        rot=Rotation.from_euler('xyz',np.fromstring(a.get('rpy','0 0 0'),sep=' '))
        shape=list(c.find('geometry'))[0];radius=float(shape.attrib['radius'])
        half=float(shape.attrib.get('length',0))/2
        points.extend([xyz+rot.apply([0,0,-half]),xyz+rot.apply([0,0,half])]);radii.extend([radius,radius])
    return np.array(points),np.array(radii)


def foot_bottom_z(position,quaternion,points,radii):
    """wxyz quaternion; exact min world Z of capsule segments minus radii."""
    # Rotation row z, no dependency on Isaac for CPU cross-checks.
    w,x,y,z=quaternion.unbind(-1)
    row=torch.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)
    return position[:,2]+(row@points.T-radii).amin(-1)


def setup_feedback(env):
    if hasattr(env,'_running_feedback'):return env._running_feedback
    from pxr import UsdPhysics
    import omni.usd
    robot=env.scene['robot'];motion=env.command_manager.get_term('motion');sensor=env.scene['contact_forces']
    ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse()
            if str(p.GetPath()).startswith(env.cfg.scene.terrain.prim_path+'/') and p.HasAPI(UsdPhysics.CollisionAPI)]
    if not ground:raise RuntimeError('No ground collision primitives')
    root=Path(__file__).resolve().parents[1]
    info={'ground':ground,'sides':[]}
    for side in ('left','right'):
        points,radii=capsule_foot_local(root/'configs/paid_original/main.urdf',side)
        name=side+'_ankle_roll_link'
        info['sides'].append(dict(name=side,body=robot.body_names.index(name),reference=motion.cfg.body_names.index(name),
            points=torch.tensor(points,dtype=torch.float32,device=env.device),radii=torch.tensor(radii,dtype=torch.float32,device=env.device),
            view=sensor._physics_sim_view.create_rigid_contact_view('/World/envs/env_*/Robot/'+name,ground,
                max_contact_data_count=32*env.num_envs)))
    env._running_feedback=info
    return info


def ground_force(env):
    info=setup_feedback(env)
    return torch.stack([s['view'].get_contact_force_matrix(env.physics_dt).reshape(env.num_envs,-1,3)[:,:,2].sum(1).clamp_min(0) for s in info['sides']],-1)


def foot_heights(env,reference=False):
    info=setup_feedback(env);robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');values=[]
    for s in info['sides']:
        if reference:
            p,q=cmd.body_pos_w[:,s['reference']],cmd.body_quat_w[:,s['reference']]
        else:p,q=robot.data.body_pos_w[:,s['body']],robot.data.body_quat_w[:,s['body']]
        values.append(foot_bottom_z(p,q,s['points'],s['radii'])-env.scene.env_origins[:,2])
    return torch.stack(values,-1)


def support_observation(env):
    forces=ground_force(env);return forces/(forces+100.)


def reference_contact(env):
    return torch.sigmoid((.020-foot_heights(env,True))/.006)


def contact_timing_reward(env):
    actual=ground_force(env);actual=actual/(actual+40.)
    return 1.-(actual-reference_contact(env)).abs().mean(-1)


def foot_clearance_reward(env):
    err=(foot_heights(env)-foot_heights(env,True)).square().mean(-1)
    return torch.exp(-err/.06**2)


def root_angular_tracking(env):
    c=env.command_manager.get_term('motion')
    err=(c.robot_anchor_ang_vel_w-c.anchor_ang_vel_w).square().sum(-1)
    return torch.exp(-err/2.5**2)


def nonfoot_contact_termination(env,threshold=20.):
    sensor=env.scene['contact_forces']
    ids=[i for i,n in enumerate(sensor.body_names) if n not in ('left_ankle_roll_link','right_ankle_roll_link')]
    # Four 5 ms samples cover the complete 20 ms policy step, including brief impacts.
    return sensor.data.net_forces_w_history[:,:,ids].norm(dim=-1).amax(dim=(1,2))>=threshold
