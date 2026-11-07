"""One shared lateral path feedback over frozen v32 strike/recovery control.

The fixed foot reference point is a kinematic marker, not a chosen contact
capsule or an impulse predictor. All bounds constrain added PD references;
physical actuator limits/colliders and ball forces are unchanged.
"""
import math

import torch

from placement_strike import BatchedLeg
from post_shot_recovery import PostShotRecoveryTask, _quintic, _rotation


class PathResidualGate:
    def __init__(self, n, device):
        self.actual = torch.zeros(n, 29, device=device)
        self.exit_delta = self.actual.clone()
        self.exit_time = torch.full((n,), -1., device=device)

    def latch(self, now, touched):
        entering = touched & (self.exit_time < 0.)
        self.exit_time = torch.where(entering, now, self.exit_time)
        self.exit_delta = torch.where(entering[:, None], self.actual, self.exit_delta)

    def update(self, now, requested, support, touched):
        n = len(self.actual)
        if (now.shape != (n,) or requested.shape != (n,29) or support.shape != (n,2)
                or touched.shape != (n,) or touched.dtype != torch.bool):
            raise ValueError('Finite batched time, 29 joint deltas, two support loads and touch mask required')
        if not all(bool(torch.isfinite(value).all()) for value in (now,requested,support)) or bool((support<0).any()):
            raise ValueError('Nonfinite path control or negative ground load')
        self.latch(now, touched | (now >= 2.1))
        support_weight=((support[:,0]-40.)/40.).clamp(0,1)*((40.-support[:,1])/40.).clamp(0,1)
        onset=_quintic((now-.6)/.12)
        magnitude=requested.abs().amax(-1,keepdim=True)
        desired=requested*torch.minimum(torch.ones_like(magnitude),.06/magnitude.clamp_min(1e-12))
        desired*= (support_weight*onset)[:,None]
        exiting=self.exit_time>=0
        fade=1.-_quintic((now-self.exit_time)/.20)
        desired=torch.where(exiting[:,None],self.exit_delta*fade[:,None],desired)
        change=desired-self.actual
        peak=change.abs().amax(-1,keepdim=True)
        self.actual=self.actual+change*torch.minimum(torch.ones_like(peak),.015/peak.clamp_min(1e-12))
        finished=exiting & (now+1e-7 >= self.exit_time+.20)
        self.actual=torch.where(finished[:,None],torch.zeros_like(self.actual),self.actual)
        return self.actual,dict(support_weight=support_weight,onset=onset,fade=fade,exiting=exiting,finished=finished)

    def reset(self, ids):
        self.actual[ids]=0.; self.exit_delta[ids]=0.; self.exit_time[ids]=-1.


def path_ik_delta(leg, q, rotation, root, delta_xy, local_point):
    """Damped increment at the measured leg pose, with no PD-target projection.

    One linearized SE(3) solve, uniformly scaled to 0.06 rad. Its position and
    orientation errors are reported after nonlinear FK; they are not promises
    about the future measured foot pose or exact reproduction of a PD target. A zero request stays exactly zero.
    """
    n=len(q)
    if q.shape!=(n,6) or delta_xy.shape!=(n,2):raise ValueError('Expected N x 6 leg and N x 2 translation')
    if not all(bool(torch.isfinite(v).all()) for v in (q,rotation,root,delta_xy,local_point)):
        raise ValueError('Nonfinite path IK input')
    result=torch.zeros_like(q);position_error=torch.zeros(n,device=q.device);angle=position_error.clone()
    ids=(delta_xy.abs().amax(-1)>0).nonzero(as_tuple=False).flatten()
    if not len(ids):return result,dict(position_error_m=position_error,orientation_error_rad=angle)
    p,r,j=leg.forward(q[ids],rotation[ids],root[ids])
    lever=(r@local_point).squeeze(-1)
    point=p+lever
    j=j.clone()
    j[:,:3]+=torch.linalg.cross(j[:,3:].transpose(1,2),lever[:,None].expand(-1,6,-1),dim=-1).transpose(1,2)
    j[:,3:]*=.15
    desired=torch.zeros(len(ids),6,device=q.device,dtype=q.dtype);desired[:,:2]=delta_xy[ids]
    normal=j.transpose(1,2)@j+torch.eye(6,device=q.device,dtype=q.dtype)[None]*1e-5
    dq=torch.linalg.solve(normal,(j.transpose(1,2)@desired[:,:,None])).squeeze(-1)
    dq*=torch.minimum(torch.ones(len(ids),1,device=q.device),.06/dq.abs().amax(-1,keepdim=True).clamp_min(1e-12))
    result[ids]=dq
    p1,r1,_=leg.forward(q[ids]+dq,rotation[ids],root[ids])
    point1=p1+(r1@local_point).squeeze(-1)
    position_error[ids]=(point+desired[:,:3]-point1).norm(dim=-1)
    cosine=((torch.diagonal(r@r1.transpose(1,2),dim1=1,dim2=2).sum(-1)-1.)*.5).clamp(-1,1)
    angle[ids]=cosine.acos()
    return result,dict(position_error_m=position_error,orientation_error_rad=angle)


class EarlyPathFeedbackTask(PostShotRecoveryTask):
    def __init__(self, env, config):
        super().__init__(env,config)
        cfg=config['early_path_feedback']
        expected={'position_gain':.25,'velocity_gain_s':.04,'maximum_translation_m':.03}
        if any(not math.isfinite(cfg[k]) or abs(cfg[k]-v)>1e-10 for k,v in expected.items()):
            raise ValueError('Fixed v34 first-candidate gains required; no per-case tuning')
        self.path_cfg=cfg
        self.path_gate=PathResidualGate(self.n,self.device)
        self.path_touch_time=torch.full((self.n,),-1.,device=self.device)
        self.path_point=torch.tensor([.06,0.,-.025],device=self.device)
        sensor=env.scene['contact_forces']
        self._path_ball_view=sensor._physics_sim_view.create_rigid_contact_view(
            '/World/envs/env_*/SoccerBall',
            ['/World/envs/env_*/Robot/'+name for name in self.robot.body_names],
            max_contact_data_count=256*self.n)

    def after_physics(self, dt):
        super().after_physics(dt)
        if dt<=0:return
        matrix=self._path_ball_view.get_contact_force_matrix(dt)
        if not bool(torch.isfinite(matrix).all()):raise ValueError('Nonfinite ball-to-robot contact evidence')
        touched=matrix.reshape(self.n,-1,3).norm(dim=-1).amax(-1)>2.
        new=touched & (self.path_touch_time<0)
        self.path_touch_time=torch.where(new,self._recovery_time_s,self.path_touch_time)
        self.path_gate.latch(self._recovery_time_s,touched)

    def commands(self, actions=None):
        if actions is not None:raise ValueError('v34 is fixed feedback, without learned action input')
        from early_path_feedback import lateral_correction
        from running_balance import ground_force
        r=self.robot.data;now=self.elapsed.clone()*.02
        index=self.feet[1];rotation=_rotation(r.body_quat_w[:,index])
        lever=(rotation@self.path_point).squeeze(-1)
        point=r.body_pos_w[:,index]+lever
        velocity=r.body_link_lin_vel_w[:,index]+torch.linalg.cross(r.body_ang_vel_w[:,index],lever,dim=-1)
        # This measured snapshot precedes exactly one original expert invocation.
        support=ground_force(self.env)
        base=super().commands(None)
        term=self.env.action_manager.get_term('joint_pos')
        scale=torch.as_tensor(term._scale,device=base.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):raise ValueError('Finite nonzero original action scale required')
        translation=torch.zeros(self.n,2,device=self.device)
        diagnostics={}
        eligible=(self.path_gate.exit_time<0)&(now>=.6)&(now<2.1)&bool(self.path_cfg['enabled'])
        ids=eligible.nonzero(as_tuple=False).flatten()
        if len(ids):
            delta,diag=lateral_correction(point[ids],velocity[ids],self.ball.data.root_pos_w[ids],
                self.ball.data.root_lin_vel_w[ids],self.goal[ids],position_gain=self.path_cfg['position_gain'],
                velocity_gain_s=self.path_cfg['velocity_gain_s'],maximum_m=self.path_cfg['maximum_translation_m'])
            translation[ids]=delta
            for key,value in diag.items():
                diagnostics[key]=torch.zeros((self.n,*value.shape[1:]),device=self.device)
                diagnostics[key][ids]=value
        joints=self.expert.ids
        q=r.joint_pos[:,joints]
        leg_delta,ik=path_ik_delta(self.leg,q,_rotation(r.root_quat_w),r.root_pos_w,translation,self.path_point)
        requested=torch.zeros_like(base);requested[:,joints]=leg_delta*self.gate[:,None]
        previous=self.path_gate.actual.clone()
        delta,gate=self.path_gate.update(now,requested,support,self.path_touch_time>=0)
        changed=delta!=0.
        out=torch.where(changed,base+delta/scale,base) if bool(changed.any()) else base
        self.path_gate.actual=(out-base)*scale
        if self.n==1 and self.debug:
            evidence={key:value[0].tolist() for key,value in diagnostics.items()}
            evidence.update({key:value[0].tolist() for key,value in ik.items()})
            evidence.update({key:value[0].tolist() for key,value in gate.items()})
            evidence.update(enabled=bool(self.path_cfg['enabled']),context_gate=float(self.gate[0]),reference_point_local_m=self.path_point.tolist(),
                measured_reference_position_w_m=point[0].tolist(),measured_reference_velocity_w_mps=velocity[0].tolist(),
                requested_translation_xy_m=translation[0].tolist(),requested_delta_rad=requested[0].tolist(),
                actual_delta_rad=self.path_gate.actual[0].tolist(),previous_delta_rad=previous[0].tolist(),
                baseline_action=base[0].tolist(),final_action=out[0].tolist(),
                first_touch_time_relative_s=float(self.path_touch_time[0]),exit_time_s=float(self.path_gate.exit_time[0]),
                post_recovery_overlap=bool((self.actual_delta[0]!=0).any() and (self.path_gate.actual[0]!=0).any()),
                ball_state_write=False)
            self.debug[-1]['early_path_feedback']=evidence
        return out

    def reset_fields(self, ids):
        super().reset_fields(ids)
        self.path_gate.reset(ids);self.path_touch_time[ids]=-1.


def get_task(env):
    if not hasattr(env,'_early_path_feedback_task'):
        env._early_path_feedback_task=EarlyPathFeedbackTask(env,env.cfg.joint_strike)
    return env._early_path_feedback_task
