"""Bounded measured-state placement/phase feedback; no simulator state writes.

This is an explicit feedback planner, not a learned human coordination model.
The running actor consumes its placement command and measured state. A captured
nominal trajectory supplies calibration, never a perturbation-case identifier.
"""
import numpy as np


class StrikeFeedback:
    def __init__(self, times, root_xy, ball_xy, config):
        self.times = np.asarray(times, float)
        self.roots = np.asarray(root_xy, float)
        self.ball = np.asarray(ball_xy, float)
        if (self.times.ndim != 1 or len(self.times)<2 or self.roots.shape!=(len(self.times),2)
                or self.ball.shape!=(2,) or not np.all(np.isfinite(self.roots))
                or not np.all(np.isfinite(self.times)) or not np.all(np.diff(self.times)>0)
                or not np.all(np.isfinite(self.ball))):
            raise ValueError('Finite ordered nominal trajectory required')
        self.max_shift=float(config.get('max_placement_m',.12))
        self.max_phase=float(config.get('max_phase_shift_s',.12))
        self.phase_rate=float(config.get('phase_shift_rate',.4))
        if not (0<self.max_shift<=.15 and 0<self.max_phase<=.15 and 0<self.phase_rate<=.5):
            raise ValueError('Feedback bounds exceeded')
        self.last_time=None;self.offset=0.;self.placement=np.zeros(2);self.foot_delta=np.zeros(2)
        self.contacted=False;self.last_phase=0.

    def update(self,time_s,root_xy,root_velocity_xy,ball_xy,left_support_N,contact):
        root=np.asarray(root_xy,float);vel=np.asarray(root_velocity_xy,float);ball=np.asarray(ball_xy,float)
        if (root.shape!=(2,) or vel.shape!=(2,) or ball.shape!=(2,)
                or not np.all(np.isfinite(np.r_[time_s,root,vel,ball,left_support_N]))
                or time_s<0 or (self.last_time is not None and time_s<self.last_time)):
            raise ValueError('Finite causal state observations required')
        dt=.02 if self.last_time is None else time_s-self.last_time
        self.last_time=time_s
        nominal=np.array([np.interp(time_s,self.times,self.roots[:,j]) for j in range(2)])
        self.contacted |= bool(contact)
        if not self.contacted and time_s<2.1:
            self.placement=np.clip(ball-self.ball,-self.max_shift,self.max_shift)
            error=self.placement-(root-nominal)
            self.foot_delta=np.clip(error,-self.max_shift,self.max_shift)
            # Forward is world -Y. A nearer ball advances the swing; a lagging
            # body delays it. Activation ramps to avoid an initial command jump.
            activation=np.clip((time_s-.6)/.6,0.,1.)
            desired=np.clip(error[1]/max(.8,-vel[1]),-self.max_phase,self.max_phase)*activation
            self.offset+=float(np.clip(desired-self.offset,-self.phase_rate*dt,self.phase_rate*dt))
        phase=max(self.last_phase,time_s+self.offset)
        self.last_phase=phase
        if not self.contacted and time_s<2.1:
            # Account for the nominal body's travel during the phase change;
            # otherwise translating the foot would double-count a near/far ball.
            phase_root=np.array([np.interp(phase,self.times,self.roots[:,j]) for j in range(2)])
            self.foot_delta=np.clip(self.placement-(root-phase_root),-self.max_shift,self.max_shift)
        return dict(time_s=float(time_s),phase_s=float(phase),phase_shift_s=float(self.offset),
                    placement_xy=self.placement.tolist(),foot_delta_xy=self.foot_delta.tolist(),nominal_root_xy=nominal.tolist(),
                    measured_root_xy=root.tolist(),measured_velocity_xy=vel.tolist(),measured_ball_xy=ball.tolist(),
                    left_support_N=float(left_support_N),contact_latched=self.contacted)


def retarget_swing_translation(leg, q0, root, delta_xy, max_delta=.25):
    """Translate a swing-foot target with the proximal four joints.

    The two ankle angles are held exactly, so reach correction cannot flatten
    the requested instep posture. Return achieved position error after bounds.
    """
    q0=np.asarray(q0,float);root=np.asarray(root,float);delta=np.asarray(delta_xy,float)
    if q0.shape!=(6,) or root.shape!=(4,4) or delta.shape!=(2,) or not np.all(np.isfinite(np.r_[q0,root.ravel(),delta])):
        raise ValueError('Finite six-joint pose, root transform and planar correction required')
    if not 0<max_delta<=.25 or max(abs(delta))>.150001:raise ValueError('Retargeting bounds exceeded')
    target,_=leg.forward(q0,root);target[:2,3]+=delta
    q=q0.copy()
    low=np.maximum(q0[:4]-max_delta,leg.limits[:4,0]);high=np.minimum(q0[:4]+max_delta,leg.limits[:4,1])
    for _ in range(20):
        pose,j=leg.forward(q,root);error=target[:3,3]-pose[:3,3]
        if np.linalg.norm(error)<.0001:break
        j=j[:3,:4]
        dq=np.linalg.solve(j.T@j+np.eye(4)*.001,j.T@error)
        q[:4]=np.clip(q[:4]+np.clip(dq,-.05,.05),low,high)
    pose,_=leg.forward(q,root)
    return q,float(np.linalg.norm(target[:3,3]-pose[:3,3]))
