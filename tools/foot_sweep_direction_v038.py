"""v38 bounded *measured contact-point sweep-direction* feedback over v32.

A fixed research control hypothesis, not a learned soccer or human-coordination
policy: use current foot/ball velocity in world coordinates and a locally
linearized 6DoF leg Jacobian to softly steer preimpact material-point motion.
Explicitly do NOT track a foot target position or assume foot-face yaw predicts
ball flight. Real qdot/PD response, normal/tangential impulse and recovery must
be independently audited after PhysX, not inferred from Jacobian setpoints.
"""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import torch
from post_shot_recovery import PostShotRecoveryTask,_rotation,_quintic
from running_balance import ground_force

ROOT=Path(__file__).resolve().parents[1]

def measured_point_velocity(foot_origin,foot_quat,link_vel,ang_vel,point_local):
    """Rigid material-point velocity, not the ankle-origin velocity."""
    n=len(foot_origin)
    if foot_origin.shape!=(n,3) or foot_quat.shape!=(n,4) or link_vel.shape!=(n,3) or ang_vel.shape!=(n,3):
        raise ValueError("Finite 3D measured foot-link state required")
    local=torch.as_tensor(point_local,device=foot_origin.device,dtype=foot_origin.dtype)
    if local.shape!=(3,):raise ValueError("Three-element foot-local point required")
    R=_rotation(foot_quat)
    lever=(R@local[None,:,None].expand(n,-1,-1)).squeeze(-1)
    point=foot_origin+lever
    velocity=link_vel+torch.linalg.cross(ang_vel,lever,dim=-1)
    if not all(bool(torch.isfinite(a).all()) for a in (point,velocity)):
        raise ValueError("Nonfinite foot point or measured velocity")
    return point,velocity,lever

def lateral_sweep_target(relative_foot_velocity_xy,ball_xy,goal_xy,lookahead_s=.012,max_translation_m=.022):
    """Signed velocity error perpendicular to current ball-to-goal direction.

    The independent measured precontact ball velocity is already subtracted
    by the caller. This is a *hypothesis*, not a learned inverse impulse model.
    """
    n=len(ball_xy)
    if (relative_foot_velocity_xy.shape!=(n,2) or ball_xy.shape!=(n,2) or goal_xy.shape!=(n,2)
        or not all(bool(torch.isfinite(a).all()) for a in (relative_foot_velocity_xy,ball_xy,goal_xy))):
        raise ValueError("Measured lateral sweep inputs must be N x 2 finite")
    ray=goal_xy-ball_xy
    length=ray.norm(dim=-1)
    if bool((length<.25).any()):raise ValueError("Goal direction must be well defined")
    direction=ray/length[:,None]
    lateral=torch.stack((-direction[:,1],direction[:,0]),dim=-1)
    unwanted=(relative_foot_velocity_xy*lateral).sum(-1)
    forward=(relative_foot_velocity_xy*direction).sum(-1)
    requested=(-lookahead_s*unwanted).clamp(-max_translation_m,max_translation_m)
    delta=lateral*requested[:,None]
    return delta,dict(goal_heading_world_xy=direction,lateral_axis_world_xy=lateral,
        lateral_velocity_mps=unwanted,forward_velocity_mps=forward,
        requested_lateral_displacement_m=requested)

def damped_point_ik_delta(leg,q,root_rot,root_pos,local_point,requested_xy,
                          max_joint_rad=.022,regularization=.005,orientation_weight=.20):
    """Small 6DoF joint adjustment for *one-step lateral target displacement*.

    Uses measured q, true source URDF Jacobian and a world-frame target. Unlike
    v34, the feedback error is relative MATERIAL-POINT SWEEP velocity, not a
    chosen foot position error. Orientation penalty mitigates yaw/normal drift.
    Returns a reference delta, not a guaranteed realized foot velocity.
    """
    n=len(q)
    if q.shape!=(n,6) or root_rot.shape!=(n,3,3) or root_pos.shape!=(n,3) or requested_xy.shape!=(n,2):
        raise ValueError("Leg state N x 6, root N x 3x3 and requested N x 2 required")
    if not all(bool(torch.isfinite(z).all()) for z in (q,root_rot,root_pos,requested_xy)):
        raise ValueError("Finite IK input required")
    point,r,j=leg.forward(q,root_rot,root_pos)
    local=torch.as_tensor(local_point,device=q.device,dtype=q.dtype)
    lever=(r@local[None,:,None].expand(n,-1,-1)).squeeze(-1)
    J=j.clone()
    J[:,:3,:]+=torch.linalg.cross(J[:,3:,:].transpose(1,2),
                 lever[:,None,:].expand(-1,6,-1),dim=-1).transpose(1,2)
    Jxy=J[:,:2,:]
    Jomega=J[:,3:,:]
    reg=torch.diag(torch.as_tensor([1.,1.,1.,3.,3.,3.],device=q.device,dtype=q.dtype))
    H=Jxy.transpose(1,2)@Jxy+orientation_weight**2*(Jomega.transpose(1,2)@Jomega)+regularization*reg[None]
    rhs=(Jxy.transpose(1,2)@requested_xy[:,:,None])
    dq=torch.linalg.solve(H,rhs).squeeze(-1)
    peak=dq.abs().amax(-1,keepdim=True)
    dq=dq*torch.minimum(torch.ones_like(peak),max_joint_rad/peak.clamp_min(1e-12))
    # Exact zero request must yield exactly zero joint change.
    dq=torch.where((requested_xy.abs().amax(-1)>0)[:,None],dq,torch.zeros_like(dq))
    if not bool(torch.isfinite(dq).all()):raise ValueError("Nonfinite IK solution")
    achieved=(Jxy@dq[:,:,None]).squeeze(-1)
    achieved_angle=(Jomega@dq[:,:,None]).squeeze(-1).norm(dim=-1)
    return dq,dict(requested_point_translation_xy_m=requested_xy,
       linearized_translation_xy_m=achieved,linearized_angular_change_rad=achieved_angle)

class SweepEnvelope:
    def __init__(self,n,device):
        self.actual=torch.zeros(n,29,device=device)
        self.exit_delta=self.actual.clone()
        self.exit_time=torch.full((n,),-1.,device=device)
    def reset(self,ids):
        self.actual[ids]=0.
        self.exit_delta[ids]=0.
        self.exit_time[ids]=-1.
    def update(self,now,raw,enabled,touched,maximum=.022,max_step=.004,fade_s=.20):
        n=len(now)
        if now.shape!=(n,) or raw.shape!=(n,29) or enabled.shape!=(n,) or touched.shape!=(n,):
            raise ValueError("Sweep envelope shapes")
        if not all(bool(torch.isfinite(a).all()) for a in (now,raw,enabled)):raise ValueError("Nonfinite sweep")
        latch=(touched|(now>=2.10)) & (self.exit_time<0)
        self.exit_delta=torch.where(latch[:,None],self.actual,self.exit_delta)
        self.exit_time=torch.where(latch,now,self.exit_time)
        active=self.exit_time<0
        desired=raw*enabled[:,None]
        desired=torch.where(active[:,None],desired,
                            self.exit_delta*(1.-_quintic((now-self.exit_time)/fade_s))[:,None])
        peak=desired.abs().amax(-1,keepdim=True)
        desired=desired*torch.minimum(torch.ones_like(peak),maximum/peak.clamp_min(1e-12))
        # Globally coupled slew scaling, not per-joint independent clipping.
        changes=desired-self.actual
        a=changes.abs().amax(-1,keepdim=True)
        self.actual+=changes*torch.minimum(torch.ones_like(a),max_step/a.clamp_min(1e-12))
        done=(self.exit_time>=0)&(now+1e-7>=self.exit_time+fade_s)
        self.actual=torch.where(done[:,None],torch.zeros_like(self.actual),self.actual)
        return self.actual.clone(),dict(exit_time_s=self.exit_time.clone(),finished=done.clone())

class FootSweepDirectionTask(PostShotRecoveryTask):
    def __init__(self,env,config):
        if config.get("prestrike_feedback") or config.get("early_path_feedback") or config.get("support_reference_frame"):
            raise ValueError("v38 is isolated over frozen v32, never mixed with rejected v34/v35")
        if config.get("post_shot_recovery",{}).get("checkpoint_sha256")!=(
              "ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"):
            raise ValueError("Frozen v32 post-shot control required")
        super().__init__(env,config)
        self.sweep_cfg=config["foot_sweep_direction"]
        if type(self.sweep_cfg.get("enabled")) is not bool:
            raise ValueError("v38 enabled must be a real boolean, never inferred from a string")
        fixed={"maximum_rad":.022,"maximum_step_rad":.004,
               "lookahead_s":.012,"max_translation_m":.022,
               "start_s":.85,"stop_s":2.10,"onset_s":.12,"fade_s":.20,
               "support_min_N":80.,"swing_max_N":40.,"minimum_forward_mps":.5,
               "minimum_range_m":.0,"maximum_range_m":.95}
        if any(k not in self.sweep_cfg or not isinstance(self.sweep_cfg[k],(int,float))
               or not math.isfinite(self.sweep_cfg[k]) or abs(self.sweep_cfg[k]-v)>1e-9
               for k,v in fixed.items()):
            raise ValueError("Prespecified v38 gains and bounds required")
        self.sweep= SweepEnvelope(self.n,self.device)
        self.point_local=torch.tensor([.115,0.,-.025],device=self.device)
        self.last_sweep_debug={}
    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.sweep.reset(ids)
        self.last_sweep_debug={}
    def commands(self,actions=None):
        if actions is not None:raise ValueError("v38 is fixed measured feedback, not a policy action")
        # HARD ZERO PATH: do not execute any new kinematics or CUDA operations.
        # Preserve original v32 commands identically when disabled. The prior
        # buggy version ignored enabled=false and is retained as a failed run.
        if self.sweep_cfg["enabled"] is False:
            base=super().commands(None)
            if self.n==1 and self.debug:
                z=torch.zeros(29,device=base.device,dtype=base.dtype)
                self.debug[-1]["foot_sweep_direction"]=dict(
                    enabled=False,eligible=False,context_gate=float(self.gate[0]),
                    previous_delta_rad=z.tolist(),actual_delta_rad=z.tolist(),
                    baseline_action=base[0].tolist(),final_action=base[0].tolist(),
                    ball_state_write=False)
            return base
        r=self.robot.data
        now=self.elapsed.clone()*.02
        # Read pre-command measured physics, one snapshot.
        right=self.feet[1]
        point,point_vel,_=measured_point_velocity(
            r.body_pos_w[:,right],r.body_quat_w[:,right],
            r.body_link_lin_vel_w[:,right],r.body_ang_vel_w[:,right],self.point_local)
        relative=point_vel[:,:2]-self.ball.data.root_lin_vel_w[:,:2]
        ray=self.goal[:,:2]-self.ball.data.root_pos_w[:,:2]
        corrections,kin=lateral_sweep_target(relative,
            self.ball.data.root_pos_w[:,:2],self.goal[:,:2],
            lookahead_s=self.sweep_cfg["lookahead_s"],
            max_translation_m=self.sweep_cfg["max_translation_m"])
        q=r.joint_pos[:,self.expert.ids]
        dq,ik=damped_point_ik_delta(self.leg,q,_rotation(r.root_quat_w),r.root_pos_w,
                                      self.point_local,corrections,
                                      max_joint_rad=self.sweep_cfg["maximum_rad"])
        support=ground_force(self.env)
        # Parent captures measured context/gate and advances the frozen v32
        # experts exactly once. All feedback features above were read BEFORE.
        base=super().commands(None)
        left=support[:,0];right_load=support[:,1]
        # Hard support and forward-motion gates protect transitions; softness
        # is ONLY the onset and inherited nominal context scalar.
        ball_range=torch.linalg.vector_norm(
            self.ball.data.root_pos_w[:,:2]-r.root_pos_w[:,:2],dim=-1)
        support_ok=(left>=self.sweep_cfg["support_min_N"]) & (right_load<=self.sweep_cfg["swing_max_N"])
        distance_ok=(ball_range>=self.sweep_cfg["minimum_range_m"]) & (ball_range<=self.sweep_cfg["maximum_range_m"])
        speed_ok=kin["forward_velocity_mps"]>=self.sweep_cfg["minimum_forward_mps"]
        window=(now>=self.sweep_cfg["start_s"])&(now<self.sweep_cfg["stop_s"])
        enabled=(support_ok & distance_ok & speed_ok & window & (~self.contact)).to(dq.dtype)
        enabled*=self.gate*_quintic((now-self.sweep_cfg["start_s"])/self.sweep_cfg["onset_s"])
        raw=torch.zeros(self.n,29,device=self.device)
        raw[:,self.expert.ids]=dq
        old=self.sweep.actual.clone()
        delta,envdiag=self.sweep.update(now,raw,enabled,self.contact,
            maximum=self.sweep_cfg["maximum_rad"],
            max_step=self.sweep_cfg["maximum_step_rad"],
            fade_s=self.sweep_cfg["fade_s"])
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
                                device=base.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):
            raise ValueError("Original action scale invalid")
        changed=delta!=0
        out=torch.where(changed,base+delta/scale,base) if bool(changed.any()) else base
        self.sweep.actual=(out-base)*scale
        if self.n==1 and self.debug:
            self.debug[-1]["foot_sweep_direction"]=dict(
                point_local_m=self.point_local.tolist(),
                point_position_w_m=point[0].tolist(),
                point_velocity_w_mps=point_vel[0].tolist(),
                ball_velocity_w_mps=self.ball.data.root_lin_vel_w[0].tolist(),
                target_goal_xy_w=kin["goal_heading_world_xy"][0].tolist(),
                lateral_velocity_mps=float(kin["lateral_velocity_mps"][0]),
                forward_velocity_mps=float(kin["forward_velocity_mps"][0]),
                predicted_requested_translation_xy_m=corrections[0].tolist(),
                predicted_achieved_translation_xy_m=ik["linearized_translation_xy_m"][0].tolist(),
                predicted_angular_change_rad=float(ik["linearized_angular_change_rad"][0]),
                support_N=support[0].tolist(),
                root_ball_distance_m=float(ball_range[0]),
                eligible=bool(enabled[0]>0),
                context_gate=float(self.gate[0]),
                enabled_amplitude=float(enabled[0]),
                previous_delta_rad=old[0].tolist(),
                actual_delta_rad=self.sweep.actual[0].tolist(),
                baseline_action=base[0].tolist(),
                final_action=out[0].tolist(),
                exit_time_s=float(envdiag["exit_time_s"][0]),
                touched=bool(self.contact[0]),
                ball_state_write=False)
        return out

def get_task(env):
    if not hasattr(env,"_foot_sweep_direction_task"):
        env._foot_sweep_direction_task=FootSweepDirectionTask(env,env.cfg.joint_strike)
    return env._foot_sweep_direction_task
