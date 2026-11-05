"""State-conditioned residuals over the frozen *closed-loop* expert."""
from pathlib import Path
import torch
from closed_loop_expert import FrozenStrideExpert,measured_context_gate
from joint_strike_core import JointStrikeTask,JointStrikeActor,bounded_phase_step,residual_action


class ClosedLoopStrikeTask(JointStrikeTask):
    def __init__(self,env,config):
        super().__init__(env,config)
        self.expert=FrozenStrideExpert(env,config)
        self.context_ready=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.velocity_error=torch.zeros(self.n,3,device=self.device)
        self.gate=torch.zeros(self.n,device=self.device)

    def reset_fields(self,ids):
        super().reset_fields(ids);self.expert.reset(ids)
        self.context_ready[ids]=False;self.velocity_error[ids]=0.;self.gate[ids]=0.

    def capture_context(self):
        ids=(~self.context_ready).nonzero(as_tuple=False).flatten()
        if len(ids):
            self.shift[ids]=self.ball.data.root_pos_w[ids,:2]-self.env.scene.env_origins[ids,:2]-torch.tensor(self.cfg['ball_xy'],device=self.device)
            self.velocity_error[ids]=self.robot.data.root_lin_vel_w[ids]-self.ref['root_velocity'][0]
            self.gate[ids]=measured_context_gate(self.shift[ids],self.velocity_error[ids]);self.context_ready[ids]=True

    def features(self):
        self.capture_context()
        return torch.cat([super().features(),self.shift/.1,self.velocity_error/.2,self.gate[:,None]],-1).clamp(-5.,5.)

    def commands(self,actions):
        if actions.shape!=(self.n,30):raise ValueError('29 joint corrections plus one phase action required')
        self.capture_context();effective=actions*self.gate[:,None]
        with torch.no_grad():base,self.step_velocity=self.expert.act(self.frame,self.elapsed,self.contact)
        term=self.env.action_manager.get_term('joint_pos')
        out=residual_action(base,effective[:,:29],term._scale,term._offset,self.robot.data.soft_joint_pos_limits,self.cfg['maximum_joint_residual_rad'])
        if self.n==1:
            self.debug.append(dict(time_s=float(self.elapsed[0])*.02,phase_s=float(self.frame[0])*.02,
                residual_rad=((out-base)*term._scale)[0].cpu().tolist(),phase_action=float(effective[0,29]),
                context_gate=float(self.gate[0]),measured_initial_velocity_error=self.velocity_error[0].cpu().tolist(),
                contact_latched=bool(self.contact[0]),measured_ball_xyz=(self.ball.data.root_pos_w[0]-self.env.scene.env_origins[0]).cpu().tolist()))
        self.frame=bounded_phase_step(self.frame,self.elapsed,effective[:,29],self.cfg['maximum_phase_shift_s']/.02,self.cfg['phase_rate_limit']);self.elapsed+=1
        self.last_action=torch.tanh(effective).detach()
        return out


def get_task(env):
    if not hasattr(env,'_closed_loop_strike_task'):env._closed_loop_strike_task=ClosedLoopStrikeTask(env,env.cfg.joint_strike)
    return env._closed_loop_strike_task


def task_features(env):return get_task(env).features()
def task_reward(env):return get_task(env).reward()
