"""v40 physical execution adapter for a frozen whole-body coordination *reference*.

This takes an OFFLINE, SHA-pinned 700x29 joint-reference delta which comes from
a human-proxy temporal coordination constraint and a G1 URDF projection.
It does NOT alter the original G1 robot model or ball, does NOT replace PPO or
write simulator state, and is NOT a direct human video controller.
Original v32 remains the only valid policy when the adapter is disabled.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,math
import numpy as np
import torch
from post_shot_recovery import PostShotRecoveryTask
from running_balance import ground_force

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_POST_SHA="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

class CoordinationReferenceTask(PostShotRecoveryTask):
    def __init__(self,env,config):
        if config.get("prestrike_feedback") or config.get("early_path_feedback") or config.get("foot_sweep_direction"):
            raise ValueError("Only frozen v32 post-shot expert allowed; never combine rejected adapters")
        if config.get("post_shot_recovery",{}).get("checkpoint_sha256")!=EXPECTED_POST_SHA:
            raise ValueError("Frozen v32 post-shot learned balance policy required")
        super().__init__(env,config)
        model=config["human_coordination_reference"]
        self.coord_enabled=model["enabled"]
        if type(self.coord_enabled) is not bool:raise ValueError("Explicit boolean flag required")
        limits={"max_joint_rad":.006,"max_step_rad":.0012,
                "first_s":1.05,"last_s":1.56,
                "support_min_N":80.,"swing_max_N":40.,"max_qdot_ratio":.90,
                "minimum_ball_distance_m":0.,"maximum_ball_distance_m":1.15}
        for k,v in limits.items():
            if k not in model or abs(float(model[k])-v)>1e-10:
                raise ValueError("Registered physical research bound changed: "+k)
        path=ROOT/model["trajectory"]
        if digest(path)!=model["trajectory_sha256"]:
            raise ValueError("Human-proxy reference artifact SHA changed")
        with np.load(path,allow_pickle=False) as z:
            delta=z["delta_joint_position_rad"]
            labels=z["joint_names"].tolist()
            times=z["controller_time_s"]
            if (delta.shape!=(700,29) or len(labels)!=29 or
                list(labels)!=list(self.robot.joint_names) or
                not np.allclose(times,np.arange(700)*.02,atol=1e-5)):
                raise ValueError("Wrong 700x29 original G1 action-order contract")
            if bool(z["source_is_verified_instep"]):raise ValueError("Cannot claim human contact truth")
        if not np.isfinite(delta).all() or abs(delta).max()>.0060005 or abs(np.diff(delta,axis=0)).max()>.0012005:
            raise ValueError("Research trajectory exceeds frozen envelope")
        self.reference=torch.tensor(delta,device=self.device,dtype=torch.float32)
        self.coord_cfg=model
        self.applied=torch.zeros(self.n,29,device=self.device)
        self.previous=self.applied.clone()
        self.coord_first_touch=torch.full((self.n,),-1.,device=self.device)
        self.coord_debug={}
    def reset_fields(self,ids):
        super().reset_fields(ids)
        self.applied[ids]=0.
        self.previous[ids]=0.
        self.coord_first_touch[ids]=-1.
        self.coord_debug={}
    def commands(self,actions=None):
        if actions is not None:raise ValueError("Frozen pre-recorded coordination reference has no external actions")
        # No-op parity must execute EXACTLY one parent call and no new math.
        if not self.coord_enabled:
            parent=super().commands(None)
            if self.n==1 and self.debug:
                z=[0.]*29
                self.debug[-1]["human_coordination_reference"]=dict(
                    enabled=False,applied_delta_rad=z,previous_delta_rad=z,
                    reference_nonzero=False,eligible=False,safety_gate=False,
                    baseline_action=parent[0].tolist(),final_action=parent[0].tolist(),
                    ball_state_write=False)
            return parent

        base=super().commands(None)
        now=self.elapsed*.02
        index=self.elapsed.long().clamp(0,699)
        raw=self.reference[index]
        self.previous=self.applied.clone()
        contact_new=self.contact&(self.coord_first_touch<0)
        self.coord_first_touch=torch.where(contact_new,now,self.coord_first_touch)
        from_previous=torch.maximum(self.applied.abs().amax(dim=-1),torch.zeros(self.n,device=self.device))
        time_ok=(now>=self.coord_cfg["first_s"])&(now<=self.coord_cfg["last_s"])
        support=ground_force(self.env)
        physical=self.robot.data
        loaded=(support[:,0]>=self.coord_cfg["support_min_N"]) & (support[:,1]<=self.coord_cfg["swing_max_N"])
        ratio=(physical.joint_vel/physical.joint_vel_limits).abs().amax(dim=-1)
        qdot_ok=(ratio<=self.coord_cfg["max_qdot_ratio"])
        ball_distance=torch.linalg.vector_norm(self.ball.data.root_pos_w[:,:2]-physical.root_pos_w[:,:2],dim=-1)
        ball_ok=(ball_distance>=self.coord_cfg["minimum_ball_distance_m"]) & (ball_distance<=self.coord_cfg["maximum_ball_distance_m"])
        eligible=time_ok & loaded & qdot_ok & ball_ok & ~self.contact
        desired=torch.where(eligible[:,None],raw*self.gate[:,None],torch.zeros_like(raw))
        diff=(desired-self.applied).clamp(-self.coord_cfg["max_step_rad"],self.coord_cfg["max_step_rad"])
        self.applied=(self.applied+diff).clamp(-self.coord_cfg["max_joint_rad"],self.coord_cfg["max_joint_rad"])
        # Bound decay after contact just as before; do not keep applying q deltas.
        done=(now>=self.coord_cfg["last_s"]+.25)&(self.applied.abs().amax(dim=-1)<1e-5)
        self.applied=torch.where(done[:,None],torch.zeros_like(self.applied),self.applied)
        scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,
                              device=self.device,dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale==0.).any()):
            raise ValueError("Original G1 29-DoF scale invalid")
        changed=self.applied!=0.
        corrected=torch.where(changed,base+self.applied/scale,base) if bool(changed.any()) else base
        self.applied=(corrected-base)*scale
        if self.n==1 and self.debug:
            self.debug[-1]["human_coordination_reference"]=dict(
                enabled=True,applied_delta_rad=self.applied[0].tolist(),
                previous_delta_rad=self.previous[0].tolist(),
                original_reference_delta_rad=raw[0].tolist(),
                reference_nonzero=bool((raw[0].abs()>1e-7).any()),
                eligible=bool(eligible[0]),safety_gate=bool(loaded[0]&qdot_ok[0]&ball_ok[0]),
                support_normal_N=support[0].tolist(),
                maximum_measured_qdot_ratio=float(ratio[0]),
                ball_distance_m=float(ball_distance[0]),
                original_context_gate=float(self.gate[0]),
                physical_contact_latched=bool(self.contact[0]),
                original_post_shot_recovery_active=bool((self.actual_delta[0].abs()>1e-7).any()),
                baseline_action=base[0].tolist(),final_action=corrected[0].tolist(),
                ball_state_write=False)
        return corrected

def get_task(env):
    if not hasattr(env,"_human_coordination_reference_task"):
        env._human_coordination_reference_task=CoordinationReferenceTask(env,env.cfg.joint_strike)
    return env._human_coordination_reference_task
