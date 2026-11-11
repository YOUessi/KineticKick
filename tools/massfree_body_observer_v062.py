#!/usr/bin/env python3
"""Read-only G1 200Hz combined body observer with CUDA mass-cache ablation.

This is a physics-observation instrument, not a natural football controller.
Never changes USD/PhysX contacts, actions or joint limits. Full-body COM *mass-
weighted* estimate is left to the existing independently recorded base trace;
don't substitute an unweighted mean and call it a center of mass.
"""
from __future__ import annotations
import math
MODES=("safe_full","mass_no_mul","mass_clone","mass_after")
BODIES=("left_wrist_yaw_link","left_hip_roll_link","left_elbow_link",
        "left_shoulder_yaw_link","torso_link","pelvis",
        "left_ankle_roll_link","right_ankle_roll_link",
        "left_knee_link","right_knee_link")
def v(x):return x.detach().cpu().tolist()
class SafeBodyObserver:
    def __init__(self,env,mode):
        if mode not in MODES:raise ValueError("Unknown frozen v062 mode")
        self.env=env;self.mode=mode;self.robot=env.scene["robot"]
        if env.num_envs!=1:raise ValueError("G1 requires one independent environment")
        self.ids={n:self.robot.body_names.index(n) for n in BODIES}
        self.rows=[];self.calls=0
        self._enter("constructor")
        if mode in ("mass_no_mul","mass_clone"):
            self._enter("mass_get_masses_ctor")
            src=self.robot.root_physx_view.get_masses()[0]
            self.masses=src.clone() if mode=="mass_clone" else src
            self._done("mass_get_masses_ctor")
            if mode=="mass_clone":
                self._enter("mass_cloned_normalize")
                self.weights=self.masses/self.masses.sum()
                self._done("mass_cloned_normalize")
        self._done("ctor_complete")
    def _enter(self,name):
        self.attempted=name
        if self.calls<2:print("V062_ENTER",self.mode,name,"frame",self.calls,flush=True)
    def _done(self,name):
        self.last_successful=name
        if self.calls<2:print("V062_OK",self.mode,name,"frame",self.calls,flush=True)
    def capture(self,dt):
        if dt<=0:return
        if abs(dt-.005)>1e-7:raise ValueError("Original 200Hz required")
        self._enter("frame_start")
        r=self.robot.data
        t=float(self.env.sim.current_time)
        row=dict(t=t)
        self._enter("com_pos")
        pos=r.body_com_pos_w[0]
        row["first_com_pos_w"]=v(pos[0])
        self._done("com_pos")
        if self.mode=="mass_after" and self.calls==0:
            self._enter("mass_after_first_com")
            self.masses=self.robot.root_physx_view.get_masses()[0].clone()
            self.weights=self.masses/self.masses.sum()
            self._done("mass_after_first_com")
        if self.mode in ("mass_clone","mass_after"):
            self._enter("weighted_com")
            row["weighted_com_w"]=v((pos*self.weights[:,None]).sum(dim=0))
            self._done("weighted_com")
        if self.mode=="mass_no_mul":
            # Prove or disprove that merely fetching the PhysX masses before
            # COM is disruptive on CUDA, independent of multiplication.
            self._enter("mass_no_mul_done")
            row["mass_sum"]=float(self.masses.sum())
            self._done("mass_no_mul_done")
        if self.mode=="safe_full":
            self._enter("com_vel")
            vel=r.body_com_lin_vel_w[0]
            row["first_com_vel_w"]=v(vel[0])
            self._done("com_vel")
            self._enter("body_pos")
            bodies=r.body_pos_w[0]
            row["bodies_w"]={n:v(bodies[i]) for n,i in self.ids.items()}
            wrist=row["bodies_w"]["left_wrist_yaw_link"]
            hip=row["bodies_w"]["left_hip_roll_link"]
            row["left_wrist_hip_link_origin_distance_m"]=math.dist(wrist,hip)
            self._done("body_pos")
            self._enter("body_vel")
            body_vel=r.body_lin_vel_w[0]
            row["velocities_w"]={n:v(body_vel[i]) for n,i in self.ids.items()}
            self._done("body_vel")
            self._enter("contact_force")
            sensor=self.env.scene["contact_forces"]
            forces=sensor.data.net_forces_w[0]
            row["nonfoot_force_w_N"]={n:v(forces[sensor.body_names.index(n)])
                                      for n in ("left_wrist_yaw_link","left_hip_roll_link","pelvis","right_hip_roll_link")}
            self._done("contact_force")
            self._enter("all_joint_state")
            row["joint_pos_rad"]=v(r.joint_pos[0])
            row["joint_vel_rad_s"]=v(r.joint_vel[0])
            self._done("all_joint_state")
        self.rows.append(row);self.calls+=1
        self._done("frame_completed")
    def result(self):
        if len(self.rows)<30:raise ValueError("Insufficient real 5ms rows")
        return dict(version="v0.1.62",mode=self.mode,actual_200hz_rows=len(self.rows),
                    rows=self.rows,last_successful_stage=self.last_successful,
                    original_G1_29dof_unchanged=True,
                    body_com_is_not_mass_weighted_in_safe_full=True,
                    use_independently_measured_base_trace_for_true_com=True,
                    actual_collider_pair_manifold_unavailable=True,
                    physical_writes=False,actuator_writes=False)
