#!/usr/bin/env python3
"""v061 strictly read-only staged whole-body G1 tensor/PhysX API isolation.

Probe *does not* change contacts, controls, physical thresholds or physics. The
trace of every stage is inspected independently, and a missing G1 JSON is never
a valid rollout. Stage prints include the precise last complete operation.
"""
from __future__ import annotations
import math
import json

MODES=("empty","mass","com_pos","com_vel","body_pos","body_vel",
       "forces","report","full_compute")
LINKS=("left_wrist_yaw_link","left_hip_roll_link")
def tensor_list(value):
    return value.detach().cpu().tolist()

class StageProbe:
    def __init__(self,env,mode):
        if mode not in MODES:raise ValueError(f"Unregistered v061 probe {mode}")
        self.env=env;self.robot=env.scene["robot"];self.mode=mode
        self.rows=[];self.calls=0;self.last_successful_stage="begin"
        self.names={n:self.robot.body_names.index(n) for n in LINKS}
        self.report_calls=0
        self._enter("constructor_start")
        if mode in ("mass","full_compute"):
            self._enter("mass_get_masses")
            self.masses=self.robot.root_physx_view.get_masses()[0]
            self._enter("mass_sum")
            self.weights=self.masses/self.masses.sum()
            self.total_mass_kg=float(self.masses.sum())
            self._done("mass_complete")
        if mode=="report":
            self._enter("reporter_import")
            import omni.physx
            self.sim=omni.physx.get_physx_simulation_interface()
            self._done("reporter_init_complete")
        self._done("constructor_complete")

    def _enter(self,s):
        if self.calls<2:
            print("V061_PROBE_ENTER",self.mode,s,"frame",self.calls,flush=True)
        self.last_attempted_stage=s
    def _done(self,s):
        self.last_successful_stage=s
        if self.calls<2:
            print("V061_PROBE_OK",self.mode,s,"frame",self.calls,flush=True)
    def capture(self,dt):
        if dt<=0:return
        if abs(dt-.005)>1e-7:raise ValueError("Original 200Hz required")
        self._enter("capture_start")
        r=self.robot.data
        t=float(self.env.sim.current_time)
        row={"t":t}
        if self.mode in ("com_pos","full_compute"):
            self._enter("body_com_pos")
            pos=r.body_com_pos_w[0]
            row["com_pos_first"]=tensor_list(pos[0])
            if self.mode=="full_compute":
                row["mass_weighted_com"]=tensor_list((pos*self.weights[:,None]).sum(0))
            self._done("body_com_pos")
        if self.mode in ("com_vel","full_compute"):
            self._enter("body_com_lin_vel")
            vel=r.body_com_lin_vel_w[0]
            row["com_vel_first"]=tensor_list(vel[0])
            if self.mode=="full_compute":
                row["mass_weighted_com_vel"]=tensor_list((vel*self.weights[:,None]).sum(0))
            self._done("body_com_lin_vel")
        if self.mode in ("body_pos","full_compute"):
            self._enter("body_pos")
            p=r.body_pos_w[0]
            coords=[tensor_list(p[self.names[name]]) for name in LINKS]
            row["body_pos"]=coords
            row["link_origin_distance_m"]=math.dist(*coords)
            self._done("body_pos")
        if self.mode in ("body_vel","full_compute"):
            self._enter("body_lin_vel")
            p=r.body_lin_vel_w[0]
            row["body_vel"]=[tensor_list(p[self.names[name]]) for name in LINKS]
            self._done("body_lin_vel")
        if self.mode in ("forces","full_compute"):
            self._enter("net_contact_forces")
            sensor=self.env.scene["contact_forces"]
            force=sensor.data.net_forces_w[0]
            row["nonfoot_forces"]=[tensor_list(force[sensor.body_names.index(name)]) for name in LINKS]
            self._done("net_contact_forces")
        if self.mode=="report" and 4.8<=t<=5.4:
            self._enter("physx_get_contact_report")
            headers,points=self.sim.get_contact_report()
            row["contact_headers"]=len(headers);row["contacts"]=len(points)
            self.report_calls+=1
            self._done("physx_get_contact_report")
        if self.mode=="full_compute":
            self._enter("joint_states")
            row["q"]=tensor_list(r.joint_pos[0])
            row["qdot"]=tensor_list(r.joint_vel[0])
            self._done("joint_states")
        self.rows.append(row)
        self.calls+=1
        self._done("capture_complete")
    def result(self):
        if not self.rows:raise ValueError("No real physics captured")
        if any(not math.isfinite(x["t"]) for x in self.rows):
            raise ValueError("Nonfinite real simulation time")
        return dict(version="v0.1.61",mode=self.mode,capture_count=self.calls,
                    last_successful_stage=self.last_successful_stage,
                    last_attempted_stage=self.last_attempted_stage,
                    report_calls=self.report_calls,rows=self.rows,
                    controller_writes=False,physx_writes=False,
                    observer_uses_only_original_robot_29dof=True)
