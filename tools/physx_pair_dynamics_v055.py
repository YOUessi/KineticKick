#!/usr/bin/env python3
"""Read-only per-5ms G1 wrist/hip contact-pair and whole-body dynamics recorder.

No USD authoring, mass edits, collision filtering, contact thresholds or control
writes. Native PhysX pair-filtered contact view provides the *link pair*, while
get_contact_report() (only when originally enabled) can provide collider paths.
A link pair is not a collider primitive path: the two proofs remain separate.
"""
from __future__ import annotations
import math

BODY_NAMES=("left_wrist_yaw_link","left_hip_roll_link","left_elbow_link",
            "left_shoulder_yaw_link","torso_link","pelvis",
            "left_ankle_roll_link","right_ankle_roll_link",
            "left_knee_link","right_knee_link")
PAIR=("left_wrist_yaw_link","left_hip_roll_link")

def vector(x):
    if hasattr(x,"detach"): x=x.detach().cpu().tolist()
    return [float(v) for v in x]

def json_vector(v):
    try: return vector(v)
    except (TypeError,ValueError): return None

def valid_pair_record(row):
    if not isinstance(row,dict) or sorted(row.get("pair",[]))!=sorted(PAIR):raise ValueError("Wrong contact body pair")
    if not math.isfinite(row.get("time_s",float("nan"))):raise ValueError("Nonfinite physics time")
    points=row.get("manifold",[])
    if not isinstance(points,list):raise ValueError("Pair manifold required")
    for p in points:
        if not isinstance(p,dict) or len(p.get("point_w",[]))!=3 or len(p.get("normal_w",[]))!=3:
            raise ValueError("Malformed contact points")
    return True

class PhysxPairDynamicsObserver:
    def __init__(self,env):
        import omni.physx
        from pxr import PhysicsSchemaTools
        self.env=env;self.robot=env.scene["robot"]
        if env.num_envs!=1:raise ValueError("Independent PhysX pair recorder needs one robot")
        self.names={name:self.robot.body_names.index(name) for name in BODY_NAMES if name in self.robot.body_names}
        if not all(n in self.names for n in PAIR):raise ValueError("Absent G1 left wrist/hip original rigid links")
        # No new PhysX contact view: self-articulation pair filtering crashed the
        # isolated instrumentation smoke test. Existing on-stage contact report
        # is a read-only collider-path source; absence means unproven pair.
        self.sim=omni.physx.get_physx_simulation_interface()
        self.decode=PhysicsSchemaTools.intToSdfPath
        self.frames=[];self.pair_events=[];self.collider_events=[]
        self.reporter_count=0;self.reporter_error=None
        self.count_capacity_exhausted=False
        self.masses=self.robot.root_physx_view.get_masses()[0]
        self.weights=self.masses/self.masses.sum()
        self.total_mass_kg=float(self.masses.sum())

    def capture(self,dt):
        if dt<=0:return
        if abs(dt-.005)>1e-7:raise ValueError("Only original 200Hz G1 step is allowed")
        r=self.robot.data;t=float(self.env.sim.current_time)
        com=(r.body_com_pos_w[0]*self.weights[:,None]).sum(dim=0)
        wrench=self.env.scene["contact_forces"].data.net_forces_w[0]
        row=dict(time_s=t,dt_s=float(dt),joint_pos_rad=vector(r.joint_pos[0]),
                 joint_vel_rad_s=vector(r.joint_vel[0]),
                 com_w=vector(com),com_vel_w=vector((r.body_com_lin_vel_w[0]*self.weights[:,None]).sum(0)),
                 body_position_w={n:vector(r.body_pos_w[0,i]) for n,i in self.names.items()},
                 body_lin_velocity_w={n:vector(r.body_lin_vel_w[0,i]) for n,i in self.names.items()},
                 root_position_w=vector(r.root_pos_w[0]),
                 nonfoot_force_net_w_N={n:vector(wrench[self.env.scene["contact_forces"].body_names.index(n)])
                                        for n in PAIR})
        a=row["body_position_w"][PAIR[0]];b=row["body_position_w"][PAIR[1]]
        row["left_wrist_hip_link_origin_distance_m"]=math.dist(a,b)
        row["direct_pair_contacts"]=0
        self.frames.append(row)
        # The *existing* Physx contact-report metadata might already be enabled.
        # This read-only query never applies ContactReportAPI or modifies stage.
        if 4.8<=t<=5.4 and self.reporter_error is None:
            try:
                headers,contacts=self.sim.get_contact_report()
                self.reporter_count+=len(headers)
                for header in headers:
                    paths=[str(self.decode(getattr(header,f))) for f in ("actor0","actor1","collider0","collider1")]
                    if not any(("left_wrist_yaw_link" in p or "left_hip_roll_link" in p) for p in paths):continue
                    k=int(getattr(header,"contact_data_offset",0));n=int(getattr(header,"num_contact_data",0))
                    details=[]
                    for contact in contacts[k:k+n]:
                        details.append(dict(point_w=json_vector(getattr(contact,"position",None)),
                            normal_w=json_vector(getattr(contact,"normal",None)),
                            impulse_raw=json_vector(getattr(contact,"impulse",None))))
                    record=dict(time_s=t,actor_paths=paths[:2],
                         collider_paths=paths[2:],manifold=details)
                    self.collider_events.append(record)
                    if (any("left_wrist_yaw_link" in p for p in paths) and
                        any("left_hip_roll_link" in p for p in paths)):
                        manifold=[]
                        for contact in details:
                            if contact["point_w"] is None or contact["normal_w"] is None:continue
                            impulse=contact["impulse_raw"] or [0.,0.,0.]
                            manifold.append(dict(point_w=contact["point_w"],
                                 normal_w=contact["normal_w"],
                                 normal_force_scalar_raw=math.sqrt(sum(x*x for x in impulse))/dt,
                                 signed_separation_m=None,
                                 raw_impulse_w_Ns=impulse))
                        if manifold:
                            event=dict(time_s=t,pair=list(PAIR),manifold=manifold,
                                  provenance="existing PhysX contact-report actor/collider paths",
                                  capacity_exhausted=False)
                            valid_pair_record(event);self.pair_events.append(event)
                            row["direct_pair_contacts"]+=len(manifold)
            except Exception as e:self.reporter_error=type(e).__name__+": "+str(e)[:250]

    def result(self):
        if not self.frames:raise ValueError("No 200Hz physics captured")
        return dict(schema_version=1,original_physics_dt_s=.005,
           wrist_hip_rigid_pair=PAIR,body_indices=self.names,
           body_mass_total_kg=self.total_mass_kg,
           frames=self.frames,pair_filtered_manifolds=self.pair_events,
           original_physx_contact_report_collider_paths=self.collider_events,
           contact_report_header_count=self.reporter_count,
           contact_report_exception=self.reporter_error,
           pair_view_buffer_full=self.count_capacity_exhausted,
           collider_paths_verified=any(
               any("left_wrist_yaw_link" in p for p in e["collider_paths"]) and
               any("left_hip_roll_link" in p for p in e["collider_paths"])
               for e in self.collider_events),
           rigid_link_pair_directly_measured=bool(self.pair_events),
           instrument_type="read-only existing PhysX contact-report poll, no new contact views",
           pair_specific_manifold_source="existing PhysX report if native headers available",
           no_new_physx_contact_view=True,
           collision_filter_or_threshold_changed=False,
           controller_or_actor_state_written=False,
           limitations=["No additional self-articulation filtered contact view, due to preflight hard crash",
                        "PhysX contact-report events require contact reporting already enabled; missing headers cannot prove absence",
                        "Net force is not uniquely ground reaction force",
                        "Body origin distance is not collider surface gap",
                        "Termination truncates actual available recovery, never imputed"])
