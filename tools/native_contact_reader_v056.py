#!/usr/bin/env python3
"""Native contact-report subscriber (read-only) for the unchanged original G1.

This does NOT apply ContactReportAPI or create new native contact views. It
captures actor/collider paths only where native PhysX itself reports them.
An empty list is UNKNOWN, never 'no collision'. Keep subscription handle alive
until the physics env closes.
"""
from __future__ import annotations
from math import isfinite

INTERESTING = ("left_wrist_yaw_link","left_hip_roll_link","pelvis",
               "right_hip_roll_link")
def vec(value):
    if value is None:return None
    try:
        items=list(value)
        if len(items)!=3:return None
        r=[float(v) for v in items]
        return r if all(isfinite(v) for v in r) else None
    except (TypeError,ValueError):
        return None

class G1ReadOnlyContactSubscriber:
    def __init__(self,env,max_events=1500):
        import omni.physx,omni.usd
        from pxr import PhysxSchema,PhysicsSchemaTools
        self.env=env
        self.events=[]
        self.collisions_dropped=0
        self.callback_calls=0
        self.native_headers=0
        self.errors=[]
        self.max_events=max_events
        self.decode=PhysicsSchemaTools.intToSdfPath
        stage=omni.usd.get_context().get_stage()
        self.report_apis={}
        for prim in stage.Traverse():
            name=prim.GetName()
            if name in INTERESTING:
                self.report_apis[str(prim.GetPath())]=bool(prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
        self.interface=omni.physx.get_physx_simulation_interface()
        self.subscription=self.interface.subscribe_contact_report_events(self._on_contacts)

    def _on_contacts(self,headers,data):
        self.callback_calls+=1
        t=float(self.env.sim.current_time)
        for header in headers:
            self.native_headers+=1
            try:
                paths=[str(self.decode(getattr(header,n))) for n in
                       ("actor0","actor1","collider0","collider1")]
                if not any(any(n in p for n in INTERESTING) for p in paths):
                    continue
                count=int(getattr(header,"num_contact_data",0))
                offset=int(getattr(header,"contact_data_offset",0))
                contacts=[]
                for cp in data[offset:offset+min(count,30)]:
                    impulse=getattr(cp,"impulse",None)
                    contacts.append(dict(point_w=vec(getattr(cp,"position",None)),
                        normal_w=vec(getattr(cp,"normal",None)),
                        impulse_xyz=vec(impulse),
                        impulse_scalar=float(impulse) if isinstance(impulse,(int,float)) else None,
                        separation_m=float(getattr(cp,"separation")) if hasattr(cp,"separation") else None))
                wrist=any("left_wrist_yaw_link" in p for p in paths)
                hip=any("left_hip_roll_link" in p for p in paths)
                pelvis=any("pelvis" in p for p in paths)
                right_hip=any("right_hip_roll_link" in p for p in paths)
                record=dict(time_s=t,actor_paths=paths[:2],collider_paths=paths[2:],
                    reported_points=count,point_sample=contacts,
                    candidate_wrist_hip=bool(wrist and hip),
                    candidate_pelvis_right_hip=bool(pelvis and right_hip))
                if len(self.events)<self.max_events:self.events.append(record)
                else:self.collisions_dropped+=1
            except Exception as ex:
                if len(self.errors)<6:self.errors.append(type(ex).__name__+":"+str(ex)[:140])

    def result(self):
        complete=not self.errors and self.collisions_dropped==0
        return dict(schema_version=1,source="native omni.physx subscribe_contact_report_events",
            physically_read_only=True, no_sensor_view_created=True,
            no_report_api_authored=True, report_api_present_on_original_bodies=self.report_apis,
            callback_calls=self.callback_calls,native_headers=self.native_headers,
            saved_interesting_events=self.events,dropped_events=self.collisions_dropped,
            errors=self.errors,report_capture_complete=complete,
            wrist_hip_direct_path_events=sum(e["candidate_wrist_hip"] for e in self.events),
            pelvis_right_hip_direct_path_events=sum(e["candidate_pelvis_right_hip"] for e in self.events),
            rigid_pair_certified=bool(complete and any(e["candidate_wrist_hip"] and
                e["reported_points"]>0 and
                any(cp["point_w"] is not None and cp["normal_w"] is not None
                    for cp in e["point_sample"]) for e in self.events)),
            limitation="No events cannot prove non-contact; callback availability and header buffer integrity must be checked with parity.")
