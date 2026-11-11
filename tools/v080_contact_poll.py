#!/usr/bin/env python3
"""Read-only PhysX existing contact-report polling on original 200Hz G1.

Never creates filtered views, applies USD APIs, changes contact thresholds,
mutates controls, or infers collider contacts from equal-opposite net forces.
Only known native reports with complete witness are eligible for certification.
"""
from __future__ import annotations
import math

PAIR=("left_wrist_yaw_link","left_hip_roll_link")

def vec3(value):
    try:
        xs=[float(x) for x in value]
        if len(xs)!=3 or not all(math.isfinite(x) for x in xs):
            return None
        return xs
    except (TypeError,ValueError):
        return None

def witness(event):
    """Require same native header with two named actors/colliders and impulse."""
    if len(event.get("actor_paths",[]))!=2 or len(event.get("collider_paths",[]))!=2:
        return False
    # Distinct native contact actors or collider slots, never both body names
    # inside a single string or across mismatched actor/collider roles.
    def distinct_pair(a,b):
        return ((PAIR[0] in a and PAIR[1] in b) or
                (PAIR[1] in a and PAIR[0] in b))
    if not (distinct_pair(*event["actor_paths"]) or
            distinct_pair(*event["collider_paths"])):
        return False
    for c in event.get("contacts",[]):
        if c.get("point_w") is None or c.get("normal_w") is None:
            continue
        imp=c.get("impulse_w")
        scalar=c.get("impulse_scalar")
        positive=(imp is not None and math.sqrt(sum(x*x for x in imp))>0)
        positive|=isinstance(scalar,(int,float)) and scalar>0
        if positive:
            return True
    return False

class ExistingNativeContactPoll:
    def __init__(self,env,start=4.70,end=5.50,max_events=500):
        import omni.physx,omni.usd
        from pxr import PhysicsSchemaTools,PhysxSchema
        self.env=env
        self.interface=omni.physx.get_physx_simulation_interface()
        self.decode=PhysicsSchemaTools.intToSdfPath
        self.start=float(start);self.end=float(end)
        self.max_events=max_events
        self.frames_seen=0;self.poll_calls=0;self.headers=0
        self.contact_points=0;self.errors=[];self.events=[]
        self.dropped_events=0
        self.times_polled=[]
        self.report_apis={}
        stage=omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.GetName() in PAIR:
                self.report_apis[str(prim.GetPath())]=bool(
                    prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
    def capture(self,dt):
        if dt<=0:return
        if abs(dt-0.005)>1e-6:raise RuntimeError("ORIGINAL_200HZ_REQUIRED")
        self.frames_seen+=1
        t=float(self.env.sim.current_time)
        if not self.start<=t<=self.end:return
        if self.errors:return
        try:
            headers,contacts=self.interface.get_contact_report()
            self.poll_calls+=1
            self.times_polled.append(round(t,6))
            self.headers+=len(headers)
            for h in headers:
                self.contact_points+=int(getattr(h,"num_contact_data",0))
                paths=[]
                for fld in ("actor0","actor1","collider0","collider1"):
                    paths.append(str(self.decode(getattr(h,fld))))
                if not any(any(target in p for p in paths) for target in PAIR):
                    continue
                offset=int(getattr(h,"contact_data_offset",0))
                n=int(getattr(h,"num_contact_data",0))
                cs=[]
                for c in contacts[offset:offset+min(n,30)]:
                    raw=getattr(c,"impulse",None)
                    iv=vec3(raw)
                    scalar=float(raw) if isinstance(raw,(int,float)) else None
                    cs.append({"point_w":vec3(getattr(c,"position",None)),
                        "normal_w":vec3(getattr(c,"normal",None)),
                        "impulse_w":iv,"impulse_scalar":scalar,
                        "separation_m":float(getattr(c,"separation"))
                        if hasattr(c,"separation") else None})
                record={"time_s":t,"actor_paths":paths[:2],
                    "collider_paths":paths[2:],"reported_points":n,
                    "contacts":cs}
                if len(self.events)<self.max_events:
                    self.events.append(record)
                else:self.dropped_events+=1
        except Exception as e:
            if len(self.errors)<5:self.errors.append(type(e).__name__+":"+str(e)[:200])
    def result(self):
        hits=[e for e in self.events if witness(e)]
        return {"version":"v0.1.80","mode":"existing_PhysX_get_contact_report_poll",
            "read_only":True,"applied_report_api":False,
            "created_contact_views":False,"controller_writes":False,
            "physics_dt_s":0.005,"window_s":[self.start,self.end],
            "frames_seen":self.frames_seen,"poll_calls":self.poll_calls,
            "native_headers":self.headers,"reported_native_contact_points":self.contact_points,
            "report_api_present_on_original_bodies":self.report_apis,
            "events_with_pair_witness":len(hits),
            "wrist_hip_direct_contact_certified":bool(
                hits and not self.errors and not self.dropped_events),
            "interesting_native_events":self.events,"dropped_events":self.dropped_events,
            "errors":self.errors,"times_polled_first_last":
            [self.times_polled[0],self.times_polled[-1]] if self.times_polled else [],
            "negative_interpretation":"UNKNOWN_NO_NATIVE_WITNESS" if not hits
                else "DIRECT_COLLIDER_PAIR_WITH_CONTACT_GEOMETRY",
            "limitation":"Absence of report headers is UNKNOWN, not absence of collision."}
