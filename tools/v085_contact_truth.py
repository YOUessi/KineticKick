#!/usr/bin/env python3
"""v85 observational native contact audit: existing PhysX reporting only.

Does not author USD schemas, change contact thresholds, or create tensor views.
A zero event stream is observer INDETERMINATE, never proof of non-collision.
"""
from __future__ import annotations
import math
from v080_contact_poll import witness, vec3

LEFT = ("left_wrist_yaw_link", "left_hip_roll_link")
POS = ("right_ankle_roll_link", "SoccerBall")
WATCH = LEFT + POS + ("left_ankle_roll_link", "right_ankle_roll_link")

def pair_in_paths(event, pair):
    a = event.get("actor_paths", ())
    b = event.get("collider_paths", ())
    def match(paths):
        if len(paths) != 2: return False
        return (pair[0] in paths[0] and pair[1] in paths[1] or
                pair[1] in paths[0] and pair[0] in paths[1])
    return match(a) or match(b)

def valid_positive(event):
    if len(event.get("actor_paths", ())) != 2 or len(event.get("collider_paths", ())) != 2:
        return False
    for point in event.get("contacts", ()):
        if point.get("point_w") is None or point.get("normal_w") is None: continue
        normal = point["normal_w"]
        if sum(float(x)**2 for x in normal) < 0.01: continue
        impulse = point.get("impulse_w")
        scalar = point.get("impulse_scalar")
        if impulse is not None and sum(float(x)**2 for x in impulse) > 0: return True
        if isinstance(scalar, (int, float)) and math.isfinite(scalar) and scalar > 0: return True
    return False

class ExistingNativeContactPoll:
    def __init__(self, env, start=0.7, end=6.2, max_events=200):
        import omni.physx, omni.usd
        from pxr import PhysxSchema, PhysicsSchemaTools, UsdPhysics
        self.env = env
        self.interface = omni.physx.get_physx_simulation_interface()
        self.decode = PhysicsSchemaTools.intToSdfPath
        self.start, self.end = float(start), float(end)
        self.poll_calls = self.callback_calls = self.native_headers = 0
        self.poll_headers = self.callback_headers = self.native_points = 0
        self.frames_seen = 0
        self.times = []
        self.events = []
        self.max_events = max_events
        self.dropped_events = 0
        self.errors = []
        self.inventory = {}
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            name = prim.GetName()
            if name not in WATCH: continue
            path = str(prim.GetPath())
            has_api = bool(prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
            threshold = None
            pair_targets = None
            if has_api:
                api = PhysxSchema.PhysxContactReportAPI(prim)
                attr = api.GetThresholdAttr()
                if attr and attr.IsValid():
                    raw = attr.Get()
                    try: threshold = float(raw) if raw is not None else None
                    except (TypeError, ValueError): threshold = None
                rel = api.GetReportPairsRel()
                if rel and rel.IsValid():
                    pair_targets = [str(v) for v in rel.GetTargets()]
            self.inventory[path] = dict(
                reported_api=has_api,
                threshold_N=threshold,
                report_pairs_targets=pair_targets,
                rigid_body_api=bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
                collision_api=bool(prim.HasAPI(UsdPhysics.CollisionAPI)))
        # A callback is observational. No schema/application or view creation.
        self.subscription = self.interface.subscribe_contact_report_events(self._callback)

    def _decode_header(self, h, data, t, source):
        paths = [str(self.decode(getattr(h, key))) for key in
                 ("actor0", "actor1", "collider0", "collider1")]
        n = int(getattr(h, "num_contact_data", 0))
        offset = int(getattr(h, "contact_data_offset", 0))
        self.native_points += n
        if not any(any(target in value for target in WATCH) for value in paths):
            return
        points = []
        for point in data[offset:offset + min(n, 12)]:
            impulse = getattr(point, "impulse", None)
            vect = vec3(impulse)
            scalar = float(impulse) if isinstance(impulse, (float, int)) else None
            points.append(dict(
                point_w=vec3(getattr(point, "position", None)),
                normal_w=vec3(getattr(point, "normal", None)),
                impulse_w=vect, impulse_scalar=scalar))
        event = dict(time_s=round(float(t), 6), source=source,
                     actor_paths=paths[:2], collider_paths=paths[2:],
                     reported_points=n, contacts=points)
        if len(self.events) < self.max_events:
            self.events.append(event)
        else:
            self.dropped_events += 1

    def _callback(self, headers, contacts):
        self.callback_calls += 1
        self.callback_headers += len(headers)
        t = float(self.env.sim.current_time)
        for h in headers:
            try:
                self._decode_header(h, contacts, t, "callback")
            except Exception as exc:
                if len(self.errors) < 10:
                    self.errors.append("callback:" + repr(exc)[:200])

    def capture(self, dt):
        if dt <= 0: return
        if abs(float(dt)-.005) > 1e-6: raise RuntimeError("V085_NOT_200HZ")
        self.frames_seen += 1
        t = float(self.env.sim.current_time)
        if not self.start <= t <= self.end: return
        self.poll_calls += 1
        try:
            headers, points = self.interface.get_contact_report()
            self.poll_headers += len(headers)
            if len(self.times) < 4: self.times.append(round(t, 4))
            for h in headers:
                self._decode_header(h, points, t, "poll")
        except Exception as exc:
            if len(self.errors) < 10:
                self.errors.append("poll:" + repr(exc)[:200])

    def result(self):
        wh = [x for x in self.events if pair_in_paths(x, LEFT) and valid_positive(x)]
        foot = [x for x in self.events if pair_in_paths(x, POS) and valid_positive(x)]
        return dict(version="v0.1.85", mode="existing_PhysX_callback_plus_poststep_poll",
                    read_only=True, no_usd_writes=True, no_view_creation=True,
                    no_controller_writes=True, physics_dt_s=.005,
                    native_headers=self.poll_headers+self.callback_headers,
                    callback_calls=self.callback_calls, callback_headers=self.callback_headers,
                    poll_calls=self.poll_calls, poll_headers=self.poll_headers,
                    native_point_count=self.native_points, window_s=[self.start, self.end],
                    frames_seen=self.frames_seen, first_poll_times=self.times,
                    report_api_inventory=self.inventory, dropped_events=self.dropped_events,
                    errors=self.errors, saved_interesting_events=self.events,
                    direct_wrist_hip_witness_count=len(wh),
                    wrist_hip_direct_contact_certified=bool(wh and not self.errors and not self.dropped_events),
                    native_foot_ball_positive_control=bool(foot and not self.errors),
                    observer_calibrated_for_contact_absence=bool(foot and not self.errors and not self.dropped_events),
                    negative_interpretation=("NATIVE_PAIR_CONTACT" if wh else
                        "UNKNOWN: zero direct pair reports cannot exclude self-collision"),
                    limitations="A stage API flag is not proof of active contact reporting; simultaneous equal-opposite force is not collider-pair evidence.")
