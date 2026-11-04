"""Explicit per-trial physical reference selection and measured sequence labels.

Safe to import before Isaac Sim. Labels describe observed execution; they do not
claim a controller transition or a successful shot merely because time elapsed.
"""
from pathlib import Path
import math


def approach_offset(entry, now):
    """World-frame virtual approach waypoint; physical ball stays unchanged."""
    offset = entry.get('approach_offset_xy', [0., 0.])
    fade = entry.get('approach_offset_fade_s', [2.5, 3.4])
    if (len(offset) != 2 or len(fade) != 2 or not all(math.isfinite(x) for x in [*offset, *fade, now])
            or not 0 <= fade[0] < fade[1] or any(abs(x) > .3 for x in offset)):
        raise ValueError('Finite approach offset within 0.3 m and ordered fade times required')
    phase = min(1., max(0., (now-fade[0])/(fade[1]-fade[0])))
    weight = 1. - phase*phase*(3.-2.*phase)
    return [offset[0]*weight, offset[1]*weight, 0.]


def resolve_motion_files(trials, root, default):
    selected = [trial.get('motion_file') for trial in (trials or [])]
    if any(value is not None for value in selected):
        if any(not isinstance(value, str) or not value for value in selected):
            raise ValueError('Specify a motion_file for every trial')
    else:
        selected = [default]
    paths = []
    for value in selected:
        path = Path(value)
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve()
        if path.suffix != '.npz' or not path.is_file():
            raise ValueError(f'Missing NPZ motion reference: {path}')
        paths.append(str(path))
    return paths


def phase_label(now, contact_time, recovered, failed):
    if failed:
        return 'failed'
    if contact_time is None:
        return 'approach'
    if now - contact_time < .25:
        return 'contact_follow_through'
    return 'stable' if recovered else 'recover'


class CoordinationTargets:
    """Support-gated preparation, extension and smooth release to the policy.

    Drives ordinary joint targets only. All unlisted joints remain controlled by
    the recurrent whole-body policy. A requested knee pattern is experimental
    until measured foot-velocity contributions and full-sequence gates pass.
    """
    def __init__(self, entries, joint_names, scale, offset, limits):
        import numpy as np
        self.np = np
        shape = (len(entries), len(joint_names))
        self.scale = np.broadcast_to(np.asarray(scale), shape)
        self.offset = np.broadcast_to(np.asarray(offset), shape)
        self.limits = np.broadcast_to(np.asarray(limits), (*shape, 2))
        self.entries = entries
        self.indices, self.preparation, self.strike = [], [], []
        self.started = [None] * len(entries)
        self.sources = [None] * len(entries)
        self.phases = ['idle'] * len(entries)
        self.velocity_targets = np.zeros(shape)
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError('Positive action scale required')
        for entry in entries:
            if not entry:
                self.indices.append([]); self.preparation.append([]); self.strike.append([])
                continue
            times = entry.get('times_s', [])
            prep, strike = entry.get('prepare', {}), entry.get('strike', {})
            if (len(times) != 3 or not np.all(np.isfinite(times)) or not 0 < times[0] < times[1] < times[2] <= 2.
                    or not np.isfinite(entry.get('start_frame', -1)) or entry.get('start_frame', -1) < 0
                    or not prep or set(prep) != set(strike) or not set(prep).issubset(joint_names)
                    or not np.all(np.isfinite([*prep.values(), *strike.values()]))):
                raise ValueError('Invalid coordination joints, targets or timing')
            for key, default in (('minimum_support_N', 80.), ('maximum_ball_distance_m', .8)):
                if not np.isfinite(entry.get(key, default)) or entry.get(key, default) <= 0:
                    raise ValueError('Positive support and ball distance gates required')
            gain = entry.get('velocity_feedforward_gain', 1.)
            if not np.isfinite(gain) or not 0 <= gain <= 2.:
                raise ValueError('Velocity feedforward gain must be within 0..2')
            ids = [joint_names.index(name) for name in prep]
            self.indices.append(ids)
            self.preparation.append(np.array(list(prep.values())))
            self.strike.append(np.array([strike[name] for name in prep]))

    def apply(self, actions, now, frames, support, ball_distance):
        np = self.np
        result = np.array(actions, dtype=float, copy=True)
        self.velocity_targets[:] = 0.
        if not np.isfinite(now) or not np.all(np.isfinite(result)):
            raise ValueError('Finite time and actions required')
        for i, entry in enumerate(self.entries):
            if not entry:
                continue
            ids = self.indices[i]
            current = result[i, ids] * self.scale[i, ids] + self.offset[i, ids]
            if self.started[i] is None:
                if not (frames[i] >= entry['start_frame'] and support[i] >= entry.get('minimum_support_N', 80.)
                        and ball_distance[i] <= entry.get('maximum_ball_distance_m', .8)):
                    continue
                self.started[i], self.sources[i] = now, current.copy()
            t = now - self.started[i]
            a, b, end = entry['times_s']
            if t < 0:
                raise ValueError('Coordination time cannot move backwards')
            if t < a:
                phase, start, finish = t/a, self.sources[i], self.preparation[i]
                duration = a
                self.phases[i] = 'prepare'
            elif t < b:
                phase, start, finish = (t-a)/(b-a), self.preparation[i], self.strike[i]
                duration = b-a
                self.phases[i] = 'strike'
            elif t < end:
                phase, start, finish = (t-b)/(end-b), self.strike[i], current
                duration = end-b
                self.phases[i] = 'follow_through'
            else:
                self.phases[i] = 'released'
                continue
            blend = phase * phase * (3 - 2 * phase)
            target = np.clip(start + blend * (finish-start), self.limits[i, ids, 0], self.limits[i, ids, 1])
            if entry.get('velocity_feedforward', False) and t < b:
                velocity = entry.get('velocity_feedforward_gain', 1.)*6*phase*(1-phase)*(finish-start)/duration
                unclipped = start + blend * (finish-start)
                self.velocity_targets[i, ids] = np.where(target == unclipped, np.clip(velocity, -12., 12.), 0.)
            result[i, ids] = (target-self.offset[i, ids]) / self.scale[i, ids]
        return result.astype(np.float32)
