"""Support-gated velocity targets for bounded strike experiments.

These are ordinary PD joint velocity targets in rad/s. This module neither
changes position/action targets nor writes robot or ball state. The caller must
pass LEFT-foot ground-filtered normal support and enforce robot velocity limits
after combining these targets with any other velocity feedforward.
"""
import numpy as np


class StrikeVelocityPulse:
    """Cosine rise, peak plateau and cosine fall in reference-frame units.

    Each entry is None, or a dict with window=[start, rise, fall, end], a
    velocities={joint_name: signed_rad_per_second} map, and optional
    minimum_support_N (default 80). Support is rechecked at every apply call;
    there is no latched activation or hidden phase state.
    """

    def __init__(self, entries, joint_names):
        if (not isinstance(entries, (list, tuple)) or not entries
                or not isinstance(joint_names, (list, tuple)) or not joint_names
                or any(not isinstance(n, str) or not n for n in joint_names)
                or len(set(joint_names)) != len(joint_names)):
            raise ValueError('Nonempty entries and unique named joints required')
        self._count = len(entries)
        self._peaks = np.zeros((self._count, len(joint_names)), dtype=np.float64)
        self._windows = np.zeros((self._count, 4), dtype=np.float64)
        self._minimum = np.zeros(self._count, dtype=np.float64)
        self._enabled = np.zeros(self._count, dtype=bool)
        for i, entry in enumerate(entries):
            if entry is None:
                continue
            if (not isinstance(entry, dict)
                    or set(entry) - {'window', 'velocities', 'minimum_support_N'}):
                raise ValueError('Expected a velocity pulse entry or None')
            try:
                window = np.asarray(entry.get('window', []), dtype=np.float64)
                minimum = float(entry.get('minimum_support_N', 80.))
                velocities = entry.get('velocities', {})
                if (window.shape != (4,) or not np.all(np.isfinite(window))
                        or window[0] < 0 or not np.all(np.diff(window) > 0)):
                    raise ValueError('Finite ordered nonnegative frame window required')
                if not np.isfinite(minimum) or minimum <= 0:
                    raise ValueError('Positive finite minimum support required')
                if (not isinstance(velocities, dict) or not velocities
                        or not set(velocities).issubset(joint_names)):
                    raise ValueError('Velocity targets must name existing joints')
                values = np.asarray(list(velocities.values()), dtype=np.float64)
                if values.shape != (len(velocities),) or not np.all(np.isfinite(values)) or np.any(abs(values) > 12.):
                    raise ValueError('Finite velocities within +/-12 rad/s required')
            except (TypeError, OverflowError) as exc:
                raise ValueError('Invalid velocity pulse parameters') from exc
            self._windows[i] = window
            self._minimum[i] = minimum
            self._peaks[i, [joint_names.index(n) for n in velocities]] = values
            self._enabled[i] = True

    def apply(self, frames, support):
        """Return fresh (num_entries, num_joints) float32 velocity targets.

        Both inputs must be finite vectors of length num_entries. Frames must
        be nonnegative. A support value below its threshold yields zeros,
        including when support disappears halfway through a pulse.
        """
        try:
            frames = np.asarray(frames, dtype=np.float64)
            support = np.asarray(support, dtype=np.float64)
        except (TypeError, OverflowError) as exc:
            raise ValueError('Finite frame and support vectors required') from exc
        if (frames.shape != (self._count,) or support.shape != (self._count,)
                or not np.all(np.isfinite(frames)) or not np.all(np.isfinite(support))
                or np.any(frames < 0)):
            raise ValueError('Finite frame/support vectors matching trial count required')
        result = np.zeros_like(self._peaks, dtype=np.float32)
        for i in np.flatnonzero(self._enabled & (support >= self._minimum)):
            start, rise, fall, end = self._windows[i]
            frame = frames[i]
            if frame <= start or frame >= end:
                continue
            if frame < rise:
                phase = (frame-start)/(rise-start)
                weight = .5*(1.-np.cos(np.pi*phase))
            elif frame <= fall:
                weight = 1.
            else:
                phase = (frame-fall)/(end-fall)
                weight = .5*(1.+np.cos(np.pi*phase))
            result[i] = self._peaks[i]*weight
        return result
