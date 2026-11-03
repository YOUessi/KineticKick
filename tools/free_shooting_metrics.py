"""Numerical acceptance rules shared by live and batch physical trials."""
from __future__ import annotations

import numpy as np


def goal_crossing(previous, current, *, goal, forward, width, height, radius):
    """The trailing edge crosses forward, with the full sphere inside the opening."""
    a, b = np.asarray(previous, float), np.asarray(current, float)
    direction = np.asarray(forward, float)
    length = np.linalg.norm(direction)
    if length == 0 or not np.all(np.isfinite([*a, *b, *direction])):
        return False
    direction /= length
    center = np.asarray(goal, float)[:2]
    da, db = float((a[:2] - center) @ direction), float((b[:2] - center) @ direction)
    if not da < radius <= db or db <= da:
        return False
    crossing = a + (b - a) * ((radius - da) / (db - da))
    lateral = np.array([-direction[1], direction[0]])
    return bool(abs((crossing[:2] - center) @ lateral) <= width / 2 - radius
                and radius - 1e-4 <= crossing[2] <= height - radius)


def contact_surface(normal, *, foot="right", minimum_cosine=.75):
    """Classify the outward contact normal in the physical foot frame.

    A force-derived normal is a surface proxy, not a precise contact-point measurement.
    Ambiguous diagonals and toe/sole contacts remain separate from the three styles.
    """
    vector = np.asarray(normal, float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        return "ambiguous"
    vector /= norm
    axis = int(np.argmax(abs(vector)))
    if abs(vector[axis]) < minimum_cosine:
        return "ambiguous"
    if axis == 0:
        return "toe" if vector[0] > 0 else "heel"
    if axis == 2:
        return "instep" if vector[2] > 0 else "sole"
    inside_sign = 1 if foot == "right" else -1
    return "inside" if vector[1] * inside_sign > 0 else "outside"


def paid_contact_patch(point, normal, *, foot="right"):
    """Check a PhysX contact against the audited PAiD cylinder-foot envelope.

    The seven cylinders span x=[-.054,.132], y=[-.036,.036],
    z=[-.035,-.015] in the ankle frame. A 15 mm tolerance allows the
    solver contact offset and motion within one 5 ms substep. Direction and
    side must agree; arbitrary outside points cannot certify a surface.
    """
    if point is None:
        return "ambiguous"
    p = np.asarray(point, float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return "ambiguous"
    low, high = np.array([-.054, -.036, -.035]) - .015, np.array([.132, .036, -.015]) + .015
    if np.any(p < low) or np.any(p > high):
        return "ambiguous"
    surface = contact_surface(normal, foot=foot)
    inside_sign = 1 if foot == "right" else -1
    if surface == "inside" and p[1] * inside_sign <= 0:
        return "ambiguous"
    if surface == "outside" and p[1] * inside_sign >= 0:
        return "ambiguous"
    if surface == "instep" and p[2] < -.03:
        return "ambiguous"
    return surface


class ShotTracker:
    def __init__(self, *, initial_root, initial_ball, goal, forward, style,
                 width=1.6, height=1.1, radius=.11):
        self.initial_root = np.asarray(initial_root, float)
        self.initial_ball = np.asarray(initial_ball, float)
        self.previous_ball = self.initial_ball.copy()
        self.goal_args = dict(goal=goal, forward=forward, width=width, height=height, radius=radius)
        self.style = style
        self.contact_time = None
        self.surface = None
        self.approached = False
        self.scored = False
        self.failed = False
        self.standing_start = None
        self.standing_seconds = 0.0
        self.longest_standing_seconds = 0.0
        self.peak_force = 0.0
        self.last_time = None

    def update(self, sample):
        now = float(sample["time_s"])
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("Trial samples must have strictly increasing times")
        self.last_time = now
        root = np.asarray(sample["root_xyz"], float)
        ball = np.asarray(sample["ball_xyz"], float)
        tilt = float(sample["root_tilt_deg"])
        speed = float(sample["root_speed_xy"])
        force = float(sample["contact_force_N"])
        finite = np.all(np.isfinite([*root, *ball, now, tilt, speed, force]))
        self.failed |= bool(sample["terminated"] or not finite or root[2] < .45 or tilt > 60)
        if self.failed:
            self.standing_start = None
            self.standing_seconds = 0.0
            return
        self.peak_force = max(self.peak_force, force)
        if force > 2 and self.contact_time is None:
            self.contact_time = now
            self.surface = paid_contact_patch(sample.get("contact_point_foot"), sample["contact_normal_foot"],
                                              foot=sample.get("foot", "right"))
            initial_distance = np.linalg.norm(self.initial_ball[:2] - self.initial_root[:2])
            distance = np.linalg.norm(ball[:2] - root[:2])
            moved = np.linalg.norm(root[:2] - self.initial_root[:2])
            self.approached = bool(moved >= .25 and initial_distance - distance >= .20)
        if self.contact_time is not None:
            self.scored |= goal_crossing(self.previous_ball, ball, **self.goal_args)
            stable = (root[2] >= .55 and tilt <= 25 and speed <= .30 and force <= 2
                      and sample["feet_support_N"] >= 80
                      and sample["nonfoot_contact_N"] < 20)
            if stable:
                if self.standing_start is None:
                    self.standing_start = now
                self.standing_seconds = now - self.standing_start
                self.longest_standing_seconds = max(self.longest_standing_seconds, self.standing_seconds)
            else:
                self.standing_start = None
                self.standing_seconds = 0.0
        self.previous_ball = ball.copy()

    def summary(self):
        recovered = self.standing_seconds >= 1.0 - 1e-8 and not self.failed
        surface_match = self.surface == self.style
        return dict(success=bool(self.approached and self.scored and recovered and surface_match and not self.failed),
                    approached=self.approached, contact_time_s=self.contact_time,
                    measured_surface=self.surface, requested_style=self.style,
                    surface_match=surface_match, goal=self.scored, recovered=recovered,
                    standing_seconds=self.standing_seconds,
                    longest_standing_seconds=self.longest_standing_seconds,
                    failed=self.failed, peak_filtered_contact_force_N=self.peak_force)
