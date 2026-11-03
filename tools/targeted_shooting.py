"""Fixed-goal aim and outgoing-speed evidence for physical shooting.

This module is safe to import before Isaac Sim starts: it uses the standard
library only, so it cannot pre-load an incompatible numerical runtime.
"""
from __future__ import annotations

import math


class TargetShotTracker:
    def __init__(self, *, target_x, speed_mps, goal_y=-3.7):
        self.target_x, self.speed_mps, self.goal_y = map(float, (target_x, speed_mps, goal_y))
        if not all(math.isfinite(x) for x in (self.target_x, self.speed_mps, self.goal_y)) or self.speed_mps <= 0:
            raise ValueError('Target and goal must be finite and requested speed must be positive')
        self.previous = None
        self.last_time = None
        self.contact_time = None
        self.crossing = None
        self.outgoing_speed = None
        self.failed = False

    def update(self, sample):
        now = float(sample['time_s'])
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Trial samples must have strictly increasing times')
        self.last_time = now
        position, velocity = sample['ball_xyz'], sample['ball_velocity']
        force = float(sample['contact_force_N'])
        valid = (len(position) == len(velocity) == 3
                 and all(math.isfinite(x) for x in (*position, *velocity, now, force)))
        self.failed |= not valid or bool(sample['terminated'])
        if self.failed:
            return
        if force > 2 and self.contact_time is None:
            self.contact_time = now
        if self.contact_time is not None and self.previous is not None:
            before = self.previous
            measure_time = self.contact_time + .10
            if self.outgoing_speed is None and before['time_s'] <= measure_time <= now + 1e-10:
                fraction = (measure_time - before['time_s']) / (now - before['time_s'])
                v = [a + (b-a)*fraction for a, b in zip(before['ball_velocity'], velocity)]
                self.outgoing_speed = math.hypot(v[0], v[1])
            a, b = before['ball_xyz'], position
            if self.crossing is None and a[1] > self.goal_y >= b[1]:
                fraction = (self.goal_y - a[1]) / (b[1] - a[1])
                self.crossing = dict(time_s=before['time_s'] + (now-before['time_s'])*fraction,
                    position=[x + (y-x)*fraction for x, y in zip(a, b)],
                    velocity=[x + (y-x)*fraction for x, y in zip(before['ball_velocity'], velocity)])
        self.previous = sample

    def summary(self):
        aim_error = abs(self.crossing['position'][0] - self.target_x) if self.crossing else None
        speed_error = abs(self.outgoing_speed - self.speed_mps) if self.outgoing_speed is not None else None
        target_hit = aim_error is not None and aim_error <= .15 + 1e-10
        speed_match = speed_error is not None and speed_error <= max(.20, .12*self.speed_mps) + 1e-10
        return dict(target_x_m=self.target_x, requested_speed_mps=self.speed_mps,
                    crossing=self.crossing, aim_error_m=aim_error,
                    outgoing_speed_mps=self.outgoing_speed, speed_error_mps=speed_error,
                    target_hit=target_hit, speed_match=speed_match,
                    target_success=bool(target_hit and speed_match and not self.failed))
