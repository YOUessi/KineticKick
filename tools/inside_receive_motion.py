"""Predict a physical ball's arrival at the original right-foot medial capsule.

Only joint targets are produced. The ball and floating base are never written.
Ground and aerial modes use their respective horizontal/ballistic predictions.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from cushion_motion import hermite


class InsideReceiveMotion:
    def __init__(self, plan):
        self.plan = plan
        self.cfg = plan['cushion']
        self.initial_foot = np.array(plan['feet']['right'], float)
        self.initial_root = np.array(plan['root_pose'], float)
        self.rotation = self.initial_foot[:3, :3]
        self.normal = self.rotation @ np.array(plan['contact_normal_local'])
        self.offset = self.rotation @ np.array(plan['contact_center_local']) + .12*self.normal
        self.impact = self.initial_foot[:3, 3] + self.offset
        self.contact_time = None
        self.hit_position = self.hit_velocity = None
        self.predictions = []
        self.angular_velocity = np.zeros(3)
        self.absorbed_displacement = 0.

    def update_prediction(self, now, ball, velocity, touched=False):
        if touched or (self.contact_time is not None and now >= self.contact_time-self.cfg['freeze_prediction_lead_s']):
            return
        ball, velocity = np.array(ball, float), np.array(velocity, float)
        acceleration = np.array([0., 0., -9.81 if self.plan['receive_mode'] == 'aerial' else 0.])
        # Allow the foot to retreat before impact. Holding identical start/hit
        # positions with positive end velocity forces a counter-motion first.
        shift = .5*self.cfg['velocity_fraction']*self.cfg['trajectory_lead_s']
        nv, na = float(velocity @ self.normal), float(acceleration @ self.normal)
        aa, bb = .5*na, nv-shift*na
        cc = float((ball-self.impact) @ self.normal)-shift*nv
        if abs(aa) < 1e-10:
            roots = [-cc/bb] if abs(bb) > 1e-10 else []
        else:
            discriminant = bb*bb-4*aa*cc
            roots = [] if discriminant < 0 else [(-bb+d*np.sqrt(discriminant))/(2*aa) for d in (-1,1)]
        roots = [dt for dt in roots if dt >= 0 and nv+na*dt < -1e-4]
        if not roots:
            return
        dt = min(roots)
        impact = ball+velocity*dt+.5*acceleration*dt*dt
        incoming = velocity+acceleration*dt
        if self.plan['receive_mode'] == 'ground':
            impact[2] = self.impact[2]
            incoming[2] = 0.
        self.hit_position = impact-self.offset
        self.hit_velocity = incoming*self.cfg['velocity_fraction']
        self.contact_time = now+dt
        self.predictions.append(dict(time_s=now, expected_contact_time_s=self.contact_time,
            incoming_velocity=incoming.tolist(), foot_hit_position=self.hit_position.tolist(),
            foot_hit_velocity=self.hit_velocity.tolist(), normal=self.normal.tolist()))

    def targets(self, now):
        foot = self.initial_foot.copy()
        shift = -self.normal*self.absorbed_displacement
        root = np.eye(4)
        root[:3, 3] = self.initial_root[:3]
        root[:3, :3] = Rotation.from_quat(self.initial_root[[4, 5, 6, 3]]).as_matrix()
        if self.contact_time is None:
            foot[:3, 3] += shift
            return foot, np.zeros(3), root, 'prepare'
        start, end = self.contact_time-self.cfg['trajectory_lead_s'], self.contact_time
        if now < start:
            foot[:3, 3] += shift
            return foot, np.zeros(3), root, 'prepare'
        if now <= end:
            foot[:3, 3], speed = hermite(self.initial_foot[:3, 3], self.hit_position,
                np.zeros(3), self.hit_velocity, end-start, now-start)
            foot[:3, 3] += shift
            return foot, speed, root, 'anticipate'
        duration = self.cfg['follow_seconds']
        dt = min(now-end, duration)
        foot[:3, 3] = self.hit_position+self.hit_velocity*(dt-.5*dt*dt/duration)
        speed = self.hit_velocity*(1-dt/duration)
        foot[:3, 3] += shift
        return foot, speed, root, 'cushion' if dt < duration else 'settle'

    def yield_to_contact(self, target, actual_position, force):
        """Keep impact-induced retreat instead of springing back into the ball.

        A bounded one-way position compliance acts through ordinary joint motors.
        It does not modify the ball, contact forces or robot physical state.
        """
        if force <= .2:
            return target
        away = -self.normal
        displacement = max(0., float((actual_position-target[:3, 3]) @ away))
        increment = min(displacement, max(0., self.cfg.get('absorption_limit_m', .04)-self.absorbed_displacement))
        self.absorbed_displacement += increment
        target = target.copy()
        target[:3, 3] += increment*away
        return target
