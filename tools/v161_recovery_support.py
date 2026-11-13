"""Double-support inverse dynamics for a measured loaded terminal stance.

The CPU dynamics worker uses Tang's existing Pinocchio ABI. The simulator only
receives 29 position targets, through its unchanged implicit drives.
"""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from v161_recovery_stance import smootherstep


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


def make_controller(ref, schema, cfg):
    final_frame = len(ref['joint_pos']) - 1
    registered = 0
    start = None
    child = None

    def close():
        if child is not None:
            child.stdin.close()
            child.wait(timeout=8)

    atexit.register(close)

    def command(ctx, target):
        nonlocal start, registered, child
        if start is None:
            first = ctx['first_contact_s']
            stable = (first is not None and ctx['t'] - first >= cfg['minimum_followthrough_s']
                      and min(ctx['foot_load']) >= cfg['minimum_bilateral_load_N']
                      and ctx['root_pos'][2] >= cfg['minimum_root_height_m']
                      and np.linalg.norm(ctx['root_velocity'][:2]) <= cfg['entry_speed_mps']
                      and rotation(ctx['root_quat'])[2, 2] >= np.cos(np.deg2rad(cfg['entry_tilt_deg'])))
            registered = registered + 1 if stable else 0
            if ctx['frame'] < final_frame or registered < cfg['stable_steps']:
                return target
            start = ctx['t']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            args = [cfg['dynamics_python'], '-s', '-u', str(Path(__file__).resolve()), '--worker']
            child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, env=env)
            child.stdin.write(json.dumps([cfg, schema['joint_names']]) + '\n')
        fields = ('q', 'dq', 'root_pos', 'root_quat', 'root_velocity',
                  'com', 'com_velocity', 'foot_load', 'limits')
        row = {key: np.asarray(ctx[key]).tolist() for key in fields}
        child.stdin.write(json.dumps(row, allow_nan=False) + '\n')
        child.stdin.flush()
        position = np.asarray(json.loads(child.stdout.readline()))
        weight = smootherstep((ctx['t'] - start) / cfg['blend_s'])
        return (1. - weight) * np.asarray(target) + weight * position if weight else target
    return command


class DoubleSupport:
    def __init__(self, cfg, names):
        import pinocchio as pin
        self.pin = pin
        self.cfg = cfg
        self.model = pin.buildModelFromUrdf(cfg['urdf'], pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.qids = np.array([self.model.idx_qs[self.model.getJointId(name)] for name in names])
        self.vids = self.qids - 1
        if sorted(self.qids) != list(range(7, 36)):
            raise ValueError('Original robot and 29 target names disagree')
        self.feet = [self.model.getFrameId(side+'_ankle_roll_link') for side in ('left', 'right')]
        self.pelvis = self.model.getFrameId('pelvis')
        self.k = np.array(cfg['drive_stiffness'])
        self.d = np.array(cfg['drive_damping'])
        self.mass = pin.computeTotalMass(self.model)
        self.reference = None

    def step(self, row):
        pin, m, d, cfg = self.pin, self.model, self.data, self.cfg
        joints, dq = np.array(row['q']), np.array(row['dq'])
        root_quat = np.array(row['root_quat'])
        orientation = rotation(root_quat)
        root_twist = np.array(row['root_velocity']).reshape(2, 3)
        q = pin.neutral(m)
        q[:7] = np.r_[row['root_pos'], root_quat[[1, 2, 3, 0]]]
        q[self.qids] = joints
        v = np.zeros(m.nv)
        v[:6] = (root_twist @ orientation).ravel()
        v[self.vids] = dq
        zero = np.zeros(m.nv)
        pin.computeJointJacobians(m, d, q)
        pin.forwardKinematics(m, d, q, v, zero)
        pin.updateFramePlacements(m, d)
        lwa = pin.LOCAL_WORLD_ALIGNED
        feet = [d.oMf[frame].copy() for frame in self.feet]
        if self.reference is None:
            self.reference = (joints.copy(), np.array(row['com']), orientation.copy(), feet)
        joint_ref, com_ref, root_ref, foot_ref = self.reference

        jfoot = np.vstack([pin.getFrameJacobian(m, d, f, lwa) for f in self.feet])
        foot_errors = np.concatenate([np.r_[goal.translation - actual.translation,
                         pin.log3(goal.rotation @ actual.rotation.T)]
                         for goal, actual in zip(foot_ref, feet)])
        foot_bias = np.concatenate([pin.getFrameClassicalAcceleration(m, d, f, lwa).vector.copy()
                                    for f in self.feet])
        foot_acc = (cfg['foot_kp'] * foot_errors - cfg['foot_kd'] * (jfoot @ v) - foot_bias)
        # Hard stationary-foot acceleration. The SVD exposes only motions compatible with both feet.
        _, singular, vh = np.linalg.svd(jfoot, full_matrices=True)
        rank = int(np.sum(singular > 1e-6))
        free = vh[rank:].T
        foot_solution = np.linalg.lstsq(jfoot, foot_acc, rcond=1e-6)[0]

        jp = pin.getFrameJacobian(m, d, self.pelvis, lwa)[3:].copy()
        pbias = pin.getFrameClassicalAcceleration(m, d, self.pelvis, lwa).angular.copy()
        jc = pin.jacobianCenterOfMass(m, d, q).copy()
        pin.centerOfMass(m, d, q, v, zero)
        cbias = d.acom[0].copy()
        posture = np.eye(m.nv)[self.vids]
        target_j = cfg['posture_kp'] * (joint_ref - joints) - cfg['posture_kd'] * dq
        target_com = (cfg['com_kp'] * (com_ref - row['com'])
                      - cfg['com_kd'] * np.array(row['com_velocity']) - cbias)
        target_root = (cfg['root_kp'] * pin.log3(root_ref @ orientation.T)
                       - cfg['root_kd'] * root_twist[1] - pbias)
        task = np.vstack([cfg['com_weight'] * jc, cfg['root_weight'] * jp, posture])
        targets = np.r_[cfg['com_weight'] * target_com, cfg['root_weight'] * target_root, target_j]
        z = np.linalg.lstsq(task @ free, targets - task @ foot_solution, rcond=1e-6)[0]
        acc = foot_solution + free @ z

        required = pin.rnea(m, d, q, v, acc).copy()
        # The motor armatures from the original 10-Hz design are excluded from the bare URDF.
        required[self.vids] += self.k / (20. * np.pi) ** 2 * acc[self.vids]
        load = np.maximum(np.asarray(row['foot_load']), 0.)
        fractions = load / load.sum() if load.sum() > 0 else np.array([.5, .5])
        f0 = np.zeros(12)
        f0[[2, 8]] = self.mass * 9.81 * fractions
        # Distribute feasible net floating-base demand, penalizing support moments by sole length.
        base = jfoot.T[:6]
        scale = np.tile([1., 1., 1., cfg['moment_arm_m'], cfg['moment_arm_m'],
                         cfg['moment_arm_m']], 2)
        weighted = base * scale
        increments = np.linalg.lstsq(weighted, required[:6] - base @ f0, rcond=1e-8)[0]
        wrench = f0 + scale * increments
        motor_torque = (required - jfoot.T @ wrench)[self.vids]
        limit = cfg['effort_fraction'] * np.array(row['limits'])[:, 3]
        motor_torque = np.clip(motor_torque, -limit, limit)
        return joints + (motor_torque + self.d * dq) / self.k


def worker():
    cfg, names = json.loads(sys.stdin.readline())
    controller = DoubleSupport(cfg, names)
    for line in sys.stdin:
        position = controller.step(json.loads(line))
        print(json.dumps(position.tolist(), allow_nan=False), flush=True)


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    worker()
