"""Whole-body inverse dynamics for a source-conditioned floating humanoid.

Ground forces are optimization variables used only to synthesize bounded motor
torques; the physical plant determines the actual contact forces.
"""

import sys
import json
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
import osqp
import pinocchio as pin
from v163_track_plan import plan

LWA = pin.LOCAL_WORLD_ALIGNED


class Tracker:
    def __init__(self, cfg, names, ref):
        self.cfg = cfg
        self.model = m = pin.buildModelFromUrdf(cfg["urdf"], pin.JointModelFreeFlyer())
        self.data = m.createData()
        self.index = np.array([m.idx_qs[m.getJointId(n)] for n in names])
        self.vi = self.index - 1
        self.feet = [
            m.getFrameId(side + "_ankle_roll_link") for side in ("left", "right")
        ]
        self.body = m.getFrameId("pelvis")
        self.p = plan(ref, cfg)
        self.mass = pin.computeTotalMass(m)
        self.K = np.array(cfg["drive_stiffness"])
        self.D = np.array(cfg["drive_damping"])
        self.eff = m.effortLimit[self.vi]
        self.b = np.eye(35)[self.vi]
        self.eye = np.eye(35)
        self.feye = np.eye(24)
        self.zero = np.zeros(35)
        self.c0 = None
        self.arms = np.array(
            [("shoulder" in n or "elbow" in n or "wrist" in n) for n in names]
        )
        self.waist = np.array(["waist" in n for n in names])
        self.shoulderRoll = [
            names.index(s + "_shoulder_roll_joint") for s in ("left", "right")
        ]

    def step(self, i, state):
        cfg = self.cfg
        m = self.model
        d = self.data
        p = self.p
        root = np.asarray(state[:7])
        rv = np.asarray(state[7:13])
        qj = np.asarray(state[13:42])
        vj = np.asarray(state[42:71])
        rot = Rotation.from_quat(root[[4, 5, 6, 3]]).as_matrix()
        q = pin.neutral(m)
        q[:7] = root[[0, 1, 2, 4, 5, 6, 3]]
        q[self.index] = qj
        v = np.zeros(35)
        v[:6] = (rv.reshape(2, 3) @ rot).ravel()
        v[self.vi] = vj
        M = pin.crba(m, d, q).copy()
        M[self.vi, self.vi] += self.K / (20 * np.pi) ** 2
        g = pin.nonLinearEffects(m, d, q, v).copy()
        pin.computeJointJacobians(m, d, q)
        pin.forwardKinematics(m, d, q, v, self.zero)
        pin.updateFramePlacements(m, d)
        jcom = pin.jacobianCenterOfMass(m, d, q).copy()
        pin.centerOfMass(m, d, q, v, self.zero)
        c = d.com[0].copy()
        cd = d.vcom[0].copy()
        cb = d.acom[0].copy()
        tar = p["c"][i].copy()
        if self.c0 is None:
            self.c0 = c - self.p["c"][0]
        tar[2] += self.c0[2]
        mats = []
        vals = []

        def task(J, w, desired):
            mats.append(w * J)
            vals.append(w * desired)

        cacc = (
            p["ca"][i]
            + cfg["com_kp"] * (tar - c)
            + cfg["com_kd"] * (p["cv"][i] - cd)
            - cb
        )
        task(jcom, cfg["cw"], np.clip(cacc, -70, 70))
        jroot = pin.getFrameJacobian(m, d, self.body, LWA)[3:].copy()
        task(
            jroot,
            cfg["rwgt"],
            np.clip(
                p["ra"][i]
                + cfg["rot_kp"] * pin.log3(p["rootR"][i] @ rot.T)
                + cfg["rot_kd"] * (p["rw"][i] - rv[3:]),
                -120,
                120,
            ),
        )
        jq = self.b
        posture_kp = cfg["post_kp"]
        posture_kd = cfg["post_kd"]
        want = (
            p["qa"][i] + posture_kp * (p["q"][i] - qj) + posture_kd * (p["qv"][i] - vj)
        )
        jf = []
        fmax = []
        contact_targets = []
        for k, f in enumerate(self.feet):
            pose = d.oMf[f]
            J = pin.getFrameJacobian(m, d, f, LWA).copy()
            b = pin.getFrameClassicalAcceleration(m, d, f, LWA).vector.copy()
            delta = p["foot"][i, k] - pose.translation
            want_xyz = (
                p["foota"][i, k]
                + cfg["foot_kp"] * delta
                + cfg["foot_kd"] * (p["footv"][i, k] - J[:3] @ v)
                - b[:3]
            )
            want_r = (
                p["fa"][i, k]
                + cfg["fr_kp"] * pin.log3(p["footR"][i, k] @ pose.rotation.T)
                + cfg["fr_kd"] * (p["fw"][i, k] - J[3:] @ v)
                - b[3:]
            )
            task(J[:3], cfg["fwgt"], np.clip(want_xyz, -110, 110))
            task(J[3:], cfg["frw"], np.clip(want_r, -150, 150))
            # Four contact patches on the existing plantar shape; decrease a
            # patch's available load as the measured toes/heels leave ground.
            for x, y in [
                (-0.043, -0.014),
                (-0.043, 0.014),
                (0.107, -0.018),
                (0.107, 0.018),
            ]:
                r = pose.rotation @ np.array([x, y, -0.035])
                z = pose.translation[2] + r[2]
                jf.append(J[:3] - pin.skew(r) @ J[3:])
                fmax.append(
                    cfg["max_ground_force"]
                    * np.clip((cfg["patch_cutoff_m"] - z) / cfg["patch_ramp_m"], 0, 1)
                    * (
                        np.clip(p["support"][i, k] * 8, 0, 1)
                        if cfg["limit_swing_load"]
                        else 1.0
                    )
                )
                contact_targets.append(p["support"][i, k] * self.mass * 9.81 / 4)
        if cfg.get("arm_hold"):
            want[self.shoulderRoll] += (
                np.array([1, -1])
                * posture_kp
                * cfg["arm_hold"]
                * min(1.0, i * 0.005 / 0.20)
            )
            weights = np.where(self.arms, cfg["arm_weight"], cfg["pw"])
            weights[self.waist] = cfg["waist_weight"]
            task(jq * weights[:, None], 1.0, np.clip(want, -250, 250) * weights)
        else:
            task(jq, cfg["pw"], np.clip(want, -250, 250))
        J = np.vstack(jf)
        nq = 35
        nf = len(J)
        B = np.c_[M, -J.T]
        taskA = np.vstack(mats)
        H = np.zeros((nq + nf, nq + nf))
        H[:nq, :nq] = taskA.T @ taskA + 1e-5 * self.eye
        wf = cfg["force_reg"]
        H[nq:, nq:] = wf**2 * self.feye
        linear = np.r_[
            -taskA.T @ np.concatenate(vals),
            -(wf**2) * np.ravel([[0, 0, w] for w in contact_targets]),
        ]
        # exact floating equations, actuator limits, friction pyramids, positive normal loads
        fr = np.array(
            [
                [1.0, 0, -0.8],
                [-1.0, 0, -0.8],
                [0, 1.0, -0.8],
                [0, -1.0, -0.8],
                [0, 0, 1.0],
            ]
        )
        F = np.kron(np.eye(8), fr)
        A = sparse.csc_matrix(
            np.vstack((B[:6], B[self.vi], np.c_[np.zeros((40, 35)), F]))
        )
        lo = np.r_[-g[:6], -self.eff - g[self.vi], np.tile([-np.inf] * 4 + [0.0], 8)]
        hi = np.r_[
            -g[:6],
            self.eff - g[self.vi],
            np.column_stack((np.zeros((8, 4)), fmax)).ravel(),
        ]
        qp = osqp.OSQP()
        qp.setup(
            P=sparse.csc_matrix(np.triu(H)),
            q=linear,
            A=A,
            l=lo,
            u=hi,
            eps_abs=0.006,
            eps_rel=0.0004,
            max_iter=1000,
            verbose=False,
            adaptive_rho=True,
        )
        sol = qp.solve()
        x = sol.x
        if x is None or (
            "solved" not in sol.info.status and "maximum" not in sol.info.status
        ):
            raise RuntimeError(sol.info.status)
        tau = np.clip((B @ x + g)[self.vi], -self.eff, self.eff)
        # Feed forward the predicted target velocity lost by the implicit drive at the next step.
        return qj + (tau + self.D * (vj + cfg["drive_dt"] * x[self.vi])) / self.K


def worker():
    cfg, names, reference = json.loads(sys.stdin.readline())
    with np.load(reference) as source:
        controller = Tracker(cfg, names, dict(source))
    for row in sys.stdin:
        i, state = json.loads(row)
        print(
            json.dumps(controller.step(i, state).tolist(), allow_nan=False), flush=True
        )


if __name__ == "__main__":
    worker()
