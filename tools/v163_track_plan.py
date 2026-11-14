"""Construct a contact-feasible full-clip task reference on the original mocap clock."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded

SIDES = ("left", "right")


def plan(src, cfg):
    ts = src["time_s"]
    dt = 0.005
    slow = cfg["slow"]
    size = cfg["stride"]
    end = ts[-1] * slow
    t = np.arange(round(14 / dt) + 1) * dt
    s = np.minimum(t / slow, ts[-1])
    lerp = lambda a: np.stack(
        [np.interp(s, ts, y) for y in np.reshape(a, (len(ts), -1)).T], axis=-1
    ).reshape((len(t),) + a.shape[1:])

    def rot_interp(arr):
        return Slerp(ts, Rotation.from_quat(arr[:, [1, 2, 3, 0]]))(s).as_matrix()

    def diff(x):
        return np.gradient(
            gaussian_filter1d(x, cfg["smooth"] / dt, axis=0, mode="nearest"), dt, axis=0
        )

    def velrot(x):
        w = np.zeros((len(t), 3))
        w[:-1] = (
            Rotation.from_matrix(x[1:] @ x[:-1].transpose(0, 2, 1)).as_rotvec() / dt
        )
        w[-1] = w[-2]
        return gaussian_filter1d(w, cfg["smooth"] / dt, axis=0)

    feet = [
        src["body_names"].tolist().index(side + "_ankle_roll_link") for side in SIDES
    ]
    pos = lerp(src["body_pos_w"][:, feet])
    rot = np.stack([rot_interp(src["body_quat_w"][:, f]) for f in feet], 1)
    z = lerp(src["urdf_sole_height_m"])
    # Grounded phases follow source kinematics with a low and near-stationary patch;
    # retarget moment-carrying foot poses onto the full plantar support surface.
    rootR = rot_interp(src["body_quat_w"][:, 0])
    root = lerp(src["body_pos_w"][:, 0])
    q = lerp(src["joint_pos"])

    def smooth01(a):
        a = np.clip(a, 0, 1)
        return a * a * (3 - 2 * a)

    lower = smooth01((t - end) / 0.65)
    for j in range(2):
        yaw = Rotation.from_matrix(rot[:, j]).as_euler("xyz")[:, 2]
        flat = Rotation.from_euler("z", yaw[:, None]).as_matrix()
        alpha = smooth01((z[:, j] - 0.008) / 0.08) * (1 - lower)
        rot[:, j] = (
            Rotation.from_rotvec(
                Rotation.from_matrix(rot[:, j] @ flat.transpose(0, 2, 1)).as_rotvec()
                * alpha[:, None]
            ).as_matrix()
            @ flat
        )
        pos[:, j, :2] = pos[0, j, :2] + size * (pos[:, j, :2] - pos[0, j, :2])
        # Set ankle height from the rotated original capsule endpoints. A
        # pitching airborne foot needs extra height to clear the pitch lever.
        pts = np.array(
            [
                [-0.054, 0, -0.025, 0.01],
                [0.132, 0, -0.025, 0.01],
                [-0.052, -0.01, -0.025, 0.01],
                [-0.052, 0.01, -0.025, 0.01],
                [0.130, -0.01, -0.025, 0.01],
                [0.130, 0.01, -0.025, 0.01],
                [0.1, -0.026, -0.025, 0.01],
                [0.1, 0.026, -0.025, 0.01],
            ]
        )
        floor = (rot[:, j, 2, :] @ pts[:, :3].T - pts[:, 3]).min(axis=1)
        pos[:, j, 2] = (-floor if cfg["rotated_sole"] else 0.035) + np.maximum(
            0.0, z[:, j] - 0.001
        ) * (1 - lower)
    if cfg.get("overlap"):
        # Insert double support at the crossover. The source G1 geometry has
        # both feet in the air while transferring to the future kicking stance.
        # Bring the landing foot ahead and keep the take-off foot planted until
        # after this new landing; retain the original swing and end poses.
        left = np.interp(
            s, [0, 2.26, 2.64, 2.77, ts[-1]], [0, 2.26, 2.79, 2.80, ts[-1]]
        )
        right = np.interp(
            s, [0, 2.37, 2.68, 2.87, ts[-1]], [0, 2.37, 2.42, 2.87, ts[-1]]
        )
        for j, clock in enumerate((left, right)):
            target = np.clip(clock * slow / dt, 0.0, len(t) - 1)
            a = np.minimum(target.astype(int), len(t) - 2)
            u = target - a
            pos[:, j] = pos[a, j] * (1 - u[:, None]) + pos[a + 1, j] * u[:, None]
            rot[:, j] = Slerp(t, Rotation.from_matrix(rot[:, j]))(
                target * dt
            ).as_matrix()
            z[:, j] = np.interp(target, np.arange(len(t)), z[:, j])
    # Restore the exact original initial poses over the first settling interval.
    ramp = smooth01(t / 0.45)
    for j, f in enumerate(feet):
        real = Rotation.from_quat(src["body_quat_w"][0, f, [1, 2, 3, 0]]).as_matrix()
        initdelta = Rotation.from_matrix(real @ rot[0, j].T).as_rotvec()
        rot[:, j] = (
            Rotation.from_rotvec((1 - ramp)[:, None] * initdelta).as_matrix()
            @ rot[:, j]
        )
        pos[:, j, 2] += (1 - ramp) * (src["body_pos_w"][0, f, 2] - pos[0, j, 2])
    # Nominal support availability; smoothly transfer load to the next planted foot.
    weights = np.exp(-np.maximum(0.0, z - 0.001) / cfg["support_width"])
    speed = np.linalg.norm(diff(pos)[:, :, :2], axis=2)
    weights *= np.exp(-speed / 0.27)
    weights = (1 - lower[:, None]) * weights + lower[:, None]
    weights /= np.maximum(1.0e-12, weights.sum(axis=1)[:, None])
    weights = gaussian_filter1d(weights, 0.045 / dt, axis=0, mode="nearest")
    cop = pos[:, :, :2] + rot[:, :, :2, 0] * cfg["cop_fwd"]
    zmp = (weights[:, :, None] * cop).sum(axis=1)
    # Bounded solution of the LIPM: COM- h/g COM'' = desired stance ZMP.
    h = cfg["com_height"]
    a = h / 9.81 / dt**2
    ab = np.zeros((3, len(t)))
    ab[1] = 1 + 2 * a
    ab[0, 1:] = -a
    ab[2, :-1] = -a
    ab[1, [0, -1]] = 1 + a
    xy = solve_banded((1, 1), ab, zmp)
    root[:, 2] -= cfg["crouch"] * ramp
    c = np.column_stack((xy, root[:, 2] + cfg["com_relative_height"]))
    result = {
        "time": t,
        "source_time": s,
        "q": q,
        "rootR": rootR,
        "c": c,
        "support": weights,
        "foot": pos,
        "footR": rot,
    }
    for k in ["q", "c", "foot"]:
        result[k + "v"] = diff(result[k])
        result[k + "a"] = diff(result[k + "v"])
    result["rw"] = velrot(rootR)
    result["ra"] = diff(result["rw"])
    result["fw"] = np.stack([velrot(rot[:, j]) for j in range(2)], 1)
    result["fa"] = diff(result["fw"])
    return result
