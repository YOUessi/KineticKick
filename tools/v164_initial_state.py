"""Project the initial joint angles into the 200 Hz drive's measured braking reserve."""

import numpy as np


def project_start(q_ref, dq_ref, limits, names, cfg, dt):
    q = np.asarray(q_ref, dtype=float)
    dq = np.asarray(dq_ref, dtype=float)
    low, high, speed = limits[:, :3].T
    reserve = float(cfg["rebound_reserve_rad"])
    reaction = int(cfg["reaction_steps"]) * dt
    if not 0 < reserve or not 0 < reaction or np.any(np.abs(dq) > speed):
        raise ValueError("Invalid joint startup state or braking settings")
    # The frozen guard reverses direction in the first two physics steps. Retain the
    # original velocity and reserve its outward travel in addition to the rebound.
    near_low = low + reserve + reaction * np.maximum(-dq, 0)
    near_high = high - reserve - reaction * np.maximum(dq, 0)
    if np.any(near_low > near_high) or not np.isfinite(q + dq).all():
        raise ValueError("Initial braking reserve does not fit the original limits")
    safe = np.clip(q, near_low, near_high)
    changed = [
        {
            "joint": names[i],
            "source_q_rad": q[i],
            "start_q_rad": safe[i],
            "source_dq_rad_s": dq[i],
        }
        for i in np.flatnonzero(safe != q)
    ]
    return safe, dq.copy(), changed
