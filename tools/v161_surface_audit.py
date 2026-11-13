"""Recompute the four material-point/time conventions on one saved physical episode."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from free_shooting_metrics import paid_contact_patch
from v160_tracking_ball_impulse import collect_contacts, point_geometry


def summarize(folder):
    saved = json.loads((folder / 'tracking_result.json').read_text())
    names = saved['source_body_names']
    with np.load(folder / 'rollout.npz', allow_pickle=False) as tr:
        times = tr['time_s']
        ankle = names.index('right_ankle_roll_link')
        positions = tr['body_position_w_m'][:, ankle]
        rotations = Rotation.from_quat(tr['body_quat_wxyz'][:, ankle][:, [1, 2, 3, 0]])
    _, _, _, points = collect_contacts(folder / 'native_contacts.jsonl.gz', times, .005)
    fractions = {}
    for timing, delay in [('pre', 2), ('post', 1)]:
        for witness in (True, False):
            total = instep = 0.
            for tick, point, sign, magnitude, _, _ in points:
                index = tick-delay
                if index < 0:
                    raise ValueError('Missing bracketing ankle pose')
                local, normal = point_geometry(point, sign, positions[index], rotations[index],
                                               foot_witness=witness)
                total += magnitude
                if paid_contact_patch(local, normal, foot='right') == 'instep':
                    instep += magnitude
            fractions[timing + ('_foot' if witness else '_native')] = instep/total if total else None
    angle = None
    if points:
        idx = points[0][0]-2
        if idx < 0:
            raise ValueError('Missing pre-contact pose')
        toe_z = rotations[idx].apply([1.,0.,0.])[2]
        angle = float(-np.degrees(np.arcsin(np.clip(toe_z,-1.,1.))))
    return {'instep_normal_impulse_fractions': fractions, 'first_precontact_toe_down_degrees': angle}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2, allow_nan=False))
