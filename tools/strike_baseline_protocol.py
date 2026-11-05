"""Frozen experiment inputs; this module never imports a simulator or numerical runtime."""
import copy
import hashlib
import math
from pathlib import Path


def verify_files(root, expected):
    for name, digest in expected.items():
        path = Path(root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen baseline fingerprint mismatch: ' + name)


def case_plan(baseline, case):
    if set(case) - {'id', 'ball_offset_xy_m', 'root_velocity_y_delta_mps'}:
        raise ValueError('Case may change only ball position and initial root velocity')
    offset = case['ball_offset_xy_m']
    speed = case['root_velocity_y_delta_mps']
    if (len(offset) != 2 or not all(math.isfinite(v) and abs(v) <= .25 for v in offset)
            or not math.isfinite(speed) or abs(speed) > .5):
        raise ValueError('Finite local position and initial velocity offsets required')
    out = copy.deepcopy(baseline)
    out['ball_xy'] = [v + d for v, d in zip(baseline['ball_xy'], offset)]
    return out
