"""Stricter physical evidence for a nearly vertical, toes-down instep shot."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from shooting_posture import audit_posture, check_posture_acceptance, rotate_vector


VERTICAL_CRITERIA = dict(max_angle_from_down_deg=15., dorsal_forward_min=.90,
                         near_contact_clearance_min_m=.01,
                         clearance_before_s=.04, clearance_after_s=.02)
# Unchanged PAiD seven-cylinder foot, padded by the same15mm conversion/contact
# tolerance used by the physical contact classifier. Clearance is an estimate.
FOOT_CORNERS = tuple(itertools.product((-.069, .147), (-.051, .051), (-.050, .0)))


def _clearance(run, index, contact_time, contact_quaternion):
    samples = run['kinematics'][index]
    times = np.array([row['time_s'] for row in samples])
    lo = contact_time - VERTICAL_CRITERIA['clearance_before_s']
    hi = contact_time + VERTICAL_CRITERIA['clearance_after_s']
    if (not len(times) or times[0] > lo+1e-8 or times[-1] < hi-1e-8 or
            np.any(np.diff(times) <= 0)):
        raise ValueError('Missing bracketing foot poses')
    first = max(int(np.searchsorted(times, lo+1e-8, side='right'))-1, 0)
    last = int(np.searchsorted(times, hi-1e-8))
    window = samples[first:last+1]
    if np.any(np.diff(times[first:last+1]) > .021):
        raise ValueError('Missing near-contact foot-pose samples')
    positions, clearances = [], []
    for sample in window:
        foot = sample['feet']['right']
        position = np.asarray(foot['position'], float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError('Invalid measured foot position')
        positions.append(position)
        clearances.append(float(position[2] + min(rotate_vector(foot['quaternion'], p)[2] for p in FOOT_CORNERS)))
    # Interpolate only translation; orientation at contact is measured every5ms.
    contact_z = np.interp(contact_time, times[first:last+1], np.array(positions)[:, 2])
    contact_clearance = float(contact_z + min(rotate_vector(contact_quaternion, p)[2] for p in FOOT_CORNERS))
    return min(*clearances, contact_clearance), contact_clearance


def audit_vertical_instep(run, index):
    result = audit_posture(run, index)
    if result['style'] != 'instep':
        raise ValueError('Vertical instep audit requires an instep trial')
    result.update(vertical_pose_ok=False, vertical_and_full_chain=False, vertical_failure=None)
    if 'toe_world_z' not in result:
        result['vertical_failure'] = result['posture_failure']
        return result
    first = next(row for row in run['traces'][index]
                 if row['contact_force_N']>2 and row.get('contacts'))
    contact = max(first['contacts'], key=lambda e:e['force_N'])
    impulse, down_impulse = 0., 0.
    for row in run['traces'][index]:
        for event in row.get('contacts', []):
            if event['foot']=='right' and event['force_N']>0:
                weight = event['force_N']*row['physics_dt_s']
                impulse += weight
                down_impulse -= rotate_vector(event['foot_quat_w'], [1,0,0])[2]*weight
    weighted_down = down_impulse/impulse if impulse else None
    result.update(toe_down_angle_deg=math.degrees(math.asin(float(np.clip(-result['toe_world_z'],-1,1)))),
                  impulse_weighted_toe_down=weighted_down)
    try:
        clearance, at_contact = _clearance(run, index, first['time_s'], contact['foot_quat_w'])
    except (KeyError, TypeError, ValueError, IndexError):
        result['vertical_failure'] = 'missing_foot_pose_coverage'
        return result
    result.update(near_contact_clearance_m=clearance, contact_clearance_m=at_contact)
    minimum_down = math.cos(math.radians(VERTICAL_CRITERIA['max_angle_from_down_deg']))
    forward = VERTICAL_CRITERIA['dorsal_forward_min']
    vertical = (-result['toe_world_z']>=minimum_down and weighted_down is not None and
                weighted_down>=minimum_down and result['top_forward']>=forward and
                result['impulse_weighted_top_forward']>=forward)
    clear = clearance >= VERTICAL_CRITERIA['near_contact_clearance_min_m']
    pose_ok = bool(vertical and clear and result['posture_ok'])
    result.update(vertical_pose_ok=pose_ok,
                  vertical_and_full_chain=bool(pose_ok and result['full_chain_success']),
                  vertical_failure=None if pose_ok else ('insufficient_clearance' if not clear else
                      'not_vertical' if not vertical else result['posture_failure']))
    return result


def check_vertical_acceptance(config, reports, config_sha256):
    if any(run.get('diagnostic_only') for run in reports):
        raise ValueError('Diagnostic-only runs cannot certify frozen acceptance')
    if config.get('vertical_instep_criteria') != VERTICAL_CRITERIA:
        raise ValueError('Missing or changed near-vertical acceptance criteria')
    result = check_posture_acceptance(config, reports, config_sha256)
    vertical_rows = []
    for run in reports:
        for i, trial in enumerate(run['trials']):
            if trial['style']=='instep':
                row = audit_vertical_instep(run, i)
                row.update(seed=run['seed'], index=i, name=trial.get('name','instep'))
                vertical_rows.append(row)
    successes = sum(row['vertical_and_full_chain'] for row in vertical_rows)
    result.update(vertical_instep_criteria=VERTICAL_CRITERIA, vertical_instep_trials=vertical_rows,
                  vertical_instep_successes=successes,
                  passed=bool(result['passed'] and len(vertical_rows)==6 and successes>=5))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--config', help='Certify the complete frozen matrix')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text()) for path in args.runs]
    if args.config:
        data = Path(args.config).read_bytes()
        result = check_vertical_acceptance(json.loads(data), runs, hashlib.sha256(data).hexdigest())
        result['sources'] = [dict(path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in args.runs]
        print(json.dumps({k:result[k] for k in ('passed','styles','vertical_instep_successes')}, indent=2))
    else:
        result = []
        for path, run in zip(args.runs, runs):
            for i, trial in enumerate(run['trials']):
                if trial.get('style',run.get('requested_style')) != 'instep': continue
                row = audit_vertical_instep(run, i)
                row.update(source=path, index=i, name=trial.get('name','instep'))
                result.append(row)
                print(row['name'], 'vertical+chain',row['vertical_and_full_chain'],
                      'angle',round(row.get('toe_down_angle_deg',0),1),
                      'top',round(row.get('top_forward',0),3),
                      'clearance',round(row.get('near_contact_clearance_m',0),3),
                      'reason',row['vertical_failure'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    return 0 if not args.config or result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
