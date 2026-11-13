"""Summarize every stored analytical-IK/whole-body-transfer development trial on the frozen ball."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from free_shooting_metrics import paid_contact_patch
from v160_tracking_ball_impulse import collect_contacts, point_geometry
from v161_evaluate import quat_rotation, report


def summarizes(folder):
    trial = json.loads((folder/'trial.json').read_text())
    audit = report(folder)
    with np.load(folder/'rollout.npz') as trace:
        times = trace['time_s']
        _, _, _, pts = collect_contacts(folder/'native_contacts.jsonl.gz', times, .005)
        names = json.loads((folder/'tracking_result.json').read_text())['source_body_names']
        foot = names.index('right_ankle_roll_link')
        origins = trace['body_position_w_m'][:,foot]
        rotations = quat_rotation(trace['body_quat_wxyz'][:,foot])
    weights = Counter()
    for tick,p,sign,scalar,*_ in pts:
        local, normal = point_geometry(p, sign, origins[tick-1], rotations[tick-1], foot_witness=True)
        weights[paid_contact_patch(local, normal)] += float(scalar)
    total = sum(weights.values())
    post = weights['instep']/total if total else None
    base = dict(audit['checks'])
    base['right_instep_proxy_poststep_all_positive'] = post is not None and post >= .95
    base.pop('complete_14s')
    base.pop('terminal_standing')
    return dict(name=folder.name, rollout_commit=trial['git_commit'], duration_s=trial['duration_s'],
                source_sha256=audit['input_sha256'], trajectory_checks_before_standing=base,
                all_development_checks_pass=all(base.values()), whole_ball_goal=audit['checks']['goal'],
                speed_mps=audit['last_exit_speed_xy_mps'], patch_pre=audit['instep_proxy']['instep_fraction'],
                patch_post=post, follow=audit['follow_through'], first_nonfoot=audit['native_pair_events']['first_nonfoot_pair'],
                first_wrong_ball=audit['native_pair_events']['first_wrong_ball_contact'],
                q_bounds_exceedance_cells=audit['q_position_violation_cells'],
                qdot_bounds_exceedance_cells=audit['qdot_violation_cells'], cpu_4s_terminal_standing_s=audit['terminal_standing_seconds'])


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    folders=sorted(p for p in args.root.iterdir() if p.is_dir() and (p/'trial.json').is_file())
    rows=[summarizes(p) for p in folders]
    counts=Counter(key for row in rows for key,passed in row['trajectory_checks_before_standing'].items() if passed)
    out=dict(complete_development_runs=len(rows), total_simulated_s=sum(row['duration_s'] for row in rows),
             all_development_checks_pass=sum(row['all_development_checks_pass'] for row in rows),
             per_check_counts=counts, rows=rows, independent_fourteen_second_runs=0,
             frozen_but_unrun=['v161_strike_u','v161_strike_v','v161_strike_w'])
    with args.output.open('x') as stream:
        json.dump(out,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
