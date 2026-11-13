"""Collect all complete speed/aim experiments, retaining failures and the matched fixed swing."""
import argparse
import hashlib
import json
from pathlib import Path
from v162_power_score import measure


def summarize(workspace):
    baseline=measure(workspace/'experiments/v162_power_phys/frozen_s0',3.74,.065)
    fixed_speed=baseline['speed_mps'];fixed_x=baseline['goal_plane_x_m']
    rows=[]
    for phase in ('v162_power_phys','v162_local_phys'):
        for file in sorted((workspace/'experiments'/phase).glob('*/power_score.json')):
            r=json.loads(file.read_text());speed,aim=r['request'];v=r['speed_mps'];x=r['goal_plane_x_m']
            rows.append(dict(run=str(file.parent.relative_to(workspace)),request=r['request'],
                        physical_failures=r['score']['failed_checks'],checks=r['checks'],passes=r['passes'],
                        speed_mps=v,goal_plane_x_m=x,fixed_exit_speed=fixed_speed,fixed_goal_plane_x=fixed_x,
                        fixed_passes=baseline['checks']['physical'] and baseline['checks']['surface'] and
                           abs(fixed_speed-speed)<=.18 and abs(fixed_x-aim)<=.10,
                        standing_s=r['score']['terminal_standing_seconds'],
                        score_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    return rows


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    result=summarize(a.workspace)
    with a.output.open('x') as o:json.dump(result,o,indent=2);o.write('\n')
