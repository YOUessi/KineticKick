#!/usr/bin/env python3
"""Build the Tang-local nominal root-path calibration from the frozen center trace."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=Path('reports/v023_baseline_center.json')
    raw=json.loads((ROOT/source).read_text())
    from strike_baseline_metrics import validated_episode
    validated_episode(raw)
    if not raw['stride_shot_success'] or raw['strike_baseline']['case']['id']!='center':
        raise ValueError('Successful frozen center evidence required')
    rows=raw['kinematics'][0]
    result=dict(source=str(source),times=[r['time_s'] for r in rows],
                root_xy=[r['root_pose'][:2] for r in rows],
                source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest())
    content=json.dumps(result,indent=2)+'\n'
    expected=json.loads((ROOT/'configs/adaptive_strike_v1.json').read_text())['feedback']['template_sha256']
    if hashlib.sha256(content.encode()).hexdigest()!=expected:
        raise ValueError('Calibration differs from the selected candidate; preserve and investigate evidence')
    target=ROOT/'configs/strike_nominal_trajectory_v1.json'
    if target.exists() and target.read_text()!=content:raise FileExistsError('Preserve different local calibration: '+str(target))
    if not target.exists():target.write_text(content)
    print(target,expected)

if __name__=='__main__':main()
