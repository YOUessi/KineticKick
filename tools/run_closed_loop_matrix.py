#!/usr/bin/env python3
"""Serial bounded physical cases, preserving every existing artifact."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--zero',action='store_true')
    p.add_argument('--prefix',required=True)
    p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args()
    if a.zero==bool(a.checkpoint):p.error('Choose one of --zero or --checkpoint')
    matrix=json.loads((ROOT/'configs/strike_baseline_v1.json').read_text())
    cases={c['id']:c for c in matrix['cases']}
    if len(set(a.cases))!=len(a.cases) or any(n not in cases for n in a.cases):p.error('Unique frozen case ids required')
    if Path(a.prefix).name!=a.prefix:p.error('Prefix must be a filename component')
    for name in a.cases:
        report=ROOT/'reports'/f'{a.prefix}_{name}.json';log=report.with_suffix('.log')
        if report.exists() or log.exists():raise FileExistsError(report)
        c=cases[name]
        cmd=[sys.executable,str(ROOT/'tools/run_closed_loop_strike.py'),'--output',str(report),'--ball-offset',*[str(x) for x in c['ball_offset_xy_m']],'--velocity-delta',str(c['root_velocity_y_delta_mps'])]
        cmd+=['--zero'] if a.zero else ['--checkpoint',str(a.checkpoint)]
        print('CASE_START',name,flush=True)
        with log.open('w') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        from strike_baseline_metrics import summarize_strike
        result=summarize_strike(json.loads(report.read_text()))
        print('CASE_RESULT',name,json.dumps({k:result[k] for k in ['outcome','contact','final_standing_seconds','completed_physics_seconds']}),flush=True)

if __name__=='__main__':main()
