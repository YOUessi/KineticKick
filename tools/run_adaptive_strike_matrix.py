#!/usr/bin/env python3
"""Serial development cases for the adaptive pilot, with frozen baseline comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'configs/adaptive_strike_v1.json')
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--disable-feedback',action='store_true')
    a=p.parse_args()
    matrix=json.loads((ROOT/'configs/strike_baseline_v1.json').read_text())
    cases={c['id']:c for c in matrix['cases']}
    if len(a.cases)!=len(set(a.cases)) or set(a.cases)-cases.keys():p.error('Unique frozen development case IDs required')
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,matrix['frozen_files_sha256'])
    plan=json.loads(a.plan.read_text());summary=[]
    for name in a.cases:
        case=cases[name];output=Path(str(a.output_prefix)+'_'+name+'.json')
        if output.exists():raise FileExistsError(output)
        cmd=[sys.executable,str(ROOT/'tools/run_adaptive_stride.py'),'--plan',str(a.plan.resolve()),
             '--output',str(output.resolve()),'--ball-offset',*[str(v) for v in case['ball_offset_xy_m']],
             '--root-velocity-y-delta',str(case['root_velocity_y_delta_mps'])]
        if a.disable_feedback:cmd.append('--disable-feedback')
        print('START',name,flush=True);start=time.monotonic()
        with output.with_suffix('.log').open('x') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(output.read_text());actual=result['adaptive_control']['observed_initial_root_velocity']
        baseline=json.loads((ROOT/('reports/v023_baseline_'+name+'.json')).read_text())
        expected=baseline['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
        if max(abs(x-y) for x,y in zip(actual,expected))>1e-6:raise ValueError('Initial velocity differs from frozen comparison')
        if result['points']!=baseline['points'] or result['goals']!=baseline['goals']:raise ValueError('Task geometry differs')
        from strike_baseline_metrics import summarize_strike
        metrics=summarize_strike(result);frozen=summarize_strike(baseline)
        row=dict(case=case,candidate=metrics,baseline=frozen,report=str(output),
                 report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-start)
        summary.append(row)
        # Save after each completed case; failures are outcomes, never skipped.
        Path(str(a.output_prefix)+'_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('RESULT',name,json.dumps(dict(outcome=metrics['outcome'],contact=metrics['contact'],
              standing_s=metrics['final_standing_seconds'],nonfoot=result['adaptive_control']['nonfoot_events'][:2])),flush=True)

if __name__=='__main__':main()
