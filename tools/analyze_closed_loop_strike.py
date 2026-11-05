#!/usr/bin/env python3
"""Audit frozen versus closed-loop residual physical development cases."""
import argparse,hashlib,json
from pathlib import Path
from strike_baseline_protocol import verify_files,case_plan
from strike_baseline_evidence import verify_case_evidence
from strike_baseline_metrics import summarize_strike
from closed_loop_evidence import verify_closed_loop,SOURCE_NAMES
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--zero',action='store_true')
    p.add_argument('--prefix',required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.zero==bool(a.checkpoint):p.error('Choose one of --zero or --checkpoint')
    if a.output.exists():raise FileExistsError(a.output)
    if Path(a.prefix).name!=a.prefix:p.error('Prefix must be a filename component')
    mp=ROOT/'configs/strike_baseline_v1.json';matrix=json.loads(mp.read_text());verify_files(ROOT,matrix['frozen_files_sha256'])
    cp=ROOT/'configs/closed_loop_strike_v1.json';config=json.loads(cp.read_text())
    for key in ['template','reference','expert_plan']:
        if sha(ROOT/config[key])!=config[key+'_sha256']:raise ValueError('Changed '+key)
    plan=json.loads((ROOT/matrix['baseline_plan']).read_text());cases={c['id']:c for c in matrix['cases']}
    if len(set(a.cases))!=len(a.cases) or any(n not in cases for n in a.cases):p.error('Unique frozen ids required')
    sources={n:sha(ROOT/'tools'/n) for n in SOURCE_NAMES}
    old_sources={k:sha(ROOT/v) for k,v in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    center=json.loads((ROOT/'reports/v023_baseline_center.json').read_text());rows=[]
    verify_case_evidence(center,cases['center'],sha(mp),old_sources,case_plan(plan,cases['center']))
    nominal=center['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    for name in a.cases:
        case=cases[name];bp=ROOT/'reports'/f'v023_baseline_{name}.json';rp=ROOT/'reports'/f'{a.prefix}_{name}.json'
        baseline=json.loads(bp.read_text());candidate=json.loads(rp.read_text())
        verify_case_evidence(baseline,case,sha(mp),old_sources,case_plan(plan,case),center)
        verify_closed_loop(candidate,baseline,case,config,sha(cp),sources,None if a.zero else sha(a.checkpoint),nominal)
        rows.append(dict(case=case,baseline=summarize_strike(baseline),candidate=summarize_strike(candidate),baseline_report=str(bp.relative_to(ROOT)),candidate_report=str(rp.relative_to(ROOT)),baseline_sha256=sha(bp),candidate_sha256=sha(rp)))
    result=dict(version='0.1.26',cases=rows,baseline_passed=sum(r['baseline']['outcome']['success'] for r in rows),candidate_passed=sum(r['candidate']['outcome']['success'] for r in rows),total=len(rows),unexecuted_cases=[n for n in cases if n not in a.cases],checkpoint_sha256=None if a.zero else sha(a.checkpoint),sources=sources,analysis_sha256=sha(__file__),validator_sha256=sha(ROOT/'tools/closed_loop_evidence.py'),interpretation='Prespecified development comparisons, not population success probabilities. Nominal uses structurally protected frozen expert; perturbed cases use state-conditioned residuals.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['baseline_passed','candidate_passed','total','unexecuted_cases']},indent=2))

if __name__=='__main__':main()
