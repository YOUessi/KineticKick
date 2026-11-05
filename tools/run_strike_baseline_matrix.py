#!/usr/bin/env python3
"""Run the preregistered sensitivity cases serially; physical failures are retained."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from strike_baseline_protocol import verify_files, case_plan
from strike_baseline_evidence import verify_case_evidence
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, default=ROOT/'configs/strike_baseline_v1.json')
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    matrix = json.loads(a.matrix.read_text())
    digest = hashlib.sha256(a.matrix.read_bytes()).hexdigest()
    verify_files(ROOT, matrix['frozen_files_sha256'])
    center = None
    baseline = json.loads((ROOT/matrix['baseline_plan']).read_text())
    source_hashes = {field:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for field,path in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    for case in matrix['cases']:
        output = Path(str(a.output_prefix)+'_'+case['id']+'.json')
        if output.exists() and not a.resume:
            raise FileExistsError(output)
        if not output.exists():
            print('START', case['id'], flush=True)
            started = time.monotonic()
            with output.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, str(ROOT/'tools/run_strike_baseline_case.py'),
                    '--matrix', str(a.matrix.resolve()), '--case', case['id'], '--output', str(output.resolve())],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('FINISHED', case['id'], 'wall_seconds', round(time.monotonic()-started, 1), flush=True)
        result = json.loads(output.read_text())
        verify_case_evidence(result, case, digest, source_hashes, case_plan(baseline, case), center)
        if case['id'] == 'center': center = result
        from strike_baseline_metrics import summarize_strike
        m = summarize_strike(result)
        print('RESULT', case['id'], json.dumps(dict(outcome=m['outcome'], contact=m['contact'],
            standing_s=m['final_standing_seconds'])), flush=True)
    print('MATRIX_COMPLETE: all cases recorded; consult outcomes, not completion count', flush=True)


if __name__ == '__main__': main()
