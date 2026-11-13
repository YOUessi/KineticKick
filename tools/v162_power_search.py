"""Optimize shot commands against whole-ball path and measured exit velocity in CPU PhysX."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def loss(row,speed,aim):
    xy=row['release_xy']
    if xy is None or row['crossing_x'] is None:
        return 25 + max(0.,-row['merit'])
    patch=min(row['instep'],row['instep_post'])
    safety=(18*max(0,.98-patch) + min(7.,row['nonfoot']/10.)
            + min(7.,max(0.,row['excess']-0.5e-6)*1e6)
            + .35*max(0.,min(1.,(row['excess']+.004)/.004)))
    follow=(1.5*max(0,.24-row['advance'])+3*max(0,.17-row['relative'])
            + .6*max(0,.75-row['v120']) + .8*max(0,row['return_speed']-1.9) + 2*row['late_contact'])
    return safety+follow+3.0*abs(np.linalg.norm(xy)-speed)+5.5*abs(row['crossing_x']-aim)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--initial',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--speed',type=float,required=True)
    cli.add_argument('--aim',type=float,required=True)
    cli.add_argument('--seed',type=int,required=True)
    cli.add_argument('--rounds',type=int,default=10)
    cli.add_argument('--samples',type=int,default=20)
    cli.add_argument('--spread',type=float,default=.18)
    a=cli.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    params=json.loads(a.initial.read_text()); cap=params['backswing_pitch_target_cap']
    command=[sys.executable,str(Path(__file__).with_name('v161_dynamics_search.py')),'--live','--feasible','--service',
             '--pitch-cap',str(cap),'--output',str(a.output/'worker')]
    rng=np.random.default_rng(a.seed); mean=np.array(params['coefficients']);sd=a.spread*np.tile([1,.50,1.15,1,.85,.45],2)
    best=np.inf
    with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1) as proc:
        if proc.stdout.readline().strip()!='ready':raise RuntimeError('Physics worker startup failed')
        try:
            with (a.output/'development.jsonl').open('x') as out:
                for g in range(a.rounds):
                    xs=rng.normal(mean,sd,(a.samples,12));xs[0]=mean
                    values=[]
                    for x in xs:
                        proc.stdin.write(json.dumps(x.tolist())+'\n');proc.stdin.flush()
                        r=json.loads(proc.stdout.readline());score=loss(r,a.speed,a.aim)
                        row={'loss':score,'round':g,'command':[a.speed,a.aim],'coefficients':x.tolist(),**r}
                        values.append(score);out.write(json.dumps(row)+'\n')
                        if score<best:
                            best=score;(a.output/'best.json').write_text(json.dumps(row,indent=2)+'\n')
                    out.flush();elites=xs[np.argsort(values)[:max(4,a.samples//5)]]
                    mean=.15*mean+.85*elites.mean(0)
                    sd=np.maximum(.014,.20*sd+.80*elites.std(0))
        finally:
            proc.stdin.close()
    if proc.returncode:raise RuntimeError('Physics worker exited with an error')


if __name__=='__main__':main()
