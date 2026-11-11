#!/usr/bin/env python3
"""Plot ONLY raw, hash-verified original G1 PhysX 5ms motion, no predicted trajectories.

A missing/terminated record is censored: never interpolate or zero fill.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import config,sha,PROTOCOL,PROGRESS

def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=config()
    state=json.loads(PROGRESS.read_text())
    if state['protocol_sha256']!=sha(PROTOCOL) or len(state['runs'])!=9:
        raise ValueError('not all preregistered cases are complete')
    by_id={}
    for case,entry in zip(c['cases'],state['runs']):
        if case['id']!=entry['id']:raise ValueError('evidence reorder')
        if entry['status']!='VALID_G1':continue
        f=ROOT/case['output']
        if sha(f)!=entry['raw_sha256']:raise ValueError('raw physics hash drift')
        raw=json.loads(f.read_text())
        by_id[case['id']]=raw
    path=ROOT/'media'
    for group in ('new_stance_a','new_stance_b','new_stance_c'):
        baseline=by_id.get(group+'_v32')
        if baseline is None:continue
        trace=baseline['traces'][0]
        times=np.asarray([r['time_s'] for r in trace],dtype=float)
        original=np.asarray([r['joint_pos'] for r in trace],dtype=float)
        fig,ax=plt.subplots(figsize=(11,4.5))
        for policy in ('synth','reverse'):
            run=by_id.get(group+'_'+policy)
            if run is None:continue
            rows=run['traces'][0]
            newtime=np.asarray([x['time_s'] for x in rows],dtype=float)
            n=min(len(newtime),len(times))
            if not np.allclose(newtime[:n],times[:n],atol=2e-6):
                raise ValueError('same-index physical time mismatch')
            q=np.asarray([r['joint_pos'] for r in rows[:n]],dtype=float)
            assert original.shape[1]==q.shape[1]==29
            difference=np.sqrt(np.mean((q-original[:n])**2,axis=1))
            ax.plot(times[:n],difference,label=policy+' vs v32, measured q RMS')
        ct=baseline['records'][0].get('contact_time_s')
        if ct is not None:ax.axvline(ct,ls='--',label='original measured ball contact')
        ax.set(xlabel='True original G1 PhysX physical time (seconds)',
            ylabel='Actual 29DoF pose RMS difference from v32 (radians)',
            title=f'v0.1.67 {group}: 200 Hz measured physical response, not human motion')
        ax.legend(loc='best');fig.tight_layout()
        dest=path/f'v067_{group}_measured_directionality.png'
        if dest.exists():raise FileExistsError('write-once physics figure '+str(dest))
        fig.savefig(dest,dpi=130)
        plt.close(fig)
        print('SAVED_VERIFIED_G1_FIGURE',dest.name)
if __name__=='__main__':main()
