#!/usr/bin/env python3
"""Show measured 200Hz early physical nonfoot, with censoring, no extrapolation."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import config,PROGRESS,PROTOCOL,sha
def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 config()
 state=json.loads(PROGRESS.read_text())
 if state['protocol_sha256']!=sha(PROTOCOL) or len(state['runs'])!=9:
  raise RuntimeError('Frozen nine-case physical protocol missing')
 fig,axs=plt.subplots(2,1,figsize=(11.8,7.8))
 palette={'v32':'tab:blue','synth':'tab:orange','reverse':'tab:green'}
 for case,row in zip(config()['cases'],state['runs']):
  f=ROOT/case['output']
  if sha(f)!=row['raw_sha256']:raise RuntimeError('Physical evidence drift')
  raw=json.loads(f.read_text())
  t=np.asarray([x['time_s'] for x in raw['traces'][0]],dtype=float)
  force=np.asarray([x['nonfoot_contact_N'] for x in raw['traces'][0]],dtype=float)
  if not np.isfinite(force).all() or not np.isfinite(t).all():raise ValueError('Nonfinite original physics')
  p=case['policy'];group=case['ball_case']
  label=f'{group}/{p} ({len(t)} real frames)'
  axs[0].plot(t,force,linewidth=1.0,label=label,color=palette[p],alpha=.50 if group!='new_stance_c' else .95)
  if 'physical_nonfoot' in raw['termination_reasons'][0]:
   axs[0].scatter([t[-1]],[force[-1]],marker='x',color=palette[p],s=55)
  ctime=raw['records'][0].get('contact_time_s')
  if ctime is not None:axs[1].scatter([ctime],[t[-1]],color=palette[p],
       marker={'new_stance_a':'o','new_stance_b':'^','new_stance_c':'x'}[group],label=label)
 axs[0].axhline(20,ls='--',color='gray',label='Original 20N termination threshold')
 axs[0].set_xlim(1.4,5.6)
 axs[0].set_yscale('symlog',linthresh=1)
 axs[0].set(xlabel='Original simulator 5ms physical state time [s]',ylabel='Recorded maximum nonfoot net force [N]',
  title='Native CUDA G1 physical nonfoot events; X=early termination, NO imputation')
 axs[0].legend(ncol=3,fontsize=7)
 axs[1].plot([1.4,1.7],[14,14],linestyle='--',color='gray',label='Required complete 14s')
 axs[1].set(xlabel='Measured foot-ball contact time [s]',ylabel='Last available physical sample [s]',
  title='Censoring: 4 early terminated native episodes ≠ 14s recovery')
 axs[1].set_ylim(0,15)
 axs[1].legend(ncol=3,fontsize=7)
 fig.tight_layout()
 output=ROOT/'media/v067_native_nonfoot_censoring.png'
 if output.exists():raise FileExistsError(output)
 fig.savefig(output,dpi=140)
 print('PRESERVED_NATIVE_G1_CENSORED_FIGURE',output)
if __name__=='__main__':main()
