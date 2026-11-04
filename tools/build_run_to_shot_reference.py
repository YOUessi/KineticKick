#!/usr/bin/env python3
"""Compose RUN and soccer commands; the controller switches references, never physical state."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--anchor-x',type=float,default=.3925)
    p.add_argument('--anchor-y',type=float,default=-6.2688)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    run_path=root/'experiments/running_v011/run_stop_handoff_reference.npz'
    shot_path=root/'third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-010_right.npz'
    run,shot=np.load(run_path),np.load(shot_path)
    shift=np.array([a.anchor_x,a.anchor_y,0.])-np.r_[shot['body_pos_w'][0,0,:2],0.]
    if not np.isfinite(shift).all():raise ValueError('Finite planned anchor required')
    arrays={}
    for key in run.files:
        if key in ('fps','kick_leg'):arrays[key]=run[key];continue
        second=shot[key].copy()
        if key=='body_pos_w':second+=shift
        arrays[key]=np.concatenate([run[key],second])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:np.savez_compressed(f,**arrays)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=dict(status='physical_handoff_candidate',reference=str(a.output.resolve()),reference_sha256=sha(a.output),
        shot_start_frame=len(run['joint_pos']),planned_anchor_xy=[a.anchor_x,a.anchor_y],
        ball_xy=(np.array([-.4475531578063965,-1.232489824295044])+shift[:2]).tolist(),
        goal_xy=(np.array([0.,-3.7])+shift[:2]).tolist(),style='inside',mode='hip',
        running_checkpoint=str(root/'experiments/running_v011/run_stop_n1024_i500/last_model.pt'),
        source_sha256={str(run_path):sha(run_path),str(shot_path):sha(shot_path)},
        scope='Moving-start run, measured-support handoff, actual ball contact and recovery; no stationary-start claim')
    a.output.with_suffix('.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))


if __name__=='__main__':main()
