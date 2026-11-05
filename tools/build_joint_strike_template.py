#!/usr/bin/env python3
"""Extract original actuator commands and observed states; no generated assets in git."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    from strike_baseline_metrics import validated_episode
    from strike_baseline_protocol import verify_files
    matrix=json.loads((ROOT/'configs/strike_baseline_v1.json').read_text());verify_files(ROOT,matrix['frozen_files_sha256'])
    source=ROOT/'reports/v023_baseline_center.json';raw=json.loads(source.read_text());trace=validated_episode(raw)
    if not raw['stride_shot_success']:raise ValueError('Successful frozen physical source required')
    k=raw['kinematics'][0];assert len(k)==700 and len(trace)==2800
    folder=ROOT/'experiments/joint_strike_v025';folder.mkdir(exist_ok=True,parents=True);out=folder/'nominal_center.npz'
    if out.exists():raise FileExistsError(out)
    root=np.array([r['root_pose'] for r in k],np.float32);velocity=np.gradient(root[:,:3],.02,axis=0)
    velocity[0]=raw['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    metadata=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  joint_names=raw['joint_names'],control_dt=.02,physics_dt=.005,velocity_reference='20ms centered root finite difference; measured reset velocity at first frame')
    np.savez_compressed(out,metadata=json.dumps(metadata),root_pose=root,root_velocity=velocity,
        joint_pos=np.array([r['joint_pos'] for r in k],np.float32),action=np.array([r['action'] for r in k],np.float32),
        velocity_command=np.array([trace[i*4]['joint_velocity_target'] for i in range(700)],np.float32))
    base=json.loads((ROOT/'configs/stride_instep_strong_v1.json').read_text())
    config=dict(version='0.1.25',status='experimental_not_accepted',template=str(out.relative_to(ROOT)),template_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        reference=base['reference'],reference_sha256=base['reference_sha256'],ball_xy=base['ball_xy'],goal_xy=base['goal_xy'],
        maximum_joint_residual_rad=.25,maximum_phase_shift_s=.15,phase_rate_limit=.30,seed=2501,
        frozen_files_sha256=matrix['frozen_files_sha256'],source_metadata=metadata,
        training=dict(nominal_fraction=.5,ball_jitter_xy_m=[.05,.10],initial_velocity_y_jitter_mps=.20,max_seconds=900,max_iterations=1000),
        scope='Single learned residual/phase controller across real approach-contact-recovery; frozen successful actuator trajectory prior, no human coordination claim')
    (ROOT/'configs/joint_strike_learning_v1.json').write_text(json.dumps(config,indent=2)+'\n')
    print(out,config['template_sha256'])

if __name__=='__main__':main()
