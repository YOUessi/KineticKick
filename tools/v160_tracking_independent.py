"""Post-run CPU audit. Rebuild ONNX inputs/actions, safety timing and a named contact from saved states."""
import argparse
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_rotate(q_wxyz, vector):
    return Rotation.from_quat(np.asarray(q_wxyz)[[1,2,3,0]]).apply(vector,inverse=True)


def check(folder, assets):
    report=json.loads((folder/'tracking_result.json').read_text()); inventory=json.loads((folder/'source_inventory.json').read_text())
    for name,expected in report['local_artifact_sha256'].items():
        if hash_file(folder/name)!=expected:raise ValueError('Archived measurement digest mismatch: '+name)
    ref=np.load(assets/report['source_clip'],allow_pickle=False); trace=np.load(folder/'rollout.npz',allow_pickle=False)
    commands=trace['policy_commands_time_frame_action_target']; observed=trace['policy_observations']
    if commands.shape!=(700,60) or observed.shape!=(700,160):raise ValueError('Incomplete action stream')
    np.testing.assert_allclose(trace['time_s'],np.arange(1,2801)*.005,rtol=0,atol=1e-12)
    sys.path.insert(0,str(assets/'experiments/runtime/onnxruntime'))
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(assets/'.pretrained_checkpoints/paid_official/policy_30000.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map; names=meta['joint_names'].split(',')
    if names!=inventory['joint_names'] or names!=report['source_joint_names']:raise ValueError('Joint name provenance mismatch')
    defaults=np.fromstring(meta['default_joint_pos'],sep=',');scale=np.fromstring(meta['action_scale'],sep=',')
    torso=inventory['body_names'].index('torso_link'); goal=np.array([0.,-3.7,.11]); last=np.zeros(29); h=np.zeros((2,1,128),np.float32);c=h.copy();max_obs=0.;max_act=0.;max_drive=0.;max_time=0.
    for tick,row in enumerate(commands):
        frame=min(tick,len(ref['joint_pos'])-1)
        if int(row[1])!=frame:raise ValueError('Incorrect held reference frame')
        max_time=max(max_time,abs(row[0]-tick*.02))
        if tick==0:
            p=ref['body_pos_w'][0,0];quat=ref['body_quat_w'][0,0];omega=ref['body_ang_vel_w'][0,0]
            q=ref['joint_pos'][0];dq=ref['joint_vel'][0];ball=np.array([.14,-.93,.113],dtype=np.float32)
        else:
            preceding=tick*4-1; p=trace['root_pose_wxyz'][preceding,:3];quat=trace['root_pose_wxyz'][preceding,3:]
            omega=trace['root_velocity_w'][preceding,3:];q=trace['joint_q_rad'][preceding];dq=trace['joint_dq_rad_s'][preceding];ball=trace['ball_position_w_m'][preceding]
        calculated=np.concatenate((ref['joint_pos'][frame],ref['joint_vel'][frame],inverse_rotate(quat,[0.,0.,-1.]),
            ref['body_ang_vel_w'][frame,torso],inverse_rotate(quat,omega),q-defaults,dq,last,
            inverse_rotate(quat,ball-p),inverse_rotate(quat,goal-p))).astype(np.float32)
        max_obs=max(max_obs,float(np.max(np.abs(calculated-observed[tick]))))
        action,h,c=session.run(['actions','h_out','c_out'],{'obs':observed[tick:tick+1], 'h_in':h,'c_in':c,'time_step':np.array([[frame]],np.float32)})
        max_act=max(max_act,float(np.max(np.abs(action[0]-row[2:31]))));last=action[0].clip(-100,100)
        max_drive=max(max_drive,float(np.max(np.abs(defaults+scale*last-row[31:]))))
    if max_obs>2e-5 or max_act>1e-6 or max_drive>1e-6 or max_time>1e-10:
        raise ValueError(f'CPU baseline command/state re-audit failed: {max_obs}, {max_act}, {max_drive}, {max_time}')
    xml=ET.parse(assets/'configs/paid_original/paid_main_resolved.urdf');joints={e.get('name'):e.find('limit') for e in xml.findall('joint') if e.find('limit') is not None}
    limits=np.asarray([[float(joints[n].get(k)) for k in ('lower','upper','velocity')] for n in names]);q=trace['joint_q_rad'];dq=trace['joint_dq_rad_s'];t=trace['time_s']
    ex=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0);fail=ex>1e-6;peak=np.unravel_index(ex.argmax(),ex.shape)
    qdot_count=int(np.count_nonzero(abs(dq)>limits[:,2]+1e-6));first_q_failure=float(t[np.flatnonzero(fail.any(axis=1))[0]]) if fail.any() else None
    # Independent impulse vector accumulation, keyed by native actor/shape identity rather than link distance.
    first_ball=first_wrist_hip=first_nonfoot=None;row_count=0
    with gzip.open(folder/'native_contacts.jsonl.gz','rt') as inp:
        for line in inp:
            step,pairs=json.loads(line);row_count+=1
            if row_count!=step:raise ValueError('Non-contiguous native contact clock')
            forces={}
            for d in pairs:
                names_pair=d['actors']; points=np.asarray(d['points'],float)
                if points.ndim!=2 or points.shape[1]!=10 or not np.isfinite(points).all():raise ValueError('Malformed native point vector')
                f=points[:,6:9].sum(0)/.005
                for actor,sign in zip(names_pair,(1.,-1.)):
                    if actor not in ('ground','SoccerBall','left_ankle_roll_link','right_ankle_roll_link'):
                        forces[actor]=forces.get(actor,np.zeros(3))+f*sign
                n=float(np.linalg.norm(f)); event={'time_s':step*.005,'actors':names_pair,'shapes':d['shapes'],'net_impulse_norm_over_dt_N':n,'contact_points':len(points)}
                if first_ball is None and 'SoccerBall' in names_pair and 'ground' not in names_pair and n>2.:first_ball=event
                if first_wrist_hip is None and set(names_pair)=={'left_wrist_yaw_link','left_hip_roll_link'} and n>20.:first_wrist_hip=event
            n=max([float(np.linalg.norm(v)) for v in forces.values()] or [0.]);
            if first_nonfoot is None and n>20.:first_nonfoot=step*.005
    if row_count!=len(t) or not np.isclose(first_nonfoot, report['first_nonfoot_net_force_above_20N_s']) or first_ball['time_s']!=report['first_robot_ball_s'] or int(fail.sum())!=report['q_position_limit_sample_joint_count'] or qdot_count!=report['qdot_limit_sample_joint_count']:
        raise ValueError('Independent raw physics audit disagrees with report')
    return {'audit':'offline reconstruction and same-state actor replay, 0 new simulations','control_steps':len(commands),'physics_steps':len(t),
        'physics_sample_zero_origin':'post-step sample 1 is 0.005s; command 0 uses raw reference initialization; every subsequent 20ms action uses the already measured preceding substep',
        'max_observation_abs_error':max_obs,'max_recurrent_action_abs_error':max_act,'max_target_abs_error_rad':max_drive,'max_control_time_abs_error_s':max_time,
        'first_native_ball_contact':first_ball,'first_native_left_wrist_hip_over20N':first_wrist_hip,'first_nonfoot_over20N_s':first_nonfoot,
        'first_actual_joint_position_excess_s':first_q_failure,'joint_position_excess_sample_joint_count':int(fail.sum()),'source_clip_position_excess_count':int(fail[t<=(len(ref['joint_pos'])-1)/50.+1e-9].sum()),
        'max_joint_position_excess_rad':float(ex[peak]),'peak_position_excess_joint':names[peak[1]],'peak_position_excess_time_s':float(t[peak[0]]),
        'raw_qdot_limit_excess_count':qdot_count,'official_MuJoCo_observation_difference':'The upstream exp/runner.py builds from zero reference channels at reset, then the previous call\'s references. This CPU PhysX port follows the project Isaac current-frame reference convention. MuJoCo applies explicitly clipped torques; this port uses effort-limited implicit position drives.',
        'scope':'Actor weights, normalization, I/O, clocks, per-joint mapping and actual native CPU contact identities are checked here. This CPU experiment is not an original USD/Isaac CUDA validation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check(a.run,a.assets)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
