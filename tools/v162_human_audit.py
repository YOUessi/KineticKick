"""Summarize frozen human-reference geometry and paired released-policy CPU runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER_FIELDS = ['runtime_code_sha256', 'runner_sha256', 'controllers', 'duration_s',
                 'ball_initial_velocity_xy_mps', 'ball_initial_xyz_m']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_round(folder, expected_npz):
    trial = json.loads((folder/'trial.json').read_text())
    report = json.loads((folder/'tracking_result.json').read_text())
    if sha(expected_npz) != trial['external_reference_sha256'] or Path(trial['clip']).resolve()!=expected_npz.resolve():
        raise ValueError('Wrong human motion for declared round')
    for name, digest in report['local_artifact_sha256'].items():
        if sha(folder / name) != digest:
            raise ValueError('Changed input '+str(folder/name))
    with np.load(folder/'rollout.npz', allow_pickle=False) as a:
        t = a['time_s']
        if not np.array_equal(t, np.arange(1, 2801) * .005):
            raise ValueError('A complete 200 Hz, 14 s round is required')
        loaded=a['robot_ball_contact']
        foot_load=a['foot_ground_normal_load_N'][0].tolist()
        nonfoot=a['nonfoot_net_contact_peak_N']
        early=float(t[np.flatnonzero(nonfoot>20.)[0]]) if np.any(nonfoot>20.) else None
        if early!=report['first_nonfoot_net_force_above_20N_s']:
            raise ValueError('Reported first nonfoot load differs')
        with gzip.open(folder/'native_contacts.jsonl.gz','rt') as f:
            tick, rows=json.loads(next(f))
        if tick!=1:
            raise ValueError('Physical contact file starts at wrong sample')
        pairs=[]
        for rec in rows:
            if not set(rec['actors'])-{'ground','SoccerBall','left_ankle_roll_link','right_ankle_roll_link'}:
                continue
            force=float(np.linalg.norm(np.array(rec['points'])[:,6:9].sum(axis=0))/.005)
            if force>20.:
                pairs.append({'actors':rec['actors'],'reported_normal_resultant_N':force})
        summary=dict(physics_steps=len(t),first_200Hz_ground_load_left_right_N=foot_load,
                   initial_200Hz_nonfoot_pairs_above_20N=pairs,robot_ball_loaded_bins=int(loaded.sum()),
                   output_report_sha256=sha(folder/'tracking_result.json'),rollout_sha256=sha(folder/'rollout.npz'),
                   contact_sha256=sha(folder/'native_contacts.jsonl.gz'),reference_sha256=trial['external_reference_sha256'])
    keys=['goal_crossed_whole_ball','first_height_below_0p45m_s','first_tilt_over_60deg_s','first_nonfoot_net_force_above_20N_s',
          'first_measured_qdot_limit_excess_s','qdot_limit_sample_joint_count','q_position_limit_sample_joint_count','peak_qdot_ratio',
          'final_standing_seconds_cpu_definition','foot_material_slip_rms_mps_when_loaded','additional_state_writes_after_initialization']
    summary.update({k:report[k] for k in keys})
    summary['source_clip_tracking']=report['phase_metrics']['source_clip']
    return {k:trial[k] for k in DRIVER_FIELDS}, summary


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--records',type=Path,default=ROOT/'experiments/v162_human/references/manifest.json')
    cli.add_argument('--base',type=Path,default=ROOT/'experiments/v160_human')
    cli.add_argument('--pair',type=Path,default=ROOT/'experiments/v162_human')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    m=json.loads(args.records.read_text())
    rows=m['records']
    for row in rows:
        f=args.records.parent/row['npz_file']
        with np.load(f,allow_pickle=False) as ref:
            if sha(f)!=row['npz_sha256'] or np.min(ref['urdf_sole_height_m'])<0:
                raise ValueError('Reference SHA or registered floor mismatch')
            if (not np.array_equal(ref['time_s'], np.arange(len(ref['joint_pos']))/50)
                    or ref['verified_contact_label_available'].any() or ref['kick_leg_verified']):
                raise ValueError('Unsupported timing or contact annotations')
    dt=1./50
    # Distinguish pose displacement speed (frozen objective) from the exported
    # central difference velocity, whose one-sided endpoint can differ.
    spec=ROOT/'configs/v162_human_support.json'
    cfg=json.loads(spec.read_text())
    import xml.etree.ElementTree as ET
    xml=ET.parse(Path('/home/you/projects/kinetickick')/cfg['urdf'])
    exported={}
    for kind, folder in [('before',args.base),('after',args.records.parent)]:
        highs=[]
        for row in rows:
            with np.load(folder/row['npz_file']) as a:
                limit=np.array([float(xml.find(f"joint[@name='{name}']/limit").get('velocity')) for name in a['joint_names']])
                highs.append(float(np.max(np.abs(a['joint_vel'])/limit)))
        exported[kind]=float(max(highs))
    pooled={}
    for subset in ['all','train','validation','test']:
        subset_rows=[r for r in rows if subset=='all' or r['split']==subset]
        pooled[subset]={}
        for label in ['before','after']:
            sel=[r[label] for r in subset_rows]
            def vals(k):
                return [r[k] for r in sel]
            frame_counts=vals('frames')
            sums=['feet_below_plane_over_10mm_frames','source_support_hover_over_10mm_samples',
                  'source_support_hover_over_20mm_samples','source_any_support_both_robot_feet_above_10mm_frames']
            checks={name:sum(vals(name)) for name in sums}
            checks.update(frames=sum(frame_counts),lowest_sole_m=min(vals('lowest_sole_capsule_m')),
                    median_of_clip_locked_material_speed_p95_mps=float(np.median(vals('locked_material_point_xy_speed_p95_mps'))),
                    median_of_clip_ankle_speed_p95_mps=float(np.median(vals('ankle_xy_speed_on_all_source_quiet_p95_mps'))),
                    max_joint_segment_speed_ratio=max(vals('max_joint_speed_ratio')),
                    direction_mean_deg=float(np.average(vals('direction_mean_deg'),weights=frame_counts)),
                    clip_direction_velocity_cosine_frame_weighted=float(np.average(vals('human_robot_direction_velocity_cosine'),weights=frame_counts)),
                    per_direction_mean_error_deg=dict(zip(m['registered_features'],
                        np.average(vals('direction_mean_by_group_deg'),weights=frame_counts,axis=0).tolist())),
                    joint_accel_rms_rad_s2=float(np.sqrt(np.average(np.asarray(vals('joint_acceleration_rms_rad_s2'))**2,
                        weights=np.asarray(frame_counts)-2))),
                    avg_root_displacement_rms_m=float(np.average(vals('root_xyz_correction_rms_m'),weights=frame_counts)))
            pooled[subset][label]=checks
    pair={}
    locked=None
    for key,folder,reference in [('old',args.pair/'physics_old_10_02',args.base/'10_02.npz'),
                               ('support',args.pair/'physics_support_10_02',args.records.parent/'10_02.npz')]:
        driver, stats=audit_round(folder,reference)
        if locked is not None and driver!=locked:
            raise ValueError('The physical pair used differing runners, scenarios or intervention')
        locked=driver
        pair[key]=stats
    out=dict(source_manifest_sha256=sha(args.records),audit_code_sha256=sha(Path(__file__)),source_license=m['source_license'],
             previously_examined_subject_11=True,physical_clip='10_02',same_physical_driver=locked,
             clip_list=[r['clip'] for r in rows],kinematic=pooled,exported_joint_vel_max_abs_over_original_urdf=exported,
             complete_14s_physics_pair=pair,
             interpretation='Sole/contact locking measures geometric material motion at 50 Hz; source force, pitch quality and true contact unknown. '
             'The two physical rounds compare each reference from its own first frame with the same unchanged CPU policy and real ball. '
             'Both have early robot self contact and falls before hitting the ball. No standing, shot or transfer claim.')
    with args.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
