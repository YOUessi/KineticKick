#!/usr/bin/env python3
"""Preserve KTH multi-view reconstruction and export scale-free relations.

No reconstruction, interpolation, smoothing, metric scaling or event detection.
Reprojection and length variation are diagnostics, not independent 3D accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/kth_coordination_v1.json'


def load_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text())


def verify_sources(directory, config):
    hashes = {}
    for name, expected in config['input_files'].items():
        payload = (Path(directory) / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected['size_bytes'] or digest != expected['sha256']:
            raise ValueError(f'Source size/hash mismatch: {name}')
        hashes[name] = digest
    return hashes


def load_inputs(directory, config=None):
    config = load_config() if config is None else config
    hashes = verify_sources(directory, config)
    directory = Path(directory)
    t, j, c = config['frames'], len(config['joint_names']), config['cameras']
    # These are the exact loop orders in the official CodeRelease/readme.txt.
    xyz = np.loadtxt(directory / 'positions3d.txt').reshape(t, j, 3)
    xy = np.loadtxt(directory / 'positions2d.txt').reshape(t, c, j, 2)
    cameras = np.loadtxt(directory / 'cameras.txt').reshape(t, c, 4, 2).transpose(0,1,3,2)
    source = {'source_class': config['source_class'], 'source_hashes': hashes,
              'professional_match_context_verified': True,
              'goal_sequence_label_verified': True,
              'evidence': config['provenance']}
    return xy, xyz, cameras, source


def _summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'rms': None, 'min': None, 'median': None, 'p95': None, 'max': None}
    return {'count': int(len(values)), 'rms': float(np.sqrt(np.mean(values**2))),
            'min': float(np.min(values)), 'median': float(np.median(values)), 'p95': float(np.quantile(values,.95)),
            'max': float(np.max(values))}


def build_coordination(positions2d, positions3d, cameras, *, source, config=None):
    config = load_config() if config is None else config
    xy, xyz, cam = [np.array(v, dtype=np.float64, copy=True)
                    for v in (positions2d, positions3d, cameras)]
    joints = config['joint_names']; j = len(joints)
    if xyz.ndim != 3 or xyz.shape[1:] != (j,3) or not len(xyz):
        raise ValueError('Invalid 3D joint shape')
    t = len(xyz); c = config['cameras']
    if xy.shape != (t,c,j,2) or cam.shape != (t,c,2,4):
        raise ValueError('Mismatched 2D/camera shape')
    fps = float(config['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source sampling rate')
    joint_valid = np.isfinite(xyz).all(axis=-1)
    observation_valid = np.isfinite(xy).all(axis=-1)
    camera_valid = np.isfinite(cam).all(axis=(-1,-2))
    homogeneous = np.concatenate([xyz, np.ones((t,j,1))], axis=-1)
    projected = np.einsum('tcij,tnj->tcni', cam, homogeneous)
    projection_valid = joint_valid[:,None,:] & camera_valid[:,:,None]
    projected[~projection_valid] = np.nan
    reprojection_valid = projection_valid & observation_valid
    residual = projected - xy
    residual[~reprojection_valid] = np.nan
    residual_norm = np.linalg.norm(residual, axis=-1)
    camera_count = reprojection_valid.sum(axis=1)
    joint_rms = np.full((t,j),np.nan)
    np.divide(np.nansum(residual_norm**2,axis=1),camera_count,
              out=joint_rms,where=camera_count>0)
    joint_rms = np.sqrt(joint_rms)

    # Missing thorax/foot landmarks stay explicit. Shoulder midpoint is a
    # separate proxy, used for a coarse trunk line only as declared in config.
    landmark_names = list(config['landmarks'])
    landmarks = np.full((t,len(landmark_names),3),np.nan)
    landmark_valid = np.zeros(landmarks.shape[:2],dtype=bool)
    landmark_defined = np.zeros(len(landmark_names),dtype=bool)
    for k,name in enumerate(landmark_names):
        contributors = config['landmarks'][name]
        if not contributors:
            continue
        indices = [joints.index(n) for n in contributors]
        good = joint_valid[:,indices].all(axis=1)
        landmarks[good,k] = xyz[good][:,indices].mean(axis=1)
        landmark_valid[:,k] = good
        landmark_defined[k] = True
    pelvis_id = landmark_names.index('pelvis_center')
    pelvis = landmarks[:,pelvis_id]
    relative_valid = joint_valid & landmark_valid[:,pelvis_id,None]
    relative = xyz - pelvis[:,None,:]
    relative[~relative_valid] = np.nan

    segment_names = list(config['segments'])
    vectors = np.full((t,len(segment_names),3),np.nan)
    lengths = np.full(vectors.shape[:2],np.nan)
    directions = np.full_like(vectors,np.nan)
    length_valid = np.zeros(lengths.shape,dtype=bool)
    segment_valid = np.zeros_like(length_valid)
    medians = np.full(len(segment_names),np.nan)
    deviation = np.full_like(lengths,np.nan)
    cvs = np.full(len(segment_names),np.nan)
    for k,name in enumerate(segment_names):
        a,b = [landmark_names.index(n) for n in config['segments'][name]]
        good = landmark_valid[:,a] & landmark_valid[:,b]
        delta = landmarks[:,b] - landmarks[:,a]
        length = np.linalg.norm(delta,axis=-1)
        vectors[good,k] = delta[good]
        lengths[good,k] = length[good]
        length_valid[:,k] = good
        directional = good & (length > config['direction_epsilon_au'])
        segment_valid[:,k] = directional
        directions[directional,k] = delta[directional]/length[directional,None]
        if good.any():
            selected = length[good]
            medians[k] = np.median(selected)
            if medians[k] > 0:
                deviation[good,k] = selected/medians[k]-1.
            if selected.mean() > 0:
                cvs[k] = selected.std()/selected.mean()

    # A dimensionless normalizer, never a body-height or meter calibration.
    chain_lengths = []
    for side in ('left','right'):
        a,b = [segment_names.index(f'{side}_{n}') for n in ('thigh','shank')]
        good = segment_valid[:,a] & segment_valid[:,b]
        chain_lengths.extend((lengths[good,a]+lengths[good,b]).tolist())
    scale = float(np.median(chain_lengths)) if chain_lengths else float('nan')
    normalized_valid = relative_valid & np.isfinite(scale) & (scale > 0)
    normalized = np.full_like(relative,np.nan)
    if np.isfinite(scale) and scale > 0:
        normalized[normalized_valid] = relative[normalized_valid]/scale
    pairs = config['relation_pairs']
    relation_names = ['__'.join(p) for p in pairs]
    cosines = np.full((t,len(pairs)),np.nan)
    relation_valid = np.zeros(cosines.shape,dtype=bool)
    for k,(left,right) in enumerate(pairs):
        a,b = segment_names.index(left),segment_names.index(right)
        good = segment_valid[:,a] & segment_valid[:,b]
        relation_valid[:,k] = good
        cosines[good,k] = np.clip(np.einsum('ti,ti->t',directions[good,a],directions[good,b]),-1,1)
    arrays = {
        'source_frame_numbers':np.arange(1,t+1,dtype=np.int64),
        'time_s':np.arange(t,dtype=np.float64)/fps, 'fps':np.array(fps),
        'joint_names':np.array(joints), 'joints_reconstructed_au':xyz,
        'joint_geometry_valid':joint_valid, 'joint_visibility_known':np.zeros((t,j),dtype=bool),
        'joint_independent_3d_accuracy_known':np.zeros((t,j),dtype=bool),
        'pipeline_joint_interpolated':np.zeros((t,j),dtype=bool),
        'source_joint_mean_au':np.mean(xyz,axis=1),
        'manual_positions2d_px':xy, 'manual_2d_finite':observation_valid,
        'orthographic_cameras':cam, 'camera_geometry_valid':camera_valid,
        'joints_reprojected_px':projected, 'projection_geometry_valid':projection_valid,
        'reprojection_residual_vector_px':residual, 'reprojection_residual_norm_px':residual_norm,
        'reprojection_valid':reprojection_valid, 'joint_reprojection_rms_px':joint_rms,
        'joint_reprojection_camera_count':camera_count,
        'landmark_names':np.array(landmark_names), 'landmarks_source_axes_au':landmarks,
        'landmark_geometry_valid':landmark_valid, 'landmark_definition_available':landmark_defined,
        'pelvis_midpoint_au':pelvis, 'joints_pelvis_relative_au':relative,
        'joint_pelvis_relative_valid':relative_valid, 'length_scale_au':np.array(scale),
        'joints_normalized':normalized, 'joint_normalized_valid':normalized_valid,
        'segment_names':np.array(segment_names), 'segment_vectors_source_axes_au':vectors,
        'segment_lengths_au':lengths, 'segment_length_valid':length_valid,
        'segment_directions_source_axes':directions, 'segment_geometry_valid':segment_valid,
        'segment_length_reference_median_au':medians,
        'segment_length_relative_deviation':deviation, 'segment_length_cv':cvs,
        'relation_names':np.array(relation_names), 'relation_cosines':cosines,
        'relation_geometry_valid':relation_valid,
        'relation_precision_known':np.zeros(cosines.shape,dtype=bool),
        'source_class':np.array(source['source_class'])}
    residual_stats = _summary(residual_norm)
    metadata = {
        'schema_version':1, 'source':source, 'frames':t, 'fps':fps,
        'coordinate_system':config['coordinate_system'],
        'time_semantics':config['time_semantics'], 'landmark_semantics':config['landmark_semantics'],
        'relation_semantics':config['relation_semantics'], 'diagnostic_semantics':config['diagnostic_semantics'],
        'license':config['license'],
        'qualification':{
            'professional_match_context_verified':bool(source.get('professional_match_context_verified',False)),
            'goal_sequence_label_verified':bool(source.get('goal_sequence_label_verified',False)),
            'named_player_verified':False, 'exact_shot_frame_verified':False,
            'metric_meter_scale_available':False, 'independent_3d_accuracy_available':False,
            'instep_verified':False, 'p2_complete':False},
        'truth_labels':{'instep':None,'kick_side':None,'effort':None,'contact_time_s':None,
                        'shot_frame':None,'ground_plane':None,'ball_center':None},
        'diagnostics':{
            'reprojection_rms_px':residual_stats['rms'],
            'reprojection_residual_px':residual_stats,
            'per_joint_residual_px':{name:_summary(residual_norm[:,:,k]) for k,name in enumerate(joints)},
            'per_segment_length':{name:{'finite_frames':int(length_valid[:,k].sum()),
                'median_au':float(medians[k]) if np.isfinite(medians[k]) else None,
                'cv':float(cvs[k]) if np.isfinite(cvs[k]) else None,
                'relative_deviation':_summary(deviation[:,k])} for k,name in enumerate(segment_names)}},
        'availability':{
            'joint_geometry_frames':int(joint_valid.sum()),
            'relation_geometry_frames':int(relation_valid.sum()),
            'relation_precision_known_frames':0,
            'per_relation_geometry_frames':{name:int(relation_valid[:,k].sum()) for k,name in enumerate(relation_names)},
            'anatomical_thorax_landmarks':False,'toe_and_heel_landmarks':False,
            'coarse_trunk_uses_shoulder_midpoint_proxy':True},
        'processing':{'interpolation':False,'smoothing':False,'quality_filtering':False,
                      'new_reconstruction':False,'metric_scaling':False},
        'npz_fields':{name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in arrays.items()}}
    return arrays,metadata


def plot_projection(arrays, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    obs, pred = arrays['manual_positions2d_px'],arrays['joints_reprojected_px']
    error = arrays['reprojection_residual_norm_px']
    score = np.nanmean(error**2,axis=(1,2))
    worst = int(np.nanargmax(score))
    frames = [0,worst,len(obs)-1]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11),
             (2,8),(3,9),(12,13)]
    fig,axes = plt.subplots(3,3,figsize=(11,10))
    for row,t in enumerate(frames):
        for c in range(3):
            ax=axes[row,c]
            for points,color in ((obs[t,c],'#2266aa'),(pred[t,c],'#d96a13')):
                for a,b in edges:
                    ax.plot(points[[a,b],0],points[[a,b],1],color=color,alpha=.55,lw=.8)
            ax.scatter(obs[t,c,:,0],obs[t,c,:,1],s=18,color='#2266aa',label='Manual 2D labels')
            ax.scatter(pred[t,c,:,0],pred[t,c,:,1],s=22,color='#d96a13',marker='x',label='Projected 3D')
            ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.grid(alpha=.2)
            rms=float(np.sqrt(np.nanmean(error[t,c]**2)))
            ax.set_title(f'Frame {t+1}, camera {c+1}; RMS {rms:.1f} px',fontsize=9)
            ax.set_xlabel('Image x (px)');ax.set_ylabel('Image y (px)')
    axes[0,0].legend(fontsize=7,loc='best')
    fig.suptitle('KTH P1S2: first / largest frame RMS / last\nIn-sample reprojection diagnostic; independent 3D accuracy unknown',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(path,dpi=140);plt.close(fig)
    return {'source_frames':[f+1 for f in frames], 'selected_by':'first, largest all-camera joint residual RMS, last'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();config=load_config(args.config)
    xy,xyz,cam,source=load_inputs(args.source_dir,config)
    arrays,metadata=build_coordination(xy,xyz,cam,source=source,config=config)
    metadata['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stem='kth_player1_sequence2_coordination'
    npz=args.output_dir/(stem+'.npz');meta=args.output_dir/(stem+'.json')
    np.savez_compressed(npz,**arrays);outputs=[npz,meta]
    if args.plot:
        figure=args.output_dir/'projection_diagnostic.png'
        metadata['projection_figure']=plot_projection(arrays,figure);outputs.append(figure)
    meta.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    manifest={'source_hashes':source['source_hashes'],'config_sha256':metadata['config_sha256'],
              'files':[{'name':p.name,'size_bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in outputs]}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'frames':len(xyz),'joint_geometry_frames':int(arrays['joint_geometry_valid'].sum()),
                      'relation_geometry_frames':int(arrays['relation_geometry_valid'].sum()),
                      'independent_3d_accuracy_known':False,'p2_complete':False,'output':str(args.output_dir)}))


if __name__=='__main__':
    main()
