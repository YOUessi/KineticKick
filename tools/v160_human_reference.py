"""Offline full-clip CMU optical-marker to original G1 29-joint reference.

Use env_isaaclab/bin/python -s with BLAS/OMP threads set to one. The runtime
sample grid is exactly 50 Hz; a separately saved source grid includes both
original endpoints. All contact, support and sole measurements here are
kinematic proxies. No physics, network or model selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import ezc3d
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CMU = Path('data/research_human_kicks/cmu_soccer_120hz_v050')
MARKERS = ('LHJ0 RHJ0 LKNE RKNE LANK RANK LTOE RTOE LHEE RHEE LMT5 RMT5 '
           'STRN LSHO RSHO LELB RELB LWRA RWRA LWRB RWRB LFIN RFIN LFWT RFWT LBWT RBWT').split()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length < 1e-7) or not np.isfinite(length).all():
        raise ValueError('Degenerate source or robot anatomical direction')
    return v / length

def marker_frame(forward, lateral):
    """Proper right-handed frame, axes forward, anatomical left, up."""
    x = unit(forward)
    z = unit(np.cross(x, lateral))
    return np.stack([x, np.cross(z, x), z], axis=-1)

def source_grid(frame_count, source_fps, target_fps):
    duration = (frame_count - 1) / source_fps
    n = round(duration * target_fps) + 1
    return np.arange(n) / target_fps, np.linspace(0., duration, n)

def interpolate_markers(xyz, valid, source_t, query_t):
    """Fill only separately flagged, bounded source gaps for IK input."""
    query = query_t * (1. / (source_t[1] - source_t[0]))
    query = np.where(np.abs(query - np.rint(query)) < 1e-9, np.rint(query), query)
    lo = np.floor(query).astype(int).clip(0, len(xyz)-1)
    hi = np.ceil(query).astype(int).clip(0, len(xyz)-1)
    observed = valid[lo] & valid[hi]
    out = np.empty((len(query_t), xyz.shape[1], 3))
    for j in range(xyz.shape[1]):
        mask = valid[:, j]
        missing = np.flatnonzero(~mask)
        groups = np.split(missing, np.flatnonzero(np.diff(missing)>1)+1)
        if mask.sum()<2 or any(len(group)>3 for group in groups):
            raise ValueError('Marker gap exceeds the fixed three-frame input policy')
        for axis in range(3):
            out[:, j, axis] = np.interp(query_t, source_t[mask], xyz[mask,j,axis])
    return out, observed

def read_mocap(path, processed_path, hz):
    c = ezc3d.c3d(str(path))
    params = c['parameters']['POINT']
    if params['RATE']['value'] != [120.] or params['UNITS']['value'] != ['mm']:
        raise ValueError('Expected original 120 Hz millimetre C3D')
    labels = [v.upper().split(':')[-1] for v in params['LABELS']['value']]
    if any(labels.count(k)!=1 for k in MARKERS):
        raise ValueError('Missing or ambiguous anatomical C3D marker')
    columns = [labels.index(k) for k in MARKERS]
    xyz = c['data']['points'][:3, columns].transpose(2,1,0)*.001
    residual = c['data']['meta_points']['residuals'][0,columns].T
    valid = np.isfinite(xyz).all(-1) & (xyz!=0).any(-1) & (residual>=0)
    ts = np.arange(len(xyz))/120.
    t, source_t = source_grid(len(ts),120.,hz)
    with np.load(processed_path,allow_pickle=False) as saved:
        if saved['primary_landmark_names'].tolist()!=MARKERS[:19] or not np.allclose(saved['time_s'],ts):
            raise ValueError('Raw and archived marker schema/time disagree')
        shared = valid[:,:19] & saved['primary_marker_valid']
        if not np.allclose(saved['primary_human_3d_m'][shared],xyz[:,:19][shared],atol=5e-7,rtol=0):
            raise ValueError('Raw and previously audited primary marker positions disagree')
    sampled, observed = interpolate_markers(xyz,valid,ts,source_t)
    return dict(zip(MARKERS,sampled.transpose(1,0,2))),t,source_t,observed,valid,c['header']['points']['first_frame']

def angle_quaternions(rotations):
    flat = Rotation.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(*rotations.shape[:-2],4)
    signs = np.where((flat[:-1]*flat[1:]).sum(-1)<0,-1.,1.)
    flat[1:] *= np.cumprod(signs,axis=0)[...,None]
    return flat[..., [3,0,1,2]]

def world_angular_velocity(rotations, dt):
    relative = rotations[1:] @ rotations[:-1].swapaxes(-1,-2)
    step = Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(*relative.shape[:-2],3)/dt
    return np.concatenate([step[:1],.5*(step[:-1]+step[1:]),step[-1:]])

def task_definitions():
    # For position-direction targets both endpoints are link origins, except for
    # anatomical midpoints explicitly represented as two equally weighted frames.
    edges = [(('left_hip_pitch_link','right_hip_pitch_link'),('left_shoulder_pitch_link','right_shoulder_pitch_link')),
             (('right_shoulder_pitch_link',),('left_shoulder_pitch_link',))]
    axes = []
    for side in ('left','right'):
        n = lambda suffix:(f'{side}_{suffix}_link',)
        edges += [(n('hip_pitch'),n('knee')),(n('knee'),n('ankle_roll')),
                  (n('shoulder_pitch'),n('elbow')),(n('elbow'),n('wrist_yaw'))]
        axes += [(n('ankle_roll')[0],0),(n('ankle_roll')[0],1),
                 (n('wrist_yaw')[0],0),(n('wrist_yaw')[0],1)]
    return edges,axes

def human_features(p):
    hip = .5*(p['LHJ0']+p['RHJ0'])
    shoulder = .5*(p['LSHO']+p['RSHO'])
    # Waist belt axes avoid using the shoulder twist to define the root yaw.
    root_r = marker_frame(p['LFWT']+p['RFWT']-p['LBWT']-p['RBWT'],
                          p['LFWT']+p['LBWT']-p['RFWT']-p['RBWT'])
    vectors=[unit(shoulder-hip),unit(p['LSHO']-p['RSHO'])];axes=[];human_sole=[]
    for label,side_sign in [('L',1.),('R',-1.)]:
        wrist=.5*(p[label+'WRA']+p[label+'WRB'])
        for a,b in [(p[label+'HJ0'],p[label+'KNE']), (p[label+'KNE'],p[label+'ANK']),
                    (p[label+'SHO'],p[label+'ELB']), (p[label+'ELB'],wrist)]:
            vectors.append(unit(b-a))
        foot=marker_frame(p[label+'TOE']-p[label+'HEE'],side_sign*(p[label+'MT5']-p[label+'TOE']))
        hand=marker_frame(p[label+'FIN']-wrist,side_sign*(p[label+'WRB']-p[label+'WRA']))
        axes.extend([foot[:,:,0],foot[:,:,1],hand[:,:,0],hand[:,:,1]])
        human_sole.append(np.stack([p[label+'HEE'],p[label+'MT5'],p[label+'TOE']],axis=1))
    world=np.stack(vectors+axes,axis=1)
    return np.einsum('tji,tkj->tki',root_r,world),root_r,hip,np.stack(human_sole,axis=1)

class Robot:
    """Analytic original-URDF link-direction Jacobians in the pelvis frame."""
    def __init__(self,urdf,schema):
        self.model=pin.buildModelFromUrdf(str(urdf));self.data=self.model.createData()
        self.joint_names=schema['joint_names'];self.body_names=schema['body_names']
        if self.model.nq!=29 or self.model.nv!=29 or len(set(self.joint_names))!=29 or self.body_names[0]!='pelvis' or len(set(self.body_names))!=30:
            raise ValueError('Unexpected original G1 schema')
        self.runtime_indices=np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names]);
        if set(self.runtime_indices)!=set(range(29)):raise ValueError('Wrong PAiD to URDF joint correspondence')
        self.ids=[int(self.model.getFrameId(n)) for n in self.body_names];self.index={n:i for i,n in enumerate(self.body_names)}
        if any(frame_id >= self.model.nframes for frame_id in self.ids):raise ValueError('Missing original runtime link')
        edges,axes=task_definitions();self.axes=[(self.index[n],k) for n,k in axes]
        self.edge_matrix=np.zeros((len(edges),30))
        for i,(start,end) in enumerate(edges):
            for names,sign in ((start,-1.),(end,1.)):
                for n in names:self.edge_matrix[i,self.index[n]]+=sign/len(names)
        self.soles=[]
        xml=ET.parse(urdf).getroot()
        for side in ('left','right'):
            shapes=[]
            for col in xml.find(f"link[@name='{side}_ankle_roll_link']").findall('collision'):
                cyl=col.find('geometry/cylinder');origin=col.find('origin')
                if cyl is None:raise ValueError('Unexpected G1 sole primitive')
                xyz=np.fromstring(origin.get('xyz'),sep=' ');rpy=np.fromstring(origin.get('rpy'),sep=' ')
                half_axis=.5*float(cyl.get('length'))*Rotation.from_euler('xyz',rpy).as_matrix()[:,2]
                for point in (xyz-half_axis,xyz+half_axis):shapes.append((*point,float(cyl.get('radius'))))
            self.soles.append(np.asarray(shapes))
        self.low=self.model.lowerPositionLimit.copy();self.high=self.model.upperPositionLimit.copy()
        self.vlim=self.model.velocityLimit.copy()

    def poses(self,q):
        pin.forwardKinematics(self.model,self.data,q);pin.updateFramePlacements(self.model,self.data)
        poses=[self.data.oMf[i] for i in self.ids]
        return np.stack([v.translation for v in poses]),np.stack([v.rotation for v in poses])

    def features(self,q):
        pos,rot=self.poses(q);pin.computeJointJacobians(self.model,self.data,q)
        jac=np.stack([pin.getFrameJacobian(self.model,self.data,i,pin.LOCAL_WORLD_ALIGNED) for i in self.ids])
        vec=self.edge_matrix@pos;length=np.linalg.norm(vec,axis=1);direction=vec/length[:,None]
        dvec=(self.edge_matrix@jac[:,:3].reshape(30,-1)).reshape(len(vec),3,29)
        proj=np.eye(3)[None]-direction[:,:,None]*direction[:,None,:]
        jd=(proj@dvec)/length[:,None,None]
        r=np.stack([rot[link,:,axis] for link,axis in self.axes]);omega=np.stack([jac[link,3:].T for link,_ in self.axes])
        jr=np.cross(omega,r[:,None,:]).swapaxes(1,2)
        return np.concatenate([direction,r]),np.concatenate([jd,jr])

def solve_ik(robot, targets, cfg):
    weights=np.sqrt(np.asarray(cfg['task_weights']))[:,None]
    if len(weights)!=targets.shape[1]:raise ValueError('Target and weights do not agree')
    zero=np.clip(np.zeros(29), robot.low, robot.high)
    q=zero.copy(); prev=q.copy();result=[]; errors=[];neutral_features=robot.features(zero)[0]
    for i,target in enumerate(targets):
        # Warm start preserves continuity; new angles must obey both velocity and position limits.
        low,high=robot.low,robot.high
        if i:
            max_step=cfg['velocity_limit_fraction']*robot.vlim/cfg['output_fps']
            low=np.maximum(low,prev-max_step);high=np.minimum(high,prev+max_step)
        for _ in range(cfg['initial_iterations'] if not i else cfg['frame_iterations']):
            value,jac=robot.features(q)
            matrix=(jac*weights[:,:,None]).reshape(-1,29)
            residual=((target-value)*weights).ravel()
            prior=cfg['pose_prior'];temporal=cfg['previous_pose_penalty'] if i else 0.
            lhs=matrix.T@matrix+(cfg['damping']+prior+temporal)*np.eye(29)
            rhs=matrix.T@residual+prior*(zero-q)+temporal*(prev-q)
            step=np.linalg.solve(lhs,rhs)
            step*=min(1.,cfg['solver_step_rad']/max(np.abs(step).max(),1e-12))
            q_next=np.clip(q+step,low,high)
            if np.max(np.abs(q_next-q))<1e-5:
                q=q_next;break
            q=q_next
        value,_=robot.features(q)
        errors.append(np.rad2deg(np.arccos(np.clip((value*target).sum(-1),-1.,1.))))
        result.append(q.copy());prev=q.copy()
    neutral_error=np.rad2deg(np.arccos(np.clip(np.einsum('tki,ki->tk',targets,neutral_features),-1.,1.)))
    return np.stack(result),np.stack(errors),neutral_error

def geometry(robot, angles, global_r, hip, markers, human_feet, dt, cfg):
    local=[robot.poses(q) for q in angles];local_p=np.stack([x[0] for x in local]);local_r=np.stack([x[1] for x in local]);
    neutral_p,_=robot.poses(np.clip(np.zeros(29),robot.low,robot.high))
    robot_leg=[];human_leg=[]
    for name,side in [('left','L'),('right','R')]:
        ix=[robot.index[f'{name}_{j}_link'] for j in ('hip_pitch','knee','ankle_roll')]
        robot_leg.append(np.linalg.norm(neutral_p[ix[0]]-neutral_p[ix[1]])+np.linalg.norm(neutral_p[ix[1]]-neutral_p[ix[2]]))
        human_leg.append(np.linalg.norm(markers[side+'HJ0']-markers[side+'KNE'],axis=1)+np.linalg.norm(markers[side+'KNE']-markers[side+'ANK'],axis=1))
    scale=float(np.mean(robot_leg)/np.median(human_leg))
    hip_center=.5*(neutral_p[robot.index['left_hip_pitch_link']]+neutral_p[robot.index['right_hip_pitch_link']]);root=scale*hip-np.einsum('tij,j->ti',global_r,hip_center)
    root[:,:2]-=root[:1,:2].copy()
    global_p=root[:,None]+np.einsum('tij,tkj->tki',global_r,local_p)
    rotations=np.einsum('tij,tkjl->tkil',global_r,local_r)
    heights=[];feet=[]
    for side,shapes in zip(('left','right'),robot.soles):
        ix=robot.index[f'{side}_ankle_roll_link'];origin=global_p[:,ix];r=rotations[:,ix]
        ends=origin[:,None]+np.einsum('tij,kj->tki',r,shapes[:,:3]);heights.append((ends[:,:,2]-shapes[:,3]).min(axis=1));feet.append(origin)
    height=np.stack(heights,axis=1);foot=np.stack(feet,axis=1)
    # One constant vertical registration for the clip. This leaves later flying,
    # penetration and drift visible rather than pinning feet every frame.
    offset=-np.median(height[:cfg['floor_registration_frames']].min(axis=1))
    root[:,2]+=offset;global_p[:,:,2]+=offset;height+=offset
    source_height=human_feet[:,:,:,2].min(-1)
    source_speed=np.linalg.norm(np.gradient(human_feet.mean(axis=2)[:,:,:2],dt,axis=0),axis=-1)
    floor=np.quantile(source_height,.05)
    support=(source_height<floor+cfg['source_low_foot_band_m']) & (source_speed<cfg['source_quiet_foot_speed_mps'])
    foot_speed=np.linalg.norm(np.gradient(foot[:,:,:2],dt,axis=0),axis=-1)
    slip=foot_speed[support]
    statistics={'map_scale':scale,'constant_root_z_offset_m':float(offset),'source_quiet_low_foot_samples':int(support.sum()),
                'ankle_xy_speed_on_source_quiet_low_foot_mps_p95':float(np.quantile(slip,.95)) if slip.size else None,
                'urdf_capsule_lowest_relative_to_registered_plane_m':float(height.min()),
                'feet_below_plane_by_over_10mm_frames':int(np.any(height<-.01,axis=1).sum()),
                'stance_inference':'source low and slow foot geometry, unsynchronized with world forces',
                'body_position_source':'original G1 URDF FK; inferred pelvic pose with a clip-level z registration'}
    return root,global_p,rotations,height,support,statistics

def clip_reference(clip,group,asset,cfg,robot,sources,record,folder):
    b=asset/CMU
    raw=b/'raw'/clip[:2]/(clip+'.c3d');processed=b/'processed_v050'/(clip+'.npz')
    if sha(raw)!=sources[raw.name] or sha(processed)!=cfg['processed_sha256'][clip]:
        raise ValueError('Changed input motion '+clip)
    p,t,st,observed,raw_valid,first=read_mocap(raw,processed,cfg['output_fps'])
    targets,root_r,hip,human_sole=human_features(p)
    angles,err,neutral_err=solve_ik(robot,targets,cfg)
    dt=1./cfg['output_fps']
    root,world,r,sole_z,support,stats=geometry(robot,angles,root_r,hip,p,human_sole,dt,cfg)
    quat=angle_quaternions(r);q=angles[:,robot.runtime_indices];dq=np.gradient(q,dt,axis=0,edge_order=2)
    limits=robot.vlim[robot.runtime_indices];edge_speed=np.abs(np.diff(q,axis=0)/dt)
    speed_ratio=float((edge_speed/limits).max());limit_error=float(np.maximum(np.maximum(robot.low-angles,angles-robot.high),0).max())
    if not np.isfinite(world).all() or limit_error>1e-6 or speed_ratio>1.+1e-6:raise ValueError('Invalid reference '+clip)
    plausible=record.get('observed_ball_kinematics_quality')=='continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth'
    event=(record.get('inferred_event') or {}) if plausible else {}
    result=folder/(clip+'.npz');span=np.ptp(q,axis=0)
    phase=np.full(len(q),-1,dtype=np.int8)
    np.savez_compressed(result,fps=np.array([cfg['output_fps']]),time_s=t,source_time_s=st,
        source_frame_float=st*120+first,source_first_frame=np.array(first),source_fps=np.array(120),
        joint_pos=q.astype('f4'),joint_vel=dq.astype('f4'),root_pos_w=root.astype('f4'),root_quat_wxyz=quat[:,0].astype('f4'),
        body_pos_w=world.astype('f4'),body_quat_w=quat.astype('f4'),body_lin_vel_w=np.gradient(world,dt,axis=0,edge_order=2).astype('f4'),
        body_ang_vel_w=world_angular_velocity(r,dt).astype('f4'),joint_names=np.array(robot.joint_names),body_names=np.array(robot.body_names),
        source_marker_names=np.array(MARKERS),source_marker_input_observed=observed,source_marker_valid_120hz=raw_valid,
        human_segment_direction_root=targets.astype('f4'),ik_feature_error_deg=err.astype('f4'),segment_names=np.array(cfg['task_names']),
        urdf_sole_height_m=sole_z.astype('f4'),source_low_quiet_foot_proxy=support,foot_side_names=np.array(['left','right']),
        verified_phase_index=phase,verified_contact_label_available=np.zeros(len(t),dtype=bool),
        kick_leg_proxy=np.array(event.get('nearest_foot_side','unknown')),kick_leg_verified=np.array(False),
        source_clip=np.array(clip),subject_id=np.array(clip[:2]),split=np.array(group))
    return dict(subject_id=clip[:2],split=group,npz_sha256=sha(result),reference_file=result.name,
        source_c3d_sha256=sha(raw),source_processed_sha256=sha(processed),source_uri=f'https://mocap.cs.cmu.edu/subjects/{clip[:2]}/{clip}.c3d',
        source_frame_range_inclusive=[int(first),int(first)+len(raw_valid)-1],source_frames=len(raw_valid),reference_frames=len(q),
        source_duration_s=float(st[-1]),reference_last_time_s=float(t[-1]),source_endpoint_time_warp_s=float(st[-1]-t[-1]),
        missing_raw_marker_samples=int((~raw_valid).sum()),target_marker_samples_using_gap_or_endpoint_fill=int((~observed).sum()),
        segment_direction_mean_error_deg=float(err.mean()),segment_direction_p95_error_deg=float(np.quantile(err,.95)),
        neutral_pose_mean_direction_error_deg=float(neutral_err.mean()),mean_direction_error_by_feature_deg=dict(zip(cfg['task_names'],err.mean(0).tolist())),
        reference_joint_position_excess_rad=limit_error,max_reference_segment_speed_relative_to_urdf=speed_ratio,
        joints_with_motion_above_5deg=int((span>np.deg2rad(5)).sum()),joint_range_rad=dict(zip(robot.joint_names,span.tolist())),
        source_ball_marker_motion_status=record['observed_ball_kinematics_quality'],approximate_launch_source_s=event.get('inferred_ball_launch_onset_s'),
        nearest_foot_side_proxy=event.get('nearest_foot_side'),anatomical_instep_verified=False,professional_player_verified=False,
        verified_first_contact_s=None,human_ground_reaction_force=None,human_CoP=None,verified_semantic_phase_count=0,
        geometry=stats,classification='OFFLINE_REFERENCE',runtime_tracking_verified=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset-root',type=Path,default=Path('/home/you/projects/kinetickick'))
    ap.add_argument('--config',type=Path,default=ROOT/'configs/v160_human_reference.json')
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/v160_human')
    args=ap.parse_args();cfg=json.loads(args.config.read_text());base=args.asset_root
    if args.output.exists():raise FileExistsError(args.output)
    for k,h in cfg['fixed_files'].items():
        if sha(base/k)!=h:raise ValueError('Source provenance mismatch '+k)
    archived=json.loads((base/CMU/'original_source_manifest.json').read_text())
    raw_hashes={Path(row['path']).name:row['sha256'] for row in archived['files']}
    audits={row['id']:row for row in json.loads((base/CMU/'processed_v050/quality_audit.json').read_text())['motions']}
    robot=Robot(base/cfg['robot_urdf'],json.loads((base/cfg['runtime_schema']).read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v160-human-build-',dir=args.output.parent) as temp:
        folder=Path(temp); clips={}
        for group,names in cfg['splits'].items():
            for name in names:
                clips[name]=clip_reference(name,group,base,cfg,robot,raw_hashes,audits[name],folder)
                print(name,clips[name]['reference_frames'],flush=True)
        manifest=dict(version='v0.1.160',method='absolute full-clip marker-direction IK on original G1',source_license=cfg['source_license'],
            clips=clips,source_file_sha256=sha(Path(__file__)),protocol_sha256=sha(args.config),source_files=cfg['fixed_files'],
            splits=cfg['splits'],registered_task_names=cfg['task_names'],subject_disjoint_test=True,validation_is_same_subject=True,
            xyz_convention='source laboratory Z-up, XY translated to first robot pelvis; quaternion wxyz, angular velocities world',
            source_duration_mapping='uniform source span including first and final frame mapped onto nearest whole number of 50Hz intervals',
            source_qualification='uncalibrated surface markers and C3D-derived human hip points; professional status, contact surface/timing/forces and seven-phase labels unavailable',
            runtime_interpretation='reference geometry only; foot slip uses ankle-link origin under source-marker proxy, no measured robot forces, no collision or feasibility certificate',
            reference_trackability_certified=False,new_physics_episodes=0,numpy_version=np.__version__,pinocchio_version=pin.__version__,ezc3d_version=ezc3d.__version__)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        folder.rename(args.output)

if __name__=='__main__':main()
