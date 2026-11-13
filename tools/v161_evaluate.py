"""Read-only evaluation of recorded G1 CPU PhysX shots and their independent reference limits."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from follow_through import FOLLOW_THROUGH_CRITERIA, _motion
from free_shooting_metrics import paid_contact_patch
from v160_tracking_ball_impulse import (BALL, FOOT, PATCHES, collect_contacts, loaded_intervals,
                                       momentum_window, point_geometry)
from v160_tracking_ball_outcomes import crossing, episode_readout

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/'configs/v160_tracking_protocol.json'
DEFAULT_URDF=Path('/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf')
RIGHT=FOOT
FEET={'left_ankle_roll_link',RIGHT}
REQUIRED_S = 14.
DT=.005


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_array(trace,name,shape):
    data=np.asarray(trace[name],dtype=float)
    if data.shape!=shape or not np.isfinite(data).all():
        raise ValueError(f'Invalid {name}: expected finite {shape}, found {data.shape}')
    return data


def quat_rotation(quats):
    q=np.asarray(quats,dtype=float)
    if not np.allclose(np.linalg.norm(q,axis=-1),1.,rtol=0,atol=2e-4):
        raise ValueError('Non-unit measured body quaternion')
    return Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4))


def tail_duration(mask,dt):
    failures=np.flatnonzero(~mask)
    count=len(mask) if not len(failures) else len(mask)-int(failures[-1])-1
    return max(0,count-1)*dt  # duration between first and last observed stable endpoint


def first_event(mask,times):
    indices=np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def any_impact_info(file,times):
    unauthorized=[]; first_nonfoot=None; peak_nonfoot=None
    with gzip.open(file,'rt') as source:
        for index,line in enumerate(source):
            tick,pairs=json.loads(line)
            if tick!=index+1 or index>=len(times):raise ValueError('Native contact log and trajectory clock differ')
            for pair in pairs:
                names=pair['actors'];points=np.asarray(pair['points'],dtype=float)
                if not len(points):continue
                if points.ndim!=2 or points.shape[1]!=10 or not np.isfinite(points).all():
                    raise ValueError('Invalid contact point record')
                if len(names)!=2 or names[0]==names[1] or not all(s.startswith(n+'/') for s,n in zip(pair['shapes'],names)):
                    raise ValueError('Native pair ownership is incomplete')
                force=float(np.linalg.norm(points[:,6:9].sum(0))/DT)
                if force<=2.:continue
                body=set(names)
                event=dict(time_s=float(times[index]),actors=names,shapes=pair['shapes'],force_N=force,
                           world_m=points[np.argmax(np.linalg.norm(points[:,6:9],axis=1)),:3].tolist())
                if BALL in body:
                    partner=next(n for n in names if n!=BALL)
                    if partner not in ('ground',RIGHT):unauthorized.append(event)
                if force>20. and bool(body-{'ground',BALL,*FEET}):
                    if first_nonfoot is None:first_nonfoot=event
                    if peak_nonfoot is None or force>peak_nonfoot['force_N']:peak_nonfoot=event
    return dict(first_wrong_ball_contact=unauthorized[0] if unauthorized else None,
                wrong_ball_pair_bins=len(unauthorized),first_nonfoot_pair=first_nonfoot,peak_nonfoot_pair=peak_nonfoot)


def patch_summary(points,times,feet,rot,ball,radius,first,last):
    """Classify every native nonzero right-foot point in the loaded span, including simultaneous points."""
    contributions=dict.fromkeys(PATCHES,0.); total=0.; errors=[]; per_tick={}
    for tick,row,sign,magnitude,foot_shape,ball_shape in points:
        if not first<=tick<=last:continue
        if tick<2:return {'available':False,'reason':'first-tick pre-contact pose unavailable','instep_fraction':None}
        index=tick-2;normal=sign*row[3:6]
        errors.append(float(np.linalg.norm(row[:3]-(ball[index]-radius*normal))))
        local,local_normal=point_geometry(row,sign,feet[index],rot[index],foot_witness=True)
        patch=paid_contact_patch(local,local_normal,foot='right')
        contributions[patch]+=float(magnitude);total+=float(magnitude)
        per_tick[tick]=per_tick.get(tick,0.)+float(magnitude)
    if not total:return {'available':False,'reason':'no finite loaded right-foot impulse','instep_fraction':None}
    return dict(available=True,instep_fraction=contributions['instep']/total,
                scalar_normal_impulse_Ns=total,per_patch_normal_impulse_Ns=contributions,
                max_prestep_ball_witness_error_m=max(errors),loaded_point_ticks=len(per_tick),
                source='pre-step native sphere point minus signed gap along foot-to-ball normal, classified in original PAiD ankle coordinates',
                qualification='geometric CPU proxy; original Isaac GPU surface equivalence unverified')


def origin(element):
    if element is None:return np.zeros(3),Rotation.identity()
    return (np.fromstring(element.get('xyz','0 0 0'),sep=' '),
            Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')))


def reconstruct_com(xml,names,positions,quats):
    """Attach omitted fixed links to their measured dynamic parent; include all URDF masses and inertial offsets."""
    owned={n:(i,np.zeros(3),Rotation.identity()) for i,n in enumerate(names)}
    joints=[j for j in xml.findall('joint') if j.get('type')=='fixed']
    for _ in joints:
        for j in joints:
            child=j.find('child').get('link');parent=j.find('parent').get('link')
            if child in owned or parent not in owned:continue
            index,p,r=owned[parent];offset,turn=origin(j.find('origin'))
            owned[child]=index,p+r.apply(offset),r*turn
    weights=np.zeros(len(names));weighted_offsets=np.zeros((len(names),3))
    for link in xml.findall('link'):
        inertial=link.find('inertial')
        if inertial is None:continue
        mass=float(inertial.find('mass').get('value'));name=link.get('name')
        if name not in owned:raise ValueError('Unmapped massive link '+name)
        i,p,r=owned[name];c,_=origin(inertial.find('origin'))
        weights[i]+=mass;weighted_offsets[i]+=mass*(p+r.apply(c))
    matrices=quat_rotation(quats).as_matrix().reshape(*quats.shape[:2],3,3)
    com=(np.einsum('tbi,b->ti',positions,weights)+np.einsum('tbij,bj->ti',matrices,weighted_offsets))/weights.sum()
    return com,float(weights.sum())


def measured_follow(times,ankle,root,hit,direction,mask):
    if hit is None or hit<.045 or hit+FOLLOW_THROUGH_CRITERIA['recovery_window_end_s']>times[-1]:
        return dict(passed=False,reason='No complete measured contact-aligned swing window')
    rows=[dict(time_s=t,feet={'right':{'position':p}},root_pose=r) for t,p,r in zip(times,ankle,root)]
    values=_motion({'kinematics':[rows]},0,hit,direction)
    c=FOLLOW_THROUGH_CRITERIA;swing_end=hit+c['advance_window_s']
    idx=np.flatnonzero((times>=hit-1e-9)&(times<=swing_end+1e-9));v=np.diff(ankle[idx]@direction)/np.diff(times[idx])
    stoppage=np.flatnonzero(v<=0.);continuing=len(v) if not len(stoppage) else int(stoppage[0])
    checks=dict(foot_advances=values['forward_advance_m']>=c['forward_advance_min_m'],
                pelvis_relative_advance=values['pelvis_relative_advance_m']>=c['pelvis_relative_advance_min_m'],
                forward_speed_at_120ms=values['forward_speed_at_120ms_mps']>=c['forward_speed_min_mps'],
                no_early_reversal=values['reversal_after_s'] is None or values['reversal_after_s']>=c['earliest_sustained_reversal_s']-1e-8,
                bounded_return=values['peak_return_speed_mps']<=c['peak_return_speed_max_mps'],
                bounded_recovery_return=values['peak_recovery_return_speed_mps']<=c['recovery_return_speed_max_mps'],
                no_late_right_foot_touch=not bool(np.any(mask&(times>=hit+c['late_contact_start_s']-1e-9)&(times<=hit+c['late_contact_end_s']+1e-9))))
    return dict(passed=all(checks.values()),criteria=c,checks=checks,continuous_initial_forward_s=continuing*DT,**values)


def raw_config(folder,config_path):
    config=json.loads(config_path.read_text())
    trial=folder/'trial.json'
    if trial.is_file():
        overrides=json.loads(trial.read_text())
        for key in ('ball_initial_xyz_m','goal_xyz_m','goal_forward_xy','goal_width_height_m','clip'):
            if key in overrides:config[key]=overrides[key]
    return config


def report(folder,config_path=DEFAULT_CONFIG,urdf=DEFAULT_URDF):
    cfg=raw_config(folder,config_path);dt=cfg['physics_dt_s']
    if dt!=DT:raise ValueError('Original 5 ms clock required')
    folder=Path(folder);paths={n:folder/n for n in ('rollout.npz','native_contacts.jsonl.gz','tracking_result.json')};contact_file=paths['native_contacts.jsonl.gz']
    saved=json.loads(paths['tracking_result.json'].read_text()); hashes={k:digest(p) for k,p in paths.items()}
    for filename,d in saved.get('local_artifact_sha256',{}).items():
        if filename in hashes and hashes[filename]!=d:raise ValueError('Stored physical evidence differs: '+filename)
    if (folder/'trial.json').is_file():hashes['trial.json']=digest(folder/'trial.json')
    if 'source_body_names' in saved:joints=saved['source_joint_names'];bodies=saved['source_body_names']
    else:
        inventory=folder/'source_inventory.json';hashes[inventory.name]=digest(inventory)
        metadata=json.loads(inventory.read_text());joints=metadata['joint_names'];bodies=metadata['body_names']
    if len(set(joints))!=29 or len(set(bodies))!=30 or bodies[0]!='pelvis':raise ValueError('Original G1 ordered schema required')
    expected=cfg.get('asset_hashes',{}).get('configs/paid_original/paid_main_resolved.urdf')
    if expected and digest(urdf)!=expected:raise ValueError('Unverified physical URDF')
    xml=ET.parse(urdf).getroot();ordered={j.get('name'):j.find('limit') for j in xml.findall('joint')}
    limits=np.array([[float(ordered[n].get(k)) for k in ('lower','upper','velocity')] for n in joints])
    with np.load(paths['rollout.npz'],allow_pickle=False) as tr:
        t=tr['time_s'];N=len(t)
        if N<3 or not np.allclose(t,np.arange(1,N+1)*dt,atol=1e-8,rtol=0):raise ValueError('Physical clock interrupted or malformed')
        q=as_array(tr,'joint_q_rad',(N,29));dq=as_array(tr,'joint_dq_rad_s',(N,29));root=as_array(tr,'root_pose_wxyz',(N,7));rv=as_array(tr,'root_velocity_w',(N,6))
        pos=as_array(tr,'body_position_w_m',(N,30,3));quat=as_array(tr,'body_quat_wxyz',(N,30,4));foot_load=as_array(tr,'foot_ground_normal_load_N',(N,2))
        ball=as_array(tr,'ball_position_w_m',(N,3));vb=as_array(tr,'ball_velocity_w_mps',(N,3));nonfoot=as_array(tr,'nonfoot_net_contact_peak_N',(N,))
    mask,impulses,counts,points=collect_contacts(contact_file,t,dt)
    episodes=episode_readout(mask,t,vb[:,:2]);ranges=loaded_intervals(mask)
    speed=episodes[-1]['velocity_xy_mps_after_interval'] if episodes else None
    first=first_event(mask,t); pair=any_impact_info(contact_file,t)
    right=bodies.index(RIGHT)
    if ranges:
        patch=patch_summary(points,t,pos[:,right],quat_rotation(quat[:,right]),ball,cfg['ball_radius_m'],ranges[0][0],ranges[-1][1])
        windows=[momentum_window(a,b,t,vb,impulses,counts,cfg['ball_mass_kg'],cfg['gravity_w_mps2'],dt) for a,b in ranges if a>=2]
    else:patch={'available':False,'instep_fraction':None,'reason':'No loaded right-foot contact'};windows=[]
    start=np.array(cfg['ball_initial_xyz_m']);goal=np.array(cfg['goal_xyz_m']);forward=np.asarray(cfg['goal_forward_xy'],float);forward/=np.linalg.norm(forward)
    whole=crossing(np.vstack([start,ball]),np.r_[0,t],goal,forward,cfg['ball_radius_m'],cfg['ball_radius_m'],cfg['goal_width_height_m'])
    goal_valid=bool(first is not None and whole and whole['whole_sphere_within_aperture'] and whole['interpolated_time_s']>=first)
    direction=np.r_[goal[:2]-start[:2],0.];direction/=np.linalg.norm(direction)
    follow=measured_follow(t,pos[:,right],root,first,direction,mask)
    com,mass=reconstruct_com(xml,bodies,pos,quat);cv=np.full_like(com,np.nan);cv[1:]=np.diff(com,axis=0)/dt
    ratio=np.abs(dq)/limits[:,2];qbad=(q<limits[:,0]-1e-6)|(q>limits[:,1]+1e-6);vbad=np.abs(dq)>limits[:,2]+1e-6
    tilt=np.rad2deg(np.arccos(np.clip(quat_rotation(root[:,3:]).as_matrix()[:,2,2],-1,1)))
    fall=(root[:,2]<.45)|(tilt>60.)
    loaded_right_any=impulses[:,0,:]
    stable=(root[:,2]>.55)&(tilt<=25.)&(np.linalg.norm(rv[:,:2],axis=1)<.30)&np.all(foot_load>30.,axis=1)&(foot_load.sum(1)>=80.)&(nonfoot<20.)
    stable &= (np.linalg.norm(cv[:,:2],axis=1)<=.30)&(np.abs(cv[:,2])<=.20)&(np.linalg.norm(loaded_right_any,axis=1)==0)&(t>(first if first is not None else np.inf))
    standing=tail_duration(stable,dt)
    checks=dict(complete_14s=N==2800,no_mid_episode_state_writes=saved.get('additional_state_writes_after_initialization') in (0,False),right_instep_proxy=patch['available'] and patch['instep_fraction']>=.95,
                legitimate_ball_body=first is not None and not pair['wrong_ball_pair_bins'],terminal_outgoing_xy=speed is not None and speed>=3.5,
                goal=goal_valid,no_nonfoot=bool(nonfoot.max()<=20.) and pair['first_nonfoot_pair'] is None,
                original_q_bounds=not bool(qbad.any()),original_qdot_bounds=not bool(vbad.any()),
                no_fall=not bool(fall.any()),continuous_follow_through=bool(follow['passed']),terminal_standing=standing>=2.5-1e-8)
    return dict(offline_audit=True,run_directory=str(folder.resolve()),protocol_config_sha256=digest(config_path),urdf_sha256=digest(urdf),
                input_sha256=hashes,seconds=float(t[-1]),physics_steps=N,checks=checks,cpu_proxy_task_pass=all(checks.values()),failed_checks=[n for n,x in checks.items() if not x],
                backend_surface_certificate=None,original_isaac_gpu_certified=None,source_clip=cfg.get('clip'),initial_ball_xyz_m=start.tolist(),
                goal_xyz_m=goal.tolist(),goal_forward_xy=forward.tolist(),goal_width_height_m=cfg['goal_width_height_m'],ball_radius_m=cfg['ball_radius_m'],
                earliest_loaded_right_foot_s=first,contact_episodes=episodes,last_exit_speed_xy_mps=speed,instep_proxy=patch,
                normal_only_momentum_windows=windows,friction_anchor_impulse_by_source=None,whole_ball_crossing=whole,follow_through=follow,
                first_fall_s=first_event(fall,t),first_nonfoot_net_force_above_20N_s=first_event(nonfoot>20.,t),
                maximum_nonfoot_net_N=float(nonfoot.max()),native_pair_events=pair,first_q_position_violation_s=first_event(qbad.any(1),t),
                first_qdot_violation_s=first_event(vbad.any(1),t),q_position_violation_cells=int(qbad.sum()),qdot_violation_cells=int(vbad.sum()),
                max_qdot_ratio=float(ratio.max()),terminal_standing_seconds=standing,standing_required_s=2.5,
                com_velocity_source='5 ms backward differences, original URDF inertial masses and offsets including fixed children transformed by measured CPU body poses',
                reconstructed_urdf_mass_kg=mass,pelvis_end_height_m=float(root[-1,2]),postinitial_state_writes_reported=saved.get('additional_state_writes_after_initialization'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--urdf',type=Path,default=DEFAULT_URDF);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=report(a.run,a.config,a.urdf)
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cpu_proxy_task_pass':out['cpu_proxy_task_pass'],'failed_checks':out['failed_checks'],'last_exit_speed_xy_mps':out['last_exit_speed_xy_mps'],'terminal_standing_seconds':out['terminal_standing_seconds']}))

if __name__=='__main__':main()
