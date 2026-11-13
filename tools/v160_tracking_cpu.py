"""W1: inventory PAiD sources and execute one full released-policy CPU PhysX baseline.

SAPIEN PhysxCpuSystem and ONNX CPU use two existing Python environments.
Initialize the original PAiD URDF once. All later actions are force-limited joint
position drives. The finite physical run continues past a safety failure to retain
raw evidence. SAPIEN shapes and native impulses are local to this CPU scene.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from audit_paid_urdf_reference import load_urdf, fk_at_frame
from free_shooting_metrics import goal_crossing

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'configs/v160_tracking_protocol.json'
DEFAULT_ASSETS = Path('/home/you/projects/kinetickick')
SCHEMA = 'reports/paid_physics_parity_clean_paid.json'
URDF = 'configs/paid_original/paid_main_resolved.urdf'
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
OBSERVATIONS = ('command,projected_gravity,motion_ref_ang_vel,base_ang_vel,joint_pos,'
                'joint_vel,actions,target_point_pos,target_destination_pos_local')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(file, data):
    with file.open('x') as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write('\n')


def rotation(q_wxyz):
    return Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]])


def source_inventory(assets, config, schema):
    paths = sorted((assets / 'third_party/HumanoidSoccer/motions').glob('*/*.npz'))
    if {str(p.relative_to(assets)) for p in paths} != set(config['motion_sha256']):
        raise ValueError('PAiD motion list disagrees with the frozen source inventory')
    tree, joints = load_urdf(assets / URDF)
    if set(joints) != set(schema['joint_names']) or len(schema['body_names']) != 30:
        raise ValueError('G1 source schema and URDF disagree')
    records = []
    for path in paths:
        rel = str(path.relative_to(assets))
        with np.load(path, allow_pickle=False) as d:
            t = len(d['joint_pos'])
            shapes = {'joint_pos': (t,29), 'joint_vel': (t,29), 'body_pos_w': (t,30,3),
                      'body_quat_w': (t,30,4), 'body_lin_vel_w': (t,30,3), 'body_ang_vel_w': (t,30,3)}
            if set(d.files) != set(shapes) | {'fps','kick_leg'} or t < 2 or d['fps'].tolist() != [50]:
                raise ValueError('Invalid PAiD file schema or frame rate: '+rel)
            arrays = {k:d[k] for k in shapes}
            if any(a.shape != shapes[k] or not np.all(np.isfinite(a)) for k,a in arrays.items()):
                raise ValueError('Invalid or incomplete full-body reference: '+rel)
            side = str(d['kick_leg'].item())
        if side not in ('left','right') or not path.stem.endswith('_'+side):
            raise ValueError('Kick-side label disagrees with released name: '+rel)
        quat = arrays['body_quat_w']; unit_error = float(np.max(abs(np.linalg.norm(quat,axis=-1)-1)))
        if unit_error > 1e-4:
            raise ValueError('Nonunit reference quaternion: '+rel)
        max_position = max_angle = 0.
        for i in range(t):
            pose = {name:(xyz,r) for name,xyz,r in fk_at_frame(tree, dict(zip(schema['joint_names'],arrays['joint_pos'][i])),
                     arrays['body_pos_w'][i,0],quat[i,0])}
            positions = np.array([pose[name][0] for name in schema['body_names']]); fkq = Rotation.from_quat([pose[name][1].as_quat() for name in schema['body_names']])
            max_position = max(max_position,float(np.linalg.norm(positions-arrays['body_pos_w'][i],axis=1).max()))
            max_angle = max(max_angle,float((fkq.inv()*rotation(quat[i])).magnitude().max()))
        records.append({'path':rel,'sha256':config['motion_sha256'][rel], 'frames':t,'fps':50,'kick_leg':side,
                        'source_class':path.parent.name,'quaternion_norm_max_error':unit_error,
                        'fk_max_position_error_m':max_position,'fk_max_orientation_error_rad':max_angle,
                        'full_frame_named_order_qualified':max_position<=config['fk_max_position_tolerance_m'] and max_angle<=config['fk_max_angle_tolerance_rad']})
    return {'upstream_revision':config['upstream_commit'],'publication_license':config['upstream_license'],
            'joint_names':schema['joint_names'],'body_names':schema['body_names'],'clips':records,
            'body_names_from':'Tang historical original Isaac runtime schema, independently checked by all-frame original-URDF FK; no embedded name arrays',
            'human_player_ids':None,'human_contact_or_force_labels':None,'total_frames':sum(r['frames'] for r in records)}


@contextmanager
def cpu_policy(assets, cfg):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    arguments = [cfg['python_for_onnx'], '-u', str(Path(__file__).with_name('v160_tracking_worker.py')), '--assets',str(assets)]
    with subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=environment) as child:
        try:
            hello = json.loads(child.stdout.readline())
            metadata = hello['metadata']
            if hello['providers'] != ['CPUExecutionProvider'] or metadata['observation_names'] != OBSERVATIONS:
                raise ValueError('Released CPU policy metadata mismatch')
            def step(frame, values):
                child.stdin.write(json.dumps([frame,values.tolist()],allow_nan=False)+'\n');child.stdin.flush()
                action = np.asarray(json.loads(child.stdout.readline()))
                if action.shape != (29,) or not np.isfinite(action).all():
                    raise ValueError('Released policy returned invalid setpoints')
                return action
            yield metadata,step
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=5)
        if child.returncode != 0:
            raise RuntimeError(f'Local CPU ONNX worker exited {child.returncode}')


def make_plant(assets, cfg, metadata, schema):
    import sapien
    from sapien import physx
    physx.set_scene_config(gravity=cfg['gravity_w_mps2'], enable_tgs=True, cpu_workers=1)
    system = physx.PhysxCpuSystem(); scene = sapien.Scene([system]); scene.set_timestep(cfg['physics_dt_s'])
    material = scene.create_physical_material(*cfg['robot_and_ground_surface'])
    ground = scene.add_ground(0,render=False,material=material)
    root = ET.parse(assets/URDF); geometry_count = 0
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
        for cylinder in link.findall('collision/geometry/cylinder'):
            cylinder.tag='capsule';geometry_count+=1
    if geometry_count!=28 or len(root.findall('.//collision/geometry/sphere'))!=1:
        raise ValueError('Unexpected original collision geometry')
    with tempfile.TemporaryDirectory(prefix='kinetickick-v160-urdf-') as scratch:
        path=Path(scratch)/'physics.urdf';root.write(path,encoding='utf-8')
        loader=scene.create_urdf_loader();loader.fix_root_link=False;loader.set_material(*cfg['robot_and_ground_surface'])
        robot=loader.load(str(path))
    robot.set_solver_position_iterations(8);robot.set_solver_velocity_iterations(4)
    names=[j.name for j in robot.active_joints]; canonical=metadata['joint_names'].split(',')
    if len(names)!=29 or len(canonical)!=29 or set(names)!=set(canonical) or canonical!=schema['joint_names']:
        raise ValueError('Drive order or named original schema mismatch')
    order=np.asarray([names.index(n) for n in canonical]);drives=[robot.active_joints[i] for i in order]
    joint_spec={j.get('name'):j.find('limit') for j in root.findall('joint') if j.find('limit') is not None}
    gains={k:np.fromstring(metadata[k],sep=',') for k in ['joint_stiffness','joint_damping','default_joint_pos','action_scale']}
    if any(v.shape!=(29,) or not np.isfinite(v).all() for v in gains.values()):
        raise ValueError('Invalid exported actuator metadata')
    limits=np.asarray([[float(joint_spec[n].get(a)) for a in ('lower','upper','velocity','effort')] for n in canonical])
    # Robot source uses stiffness = armature * (2*pi*10 Hz)^2; metadata is printed to 3 decimals.
    # Select the exact source constants and verify that the derived K and D round to the checkpoint values.
    arm_options=np.array([.003609725,.010177520,.025101925,.00425,2*.003609725]);omega=20.*np.pi
    armature=arm_options[np.abs(gains['joint_stiffness'][:,None]-omega**2*arm_options).argmin(axis=1)]
    k=armature*omega**2;d=4*armature*omega
    if not np.allclose(k,gains['joint_stiffness'],atol=.00051,rtol=0) or not np.allclose(d,gains['joint_damping'],atol=.00051,rtol=0):
        raise ValueError('Declared source gains disagree with PAiD metadata')
    # Upstream sim2sim uses exported (rounded) stiffness/damping and action scale.
    for i,joint in enumerate(drives):
        joint.set_drive_properties(gains['joint_stiffness'][i],gains['joint_damping'][i],limits[i,3],mode='force')
        joint.set_armature([armature[i]]);joint.set_max_velocity(limits[i,2]);joint.set_drive_velocity_target(0.)
        if not np.allclose(joint.limits[0],limits[i,:2],atol=1e-5) or not np.isclose(joint.max_velocity,limits[i,2]) or not np.isclose(joint.force_limit,limits[i,3]):
            raise ValueError('SAPIEN original joint/drive limits mismatch: '+canonical[i])
    for link in robot.links:
        link.set_linear_damping(0.);link.set_angular_damping(0.);link.set_max_depenetration_velocity(1.)
        link.set_max_linear_velocity(1000.);link.set_max_angular_velocity(1000.)
    builder=scene.create_actor_builder();radius=cfg['ball_radius_m'];mass=cfg['ball_mass_kg']
    builder.add_sphere_collision(radius=radius,material=scene.create_physical_material(*cfg['ball_surface']))
    builder.set_mass_and_inertia(mass,sapien.Pose(),np.repeat(.4*mass*radius**2,3));builder.set_initial_pose(sapien.Pose(cfg['ball_initial_xyz_m']))
    ball_entity=builder.build(name='SoccerBall');ball=ball_entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
    ball.set_linear_damping(0.);ball.set_angular_damping(.05)
    entities={link.name:link for link in robot.links}; ground_body=ground.find_component_by_type(physx.PhysxRigidStaticComponent)
    if not set(schema['body_names'])<=set(entities):raise ValueError('Missing source-mapped robot links')
    shape_ids={}
    for name,actor in list(entities.items())+[('ground',ground_body),('SoccerBall',ball)]:
        for j,shape in enumerate(actor.collision_shapes):shape_ids[shape]=f'{name}/{type(shape).__name__}[{j}]'
    return scene,system,robot,ball,drives,order,limits,gains,entities,shape_ids


def observation(raw_q,raw_dq,root,omega,ball,goal,previous,ref,frame,anchor,defaults):
    into_body=rotation(root.q).inv()
    return np.concatenate([ref['joint_pos'][frame],ref['joint_vel'][frame],into_body.apply([0,0,-1]),
        ref['body_ang_vel_w'][frame,anchor],into_body.apply(omega),raw_q-defaults,raw_dq,previous,
        into_body.apply(ball-root.p),into_body.apply(goal-root.p)]).astype(np.float32)


def contacts_for_step(system,shape_ids,dt):
    """Keep native points; report supported material-point slip in the horizontal plane."""
    rows=[];weights=np.zeros(2);slip=np.zeros(2);other_peak=0.;ball_links=[]
    body_forces={}
    for c in system.get_contacts():
        b0,b1=c.bodies; n0,n1=b0.entity.name,b1.entity.name
        geom=[shape_ids[s] for s in c.shapes]; points=[];pair_vec=np.zeros(3)
        for p in c.points:
            loc=np.asarray(p.position,dtype=float);imp=np.asarray(p.impulse,dtype=float);normal=np.asarray(p.normal,dtype=float)
            points.append([*loc,*normal,*imp,float(p.separation)]); pair_vec+=imp/dt
            for j,foot in enumerate(FEET):
                if {n0,n1}!={foot,'ground'}:continue
                link=b0 if n0==foot else b1; com=(link.entity.pose*link.cmass_local_pose).p
                v=np.asarray(link.linear_velocity)+np.cross(link.angular_velocity,loc-com)
                load=abs(float(np.dot(imp,normal)*normal[2]))/dt
                weights[j]+=load;slip[j]+=load*np.linalg.norm(v[:2])
        if not points:continue
        rows.append({'actors':[n0,n1],'shapes':geom,'points':points})
        for name,force in [(n0,pair_vec),(n1,-pair_vec)]:
            if name not in ('ground','SoccerBall',*FEET):body_forces[name]=body_forces.get(name,np.zeros(3))+force
        if 'SoccerBall' in (n0,n1) and 'ground' not in (n0,n1) and np.linalg.norm(pair_vec)>2.:
            ball_links.append(n1 if n0=='SoccerBall' else n0)
    other_peak=max([float(np.linalg.norm(v)) for v in body_forces.values()] or [0.])
    np.divide(slip,weights,out=slip,where=weights>0.)
    return rows,weights,slip,other_peak,ball_links


def interpolate(values, clock):
    index=np.minimum(clock,len(values)-1); before=np.floor(index).astype(int)
    after=np.minimum(before+1,len(values)-1)
    fraction=(index-before).reshape((len(index),)+(1,)*(values.ndim-1))
    return values[before]*(1.-fraction)+values[after]*fraction


def metrics(trace, ref, cfg, schema, limits, first_ball_step, scored, contact_counts):
    # Post-step sample i corresponds to (i+1)*dt. Stop the reference clock at its true end.
    t=trace['time_s']; q=trace['joint_q_rad']; dq=trace['joint_dq_rad_s']; pos=trace['body_position_w_m']
    clock=t/float(cfg['control_dt_s']); rq=interpolate(ref['joint_pos'],clock); rdq=interpolate(ref['joint_vel'],clock)
    rpos=interpolate(ref['body_pos_w'],clock); ref_end=(len(ref['joint_pos'])-1)*cfg['control_dt_s']
    groups={'all_14s':np.ones(len(t),bool),'source_clip':t<=ref_end+1e-9,'held_last_frame':t>ref_end+1e-9}
    if first_ball_step is not None:
        impact_t=first_ball_step*cfg['physics_dt_s'];groups.update(before_first_robot_ball=t<impact_t,first_robot_ball_to_300ms=(t>=impact_t)&(t<impact_t+.3),after_first_robot_ball_300ms=t>=impact_t+.3)
    else:impact_t=None
    # Pelvis-relative vectors remove both translation and orientation using measured and reference quaternions independently.
    # At 50Hz exact source endpoints avoid approximating quaternion interpolation.
    stride=round(cfg['control_dt_s']/cfg['physics_dt_s']);ends=np.arange(stride-1,len(t),stride)
    ref_index=np.minimum(np.rint(clock[ends]).astype(int),len(ref['joint_pos'])-1)
    measured_rot=rotation(trace['root_pose_wxyz'][ends,3:]);rrot=rotation(ref['body_quat_w'][ref_index,0])
    local=np.einsum('tji,tbj->tbi',measured_rot.as_matrix(),pos[ends]-pos[ends,:1]);rlocal=np.einsum('tji,tbj->tbi',rrot.as_matrix(),ref['body_pos_w'][ref_index]-ref['body_pos_w'][ref_index,:1]);err=np.linalg.norm(local-rlocal,axis=2)
    primary=np.array([n in cfg['tracking_links'] for n in schema['body_names']]);foot_indices=[schema['body_names'].index(n) for n in FEET]
    relative_error=np.linalg.norm(trace['root_pose_wxyz'][:,:3]-rpos[:,0],axis=1)
    stage={}
    for k,mask in groups.items():
        if not mask.any():continue
        on_grid=mask[ends];seg={'physics_samples':int(mask.sum()),'joint_position_rmse_rad':float(np.sqrt(np.mean((q[mask]-rq[mask])**2))),
            'joint_velocity_rmse_rad_s':float(np.sqrt(np.mean((dq[mask]-rdq[mask])**2))),
            'absolute_root_position_rmse_m':float(np.sqrt(np.mean(relative_error[mask]**2))),
            'pelvis_relative_body_mpjpe_m':float(err[on_grid][:,primary].mean()) if on_grid.any() else None}
        stage[k]=seg
    ratio=np.abs(dq)/limits[:,2];where=np.unravel_index(int(ratio.argmax()),ratio.shape)
    q_viol=(q<limits[:,0]-1e-6)|(q>limits[:,1]+1e-6);drop=trace['root_pose_wxyz'][:,2]<.45
    up=rotation(trace['root_pose_wxyz'][:,3:]).as_matrix()[:,2,2];tilt=np.degrees(np.arccos(np.clip(up,-1,1)))
    wrong=trace['nonfoot_net_contact_peak_N']>20.;feet_load=trace['foot_ground_normal_load_N'];loaded=feet_load>30.
    stand=(trace['root_pose_wxyz'][:,2]>.55)&(tilt<=25.)&(np.linalg.norm(trace['root_velocity_w'][:,:2],axis=1)<.30)&np.all(loaded,axis=1)&(~wrong)
    standing_tail=0
    for yes in stand[::-1]:
        if not yes:break
        standing_tail+=1
    evt=lambda mask: float(t[np.flatnonzero(mask)[0]]) if mask.any() else None
    source_mask=t<=ref_end+1e-9; foot_load=trace['foot_ground_normal_load_N'];strike_indices=np.where(trace['robot_ball_contact'])[0]
    first_speed=None
    if len(strike_indices):
        sample=next((i for i in range(strike_indices[0]+1,len(t)) if not trace['robot_ball_contact'][i]),None)
        if sample is not None:first_speed=float(np.linalg.norm(trace['ball_velocity_w_mps'][sample,:2]))
    return {'completed_seconds':float(t[-1]),'completed_physics_steps':len(t),'reference_last_timestamp_s':ref_end,
            'first_robot_ball_s':impact_t,'first_separated_ball_speed_xy_mps':first_speed,'goal_crossed_whole_ball':scored,
            'first_nonfoot_net_force_above_20N_s':evt(wrong),'first_height_below_0p45m_s':evt(drop),'first_tilt_over_60deg_s':evt(tilt>60.),
            'first_measured_qdot_limit_excess_s':evt(np.any(abs(dq)>limits[:,2]+1e-6,axis=1)),
            'qdot_limit_sample_joint_count':int(np.sum(abs(dq)>limits[:,2]+1e-6)),'q_position_limit_sample_joint_count':int(q_viol.sum()),
            'peak_qdot_ratio':float(ratio[where]),'peak_qdot_joint':schema['joint_names'][where[1]],'peak_qdot_time_s':float(t[where[0]]),
            'any_fall_during_reference':bool(np.any((drop|(tilt>60.))[source_mask])),'any_nonfoot_over_20N_during_reference':bool(wrong[source_mask].any()),
            'final_standing_seconds_cpu_definition':standing_tail*cfg['physics_dt_s'],'max_nonfoot_net_force_N':float(trace['nonfoot_net_contact_peak_N'].max()),
            'foot_ground_loaded_s':[float(x) for x in np.sum(loaded,axis=0)*cfg['physics_dt_s']],
            'foot_material_slip_rms_mps_when_loaded':[float(np.sqrt(np.mean(trace['foot_patch_slide_mps'][loaded[:,j],j]**2))) if loaded[:,j].any() else None for j in range(2)],
            'ankle_link_origin_horizontal_speed_rms_when_loaded_mps':[float(np.sqrt(np.mean(trace['foot_link_origin_speed_xy_mps'][loaded[:,j],j]**2))) if loaded[:,j].any() else None for j in range(2)],
            'native_robot_ball_contact_link_bins':contact_counts,'phase_metrics':stage,
            'measurement_scope':'SAPIEN CPU PhysX, original source URDF with fixed links retained; action clip is held only as reference time saturates. No original Isaac USD collision/manifold equivalence, no certified instep fraction, no current actuator torque observation. Loaded when normal vertical magnitude >30 N. Terminal standing uses bilateral support >30 N, root z>.55 m, tilt<=25°, horizontal root speed<.3 m/s and nonfoot net contact <=20 N.'}


def run(assets, output, config, schema, metadata, act, *, controller_factory=None):
    import sapien
    scene,system,robot,ball,drives,order,limits,gains,links,shape_ids=make_plant(assets,config,metadata,schema)
    with np.load(assets/config['clip'],allow_pickle=False) as source:
        ref={key:source[key] for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
    link_list=[links[n] for n in schema['body_names']];anchor=schema['body_names'].index(metadata['anchor_body_name'])
    robot.set_root_pose(sapien.Pose(ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]));robot.set_root_linear_velocity(ref['body_lin_vel_w'][0,0]);robot.set_root_angular_velocity(ref['body_ang_vel_w'][0,0]);
    robot.set_qpos(ref['joint_pos'][0,np.argsort(order)]);robot.set_qvel(ref['joint_vel'][0,np.argsort(order)])
    control = None
    if controller_factory is not None:
        # Initialization above always uses the unedited source and original velocities.
        ref, control = controller_factory(ref, schema)
        com_links = list(robot.links)
        masses = np.asarray([link.mass for link in com_links])
        load = np.zeros(2)
    raw_targets = []
    goal=np.asarray(config['goal_xyz_m']);previous=np.zeros(29);dt=config['physics_dt_s'];policy_dt=config['control_dt_s'];decimation=round(policy_dt/dt)
    nsteps=round(config['duration_s']/dt);step_count=0;contact_first=None;scored=False;counts={};ball_prev=ball.entity.pose.p.copy();rows={};commands=[];observations=[]
    import gzip
    with gzip.open(output/'native_contacts.jsonl.gz','xt',encoding='utf8',compresslevel=2) as raw_contacts:
        for tick in range(nsteps//decimation):
            frame=min(tick,len(ref['joint_pos'])-1); root=robot.root_pose
            obs=observation(robot.qpos[order],robot.qvel[order],root,robot.root_angular_velocity,ball.entity.pose.p,goal,previous,ref,frame,anchor,gains['default_joint_pos'])
            action=act(frame,obs);prev_raw=action.copy(); target=gains['default_joint_pos']+gains['action_scale']*np.clip(action,-100.,100.)
            if control is not None:
                raw_targets.append(target.copy())
                com_points = [(link.entity.pose * link.cmass_local_pose).p for link in com_links]
                ctx = dict(t=tick*policy_dt, dt=policy_dt, frame=frame,
                    q=robot.qpos[order].copy(), dq=robot.qvel[order].copy(),
                    root_pos=root.p.copy(), root_quat=root.q.copy(),
                    root_velocity=np.r_[robot.root_linear_velocity, robot.root_angular_velocity],
                    body_pos=np.array([link.entity.pose.p for link in link_list]),
                    body_quat=np.array([link.entity.pose.q for link in link_list]),
                    ball_pos=ball.entity.pose.p.copy(), ball_velocity=ball.linear_velocity.copy(),
                    goal=goal.copy(), foot_load=load.copy(),
                    first_contact_s=contact_first*dt if contact_first is not None else None,
                    com=np.average(com_points,axis=0,weights=masses),
                    com_velocity=np.average([link.linear_velocity for link in com_links],axis=0,weights=masses),
                    default_q=gains['default_joint_pos'].copy(), limits=limits.copy())
                target=np.asarray(control(ctx,target.copy()),dtype=float)
                if target.shape!=(29,) or not np.isfinite(target).all():
                    raise ValueError('Controller must return 29 finite joint targets')
            for joint,value in zip(drives,target):joint.set_drive_target(float(value))
            previous=np.clip(prev_raw,-100.,100.);commands.append(np.r_[tick*policy_dt,frame,action,target]);observations.append(obs)
            for _ in range(decimation):
                scene.step();step_count+=1;root=robot.root_pose
                contacts,load,slip,nonfoot,ball_links=contacts_for_step(system,shape_ids,dt)
                if ball_links and contact_first is None:contact_first=step_count
                for name in ball_links:counts[name]=counts.get(name,0)+1
                raw_contacts.write(json.dumps([step_count,contacts],separators=(',',':'),allow_nan=False)+'\n')
                ball_pos=ball.entity.pose.p
                scored |= goal_crossing(ball_prev,ball_pos,goal=goal,forward=config['goal_forward_xy'],width=config['goal_width_height_m'][0],height=config['goal_width_height_m'][1],radius=config['ball_radius_m'])
                ball_prev=ball_pos.copy()
                foot_origin=[]
                for side in FEET:
                    l=links[side]; com=(l.entity.pose*l.cmass_local_pose).p
                    v=l.linear_velocity+np.cross(l.angular_velocity,l.entity.pose.p-com)
                    foot_origin.append(float(np.linalg.norm(v[:2])))
                row={'time_s':step_count*dt,'joint_q_rad':robot.qpos[order],'joint_dq_rad_s':robot.qvel[order],
                     'root_pose_wxyz':np.r_[root.p,root.q],'root_velocity_w':np.r_[robot.root_linear_velocity,robot.root_angular_velocity],
                     'body_position_w_m':np.array([l.entity.pose.p for l in link_list]),'body_quat_wxyz':np.array([l.entity.pose.q for l in link_list]),
                     'ball_position_w_m':ball_pos,'ball_velocity_w_mps':ball.linear_velocity,'ball_angular_velocity_w_rad_s':ball.angular_velocity,
                     'foot_ground_normal_load_N':load,'foot_patch_slide_mps':slip,'foot_link_origin_speed_xy_mps':foot_origin,
                     'nonfoot_net_contact_peak_N':nonfoot,'robot_ball_contact':bool(ball_links)}
                for key,value in row.items():
                    array=np.asarray(value)
                    if not np.isfinite(array).all():raise FloatingPointError(f'Nonfinite value at step {step_count}: {key}')
                    rows.setdefault(key,[]).append(array.copy())
    trace={k:np.asarray(v) for k,v in rows.items()};trace['policy_commands_time_frame_action_target']=np.asarray(commands);trace['policy_observations']=np.asarray(observations)
    if raw_targets:
        trace['paid_targets_before_controller']=np.asarray(raw_targets)
    np.savez_compressed(output/'rollout.npz',**trace)
    report=metrics(trace,ref,config,schema,limits,contact_first,scored,counts)
    report.update({'runtime':'SAPIEN '+sapien.__version__+' / PhysxCpuSystem, CPU ONNX Runtime', 'actions':len(commands),'original_physx_gpu_runs':0,
        'new_cpu_physical_runs':1,'initialization_pose_wxyz':np.r_[ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]].tolist(),
        'source_clip':config['clip'],'source_joint_names':schema['joint_names'],'source_body_names':schema['body_names'],
        'cpu_links':len(robot.links),'cpu_active_joints':robot.dof,'cpu_urdf_shape_count':int(sum(len(l.collision_shapes) for l in robot.links)),
        'cpu_robot_mass_kg':float(sum(l.mass for l in robot.links)),'first_eligible_contact_definition':'norm of the sum of reported contact-point normal impulse vectors / dt >2 N; friction-anchor impulses are unavailable',
        'additional_state_writes_after_initialization':0,'input_digest':config['asset_hashes'],'protocol_sha256':digest(PROTOCOL),
        'code_sha256':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('v160_tracking_worker.py')]},
        'local_artifact_sha256':{name:digest(output/name) for name in ('rollout.npz','native_contacts.jsonl.gz')}})
    save_json(output/'tracking_result.json',report)
    print(json.dumps({k:report[k] for k in ('completed_seconds','actions','first_robot_ball_s','first_height_below_0p45m_s','first_tilt_over_60deg_s','first_nonfoot_net_force_above_20N_s','peak_qdot_ratio','final_standing_seconds_cpu_definition','phase_metrics')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assets=args.assets.resolve();cfg=json.loads(PROTOCOL.read_text())
    for k,expected in (cfg['asset_hashes']|cfg['motion_sha256']).items():
        if digest(assets/k)!=expected:raise ValueError('Frozen asset digest mismatch: '+k)
    schema=json.loads((assets/SCHEMA).read_text());args.output.mkdir(parents=True,exist_ok=False)
    inventory=source_inventory(assets,cfg,schema);save_json(args.output/'source_inventory.json',inventory)
    chosen=next(r for r in inventory['clips'] if r['path']==cfg['clip'])
    if not chosen['full_frame_named_order_qualified']:raise ValueError('Selected motion failed named-order FK qualification')
    with cpu_policy(assets,cfg) as (meta,act):
        if meta['body_names'].split(',')!=cfg['tracking_links']:raise ValueError('Mismatch between released policy tracking bodies and protocol')
        run(assets,args.output,cfg,schema,meta,act)


if __name__=='__main__':main()
