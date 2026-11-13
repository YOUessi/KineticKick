"""Measure native rigid-body and shape contacts in a renderer-free CPU fixture.

The original G1 foot link supplies the collision transforms and inertial tensor.
SAPIEN's PxCapsule uses local X; the URDF cylinder uses local Z. We follow the
PAiD importer's cylinder-to-capsule convention. Shape IDs below are local asset
indices checked against the native shape objects, never alleged USD prim paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/v160_cpu_contact.json'
DEFAULT_URDF = Path('/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf')


def numbers(element, name):
    return np.asarray([float(v) for v in element.get(name, '0 0 0').split()])


def pose(sapien, position, rotation):
    x, y, z, w = rotation.as_quat()
    return sapien.Pose(position, [w, x, y, z])


def add_foot(scene, sapien, xml, side, height, material, fixed):
    name = f'{side}_ankle_roll_link'
    link = xml.find(f"link[@name='{name}']")
    builder = scene.create_actor_builder()
    capsule_frame = Rotation.from_euler('y', -np.pi / 2)
    for collider in link.findall('collision'):
        origin, cylinder = collider.find('origin'), collider.find('geometry/cylinder')
        axis_frame = Rotation.from_euler('xyz', numbers(origin, 'rpy')) * capsule_frame
        builder.add_capsule_collision(pose(sapien, numbers(origin, 'xyz'), axis_frame),
                                      radius=float(cylinder.get('radius')),
                                      half_length=float(cylinder.get('length')) / 2, material=material)
    inertial = link.find('inertial')
    origin, inertia = inertial.find('origin'), inertial.find('inertia').attrib
    tensor = np.array([[float(inertia['i' + ''.join(sorted((a,b)))]) for b in 'xyz'] for a in 'xyz'])
    principal, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    inertial_frame = Rotation.from_euler('xyz', numbers(origin,'rpy')) * Rotation.from_matrix(axes)
    if not fixed:
        builder.set_mass_and_inertia(float(inertial.find('mass').get('value')),
                                     pose(sapien, numbers(origin, 'xyz'), inertial_frame), principal)
    builder.set_initial_pose(sapien.Pose([0, 0, height]))
    return builder.build_static(name) if fixed else builder.build(name)


def rigid_state(body):
    p = body.entity.pose
    return np.r_[p.p, p.q, body.linear_velocity, body.angular_velocity].astype(float)


def build_case(sapien, case, xml, cfg):
    system = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([system]); scene.timestep = cfg['physics_dt_s']
    # A standalone CPU scene inherits the runtime's default world gravity. Check it explicitly.
    gravity = list(sapien.physx.get_scene_config().gravity)
    if not np.allclose(gravity, cfg['gravity_mps2'], atol=1e-6, rtol=0):
        raise ValueError('Unexpected physical scene gravity')
    material = scene.create_physical_material(*cfg['material'])
    ground = scene.add_ground(0, render=False, material=material)
    ball_case = case.startswith('ball_')
    side = 'left' if case.startswith('left_') else 'right'
    foot = add_foot(scene, sapien, xml, side,
                    cfg['fixed_foot_z_m'] if ball_case else cfg['free_foot_start_z_m'], material, ball_case)
    entities = [ground, foot]
    if ball_case:
        builder = scene.create_actor_builder()
        radius, mass = cfg['ball_radius_m'], cfg['ball_mass_kg']
        builder.add_sphere_collision(radius=radius, material=material)
        builder.set_mass_and_inertia(mass, sapien.Pose(), [0.4 * mass * radius**2] * 3)
        lateral = cfg['miss_ball_y_m'] if case == 'ball_clear_of_right_foot' else 0.0
        builder.set_initial_pose(sapien.Pose([.035, lateral, cfg['ball_start_z_m']]))
        dynamic_entity = builder.build('soccer_ball'); entities.append(dynamic_entity)
    else:
        dynamic_entity = foot
    dynamic = dynamic_entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    dynamic.linear_damping = 0.; dynamic.angular_damping = 0.; dynamic.sleep_threshold = 0.
    bodies, shapes = {}, {}
    for entity in entities:
        body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
        bodies[body] = entity.name
        for index, shape in enumerate(body.get_collision_shapes()):
            shapes[shape] = (entity.name, index)
    return scene, system, dynamic, bodies, shapes


def copy_contacts(system, bodies, shapes):
    rows = []
    for contact in system.get_contacts():
        owners = [bodies[b] for b in contact.bodies]
        collider = [shapes[s] for s in contact.shapes]
        if owners != [entry[0] for entry in collider]:
            raise ValueError('Native collider is not owned by the reported rigid body')
        for point in contact.points:
            values = np.r_[point.position, point.normal, point.impulse, point.separation].astype(float)
            if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:6])-1)>1e-4:
                raise ValueError('Invalid native contact vector')
            rows.append(dict(bodies=owners, shape_indices=[entry[1] for entry in collider],
                             position=values[:3].tolist(), normal=values[3:6].tolist(),
                             impulse_on_first_Ns=values[6:9].tolist(), separation_m=values[9]))
    return rows


def step_metrics(points, moving_name, before, after, mass, gravity, dt):
    total = np.zeros(3); centroid_sum = np.zeros(3); sole_load = 0.
    for point in points:
        owners = point['bodies']; impulse = np.asarray(point['impulse_on_first_Ns'])
        if moving_name in owners:
            total += impulse if owners[0] == moving_name else -impulse
        if 'ground' in owners and any(s.endswith('_ankle_roll_link') for s in owners):
            sign = -1 if owners[0] == 'ground' else 1
            normal = np.asarray(point['normal']); normal_for_foot = sign*normal
            normal_impulse = max(0., float(impulse@normal))*max(0., float(normal_for_foot[2]))
            centroid_sum += normal_impulse * np.asarray(point['position']); sole_load += normal_impulse
    closure = mass*(after[7:10]-before[7:10])-mass*gravity*dt-total
    return dict(impulse_on_dynamic_Ns=total.tolist(), closure_residual_Ns=closure.tolist(),
                sole_upward_normal_impulse_Ns=sole_load,
                sole_normal_centroid_world_m=(centroid_sum/sole_load).tolist() if sole_load > 0 else None)


def run_fixture(sapien, cfg, xml, case, on):
    scene, system, moving, bodies, shapes = build_case(sapien, case, xml, cfg)
    rows=[]; previous=rigid_state(moving); dt=cfg['physics_dt_s']
    for index in range(1,cfg['steps']+1):
        scene.step()
        current=rigid_state(moving)
        if not np.isfinite(current).all(): raise ValueError('Nonfinite measured rigid state')
        row=dict(step=index, time_s=index*dt, before=previous.tolist(), after=current.tolist())
        if on:
            row['points']=copy_contacts(system,bodies,shapes)
            row.update(step_metrics(row['points'],moving.entity.name,previous,current,moving.mass,
                                    np.asarray(cfg['gravity_mps2']),dt))
        rows.append(row);previous=current
    return rows


def assess(cfg, case, off, on, xml):
    if len(off)!=cfg['steps'] or len(on)!=cfg['steps']:raise ValueError('Incomplete frozen duration')
    max_diff=0.; loaded=[]; max_closure=0.; sole_ticks=0; min_clearance=None
    ball_case=case.startswith('ball_'); side='left' if case.startswith('left_') else 'right'
    name=f'{side}_ankle_roll_link'; target={name,'soccer_ball' if ball_case else 'ground'}
    link=xml.find(f"link[@name='{name}']")
    # In the stationary-foot negative, a conservative bound valid at every Z is the
    # measured lateral ball-to-capsule-axis distance minus both radii, including axis tilt.
    lateral_extent=max(abs(numbers(c.find('origin'),'xyz')[1])+float(c.find('geometry/cylinder').get('radius'))+
        abs(Rotation.from_euler('xyz',numbers(c.find('origin'),'rpy')).as_matrix()[1,2])*
        float(c.find('geometry/cylinder').get('length'))/2 for c in link.findall('collision'))
    for i,(a,b) in enumerate(zip(off,on),1):
        if any(row['step']!=i or abs(row['time_s']-i*cfg['physics_dt_s'])>1e-12 for row in (a,b)):
            raise ValueError('Nonconsecutive physical step')
        max_diff=max(max_diff,*(float(np.max(np.abs(np.subtract(a[k],b[k])))) for k in ('before','after')))
        max_closure=max(max_closure,float(np.linalg.norm(b['closure_residual_Ns'])))
        if b['sole_upward_normal_impulse_Ns']>cfg['positive_min_impulse_Ns']:sole_ticks+=1
        positive_points=[p for p in b['points'] if set(p['bodies'])==target and np.linalg.norm(p['impulse_on_first_Ns'])>cfg['positive_min_impulse_Ns']]
        if positive_points:loaded.append(i)
        if case=='ball_clear_of_right_foot':
            gap=min(abs(b[k][1])-lateral_extent-cfg['ball_radius_m'] for k in ('before','after'))
            min_clearance=float(gap if min_clearance is None else min(min_clearance,gap))
    checks={'read_parity':max_diff<=cfg['state_parity_max_abs'],
            'linear_impulse_closure':max_closure<=cfg['per_step_linear_impulse_closure_tolerance_Ns']}
    if case=='ball_clear_of_right_foot':
        checks['separated_negative']=not loaded and min_clearance>cfg['negative_min_geometric_clearance_m']
    else:
        checks['positive_loaded_pair']=bool(loaded)
    if not ball_case:checks['measured_sole_patch']=sole_ticks>0
    return dict(checks=checks,passed=all(checks.values()),max_state_abs_difference=max_diff,
                max_5ms_linear_residual_Ns=max_closure,first_loaded_step=loaded[0] if loaded else None,
                loaded_ticks=len(loaded),sole_patch_loaded_ticks=sole_ticks,
                geometric_lateral_clearance_lower_bound_m=min_clearance)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path); parser.add_argument('--urdf',type=Path,default=DEFAULT_URDF)
    parser.add_argument('--completed-soles', type=Path, help='Reuse both complete pre-amendment OFF/ON sole records under their registered SHA256s')
    parser.add_argument('--rescore-from', type=Path, help='Only read already captured JSON records from this directory and --completed-soles')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Set CUDA_VISIBLE_DEVICES to empty for this CPU fixture')
    if not args.rescore_from:
        import sapien
        if sapien.__version__!='3.0.3':raise ValueError('Requalify the new native runtime version')
    cfg=json.loads(PLAN.read_text()); digest=hashlib.sha256(args.urdf.read_bytes()).hexdigest()
    if digest!=cfg['urdf_sha256']:raise ValueError('Original G1 URDF hash mismatch')
    xml=ET.parse(args.urdf).getroot(); args.output.mkdir(parents=True,exist_ok=False)
    summary={}; input_records={}
    locked = json.loads((ROOT/'configs/v160_completed_soles.json').read_text()) if args.completed_soles else {}
    for case in cfg['cases']:
        series={}
        for mode in cfg['read_modes']:
            filename=f'{case}_{mode}.json'
            if args.completed_soles and case.endswith('_sole_on_ground'):
                path=args.completed_soles/filename
                if hashlib.sha256(path.read_bytes()).hexdigest()!=locked[filename]:raise ValueError('Registered original sole record changed')
                rows=json.loads(path.read_text())
            elif args.rescore_from:
                path=args.rescore_from/filename
                rows=json.loads(path.read_text())
            else:
                path=args.output/filename
                rows=run_fixture(sapien,cfg,xml,case,mode=='on')
                path.write_text(json.dumps(rows,allow_nan=False)+'\n')
            series[mode]=rows
            input_records[filename]={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        summary[case]=assess(cfg,case,series['off'],series['on'],xml)
    document=dict(backend='SAPIEN PhysX CPU 3.0.3, isolated rigid shapes',urdf_sha256=digest,
                  protocol_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),cases=summary,input_records=input_records,
                  passed=all(s['passed'] for s in summary.values()),original_cuda_pair_certificate='UNKNOWN',
                  new_full_G1_rollouts=0)
    (args.output/'summary.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n');print(json.dumps(document))
    raise SystemExit(0 if document['passed'] else 2)

if __name__=='__main__':main()
