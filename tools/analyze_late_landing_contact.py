#!/usr/bin/env python3
"""Bounded CPU reanalysis of the v30 late-landing contact observation.

This diagnostic reads existing reports only. SDK point staging remains
unknown; pre/post reconstructions are alternatives, not time certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_landing_release import _probe
from explosive_strike_metrics import landing_window_dynamics, joint_velocity_scaling_hypothesis, summarize_explosive_strike
from strike_baseline_metrics import validated_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(state):
    q = state['quaternion_wxyz']
    return Rotation.from_quat(q[1:] + q[:1])


def capsules(path, side):
    link = ET.parse(path).getroot().find("link[@name='" + side + "_ankle_roll_link']")
    result = []
    for i, c in enumerate(link.findall('collision')):
        origin = c.find('origin').attrib
        shape = list(c.find('geometry'))[0]
        if shape.tag not in ('cylinder', 'capsule', 'sphere'):
            raise ValueError('Unexpected original foot primitive')
        p = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        r = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
        half = float(shape.attrib.get('length', 0.)) / 2
        result.append(dict(index=i, a=(p+r.apply([0, 0, -half])).tolist(),
            b=(p+r.apply([0, 0, half])).tolist(), radius=float(shape.attrib['radius']), urdf_type=shape.tag))
    return result


def nearest_capsule(point, geometry):
    results = []
    for c in geometry:
        a, b = np.array(c['a']), np.array(c['b'])
        axis = b-a
        u = float(np.clip(np.dot(point-a, axis)/max(np.dot(axis, axis), 1e-20), 0., 1.))
        radial = point-(a+u*axis)
        signed = float(np.linalg.norm(radial)-c['radius'])
        results.append(dict(geometric_primitive_index=c['index'], axis_fraction=u,
                           signed_surface_distance_m=signed, absolute_surface_distance_m=abs(signed)))
    return min(results, key=lambda x: x['absolute_surface_distance_m'])


def point_measurement(point, normal, state, geometry):
    p, origin = np.array(point), np.array(state['position_w_m'])
    lever = p-origin
    local = rotation(state).inv().apply(lever)
    v = np.array(state['link_origin_linear_velocity_w_mps']) + np.cross(state['angular_velocity_w_rad_s'], lever)
    n = np.array(normal)/np.linalg.norm(normal)
    vn = float(v@n)
    return dict(local_point_m=local.tolist(), lever_w_m=lever.tolist(), lever_length_m=float(np.linalg.norm(lever)),
                nearest_capsule=nearest_capsule(local, geometry), velocity_at_raw_world_point_w_mps=v.tolist(),
                normal_velocity_mps=vn, tangential_velocity_w_mps=(v-vn*n).tolist(),
                tangential_speed_mps=float(np.linalg.norm(v-vn*n)))


def wrench(contacts, state):
    f, moment = np.zeros(3), np.zeros(3)
    origin = np.array(state['position_w_m'])
    for c in contacts:
        force = np.array(c['force_w_N'])
        f += force
        moment += np.cross(np.array(c['point_w'])-origin, force)
    return dict(force_w_N=f.tolist(), moment_about_link_origin_w_Nm=moment.tolist(),
                moment_about_link_origin_local_Nm=rotation(state).inv().apply(moment).tolist(),
                reference_origin_w_m=origin.tolist())


def measure_contact(c, normal, before, after, geometry):
    p = np.array(c['point_w'])
    pre = point_measurement(p, normal, before, geometry)
    post = point_measurement(p, normal, after, geometry)
    local = np.array(pre['local_point_m'])
    advected = np.array(after['position_w_m']) + rotation(after).apply(local)
    v_after = np.array(after['link_origin_linear_velocity_w_mps']) + np.cross(after['angular_velocity_w_rad_s'], rotation(after).apply(local))
    return dict(raw=c, before=pre, after=post,
                conditional_same_material_point_after_w_m=advected.tolist(),
                conditional_same_material_point_after_velocity_w_mps=v_after.tolist(),
                conditional_material_interpretation='Assumes the raw SDK point corresponds to this pre-step local point. Strong surface agreement is supporting evidence, not SDK time/material-point certification.')


def analyze_detail(row, geometry):
    d = row['ground_contact_detail']
    if abs(d['step_end_s']-row['time_s']) > 1e-9 or abs(d['step_end_s']-d['step_start_s']-.005) > 1e-8:
        raise ValueError('Unmatched contact bin')
    out = dict(step_start_s=d['step_start_s'], step_end_s=d['step_end_s'], feet={})
    for side in ('left', 'right'):
        c = d['ground_filtered_contacts'][side]
        if c['capacity_exhausted']:
            raise ValueError('Contact buffer capacity exhausted')
        before, after = d['feet_before'][side], d['feet_after'][side]
        normals = [x for x in c['normal_contacts'] if x['force_scalar_N'] > 0.]
        friction = [x for x in c['friction_anchors'] if np.linalg.norm(x['force_w_N']) > 0.]
        all_normals = np.array([x['normal_w'] for x in c['normal_contacts']])
        plane_normal = all_normals[0] if len(all_normals) else None
        if len(all_normals) and np.max(np.linalg.norm(all_normals-plane_normal, axis=1)) > 1e-6:
            raise ValueError('The recorded ground patch has nonparallel normals')
        if friction and plane_normal is None:
            raise ValueError('Friction decomposition lacks a recorded plane normal')
        force_errors = {}
        for kind, contacts, field in [('normal', c['normal_contacts'], 'support_foot_normal_force_w_N'),
                                      ('friction', c['friction_anchors'], 'support_foot_friction_force_w_N')]:
            summed = np.sum([x['force_w_N'] for x in contacts], axis=0) if contacts else np.zeros(3)
            force_errors[kind] = float(np.linalg.norm(summed-row[field][side]))
            if force_errors[kind] > .001:
                raise ValueError('Raw contact forces differ from certified observer forces')
        measured_normals = [measure_contact(x, x['normal_w'], before, after, geometry[side]) for x in normals]
        measured_friction = [measure_contact(x, plane_normal, before, after, geometry[side]) for x in friction]
        data = dict(before_state=before, after_state=after,
            counts={k: c[k] for k in ('normal_counts', 'normal_starts', 'friction_counts', 'friction_starts')},
            raw_force_reconstruction_errors_N=force_errors, normal_contacts=measured_normals, friction_anchors=measured_friction,
            wrench={stage: {kind: wrench(points, state) for kind, points in
                    [('normal', c['normal_contacts']), ('friction', c['friction_anchors']), ('sum', c['normal_contacts']+c['friction_anchors'])]}
                    for stage, state in [('before', before), ('after', after)]})
        if normals:
            weights = np.array([x['force_scalar_N'] for x in normals])
            data['normal_force_weighted_local_centroid_before_m'] = np.average(
                [x['before']['local_point_m'] for x in measured_normals], axis=0, weights=weights).tolist()
            data['normal_force_weighted_pre_velocity'] = dict(
                normal_mps=float(np.average([x['before']['normal_velocity_mps'] for x in measured_normals], weights=weights)),
                tangential_speed_mps=float(np.average([x['before']['tangential_speed_mps'] for x in measured_normals], weights=weights)))
        out['feet'][side] = data
    return out


def integrated_wrenches(details, start, end):
    selected = [d for d in details if d['step_start_s'] >= start-1e-9 and d['step_end_s'] <= end+1e-9]
    if abs(sum(d['step_end_s']-d['step_start_s'] for d in selected)-(end-start)) > 1e-8:
        raise ValueError('Incomplete integration interval')
    result = dict(interval_s=[start, end], feet={})
    for side in ('left', 'right'):
        result['feet'][side] = {}
        for stage in ('before', 'after'):
            result['feet'][side][stage] = {}
            for kind in ('normal', 'friction', 'sum'):
                impulse = sum((np.array(d['feet'][side]['wrench'][stage][kind]['force_w_N'])*.005 for d in selected), start=np.zeros(3))
                angular = sum((np.array(d['feet'][side]['wrench'][stage][kind]['moment_about_link_origin_w_Nm'])*.005 for d in selected), start=np.zeros(3))
                result['feet'][side][stage][kind] = dict(impulse_w_Ns=impulse.tolist(),
                    snapshot_moment_integral_w_Nms=angular.tolist(), mean_snapshot_moment_w_Nm=(angular/(end-start)).tolist())
    result['interpretation'] = 'Snapshot lever arms crossed with step-average reported forces. The link origin moves and force/point substep history is absent; this is not a separately calibrated torque or rigid-foot angular-momentum closure.'
    return result


def analytic_checks():
    state = dict(position_w_m=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.],
                 link_origin_linear_velocity_w_mps=[1., 0., -2.], angular_velocity_w_rad_s=[0., 0., 2.])
    shape = [dict(index=0, a=[0., 0., 0.], b=[1., 0., 0.], radius=1.)]
    assert nearest_capsule(np.array([.5, 1., 0.]), shape)['absolute_surface_distance_m'] == 0.
    m = point_measurement([1., 0., 0.], [0., 0., 1.], state, shape)
    assert np.allclose(m['velocity_at_raw_world_point_w_mps'], [1., 2., -2.])
    assert m['normal_velocity_mps'] == -2. and np.isclose(m['tangential_speed_mps'], np.sqrt(5.))
    w = wrench([dict(point_w=[1., 0., 0.], force_w_N=[0., 3., 0.])], state)
    assert w['moment_about_link_origin_w_Nm'] == [0., 0., 3.]
    turned = dict(state, quaternion_wxyz=[np.sqrt(.5), 0., 0., np.sqrt(.5)])
    assert np.allclose(point_measurement([0., 1., 0.], [0., 0., 1.], turned, shape)['local_point_m'], [1., 0., 0.])
    return dict(capsule_distance=True, point_velocity_and_normal_sign=True, moment_cross_product_sign=True, quaternion_world_to_local=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/v030_late_contact_heldout_left_near_slower.json')
    parser.add_argument('--previous', type=Path, default=ROOT/'reports/v030_lag_pose_heldout_left_near_slower.json')
    parser.add_argument('--baseline', type=Path, default=ROOT/'reports/v028_final_heldout_left_near_slower.json')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/late_landing_contact_diagnostic.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if args.output.resolve() in {p.resolve() for p in (args.report, args.previous, args.baseline, Path(__file__))}:
        raise ValueError('Cannot overwrite input')
    report, previous, baseline = [json.loads(p.read_text()) for p in (args.report, args.previous, args.baseline)]
    identity = dict(trace_count=len(report['traces'][0]), original_trace_fields_equal=(len(report['traces'][0]) == len(previous['traces'][0]) and
        all(all(k in b and b[k] == v for k, v in a.items()) for a, b in zip(previous['traces'][0], report['traces'][0]))),
        **{key+'_equal': report[key] == previous[key] for key in ('records', 'kinematics', 'sequence_records')})
    if not all(v for k, v in identity.items() if k.endswith('_equal')):
        raise ValueError('Read-only observation changed original outputs')
    urdf = ROOT/'configs/paid_original/paid_main_resolved.urdf'
    if sha(urdf) != report['asset_sha256']:
        raise ValueError('Asset hash mismatch')
    probe, probe_provenance = _probe(report)
    geometry = {side: capsules(urdf, side) for side in ('left', 'right')}
    rows = validated_rows(report['traces'][0])
    details = [analyze_detail(r, geometry) for r in rows if 'ground_contact_detail' in r]
    if len(details) != 17:
        raise ValueError('Expected the declared 17 observed contact bins')
    direction = summarize_explosive_strike(report)['direction_w']
    window = landing_window_dynamics(rows, report['dynamics_telemetry'], 2.15, 2.35, direction,
        joint_names=report['joint_names'], joint_velocity_limits=report['joint_velocity_limits'], gravity_w=probe['gravity_w'])
    residual = next(b for b in window['momentum_balance']['bins'] if abs(b['interval_s'][0]-2.25) < 1e-9)
    state = next(r for r in rows if abs(r['time_s']-2.25) < 1e-9)
    hypothesis = joint_velocity_scaling_hypothesis(state, report['joint_names'], report['joint_velocity_limits'],
        report['dynamics_telemetry']['robot_total_mass_kg'], [0., 0., -.07605], residual['unaccounted_impulse_w_Ns'])
    inputs = [args.report, args.previous, args.baseline, urdf, Path(__file__)] + [ROOT/name for name in
        ['tools/audit_landing_release.py', 'tools/explosive_strike_metrics.py', 'tools/strike_baseline_metrics.py',
         'tools/strike_dynamics_telemetry.py', 'tools/ground_contact_observer.py',
         'third_party/HumanoidSoccer/source/whole_body_tracking/soccer/robots/g1.py']]
    result = dict(schema_version=1, analytic_checks=analytic_checks(), identity=identity,
        source_sha256={str(p.resolve().relative_to(ROOT)): sha(p) for p in inputs},
        probe_provenance=probe_provenance, recorded_detail_semantics=report['dynamics_telemetry']['ground_contact_detail'],
        capsule_geometry=geometry, capsule_interpretation='Original URDF cylinders are imported as capsules, as declared by the original g1.py replace_cylinders_with_capsules=True. Geometric nearest-index is not an SDK shape ID.',
        details=details, integrations=[integrated_wrenches(details, a, b) for a, b in [(2.24, 2.245), (2.245, 2.25), (2.24, 2.25), (2.2, 2.28)]],
        late_window_dynamics=window, velocity_scaling_hypothesis=hypothesis,
        original_task={name: payload['records'][0] for name, payload in [('v28', baseline), ('v30', report)]},
        limitations=['SDK point/anchor staging and material identity remain unverified; pre/post surface residuals are diagnostic only.',
            'Normal points and friction anchors are independent; no pairing by buffer index or identical count is assumed.',
            'Reported contact wrenches about the ankle-roll link origin are not actuator/joint reaction torque or energy-transfer shares.',
            'Pre/post snapshot moment integration omits force-point substep history and is not an angular-momentum balance.',
            'No baseline vector-ground-force or contact-point replay exists for original v28.'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), samples=len(details), identity=identity, analytic_checks=result['analytic_checks'])))


if __name__ == '__main__':
    main()
