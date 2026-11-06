#!/usr/bin/env python3
"""Reanalyze frozen v15/v27/v28 episodes without loading a simulator or policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from explosive_strike_metrics import (joint_velocity_scaling_hypothesis, landing_window_dynamics,
                                      summarize_explosive_strike)


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified_report(root, relative_path, expected_sha256):
    path = (Path(root) / relative_path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Frozen report must remain inside the project')
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Frozen report digest mismatch: ' + str(relative_path))
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError('Physical report must be an object')
    return result, digest


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


PAIR_FIELDS = {
    'preimpact_foot_speed_mps': ('contact', 'preimpact_foot', 'speed_mps'),
    'normal_impulse_Ns': ('contact', 'normal_impulse_Ns'),
    'outgoing_ball_speed_xy_mps': ('contact', 'outgoing_ball', 'speed_xy_mps'),
    'support_com_delta_forward_velocity_mps': ('support_braking', 'com_kinematics', 'delta_forward_velocity_mps'),
    'right_hip_positive_work_estimate_J': ('precontact_joint_dynamics', 'joints', 'right_hip_pitch_joint', 'power', 'positive_work_J'),
    'final_standing_s': ('recovery', 'original_recovery', 'final_standing_seconds'),
}


def paired_changes(cases, before, after):
    a = {c['case']['id']: c for c in cases if c['version'] == before}
    b = {c['case']['id']: c for c in cases if c['version'] == after}
    common = sorted(set(a) & set(b))
    result = dict(before=before, after=after, matched_case_count=len(common),
                  before_passed=sum(a[k]['metrics']['outcome']['success'] for k in common),
                  after_passed=sum(b[k]['metrics']['outcome']['success'] for k in common), metrics={})
    for label, path in PAIR_FIELDS.items():
        delta = []
        for key in common:
            av, bv = nested(a[key]['metrics'], *path), nested(b[key]['metrics'], *path)
            if av is not None and bv is not None:
                delta.append(dict(case=key, before=av, after=bv, change=bv - av))
        result['metrics'][label] = dict(available_pairs=len(delta), unavailable_pairs=len(common) - len(delta),
            mean_change=float(np.mean([r['change'] for r in delta])) if delta else None,
            min_change=min((r['change'] for r in delta), default=None),
            max_change=max((r['change'] for r in delta), default=None), cases=delta)
    return result


def compare_frozen_replay(observed, original):
    old_rows, new_rows = original['traces'][0], observed['traces'][0]
    differences = [b['time_s'] for a, b in zip(old_rows, new_rows)
                   if any(key not in b or b[key] != value for key, value in a.items())]
    fields = ('asset_sha256', 'policy_sha256', 'goals', 'points', 'initial_robot_state',
              'joint_velocity_limits', 'joint_effort_limits', 'controller_sources_sha256',
              'motion_sources_sha256', 'sequence_sources_sha256', 'root_fixed', 'post_initialization_state_writes')
    return dict(records_equal=observed.get('records') == original.get('records'),
                kinematics_equal=observed.get('kinematics') == original.get('kinematics'),
                legacy_trace_fields_equal=len(old_rows) == len(new_rows) and not differences,
                equal_trace_length=len(old_rows) == len(new_rows), first_trace_difference_s=min(differences, default=None),
                frozen_input_fields_equal={key: observed.get(key) == original[key] for key in fields if key in original},
                definition='Exact JSON-value comparison for original records, kinematics and every legacy trace field; added observation fields do not count as motion changes')


def analyze_instrumented_report(path, root=ROOT, start_s=1.80, end_s=1.90):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Instrumented report must be inside the project')
    payload = path.read_bytes()
    result = json.loads(payload)
    metadata = result['dynamics_telemetry']
    if metadata.get('physical_writes') is not False or metadata.get('controller_changed') is not False:
        raise ValueError('Instrumented report must certify observation-only operation')
    if metadata['controller'] != 'v28':
        raise ValueError('This comparison is scoped to the three observed v28 episodes')
    registry_path = root / 'reports/v028_final_comparison.json'
    registry = json.loads(registry_path.read_text())
    case = next(c for c in registry['cases'] if c['case']['id'] == metadata['registered_case'])
    frozen, frozen_digest = read_verified_report(root, case['candidate_report'], case['candidate_sha256'])
    probe_path = Path(metadata['probe_path'])
    if not probe_path.is_absolute():
        probe_path = root / probe_path
    probe, probe_digest = read_verified_report(root, probe_path, metadata['probe_sha256'])
    if probe['certificate'].get('verified') is not True or probe['certificate'].get('matrix_semantics') != 'normal_only':
        raise ValueError('A passed independent normal/friction force probe is required')
    urdf_path = root / 'configs/paid_original/paid_main_resolved.urdf'
    urdf_digest = sha(urdf_path)
    if urdf_digest != result['asset_sha256']:
        raise ValueError('URDF for the scaling hypothesis differs from the recorded asset')
    origin = ET.fromstring(urdf_path.read_bytes()).find("./link[@name='pelvis']/inertial/origin")
    offset = [float(v) for v in origin.attrib['xyz'].split()]
    metrics = summarize_explosive_strike(result)
    landing = landing_window_dynamics(result['traces'][0], metadata, start_s, end_s, metrics['direction_w'],
        joint_names=result['joint_names'], joint_velocity_limits=result['joint_velocity_limits'], gravity_w=probe['gravity_w'])
    balance = landing['momentum_balance']
    if balance['status'] == 'available':
        interval = balance['max_residual_bin_s']
        before = next(r for r in result['traces'][0] if abs(r['time_s'] - interval[0]) < 1e-9)
        residual = next(b['unaccounted_impulse_w_Ns'] for b in balance['bins'] if b['interval_s'] == interval)
        landing['joint_velocity_scaling_hypothesis'] = joint_velocity_scaling_hypothesis(before,
            result['joint_names'], result['joint_velocity_limits'], metadata['robot_total_mass_kg'], offset, residual)
        landing['max_residual_joint_limit_context'] = [r for r in landing['joint_speed_limits'].get('samples', [])
                                                       if any(abs(r['time_s'] - t) < 1e-9 for t in interval)]
    return dict(case=metadata['registered_case'], report=str(path.relative_to(root)),
                sha256=hashlib.sha256(payload).hexdigest(), frozen_report=case['candidate_report'], frozen_sha256=frozen_digest,
                frozen_replay_match=compare_frozen_replay(result, frozen), dynamics_telemetry=metadata,
                probe_report=str(probe_path.resolve().relative_to(root)), probe_sha256=probe_digest,
                probe_certificate=probe['certificate'], hypothesis_urdf=str(urdf_path.relative_to(root)),
                hypothesis_urdf_sha256=urdf_digest, original_metrics=metrics, landing_window=landing)


def aggregate_reports(root=ROOT):
    root = Path(root)
    config_path = root / 'configs/strike_baseline_v1.json'
    legacy_path = root / 'reports/v023_baseline_final_summary.json'
    comparison_path = root / 'reports/v028_final_comparison.json'
    config, legacy, comparison = [json.loads(p.read_text()) for p in (config_path, legacy_path, comparison_path)]
    config_digest = sha(config_path)
    if legacy['matrix_sha256'] != config_digest:
        raise ValueError('Frozen v15 matrix fingerprint differs from its summary')
    if comparison['version'] != '0.1.28' or comparison['baseline_version'] != 'v0.1.27':
        raise ValueError('Expected the declared v27/v28 comparison')
    cases, seen = [], set()
    source_inputs = {str(p.relative_to(root)): sha(p) for p in (config_path, legacy_path, comparison_path)}
    legacy_by_id = {case['id']: case for case in legacy['cases']}

    def add(version, case, group, path, digest, expected_outcome, frozen=False):
        key = (version, case['id'])
        if key in seen:
            raise ValueError('Duplicate version/case in source manifest')
        seen.add(key)
        episode, observed = read_verified_report(root, path, digest)
        if frozen:
            evidence = episode.get('strike_baseline', {})
            if (evidence.get('matrix_sha256') != config_digest or evidence.get('case') != case
                    or evidence.get('frozen_files_sha256') != config['frozen_files_sha256']):
                raise ValueError('Frozen baseline report provenance differs: ' + path)
        metrics = summarize_explosive_strike(episode, expected_duration_s=config['steps'] * .02,
                                            window_s=config['precontact_window_s'])
        if metrics['outcome'] != expected_outcome:
            raise ValueError('Recomputed original outcome differs from frozen summary: ' + path)
        cases.append(dict(version=version, case=case, case_group=group, report=path, sha256=observed,
                          source_torque_evidence=episode.get('torque_evidence'), metrics=metrics))

    for case in config['cases']:
        old = legacy_by_id[case['id']]
        add('v0.1.15', case, 'frozen_sensitivity', old['report'], old['sha256'], old['metrics']['outcome'], frozen=True)
    for case in comparison['cases']:
        if case['baseline_report'] is not None:
            add('v0.1.27', case['case'], case['case_group'], case['baseline_report'], case['baseline_sha256'],
                case['baseline']['outcome'])
        add('v0.1.28', case['case'], case['case_group'], case['candidate_report'], case['candidate_sha256'],
            case['candidate']['outcome'])
    groups = {}
    for version in ('v0.1.15', 'v0.1.27', 'v0.1.28'):
        selected = [c for c in cases if c['version'] == version]
        groups[version] = dict(total=len(selected), passed=sum(c['metrics']['outcome']['success'] for c in selected),
            contacts=sum(c['metrics']['contact']['status'] == 'available' for c in selected),
            failures=dict(Counter(reason for c in selected for reason in c['metrics']['outcome']['failure_reasons'])),
            horizontal_grf_available=sum(nested(c['metrics'], 'support_braking', 'horizontal_ground_reaction', 'status') == 'available' for c in selected),
            thorax_orientation_available=sum(nested(c['metrics'], 'segment_sequencing', 'segments', 'thorax', 'status') == 'available' for c in selected))
    paths = ['tools/explosive_strike_metrics.py', 'tools/analyze_explosive_strike.py', 'tools/strike_baseline_metrics.py']
    return dict(schema_version=1, analysis_version='0.1.29', source_inputs_sha256=source_inputs,
        analysis_sources_sha256={p: sha(root / p) for p in paths}, groups=groups, cases=cases,
        paired_comparisons=[paired_changes(cases, 'v0.1.15', 'v0.1.27'), paired_changes(cases, 'v0.1.27', 'v0.1.28')],
        scope='Offline reanalysis of deterministic, already executed physical episodes; no training, simulation, controller edits, or population-rate claim',
        interpretation='Different versions have different case sets. Use matched comparisons; unavailable measurements are not zero or failure outcomes.')


def _format(value, precision=3):
    return 'unavailable' if value is None else f'{value:.{precision}f}'


def render_instrumented_markdown(cases, artifact_path):
    def vector(v):
        return '[' + ', '.join(f'{a:.6f}' for a in v) + ']'
    lines = ['## Three observed v28 replays: actual force and segment telemetry', '',
        'These are additional observation-only replays of the v0.1.28 controller. They preserve the original '
        'limits and acceptance criteria. Their original task pass does not erase a landing shock, overspeed, '
        'or unaccounted momentum change. Each report retains its own probe and collector source SHA; probe '
        'files are allowed to differ. The original and clean-exit probe versions have the same force-sign result.', '',
        f'Independent instrumented-analysis artifact: `{artifact_path}`.', '',
        '| Case | Original full task | Records exact | Kinematics exact | All legacy trace fields exact |',
        '|---|---|---|---|---|']
    for c in cases:
        match = c['frozen_replay_match']
        lines.append(f"| {c['case']} | {'pass' if c['original_metrics']['outcome']['success'] else 'fail'} | {match['records_equal']} | {match['kinematics_equal']} | {match['legacy_trace_fields_equal']} |")
    lines += ['', 'The fixed diagnostic interval is **1.80–1.90 s**. Peak force below includes observations at '
        'both endpoints. Impulses integrate only the overlapping post-step bins: the force sample at 1.800 s '
        'belongs to the preceding 1.795–1.800 s bin and is excluded from the interval impulse. This distinction '
        'matters for center, whose 1.800 s sample is its largest observed force in this interval. These are fixed '
        'absolute-time windows, not identical phases of every case’s landing.', '',
        '| Case | Total force peak N @ s (body weights) | Foot-point speed peak m/s @ s | Max observed speed/limit | Window momentum residual norm (N·s) |',
        '|---|---:|---:|---:|---:|']
    for c in cases:
        w = c['landing_window']; f = w['endpoint_inclusive_total_force_peak']; v = w['foot_point_motion']
        lines.append(f"| {c['case']} | {f['peak_norm_force_N']:.3f} @ {f['time_s']:.3f} ({f['peak_norm_body_weights']:.3f}) | {v['peak_speed_mps']:.3f} @ {v['peak_speed_time_s']:.3f} | {w['joint_speed_limits']['max_speed_to_limit_ratio']:.6f} | {w['momentum_balance']['residual_norm_Ns']:.6g} |")
    lines += ['', 'World-vector component impulses over the same interval; X/Y/Z axes are kept explicit. '
        'Normal and friction are separately measured, not reconstructed from vertical support or COM changes.', '',
        '| Case | Foot | Normal impulse [X,Y,Z] N·s | Friction impulse [X,Y,Z] N·s | Total impulse [X,Y,Z] N·s |',
        '|---|---|---|---|---|']
    for c in cases:
        for side in ('left', 'right'):
            parts = c['landing_window']['ground_reaction']['feet'][side]['components']
            lines.append('| ' + c['case'] + ' | ' + side + ' | ' + ' | '.join(vector(parts[p]['impulse_w_Ns']) for p in ('normal', 'friction', 'total')) + ' |')
    lines += ['', '| Case | Pelvis net 3-D rotation / projected yaw (deg) | Thorax net 3-D rotation / projected yaw (deg) | Relative pelvis→thorax rotation / projected-yaw change (deg) |',
        '|---|---:|---:|---:|']
    for c in cases:
        w = c['landing_window']; pelvis = w['segments']['pelvis']; thorax = w['segments']['thorax']; relative = w['pelvis_thorax_relative_heading']
        lines.append(f"| {c['case']} | {pelvis['net_rotation_angle_deg']:.3f} / {pelvis['net_projected_yaw_deg']:.3f} | {thorax['net_rotation_angle_deg']:.3f} / {thorax['net_projected_yaw_deg']:.3f} | {relative['net_relative_rotation_angle_deg']:.3f} / {relative['net_change_deg']:.3f} |")
    lines += ['', 'Three-dimensional net rotation is the angle of the relative orientation between observed '
        'endpoints, not an Euler-angle sum. Projected yaw uses the world-horizontal projection of each body '
        'X axis. Rotation magnitudes, relative headings, and observed sequencing do not establish energy transfer.', '',
        '| Case | Precontact pelvis / thorax / left-arm / right-arm peak projected yaw-rate times (ms) |',
        '|---|---:|']
    for c in cases:
        segments = c['original_metrics']['segment_sequencing']['segments']
        times = [segments[k]['projected_yaw_rate_peaks_rad_s']['absolute']['time_to_contact_s'] * 1000
                 for k in ('pelvis', 'thorax', 'left_upper_arm', 'right_upper_arm')]
        lines.append('| ' + c['case'] + ' | ' + ' / '.join(f'{v:.1f}' for v in times) + ' |')
    lines += ['', 'Precontact values use the earlier 300 ms protocol and finite-difference interval midpoints. '
        'They are descriptive peak timings; the JSON retains signed extrema, all ties, and observed world '
        'angular velocities separately.', '',
        '### Momentum residual and speed-limit observations', '',
        'The diagnostic residual is `m(v_COM,end − v_COM,start) − J_left − J_right − m g Δt`. '
        'Mass is the measured summed PhysX body mass. Gravity is assumed equal to the recorded probe '
        'gravity `[0,0,−9.81] m/s²`; it is not independently stored as robot-world telemetry. All per-bin '
        'residual vectors and every overspeed joint are retained. A residual is not reassigned to friction or '
        'removed from task results.', '',
        '| Case | Largest residual bin (s) | Original residual [X,Y,Z] N·s | Overspeed joints at bin start → end |',
        '|---|---|---|---|']
    for c in cases:
        w = c['landing_window']; b = w['momentum_balance']; interval = b['max_residual_bin_s']
        residual = next(r['unaccounted_impulse_w_Ns'] for r in b['bins'] if r['interval_s'] == interval)
        contexts = []
        for r in w['max_residual_joint_limit_context']:
            joints = ', '.join(f"{j['joint']} {j['signed_velocity_rad_s']:.5f}/{j['limit_rad_s']:.1f} rad/s ({j['ratio']:.6f}×)" for j in r['overspeed_joints']) or 'none'
            contexts.append(f"{r['time_s']:.3f}: {joints}; max ratio {r['max_speed_to_limit_ratio']:.6f}")
        lines.append(f"| {c['case']} | {interval[0]:.3f}–{interval[1]:.3f} | {vector(residual)} | " + ' → '.join(contexts) + ' |')
    lines += ['', 'The `initial_slower_02` replay retains a full task pass but shows a large landing shock, '
        'right-knee overspeed, and a subsequent non-closing momentum bin. The large impact bin itself '
        'matches the measured contact impulse closely; the large unexplained change occurs in the following '
        '5 ms bin. Temporal coincidence supports investigation, not a causal conclusion.', '',
        '### Published solver mechanism: supporting inference, not a confirmed local bug', '',
        'An open, unmerged PhysX proposal describes a pre-solve mechanism in which one joint exceeding '
        'its velocity limit scales all articulation joint velocities by the smallest limit-to-speed ratio. '
        'The proposal changes that behavior to separate per-DOF clipping. The installed solver binary has '
        'not been matched to that source. [PhysX PR #534](https://github.com/NVIDIA-Omniverse/PhysX/pull/534) '
        '(status checked 2026-11-07).', '',
        'The following model is evaluated from the observation at the start of the largest residual bin:', '',
        '```text', 'gamma = min(1, min_j limit_j / |qdot_j|)',
        'x_rootCOM = x_root + R_pelvis * pelvis_inertial_origin',
        'v_internal = v_COM - v_rootCOM - omega_root × (x_COM - x_rootCOM)',
        'predicted impulse = mass * (gamma - 1) * v_internal', '```', '',
        '| Case | gamma | Predicted impulse [X,Y,Z] N·s | Prediction vs original residual error norm (N·s) |',
        '|---|---:|---|---:|']
    for c in cases:
        h = c['landing_window']['joint_velocity_scaling_hypothesis']
        lines.append(f"| {c['case']} | {h['gamma']:.9f} | {vector(h['predicted_impulse_w_Ns'])} | {h['prediction_residual_error_norm_Ns']:.6g} |")
    lines += ['', 'The pelvis inertial origin `[0,0,−0.07605] m` is read from the SHA-matched URDF, not queried '
        'directly from PhysX. The model treats recorded root linear velocity as root-COM velocity and assumes '
        'the proposed uniform scaling applies while the base twist is unchanged. Close agreement in the '
        'overspeed case is evidence consistent with that mechanism; it does not prove which local binary '
        'operation caused the residual. No engine version, force, joint limit, state, or acceptance result is '
        'changed by this analysis.', '', '### Instrumented provenance', '',
        '| Case | Physical report SHA-256 | Frozen v28 SHA-256 | Probe file and SHA-256 |', '|---|---|---|---|']
    for c in cases:
        lines.append(f"| {c['case']} | `{c['sha256']}` | `{c['frozen_sha256']}` | `{c['probe_report']}` — `{c['probe_sha256']}` |")
    lines += ['', 'Collector and wrapper source hashes are copied unchanged from each physical report’s '
        '`dynamics_telemetry`; later wrapper cleanup does not retroactively change acquisition provenance. '
        'The JSON also records the URDF hash, original probe certificate, baseline-match results, and current '
        'analysis-source hashes.', '', 'Reproduce the baseline plus these three instrumented analyses:', '',
        '```sh', 'python3 tools/analyze_explosive_strike.py \\',
        '  --output reports/explosive_strike_baseline_v029.json \\',
        f'  --instrumented-output {artifact_path} \\',
        '  --instrumented-reports ' + ' '.join(c['report'] for c in cases) + ' \\',
        '  --markdown reports/2026-11-07-explosive-strike-baseline.md --overwrite', '```', '']
    return '\n'.join(lines)


def render_markdown(summary, artifact_path):
    lines = [
        '# Explosive-strike offline baseline — 2026-11-07', '',
        'This v0.1.29 analysis measures existing physical episodes. It does not change the controller, '
        'rerun a simulator, train a model, or assign a naturalness score. Frozen v0.1.15 is the seven-case '
        'v0.1.23 replay, including `reports/v023_baseline_center.json`; v0.1.27 and v0.1.28 are loaded '
        'from the paths and SHA-256 digests in `reports/v028_final_comparison.json`.', '',
        f"Full per-joint extrema, unavailable-evidence reasons, paired deltas, and all {len(summary['cases'])} episode digests: `{artifact_path}`.", '',
        '## Original task outcomes retained', '',
        '| Version | Episodes | Full task passes | Actual foot contacts | Certified horizontal GRF | Thorax orientation |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for version, group in summary['groups'].items():
        lines.append(f"| {version} | {group['total']} | {group['passed']} | {group['contacts']} | {group['horizontal_grf_available']} | {group['thorax_orientation_available']} |")
    lines += ['', 'The case counts differ and are deterministic sensitivity results, not population success rates. '
              'The v0.1.28 matrix has 15 development cases and three later evaluation cases; all original goal, '
              'running, contact-surface, recovery, illegal-contact, and physical-integrity gates remain in force.', '',
              '## Center case: speed, impulse, support and recovery', '',
              '| Version | Preimpact fixed-foot-point speed (m/s) | Outgoing ball horizontal speed (m/s) | First normal impulse (N·s) | Loaded-support COM Δv forward (m/s) | Final sequence standing (s) |',
              '|---|---:|---:|---:|---:|---:|']
    centers = [c for c in summary['cases'] if c['case']['id'] == 'center']
    for c in centers:
        m = c['metrics']
        values = [nested(m, *PAIR_FIELDS[key]) for key in ['preimpact_foot_speed_mps', 'outgoing_ball_speed_xy_mps', 'normal_impulse_Ns', 'support_com_delta_forward_velocity_mps', 'final_standing_s']]
        lines.append('| ' + c['version'] + ' | ' + ' | '.join(_format(v) for v in values) + ' |')
    lines += ['', 'The COM quantity is a measured velocity change during the last continuous ≥80 N left-support '
              'episode intersecting the preceding 300 ms. A negative value means a loss of forward COM velocity '
              'along the fixed ball-to-goal direction. It does not identify the support foot’s horizontal force, '
              'a force path, or energy transferred to the kicking leg. Standing durations here use the original '
              '`sequence_records[0].recovery` criterion. The shot tracker has a separate standing counter '
              '(center v0.1.27/v0.1.28: 8.965 s versus 8.910 s for the sequence criterion); both are retained '
              'in the JSON, without changing either gate.', '',
              '## Center case: signed joint timing', '',
              'Both positive and negative peaks are retained because the largest absolute knee speed can occur '
              'in a different motion phase than the late hip-speed peak. Times below are relative to '
              'the first measured foot-ball normal-force sample; negative times precede contact.', '',
              '| Version | Joint | Positive q̇ peak (rad/s @ ms) | Negative q̇ peak (rad/s @ ms) | Positive estimated-power peak (W @ ms) | Negative estimated-power peak (W @ ms) |',
              '|---|---|---:|---:|---:|---:|']
    joint_names = ['right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint',
                   'waist_yaw_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint']
    def peak_cell(joint, kind, sign):
        p = nested(joint, kind, 'peaks', sign)
        return 'absent' if p is None else f"{p['value']:.2f} @ {p['time_to_contact_s'] * 1000:.1f}"
    for c in centers:
        for name in joint_names:
            j = nested(c['metrics'], 'precontact_joint_dynamics', 'joints', name)
            cells = [peak_cell(j, kind, sign) for kind, sign in [('velocity', 'positive'), ('velocity', 'negative'), ('power', 'positive'), ('power', 'negative')]]
            lines.append('| ' + c['version'] + ' | ' + name.removesuffix('_joint') + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'The 300 ms window includes approach motion before the final plant. The following view uses '
              'only the loaded-support subset of that window, including the first loaded bin and excluding '
              'the collision bin. In the center case the first loaded sample is 1.395 s, with onset bracket '
              '1.390–1.395 s; contact is 1.570 s.', '',
              '| Version | Joint | Post-plant positive q̇ peak (rad/s @ ms) | Post-plant negative q̇ peak (rad/s @ ms) | Positive estimated work (J) | Negative estimated work (J) |',
              '|---|---|---:|---:|---:|---:|']
    for c in centers:
        for name in joint_names[:3]:
            j = nested(c['metrics'], 'post_plant_joint_dynamics', 'joints', name)
            cells = [peak_cell(j, 'velocity', sign) for sign in ('positive', 'negative')]
            cells += [_format(nested(j, 'power', field)) for field in ('positive_work_J', 'negative_work_J')]
            lines.append('| ' + c['version'] + ' | ' + name.removesuffix('_joint') + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'These signs are robot joint coordinates. Shoulder and waist values are relative-joint proxies, '
              'not world upper-arm/thorax angular velocities. Pelvis projected heading is available from '
              '20 ms root orientations; thorax and upper-arm world orientations are unavailable in these '
              'frozen reports, so a full pelvis–thorax–arm sequence cannot be concluded.', '',
              '| Version | Pelvis projected yaw net / range (deg) | Peak absolute projected yaw rate (rad/s @ ms) | Positive hip / knee / ankle forward integrals (m) |',
              '|---|---:|---:|---:|']
    for c in centers:
        pelvis = nested(c['metrics'], 'segment_sequencing', 'segments', 'pelvis')
        p = nested(pelvis, 'projected_yaw_rate_peaks_rad_s', 'absolute')
        k = c['metrics']['kinematic_foot_contributions']
        integrals = ' / '.join(_format(nested(k, 'parts', part, 'positive_integral_m')) for part in ('hip', 'knee', 'ankle'))
        lines.append(f"| {c['version']} | {_format(pelvis.get('net_projected_yaw_deg'))} / {_format(pelvis.get('projected_yaw_range_deg'))} | {p['value']:.3f} @ {p['time_to_contact_s'] * 1000:.1f} | {integrals} |")
    lines += ['', 'The integrals are signed/positive/negative Jacobian foot-velocity components in the 300 ms '
              'window; the table shows positive components only. The JSON retains all signs, the floating-base '
              'remainder, and numerical closure against the measured fixed-point velocity. These are '
              'kinematic displacement contributions, not energy percentages or ball-work shares.', '',
              '## Matched changes', '',
              '| Comparison | Matched cases | Full passes before → after | Mean foot-speed change (m/s) | Mean normal-impulse change (N·s) | Mean final-sequence-standing change (s) |',
              '|---|---:|---:|---:|---:|---:|']
    for pair in summary['paired_comparisons']:
        means = [pair['metrics'][key]['mean_change'] for key in ['preimpact_foot_speed_mps', 'normal_impulse_Ns', 'final_standing_s']]
        lines.append(f"| {pair['before']} → {pair['after']} | {pair['matched_case_count']} | {pair['before_passed']} → {pair['after_passed']} | " + ' | '.join(_format(v) for v in means) + ' |')
    lines += ['', 'Means include only pairs with both measurements available; the JSON records pair counts and '
              'each case’s delta. No significance test or human-biomechanics match is implied.', '',
              '## Per-case contact and follow-through', '',
              '| Version | Case | Full task | Preimpact foot speed (m/s) | Normal impulse (N·s) | Initial forward follow-through (ms) | Final sequence standing (s) |',
              '|---|---|---|---:|---:|---:|---:|']
    for c in summary['cases']:
        m = c['metrics']
        values = [nested(m, 'contact', 'preimpact_foot', 'speed_mps'), nested(m, 'contact', 'normal_impulse_Ns'),
                  nested(m, 'follow_through', 'continuous_forward_duration_s'), nested(m, 'recovery', 'original_recovery', 'final_standing_seconds')]
        if values[2] is not None:
            values[2] *= 1000
        outcome = 'pass' if m['outcome']['success'] else ', '.join(m['outcome']['failure_reasons'])
        lines.append('| ' + c['version'] + ' | ' + c['case']['id'] + ' | ' + outcome + ' | ' + ' | '.join(_format(v) for v in values) + ' |')
    lines += ['', 'Follow-through starts at the first zero-force sample after the primary contact episode and '
              'runs for 300 ms, unless the trace ends or another foot-ball contact occurs. Its continuous '
              'forward duration ends at the last positive sample before the first nonpositive forward '
              'velocity; reversal is descriptive, not a failure. Integrated path length is a fixed-reference-point '
              'kinematic estimate. Original recovery duration and acceptance are retained; final-standing onset '
              'is derived from the original final duration, not from a new recovery threshold.', '',
              '## Methods and evidence boundaries', '',
              '- Physical trace validation rejects gaps, nonfinite force/time, inconsistent stored contact, and '
              'unexplained early truncation. The legacy duplicated terminal marker is excluded from integration.',
              '- Precontact windows are the 300 ms preceding the first positive measured foot-ball force. '
              'Samples at contact are excluded, including their entire collision bin. At 5 ms sampling, a '
              'complete 300 ms window consequently provides 295 ms of strictly preimpact rectangular bins. '
              'Fractional first bins are clipped; window coverage is recorded.',
              '- Signed velocity and estimated-power extrema retain all exact global ties and every local '
              'positive/negative plateau, with boundary flags. Representative global timing is the earliest '
              'tie. There is no smoothing, so small local extrema may reflect numerical noise. Absent signs '
              'have no fabricated zero-valued peak time. A 5 ms observation is not a continuous-time optimum.',
              '- Power is `joint_vel × joint_torque_estimate`; positive and negative work are integrated '
              'separately. The frozen producer calls the implicit-actuator applied torque an estimated clipped '
              'PD torque. These are approximate actuator quantities, not measured hardware power, inverse '
              'dynamics, muscle power, or intersegment energy-transfer shares.',
              '- Support onset is bracketed between the first ≥80 N sample and its preceding physics boundary. '
              'The last continuous left support reaching the last precontact sample is used. COM acceleration '
              'uses finite differences between two loaded observations; it cannot include the collision bin. '
              'Clipped positive vertical ground force only determines the support gate.',
              '- Optional horizontal GRF is admitted only when `dynamics_telemetry.support_ground_force` '
              'certifies world force ON the robot, normal plus friction, ground filtering, complete coverage, '
              'and a verified sign probe. Missing, partial, or uncertified vectors remain unavailable. '
              'No horizontal force is reconstructed from vertical support or COM deceleration.',
              '- First-contact impulse integrates actual sampled contact-point normal-force magnitudes '
              'over the first contiguous contact episode. Separate episodes are counted independently. '
              'This scalar normal impulse is not the net vector impulse or the ball momentum change. '
              'Preimpact foot speed is the last noncontact sample of the fixed foot point '
              '`[0.06, 0, -0.025] m`, not the collision-point speed.',
              '- Segment headings project each measured body X axis onto the world horizontal plane and '
              'unwrap yaw. Heading rates are finite differences at interval midpoints, distinct from world-Z '
              'angular velocity. Relative pelvis/thorax projected heading requires common observed timestamps. '
              'A measured segment sequence does not establish causal energy flow.',
              '- Missing quantities carry `status: unavailable`, a reason, and evidence needed. Invalid '
              'mandatory physical traces or changed source digests stop analysis. Optional missing fields '
              'do not silently become zero. The JSON preserves the original outcome for every case.', '',
              '## Reproduction and provenance', '',
              '```sh', 'python3 tools/analyze_explosive_strike.py \\',
              f'  --output {artifact_path} \\',
              '  --markdown reports/2026-11-07-explosive-strike-baseline.md', '```', '',
              'Outputs are new analysis artifacts. Existing files require explicit `--overwrite`; physical '
              'source reports cannot be output destinations. No local motion/model assets are exported.', '',
              '| Input or analysis source | SHA-256 |', '|---|---|']
    for path, digest in {**summary['source_inputs_sha256'], **summary['analysis_sources_sha256']}.items():
        lines.append(f'| `{path}` | `{digest}` |')
    for c in centers:
        lines.append(f"| `{c['report']}` | `{c['sha256']}` |")
    lines += ['', 'All source episodes are digest-checked against the existing summaries before metrics are '
              'computed; original task outcomes are recomputed and checked against those summaries. '
              'The full per-episode digest manifest is in the JSON. This verifies consistency with the '
              'local recorded evidence; it is not independent validation of the simulator or a fresh replay.', '']
    if summary.get('instrumented_cases'):
        lines += [render_instrumented_markdown(summary['instrumented_cases'], summary['instrumented_artifact_path'])]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/explosive_strike_baseline_v029.json')
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--instrumented-reports', nargs='+', type=Path)
    parser.add_argument('--instrumented-output', type=Path)
    args = parser.parse_args()
    if bool(args.instrumented_reports) != bool(args.instrumented_output):
        parser.error('--instrumented-reports and --instrumented-output must be supplied together')
    outputs = [p.resolve() for p in (args.output, args.markdown, args.instrumented_output) if p is not None]
    if len(set(outputs)) != len(outputs):
        parser.error('JSON and Markdown destinations must differ')
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    summary = aggregate_reports()
    if args.instrumented_reports:
        summary['instrumented_cases'] = [analyze_instrumented_report(p) for p in args.instrumented_reports]
        summary['instrumented_artifact_path'] = str(args.instrumented_output.resolve().relative_to(ROOT))
    protected = {str((ROOT / p).resolve()) for p in summary['source_inputs_sha256']}
    protected.update(str((ROOT / c['report']).resolve()) for c in summary['cases'])
    protected.update(str((ROOT / p).resolve()) for p in summary['analysis_sources_sha256'])
    for c in summary.get('instrumented_cases', []):
        protected.update(str((ROOT / c[key]).resolve()) for key in ('report', 'frozen_report', 'probe_report', 'hypothesis_urdf'))
    if any(str(p) in protected for p in outputs):
        raise ValueError('Analysis cannot overwrite any source input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if args.instrumented_output:
        args.instrumented_output.parent.mkdir(parents=True, exist_ok=True)
        args.instrumented_output.write_text(json.dumps(dict(schema_version=1,
            analysis_sources_sha256=summary['analysis_sources_sha256'], cases=summary['instrumented_cases']),
            indent=2, allow_nan=False) + '\n')
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        label = str(args.output.resolve().relative_to(ROOT)) if args.output.resolve().is_relative_to(ROOT) else str(args.output.resolve())
        args.markdown.write_text(render_markdown(summary, label))
    print(json.dumps(dict(output=str(args.output), groups=summary['groups'], report_count=len(summary['cases']),
                         instrumented_count=len(summary.get('instrumented_cases', [])))))


if __name__ == '__main__':
    main()
