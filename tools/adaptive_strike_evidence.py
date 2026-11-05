"""Reject mixed controller modes and incomplete evidence in the adaptive study."""
import math


def verify_candidate_runtime(candidate, baseline, sources):
    ev=candidate.get('adaptive_control',{})
    if ev.get('enabled') is not True:
        raise ValueError('Candidate C requires feedback enabled; keep ablations separate')
    if ev.get('source_sha256')!=sources or candidate.get('stride_shot_source_sha256')!=sources['run_adaptive_stride.py']:
        raise ValueError('Candidate execution source differs')
    for key in ('config_sha256','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256',
                'contact_frame_sampler_sha256','termination_policy','contact_geometry'):
        if key not in candidate or key not in baseline or candidate[key]!=baseline[key]:
            raise ValueError('Immutable runtime evidence differs: '+key)


def validate_feedback_trace(candidate, config):
    feedback=candidate.get('adaptive_control',{}).get('feedback',[])
    poses=candidate['kinematics'][0]
    if not feedback or len(feedback)!=len(poses):
        raise ValueError('One feedback sample per executed policy step required')
    physical=[r for r in candidate['traces'][0] if not r.get('terminated',False)]
    if any('robot_ball_contact_force_N' not in r for r in physical):
        raise ValueError('Physical contact evidence required to check the latch')
    contacts=[r['time_s'] for r in physical if r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0.)>2.]
    first=min(contacts) if contacts else math.inf
    last=-math.inf
    try:
        for i,(r,k) in enumerate(zip(feedback,poses)):
            scalars=[r[n] for n in ('time_s','phase_s','phase_shift_s','left_support_N')]
            for key in ('placement_xy','foot_delta_xy','measured_root_xy','measured_velocity_xy','measured_ball_xy'):
                if len(r[key])!=2:raise ValueError('Planar state evidence required')
                scalars.extend(r[key])
            if not all(math.isfinite(v) for v in scalars):raise ValueError('Non-finite controller evidence')
            t=r['time_s'];phase=r['phase_s']
            if abs(t-k['time_s'])>1e-8 or abs(t-i*.02)>1e-8 or phase<last-1e-8:
                raise ValueError('Missing, shifted or noncausal controller sample')
            if abs(phase-t-r['phase_shift_s'])>1e-8 or abs(phase-t)>config['max_phase_shift_s']+1e-8:
                raise ValueError('Controller phase evidence exceeds its declared bound')
            if max(map(abs,r['placement_xy']+r['foot_delta_xy']))>config['max_placement_m']+1e-8:
                raise ValueError('Placement evidence exceeds its declared bound')
            if type(r['contact_latched']) is not bool or r['contact_latched']!=(first<=t+1e-8):
                raise ValueError('Contact latch disagrees with prior physical right-foot contact')
            last=phase
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete controller evidence') from exc
