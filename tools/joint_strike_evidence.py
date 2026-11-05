"""Audit measured feedback and immutable physics for learned residual rollouts."""
import math
from strike_baseline_metrics import validated_rows


def validate_feedback(result,config):
    feedback=result.get('joint_strike',{}).get('feedback',[])
    poses=result['kinematics'][0]
    if not feedback or len(feedback)!=len(poses):raise ValueError('One feedback sample per executed command required')
    rows=validated_rows(result['traces'][0])
    commands=math.ceil(rows[-1]['time_s']/.02-1e-7)
    if len(feedback)!=commands:raise ValueError('Controller evidence does not cover executed physics')
    if any('robot_ball_contact_force_N' not in r for r in rows):raise ValueError('All-body contact evidence required')
    first=min((r['time_s'] for r in rows if r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0.)>2.),default=math.inf)
    states={round(r['time_s'],7):r['ball_xyz'] for r in rows}
    states[0.]=[*result['points'][0],.113]
    expected_phase=0.
    try:
        for i,(row,pose) in enumerate(zip(feedback,poses)):
            t=row['time_s'];phase=row['phase_s'];action=row['phase_action'];residual=row['residual_rad'];ball=row['measured_ball_xyz']
            if len(residual)!=29 or len(ball)!=3 or not all(math.isfinite(x) for x in [t,phase,action,*residual,*ball]):raise ValueError('Finite, correctly shaped feedback required')
            if abs(t-i*.02)>1e-7 or abs(t-pose['time_s'])>1e-7:raise ValueError('Shifted command timestamp')
            if abs(phase-expected_phase)>2e-6 or abs(phase-t)>config['maximum_phase_shift_s']+2e-6:raise ValueError('Incorrect phase recurrence or bound')
            if max(map(abs,residual))>config['maximum_joint_residual_rad']+1e-6:raise ValueError('Joint correction exceeds bound')
            if type(row['contact_latched']) is not bool or row['contact_latched']!=(first<=t+1e-8):raise ValueError('Contact latch disagrees with measured prior contact')
            measured=states.get(round(t,7))
            if measured is None or any(abs(x-y)>1e-5 for x,y in zip(measured,ball)):raise ValueError('Feedback ball differs from physical state')
            rate=config['phase_rate_limit']*math.tanh(action) if i<100 else 0.
            expected_phase=max(phase,min(max(phase+.02*(1+rate),t+.02-config['maximum_phase_shift_s']),t+.02+config['maximum_phase_shift_s']))
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete feedback evidence') from exc


def verify_runtime(candidate,baseline,case,config,config_sha,sources,checkpoint_sha):
    ev=candidate.get('joint_strike',{})
    if ev.get('mode')!='learned_residual' or ev.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('Learned checkpoint differs')
    if ev.get('config')!=config or ev.get('config_sha256')!=config_sha or ev.get('source_sha256')!=sources:raise ValueError('Candidate source/configuration differs')
    if ev.get('template_sha256')!=config['template_sha256'] or candidate.get('policy_sha256')!=checkpoint_sha:raise ValueError('Active policy/template differs')
    if ev.get('ball_offset_xy')!=case['ball_offset_xy_m'] or ev.get('velocity_y_delta')!=case['root_velocity_y_delta_mps']:raise ValueError('Requested case differs')
    expected=baseline['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    velocity=ev.get('initial_root_velocity',[])
    if len(velocity)!=3 or any(not math.isfinite(v) or abs(v-e)>1e-6 for v,e in zip(velocity,expected)):raise ValueError('Measured initial velocity differs')
    for key in ['initial_robot_state','asset_sha256','joint_effort_limits','joint_velocity_limits','points','goals','goal_width_m','goal_height_m','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','termination_policy','contact_geometry','config_sha256']:
        if key not in candidate or key not in baseline or candidate[key]!=baseline[key]:raise ValueError('Frozen physical runtime differs: '+key)
    if candidate.get('root_fixed') is not False or candidate.get('post_initialization_state_writes') is not False:raise ValueError('Physical integrity differs')
    validate_feedback(candidate,config)
