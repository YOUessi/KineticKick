"""Independent physical/runtime audit for protected closed-loop residual runs."""
import math
from joint_strike_evidence import validate_feedback
from closed_loop_acceptance import full_task_success

SOURCE_NAMES=['run_closed_loop_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','closed_loop_acceptance.py']


def verify_closed_loop(candidate,baseline,case,config,config_sha,sources,checkpoint_sha,nominal_velocity):
    ev=candidate.get('joint_strike',{})
    mode='zero_reference' if checkpoint_sha is None else 'learned_residual'
    if ev.get('mode')!=mode or ev.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('Wrong learned/zero controller')
    if ev.get('config')!=config or ev.get('config_sha256')!=config_sha or ev.get('source_sha256')!=sources:raise ValueError('Execution source/configuration differs')
    expected_policy=checkpoint_sha or config['expert_plan_sha256']
    if candidate.get('policy_sha256')!=expected_policy or ev.get('template_sha256')!=config['template_sha256']:raise ValueError('Wrong active policy or template')
    if ev.get('expert_backend')!='onnx':raise ValueError('Single-env independent evaluation must record exact ONNX expert')
    if ev.get('ball_offset_xy')!=case['ball_offset_xy_m'] or ev.get('velocity_y_delta')!=case['root_velocity_y_delta_mps']:raise ValueError('Requested case differs')
    expected=baseline['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    velocity=ev.get('initial_root_velocity',[])
    if len(velocity)!=3 or any(not math.isfinite(x) or abs(x-y)>1e-6 for x,y in zip(velocity,expected)):raise ValueError('Wrong measured initial velocity')
    for key in ['initial_robot_state','asset_sha256','joint_effort_limits','joint_velocity_limits','points','goals','goal_width_m','goal_height_m','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','termination_policy','contact_geometry','config_sha256']:
        if key not in candidate or key not in baseline or candidate[key]!=baseline[key]:raise ValueError('Frozen physical/runtime evidence differs: '+key)
    if candidate.get('root_fixed') is not False or candidate.get('post_initialization_state_writes') is not False:raise ValueError('Physical integrity violated')
    expected_success=full_task_success(candidate['running_evidence'],candidate['stride_shot_evidence'],candidate['sequence_records'][0],ev.get('expert_handoffs',[]))
    if candidate.get('stride_shot_success') is not expected_success:raise ValueError('Full success must preserve the frozen recovery-handoff gate')
    validate_feedback(candidate,config)
    delta=[x-y for x,y in zip(velocity,nominal_velocity)]
    shift=[candidate['points'][0][i]-config['ball_xy'][i] for i in range(2)]
    dead=lambda x:0. if abs(x)<=1e-4 else x
    normalized=[dead(shift[0])/.05,dead(shift[1])/.10,*[dead(x)/.20 for x in delta]]
    gate=min(1.,math.sqrt(sum(x*x for x in normalized)))
    for row in ev['feedback']:
        actual=row.get('context_gate')
        observed=row.get('measured_initial_velocity_error',[])
        if actual is None or not math.isfinite(actual) or abs(actual-gate)>1e-5:raise ValueError('Gate differs from measured reset state')
        if len(observed)!=3 or any(not math.isfinite(x) or abs(x-y)>1e-6 for x,y in zip(observed,delta)):raise ValueError('Gate velocity error is unmeasured')
        if gate==0. and (any(row['residual_rad']) or row['phase_action']!=0.):raise ValueError('Nominal protection violated')
