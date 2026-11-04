"""Inside-foot trap evidence; ground stopping and aerial interception are separate.

Success requires active withdrawal, actual medial contact, retained ball, balanced
recovery, and no other body contacting the ball. No shooting score is reused.
"""
import math


def summarize_inside_receive(result, mode):
    if mode not in ('ground', 'aerial'):
        raise ValueError(mode)
    rows = [r for r in result['traces'][0] if not r.get('terminated')]
    sensors = result['inside_sensor_trace']
    if not rows or len(rows) != len(sensors):
        raise ValueError('Nonempty aligned physics and inside-foot sensor traces required')
    norm = lambda v: math.sqrt(sum(x*x for x in v))
    foot = [i for i,r in enumerate(rows) if r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0.) > 2.]
    ground = [i for i,s in enumerate(sensors) if s['ball_ground_force_N'] > 1.]
    first = foot[0] if foot else None
    end = first
    if first is not None:
        while end+1 < len(rows) and end-first < 30 and rows[end+1]['robot_ball_contact_force_N'].get('right_ankle_roll_link',0.) > 2.:
            end += 1
    after = min(end+1,len(rows)-1) if end is not None else None
    if mode == 'aerial' and after is not None and ground:
        after = min(after, ground[0]-1)
    valid_after = after is not None and after > first
    incoming = norm(rows[first-1]['ball_velocity']) if first is not None and first > 0 else None
    outgoing = norm(rows[after]['ball_velocity']) if valid_after else None
    reduction = 1-outgoing/incoming if incoming and outgoing is not None else None
    withdrawal = None
    relative = None
    if incoming:
        v = rows[first-1]['ball_velocity']
        fv = sensors[first-1]['inside_point_velocity']
        withdrawal = sum(x*y for x,y in zip(v,fv))/incoming
        relative = norm([x-y for x,y in zip(v,fv)])
    anticipated = bool(withdrawal is not None and withdrawal > .05 and relative < incoming)
    airborne = bool(first is not None and (not ground or first < ground[0]) and rows[first]['ball_xyz'][2] > .16)
    correct_arrival = airborne if mode == 'aerial' else bool(first is not None and ground and ground[0] < first)
    illegal = any(any(n != 'right_ankle_roll_link' and f > 2. for n,f in r['robot_ball_contact_force_N'].items()) for r in rows)
    collision = any(r['nonfoot_contact_N'] >= 20. for r in rows)
    terminated = any(r.get('terminated') for r in result['traces'][0])
    held = balanced = 0.
    for r,s in reversed(list(zip(rows,sensors))):
        if (math.dist(r['ball_xyz'][:2], r['root_xyz'][:2]) < .65 and r['ball_xyz'][2] < .145
                and norm(r['ball_velocity']) < .25 and norm(s['ball_angular_velocity']) < 3.):
            held += r['physics_dt_s']
        else:
            break
    for r in reversed(rows):
        if (r['root_xyz'][2] > .65 and r['root_tilt_deg'] < 12 and r['root_speed_xy'] < .15
                and min(r['support_foot_force_N'].values()) > 20):
            balanced += r['physics_dt_s']
        else:
            break
    fraction = result['records'][0]['requested_surface_impulse_fraction']
    safe = not (illegal or collision or terminated)
    cushioned = bool(correct_arrival and anticipated and reduction is not None and reduction > .4)
    return dict(metrics_version=1, task='inside_foot_'+mode+'_trap', success=bool(cushioned and fraction>=.7 and held>=.75 and balanced>=.5 and safe),
        receive_mode=mode, contact_time_s=rows[first]['time_s'] if first is not None else None,
        first_ground_time_s=rows[ground[0]]['time_s'] if ground else None, airborne_contact=airborne,
        correct_arrival=correct_arrival, inside_surface_impulse_fraction=fraction, anticipatory_withdrawal=anticipated,
        pre_contact_inside_point_speed_along_ball_mps=withdrawal, pre_contact_relative_speed_mps=relative,
        incoming_speed_mps=incoming, outgoing_speed_mps=outgoing, speed_reduction_fraction=reduction,
        outgoing_measurement_time_s=rows[after]['time_s'] if valid_after else None,
        outgoing_measured_before_ground_contact=bool(mode=='aerial' and valid_after),
        speed_reduction_includes_ground=mode=='ground', cushioned=cushioned,
        ball_controlled_final_seconds=held, balanced_final_seconds=balanced,
        final_ball_speed_mps=norm(rows[-1]['ball_velocity']),final_ball_spin_rad_s=norm(sensors[-1]['ball_angular_velocity']),
        final_ball_distance_from_root_m=math.dist(rows[-1]['ball_xyz'][:2],rows[-1]['root_xyz'][:2]),
        unintended_body_ball_contact=illegal,nonfoot_collision=collision,terminated=terminated)
