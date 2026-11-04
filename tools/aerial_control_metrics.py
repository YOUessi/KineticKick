"""Separate aerial contact, anticipatory cushioning, ball control, and balance.

A collision slowing the ball is not sufficient evidence of active cushioning.
Ground contact must not contribute to the reported foot-only speed reduction.
"""
import math


def summarize_aerial_control(result):
    rows = [r for r in result['traces'][0] if not r.get('terminated')]
    sensors = result['aerial_sensor_trace']
    if not rows or len(rows) != len(sensors):
        raise ValueError('Nonempty, aligned aerial and physics traces required')
    foot = [i for i, r in enumerate(rows)
            if r['robot_ball_contact_force_N'].get('right_ankle_roll_link', 0) > 2]
    ground = [i for i, s in enumerate(sensors) if s['ball_ground_force_N'] > 1]
    first = foot[0] if foot else None
    airborne = (first is not None and (not ground or first < ground[0])
                and rows[first]['ball_xyz'][2] > .16)
    illegal = any(any(n != 'right_ankle_roll_link' and f > 2
                      for n, f in r['robot_ball_contact_force_N'].items()) for r in rows)
    collision = any(r['nonfoot_contact_N'] >= 20 for r in rows)
    terminated = any(r.get('terminated') for r in result['traces'][0])
    norm = lambda x: math.sqrt(sum(v*v for v in x))
    incoming = norm(rows[first-1]['ball_velocity']) if first is not None and first > 0 else None
    end_contact = first
    if first is not None:
        while (end_contact+1 < len(rows) and end_contact-first < 30
               and rows[end_contact+1]['robot_ball_contact_force_N'].get('right_ankle_roll_link', 0) > 2):
            end_contact += 1
    measured_after = min(end_contact+1, len(rows)-1) if first is not None else None
    if measured_after is not None and ground:
        measured_after = min(measured_after, ground[0]-1)
    measured_in_air = measured_after is not None and measured_after > first
    outgoing = norm(rows[measured_after]['ball_velocity']) if measured_in_air else None
    reduction = 1-outgoing/incoming if incoming and outgoing is not None else None
    pre_contact_vz = rows[first-1]['right_foot_point_velocity'][2] if first is not None and first > 0 else None
    contact_vz = min((rows[i]['right_foot_point_velocity'][2]
                      for i in range(first, end_contact+1)), default=0.) if first is not None else None
    # Fixed point [0.06, 0, -0.025] in the foot, already recorded by the physics logger.
    anticipated = pre_contact_vz is not None and pre_contact_vz < -.1
    slowed = bool(airborne and measured_in_air and reduction is not None and reduction > .4)
    cushioned = bool(slowed and anticipated and contact_vz < -.25)
    ball_hold = standing = 0.
    for row in reversed(rows):
        near = math.dist(row['ball_xyz'][:2], row['root_xyz'][:2]) < .65
        if near and row['ball_xyz'][2] < .145 and norm(row['ball_velocity']) < .25:
            ball_hold += row['physics_dt_s']
        else:
            break
    for row in reversed(rows):
        if (row['root_xyz'][2] > .65 and row['root_tilt_deg'] < 12
                and row['root_speed_xy'] < .15 and min(row['support_foot_force_N'].values()) > 20):
            standing += row['physics_dt_s']
        else:
            break
    safe = not (illegal or collision or terminated)
    surface_fraction = result['records'][0]['requested_surface_impulse_fraction']
    success = bool(cushioned and surface_fraction >= .7 and ball_hold >= .75 and standing >= .5 and safe)
    return dict(
        metrics_version=2, task='single_aerial_foot_trap', success=success,
        air_contact_and_recovery=bool(airborne and standing >= .5 and safe),
        airborne_contact=airborne,
        contact_time_s=rows[first]['time_s'] if first is not None else None,
        first_ground_time_s=rows[ground[0]]['time_s'] if ground else None,
        incoming_speed_mps=incoming, outgoing_speed_mps=outgoing,
        outgoing_measurement_time_s=rows[measured_after]['time_s'] if measured_in_air else None,
        outgoing_measured_before_ground_contact=measured_in_air,
        speed_reduction_fraction=reduction, speed_reduced_in_air=slowed,
        pre_contact_foot_point_vertical_speed_mps=pre_contact_vz,
        min_contact_foot_point_vertical_speed_mps=contact_vz,
        anticipatory_foot_lowering=anticipated, cushioned=cushioned,
        instep_surface_impulse_fraction=surface_fraction,
        ball_controlled_final_seconds=ball_hold, balanced_final_seconds=standing,
        final_ball_speed_mps=norm(rows[-1]['ball_velocity']),
        final_ball_distance_from_root_m=math.dist(rows[-1]['ball_xyz'][:2], rows[-1]['root_xyz'][:2]),
        unintended_body_ball_contact=illegal, nonfoot_collision=collision, terminated=terminated,
        continuous_juggling_demonstrated=False)
