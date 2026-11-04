"""Require retained near-rest ball and no new foot strike during standing recovery."""
import math
from inside_receive_metrics import summarize_inside_receive


def summarize_balanced_receive(result,mode):
    value=summarize_inside_receive(result,mode)
    rows=[r for r in result['traces'][0] if not r.get('terminated')]
    events=result['inside_control_events']
    handoffs=[e for e in events if e['kind'] in ('learned_standing_recovery','cartesian_standing_recovery')]
    handoff=handoffs[0] if handoffs else None
    after=[r for r in rows if handoff is not None and r['time_s']>=handoff['time_s']]
    norm=lambda x:math.sqrt(sum(v*v for v in x))
    repeated=any(r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0.)>2. for r in after)
    speed=max((norm(r['ball_velocity']) for r in after),default=None)
    clearance=handoff.get('ball_foot_clearance_m') if handoff else None
    held=0.
    for row,sensor in reversed(list(zip(rows,result['inside_sensor_trace']))):
        if (math.dist(row['ball_xyz'][:2],row['root_xyz'][:2])<.65 and row['ball_xyz'][2]<.145
                and norm(row['ball_velocity'])<.05 and norm(sensor['ball_angular_velocity'])<.5):
            held+=row['physics_dt_s']
        else:break
    safe_release=bool(after and clearance is not None and clearance>=.025 and not repeated and speed<=.1)
    value.update(metrics_version=2,loose_v1_success=value['success'],near_rest_final_seconds=held,
        near_rest_speed_limit_mps=.05,near_rest_spin_limit_rad_s=.5,
        recovery_handoff_time_s=handoff['time_s'] if handoff else None,
        ball_foot_clearance_at_handoff_m=clearance,foot_recontact_after_handoff=repeated,
        maximum_ball_speed_after_handoff_mps=speed,safe_release=safe_release,
        success=bool(value['success'] and held>=.75 and safe_release))
    return value
