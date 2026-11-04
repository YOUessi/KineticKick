"""Require observed-ball reaction from grounded feet and lower residual motion."""
import math
from balanced_receive_metrics import summarize_balanced_receive


def summarize_reactive_receive(result,mode):
    value=summarize_balanced_receive(result,mode)
    rows=[r for r in result['traces'][0] if not r.get('terminated')]
    sensors=result['inside_sensor_trace']
    events=result.get('reactive_events',[])
    reaction=next((e for e in events if e['kind']=='react_to_observed_ball'),None)
    start=reaction['time_s'] if reaction else None
    pre=[(r,s) for r,s in zip(rows,sensors) if start is not None and r['time_s']<=start]
    grounded=bool(pre and all(s['right_foot_position'][2]<.065 for _,s in pre)
        and any(min(r['support_foot_force_N'].values())>2. for r,_ in pre))
    observed_reaction=bool(reaction and grounded and value['contact_time_s'] is not None
        and 0.<value['contact_time_s']-start<=.75)
    held=0.
    for r,s in reversed(list(zip(rows,sensors))):
        if (math.dist(r['ball_xyz'][:2],r['root_xyz'][:2])<.65 and r['ball_xyz'][2]<.145
                and math.dist(r['ball_velocity'],[0,0,0])<.02
                and math.dist(s['ball_angular_velocity'],[0,0,0])<.3):held+=r['physics_dt_s']
        else:break
    release=next((e['time_s'] for e in result['inside_control_events'] if e['kind']=='separate_foot_before_recovery'),None)
    recontact=any(r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0)>2.
        for r in rows if release is not None and r['time_s']>=release)
    value.update(metrics_version=3,balanced_v2_success=value['success'],reactive_from_grounded_feet=observed_reaction,
        reaction_time_s=start,low_residual_final_seconds=held,low_residual_speed_limit_mps=.02,
        low_residual_spin_limit_rad_s=.3,foot_recontact_after_separation=recontact,
        success=bool(value['success'] and observed_reaction and held>=.75 and not recontact))
    return value
