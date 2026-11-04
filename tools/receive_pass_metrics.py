"""Separate reception, measured footwork, directional travel and recovery evidence."""
import math
import numpy as np
from free_shooting_metrics import paid_contact_patch
from reactive_receive_metrics import summarize_reactive_receive


def directional_travel(rows,direction,start):
    after=[r for r in rows if r['time_s']>=start]
    d=np.asarray(direction,float)[:2];d/=np.linalg.norm(d)
    value=dict(crossed=False,heading_error_deg=None,crossing_time_s=None)
    if not after:return value
    origin=np.asarray(after[0]['ball_xyz'],float);previous=after[0]
    for row in after[1:]:
        v=np.asarray(row['ball_xyz'],float)-origin;along=float(v[:2]@d)
        if along>=1.-1e-9:
            p=np.asarray(previous['ball_xyz'],float)-origin;old=float(p[:2]@d)
            u=np.clip((1.-old)/max(1e-9,along-old),0.,1.);at=p+u*(v-p)
            across=float(at[0]*d[1]-at[1]*d[0])
            value.update(crossed=True,heading_error_deg=math.degrees(math.atan2(abs(across),1.)),
                crossing_time_s=previous['time_s']+u*(row['time_s']-previous['time_s']),lateral_error_m=abs(across))
            break
        previous=row
    value['peak_ball_height_m']=max(r['ball_xyz'][2] for r in after)
    value['peak_ball_speed_mps']=max(math.dist(r['ball_velocity'],[0,0,0]) for r in after)
    return value


def measured_step(rows,sensors,start,end):
    samples=[(r,s) for r,s in zip(rows,sensors) if start<=r['time_s']<end]
    value=dict(success=False,lifted=False,displacement_m=0.,landing_time_s=None)
    if not samples:return value
    initial=np.array(samples[0][1]['right_foot_position']);lifted=False
    for r,s in samples:
        p=np.array(s['right_foot_position']);force=r['support_foot_force_N']['right']
        lifted |= bool(p[2]>initial[2]+.025 and force<5.)
        if lifted and p[2]<.07 and force>20.:
            distance=float(np.linalg.norm(p[:2]-initial[:2]))
            if distance>=.03:
                value.update(success=True,lifted=True,displacement_m=distance,landing_time_s=r['time_s']);return value
    value['lifted']=lifted
    return value


def summarize_receive_pass(result,mode):
    rows=result['traces'][0];sensors=result['inside_sensor_trace'];events=result['passing_events']
    plan=next((e for e in events if e['kind']=='plan_pass_from_stopped_ball'),None)
    strike=next((e for e in events if e['kind']=='pass_strike'),None)
    cutoff=plan['time_s'] if plan else rows[-1]['time_s']+.001
    n=sum(r['time_s']<cutoff for r in rows);prefix=dict(result)
    prefix['traces']=[rows[:n]];prefix['inside_sensor_trace']=sensors[:n]
    contacts=[c for c in result['contacts'][0] if c['time_s']<cutoff and c['force_N']>0]
    total=sum(c['force_N'] for c in contacts)
    inside=sum(c['force_N'] for c in contacts if paid_contact_patch(c['point_foot'],c['normal_foot'],foot=c['foot'])=='inside')
    prefix['records']=[dict(result['records'][0],requested_surface_impulse_fraction=inside/total if total else 0.)]
    try:received=summarize_reactive_receive(prefix,mode)
    except ValueError as exc:received=dict(success=False,incomplete_evidence=str(exc))
    value=dict(success=False,receive=received,planning_time_s=plan['time_s'] if plan else None,
        requested_direction=plan['direction'] if plan else None,pass_contact_time_s=None)
    if not plan or not strike:return value
    start=strike['time_s'];footwork=measured_step(rows,sensors,plan['time_s'],start)
    early=any(r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0)>2. for r in rows if plan['time_s']<=r['time_s']<start)
    touch=next((r['time_s'] for r,s in zip(rows,sensors) if r['time_s']>=start
        and s['phase']=='pass_strike' and r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0)>2.),None)
    value.update(footwork=footwork,unintended_contact_during_footwork=early,pass_contact_time_s=touch)
    if touch is None:return value
    flight=directional_travel(rows,plan['direction'],touch)
    after=[c for c in result['contacts'][0] if c['time_s']>=start and c['force_N']>0]
    total=sum(c['force_N'] for c in after)
    inside=sum(c['force_N'] for c in after if paid_contact_patch(c['point_foot'],c['normal_foot'],foot=c['foot'])=='inside')
    fraction=inside/total if total else 0.
    illegal=any(any(n!='right_ankle_roll_link' and f>2. for n,f in r['robot_ball_contact_force_N'].items()) for r in rows)
    unsafe=illegal or any(r['terminated'] or r['nonfoot_contact_N']>=20. for r in rows)
    standing=0.
    for r in reversed(rows):
        if (r['root_xyz'][2]>.65 and r['root_tilt_deg']<12. and r['root_speed_xy']<.15
                and min(r['support_foot_force_N'].values())>20.):standing+=r['physics_dt_s']
        else:break
    success=bool(received['success'] and footwork['success'] and not early and not unsafe and fraction>=.7
        and flight['crossed'] and flight['heading_error_deg']<=8. and flight['peak_ball_height_m']<.18
        and flight['peak_ball_speed_mps']>.5 and standing>=.5)
    value.update(success=success,travel=flight,inside_surface_impulse_fraction=fraction,
        unintended_body_ball_contact=illegal,unsafe=unsafe,balanced_final_seconds=standing)
    return value
