"""Physical recovery gate after a finite running command; no shooting claim."""
import math


def summarize_recovery(rows, *, expected_duration_s, window_s=.8):
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('Positive finite recovery window required')
    coverage=bool(rows);previous=0.
    for r in rows:
        t=r['time_s'];dt=r.get('physics_dt_s',0.)
        coverage &= bool(math.isfinite(t) and 0<dt<=.0075 and abs(t-previous-dt)<1e-8)
        previous=t
    tail=[r for r in rows if r['time_s'] > expected_duration_s-window_s-1e-8]
    complete=bool(coverage and tail and abs(rows[-1]['time_s']-expected_duration_s)<.0075
        and tail[0]['time_s'] <= expected_duration_s-window_s+.0075)
    clean=bool(rows) and all(not r.get('terminated',False) and r['nonfoot_contact_N']<20
        and r['root_xyz'][2]>=.45 and r['root_tilt_deg']<=60 for r in rows)
    speeds=[math.hypot(*r['root_linear_velocity'][:2]) for r in tail]
    spins=[math.sqrt(sum(x*x for x in r['root_angular_velocity'])) for r in tail]
    tilts=[r['root_tilt_deg'] for r in tail]
    support=sum(set(r['support_foot_force_N'])=={'left','right'}
        and all(math.isfinite(f) and f>=30 for f in r['support_foot_force_N'].values())
        for r in tail)/max(len(tail),1)
    values=speeds+spins+tilts
    finite=bool(values) and all(math.isfinite(x) for x in values)
    return dict(window_s=window_s,coverage_complete=complete,clean_episode=clean,
        max_planar_speed_m_s=max(speeds,default=None),max_angular_speed_rad_s=max(spins,default=None),
        max_tilt_deg=max(tilts,default=None),double_support_fraction=support,
        standing_success=bool(complete and clean and finite and max(speeds)<=.25
            and max(spins)<=.5 and max(tilts)<=15 and support>=.8),
        thresholds=dict(planar_speed_m_s=.25,angular_speed_rad_s=.5,tilt_deg=15,double_support_fraction=.8,each_foot_force_N=30),
        scope='End-of-locomotion root motion and two-foot support; not post-shot recovery')
