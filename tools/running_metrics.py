"""Conservative repeated-flight evidence from ground-filtered physical telemetry.

A running pass is a locomotion gate, not a complete shooting-sequence pass.
Initial free fall, same-leg hops, missing ground samples and late falls fail.
"""
import math


def summarize_running(trace, *, initial_root, forward):
    direction=list(forward[:2])
    if len(direction)!=2 or not all(math.isfinite(x) for x in direction):
        raise ValueError('Finite horizontal direction required')
    norm=math.hypot(*direction)
    if norm<=1e-12 or len(initial_root)!=3 or not all(math.isfinite(x) for x in initial_root):
        raise ValueError('Nonzero direction and finite initial root required')
    direction=[x/norm for x in direction]
    rows=list(trace)
    coverage=bool(rows)
    failed=False
    previous=0.
    stance=None
    air=None
    landing=None
    flights=[]
    for row in rows:
        t,dt=float(row['time_s']),float(row['physics_dt_s'])
        root=row['root_xyz']
        scalars=[t,dt,*root,row['root_tilt_deg'],row['nonfoot_contact_N']]
        if len(root)!=3 or not all(math.isfinite(x) for x in scalars) or dt<=0 or dt>.0075 or t<=previous:
            raise ValueError('Finite increasing 5 ms telemetry required')
        coverage &= abs(t-dt-previous)<=1e-8
        previous=t
        failed |= bool(row.get('terminated',False) or root[2]<.45 or row['root_tilt_deg']>60
                       or row['nonfoot_contact_N']>=20)
        forces=row.get('support_foot_force_N')
        if not isinstance(forces,dict) or set(forces)!={'left','right'}:
            coverage=False
            stance=air=landing=None
            continue
        if not all(math.isfinite(x) and x>=0 for x in forces.values()):
            raise ValueError('Finite nonnegative ground force required')
        if t<.3:
            continue
        side=next((s for s,o in [('left','right'),('right','left')]
                   if forces[s]>=80 and forces[o]<20),None)
        floating=max(forces.values())<5
        if landing is not None:
            if t-landing['end_s']>.05 or (side is not None and side==landing['launch_side']):
                landing=None
            elif side is not None:
                flights.append(dict(**landing,landing_side=side,end_xyz=list(root)))
                landing=None
        if floating:
            if air is None and landing is None and stance is not None and t-stance['time_s']<=.035:
                air=dict(start_s=t-dt,launch_side=stance['side'],start_xyz=stance['root'])
        else:
            if air is not None:
                duration=t-dt-air['start_s']
                if duration>=.02-1e-9:
                    landing=dict(**air,end_s=t-dt,duration_s=duration)
                    if side is not None:
                        if side!=air['launch_side']:
                            flights.append(dict(**landing,landing_side=side,end_xyz=list(root)))
                        landing=None
                air=None
            if side is not None:
                stance=dict(side=side,time_s=t,root=list(root))
    longest=[]
    chain=[]
    for flight in flights:
        if chain and (flight['launch_side']==chain[-1]['launch_side']
                      or flight['start_s']-chain[-1]['end_s']>.6):
            chain=[]
        chain.append(flight)
        if len(chain)>len(longest): longest=list(chain)
    progress=0.
    if longest:
        progress=sum((longest[-1]['end_xyz'][i]-longest[0]['start_xyz'][i])*direction[i] for i in range(2))
    return dict(coverage_complete=bool(coverage),failed=bool(failed),flight_intervals=flights,
        alternating_flight_count=len(longest),running_progress_m=progress,
        running_success=bool(coverage and not failed and len(longest)>=3 and progress>=1.),
        scope='Locomotion only; repeated alternating flights after 0.30 s, not shot/recovery acceptance')
