"""Measure direct running contact, rejecting run-stop-kick episodes."""
import math


def summarize_stride_shot(trace, *, contact_time, running):
    limits = dict(minimum_forward_contact_speed_m_s=1., maximum_flight_to_contact_s=.45,
                  stop_speed_m_s=.4, maximum_stop_duration_s=.15)
    if contact_time is None:
        return dict(continuous_running_contact=False, reason='no actual foot contact', limits=limits)
    flights = [f for f in running['flight_intervals'] if f['end_s'] <= contact_time]
    before = [r for r in trace if r['time_s'] <= contact_time + 1e-8]
    if len(before) < 2 or len(flights) < 3:
        return dict(continuous_running_contact=False, reason='fewer than three pre-contact alternating flights', limits=limits)
    start = flights[-3]['start_s']
    stop = longest_stop = 0.
    previous = None
    contact_velocity = None
    for row in before:
        if previous is not None:
            dt = row['time_s'] - previous['time_s']
            forward = -(row['root_xyz'][1] - previous['root_xyz'][1]) / dt
            contact_velocity = forward
            if row['time_s'] >= start:
                stop = stop + dt if forward < limits['stop_speed_m_s'] else 0.
                longest_stop = max(stop, longest_stop)
        previous = row
    gap = contact_time - flights[-1]['end_s']
    alternating = all(a['launch_side'] != b['launch_side'] and b['start_s']-a['end_s'] <= .6
                      for a,b in zip(flights[-3:],flights[-2:]))
    passed = bool(alternating and math.isfinite(contact_velocity)
                  and contact_velocity >= limits['minimum_forward_contact_speed_m_s']
                  and gap <= limits['maximum_flight_to_contact_s']
                  and longest_stop < limits['maximum_stop_duration_s'])
    return dict(continuous_running_contact=passed, pre_contact_alternating_flights=len(flights),
                forward_contact_speed_m_s=contact_velocity, last_flight_to_contact_s=gap,
                longest_pre_contact_stop_s=longest_stop, limits=limits)
