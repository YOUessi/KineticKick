import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from running_metrics import summarize_running


def trace():
    rows=[]
    for i in range(600):
        t=(i+1)*.005
        phase=int((t-.5+1e-8)/.25) if t>=.5 else -1
        local=(t-.5)% .25
        side='left' if phase%2==0 else 'right'
        forces={'left':150.,'right':150.}
        if 0<=phase<6:
            forces={side:300.,'right' if side=='left' else 'left':0.}
            if .15<local<.25: forces={'left':0.,'right':0.}
        rows.append(dict(time_s=t,physics_dt_s=.005,root_xyz=[t,0.,.8],root_tilt_deg=5.,
            nonfoot_contact_N=0.,support_foot_force_N=forces,terminated=False))
    return rows


def measure(rows):
    return summarize_running(rows,initial_root=[0,0,.8],forward=[1,0])


class RunningMetricsTests(unittest.TestCase):
    def test_alternating_repeated_flights(self):
        result=measure(trace())
        self.assertTrue(result['running_success'])
        self.assertGreaterEqual(result['alternating_flight_count'],3)
    def test_initial_drop_is_not_running(self):
        rows=trace()
        for row in rows:
            row['support_foot_force_N']={'left':0.,'right':0.} if row['time_s']<.25 else {'left':150.,'right':150.}
        self.assertFalse(measure(rows)['running_success'])
    def test_same_leg_hops_not_alternating_running(self):
        rows=trace()
        for row in rows:
            f=row['support_foot_force_N']
            if f['left']+f['right']>0: row['support_foot_force_N']={'left':300.,'right':0.}
        self.assertFalse(measure(rows)['running_success'])
    def test_gaps_and_missing_ground_evidence_rejected(self):
        rows=trace();del rows[150:155]
        self.assertFalse(measure(rows)['running_success'])
        rows=trace();rows[150].pop('support_foot_force_N')
        self.assertFalse(measure(rows)['running_success'])
    def test_no_flight_or_fall_rejected(self):
        rows=trace()
        for row in rows: row['support_foot_force_N']={'left':150.,'right':150.}
        self.assertFalse(measure(rows)['running_success'])
        for key,value in [('terminated',True),('root_xyz',[2.,0.,.3]),('nonfoot_contact_N',100.)]:
            rows=trace();rows[-1][key]=value
            self.assertFalse(measure(rows)['running_success'])
    def test_progress_must_occur_during_flight_sequence(self):
        rows=trace()
        for row in rows: row['root_xyz'][0]=0. if row['time_s']<2. else 3.
        self.assertFalse(measure(rows)['running_success'])
    def test_landing_force_ramp_keeps_real_flight_duration(self):
        rows=trace()
        for i in range(1,len(rows)):
            old=rows[i-1]['support_foot_force_N'];cur=rows[i]['support_foot_force_N']
            if max(old.values())<5 and max(cur.values())>=80:
                rows[i]['support_foot_force_N']={k:v/10 for k,v in cur.items()}
        result=measure(rows)
        self.assertTrue(result['running_success'])
        self.assertTrue(all(f['duration_s']<.11 for f in result['flight_intervals']))
    def test_negative_or_nan_force_rejected(self):
        for value in [-1.,float('nan')]:
            rows=trace();rows[100]['support_foot_force_N']['left']=value
            with self.assertRaises(ValueError): measure(rows)
