"""Synthetic contract tests, not physical calibration or acceptance evidence."""
import copy
import hashlib
import importlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE/'tools'), str(HERE), '/home/you/projects/kinetickick/tools']
try:
    api = importlib.import_module('runup_targeting')
except ModuleNotFoundError:
    api = None
from targeted_shooting import POLICY_SHA256, ASSET_SHA256

SOURCES = ['tools/eval_free_shooting.py', 'tools/runup_sequence.py', 'tools/runup_metrics.py',
           'tools/runup_targeting.py', 'tools/targeted_shooting.py', 'tools/free_shooting_metrics.py',
           'tools/shooting_control.py', 'tools/cartesian_strike.py', 'tools/soccer_onnx_policy.py',
           'tools/strike_velocity.py', 'tools/running_metrics.py', 'tools/show_targeted_runup.py']


def config_fixture():
    return dict(schema_version=1, version='0.1.8', trial_steps=200, policy_dt_s=.02, physics_dt_s=.005,
        goal=[0., -3.7], goal_forward=[0., -1.], goal_width_m=1.6, goal_height_m=1.1,
        target_x_m=[-.3, 0., .3], acceptance_seeds=[1811, 1812], interpolation=False,
        policy_sha256=POLICY_SHA256, asset_sha256=ASSET_SHA256,
        source_sha256={name: 'a'*64 for name in SOURCES}, upstream_sources_sha256='b'*64,
        modes={'hip': {'inside': dict(base_trial=dict(mode='hip', style='inside', ball=[0., -.93],
            motion_file='motions/010.npz', strike={}), motion_sha256='c'*64,
            speeds_mps={'low': 1.6, 'medium': 2., 'high': 2.4}, nodes=[
                dict(power=p, target_x_m=x, control=dict(policy_goal=[x, -3.7]))
                for p in ['low', 'medium', 'high'] for x in [-.3, 0., .3]])}})


def trace_fixture(target_x=0., speed=2.):
    """Analytic forward kick at t=1, with 0.8m approach and stable final 3s."""
    vx = speed*target_x/math.hypot(target_x, 2.77)
    vy = -speed*2.77/math.hypot(target_x, 2.77)
    rows=[]
    for step in range(1, 801):
        t=step*.005; contact=step==200; moving=t<1.; root_y=-min(t*.8, .8)
        point=[.075,.036,-.025]; normal=[0.,1.,0.]
        event=dict(time_s=t, foot='right', force_N=10., point_foot=point, normal_foot=normal)
        rows.append(dict(time_s=t, physics_dt_s=.005,
            ball_xyz=[max(0.,t-1)*vx, -.93+max(0.,t-1)*vy, .11],
            ball_velocity=[vx,vy,0.] if t>=1 else [0.,0.,0.],
            root_xyz=[0.,root_y,.8], root_speed_xy=.8 if moving else .05,root_tilt_deg=0.,
            feet_support_N=300.,nonfoot_contact_N=0., support_foot_force_N={'left':150.,'right':150.},
            whole_body_com_xyz=[0.,root_y,.7],whole_body_com_velocity=[0.,-.8 if moving else 0.,0.],
            joint_pos=[0.,.5,0.],joint_vel=[1.,2.,0.],
            foot_velocity_contributions={'hip':[0.,-2.,0.],'knee':[0.,-1.,0.], 'ankle':[0.,0.,0.],'base':[0.,0.,0.]},
            contact_force_N=10. if contact else 0.,contact_point_foot=point if contact else None,
            contact_normal_foot=normal if contact else [0.,0.,0.],foot='right',
            contacts=[event] if contact else [],robot_ball_contact_force_N={'right_ankle_roll_link':10.} if contact else {},
            terminated=False))
    return rows


def runs_fixture(config):
    runs=[]
    for seed in config['acceptance_seeds']:
        trials=[api.resolve_runup_targeted_trial(config,'hip','inside',x,power=p)
                for p in ['low','medium','high'] for x in config['target_x_m']]
        runs.append(dict(seed=seed,trials=trials,traces=[trace_fixture(t['target']['target_x'],t['target']['speed_mps']) for t in trials],
            points=[t['ball'] for t in trials],goals=[t['goal'] for t in trials],policy_goals=[t['policy_goal'] for t in trials],
            records=[{'success':False,'contact_time_s':None} for _ in trials],
            initial_robot_state={'root_position':[[0.,0.,.8] for _ in trials]},
            joint_names=['right_hip_pitch_joint','right_knee_joint','right_ankle_pitch_joint'],
            root_fixed=False,post_initialization_state_writes=False,contribution_foot='right',
            perturb_reset=True,contact_evidence_version=2,sequence_evidence_version=1,
            goal_width_m=1.6,goal_height_m=1.1,policy_sha256=POLICY_SHA256,asset_sha256=ASSET_SHA256,
            runup_targeting_config_sha256='f'*64,runup_targeting_sources_sha256=config['source_sha256'],
            upstream_sources_sha256=config['upstream_sources_sha256'],
            motion_files=['/repo/motions/010.npz']*len(trials),motion_sources_sha256={'/repo/motions/010.npz':'c'*64}))
    return runs


class RunupTargetingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api, 'Discrete runup targeting module is not implemented yet')

    def test_resolver_preserves_fixed_geometry_and_returns_independent_commands(self):
        c=config_fixture();a=api.resolve_runup_targeted_trial(c,'hip','inside',-.3,power='high')
        b=api.resolve_runup_targeted_trial(c,'hip','inside',.3,speed_mps=1.6)
        self.assertEqual(a['goal'],[0.,-3.7]);self.assertEqual(a['ball'],[0.,-.93])
        self.assertEqual(a['target'],dict(target_x=-.3,speed_mps=2.4))
        self.assertEqual(b['target'],dict(target_x=.3,speed_mps=1.6))
        a['strike']['bad']=3
        self.assertEqual(c['modes']['hip']['inside']['base_trial']['strike'],{})
        self.assertEqual(b['strike'],{})

    def test_unfrozen_and_ambiguous_commands_are_rejected(self):
        c=config_fixture()
        for args in [dict(power='low',speed_mps=1.6),{},dict(power='huge'),dict(speed_mps=1.7)]:
            with self.assertRaises(ValueError):api.resolve_runup_targeted_trial(c,'hip','inside',0.,**args)
        for mode,style,x in [('knee','inside',0.),('hip','outside',0.),('hip','inside',.1),('hip','inside',float('nan'))]:
            with self.assertRaises(ValueError):api.resolve_runup_targeted_trial(c,mode,style,x,power='low')

    def test_config_rejects_geometry_overrides_unbounded_controls_and_missing_provenance(self):
        mutations=[lambda c:c['modes']['hip']['inside']['nodes'][0]['control'].update(ball=[.1,-.93]),
            lambda c:c['modes']['hip']['inside']['nodes'][0]['control'].update(goal=[.2,-3.7]),
            lambda c:c['modes']['hip']['inside']['nodes'][0]['control'].update(coordination={'times_s':[.1,.1,.7]}),
            lambda c:c['modes']['hip']['inside']['nodes'][0]['control'].update(strike={'offsets':{'right_hip_yaw_joint':100.}}),
            lambda c:c.update(goal_forward=[.1,-1.]),lambda c:c.update(target_x_m=[-.3,0.,.7]),
            lambda c:c.update(target_x_m=[-.1,0.,.1]),lambda c:c.update(acceptance_seeds=[1811,1811]),
            lambda c:c['modes']['hip']['inside']['nodes'].pop(),lambda c:c['source_sha256'].pop('tools/runup_metrics.py'),
            lambda c:c['modes']['hip']['inside']['speeds_mps'].update(high=2.1)]
        for mutate in mutations:
            c=config_fixture();mutate(c)
            with self.assertRaises(ValueError):api.validate_runup_target_config(c)
        c=config_fixture();c['target_x_m']=[-.4,0.,.4]
        for n in c['modes']['hip']['inside']['nodes']:n['target_x_m']={-.3:-.4,0.:0.,.3:.4}[n['target_x_m']]
        api.validate_runup_target_config(c)

    def test_manifest_requires_complete_unique_matrix_exact_controls_and_fingerprints(self):
        c=config_fixture();runs=runs_fixture(c)
        self.assertEqual(len(api.validate_manifest(c,runs,'f'*64)),18)
        changes=['missing','duplicate','control','goal','hash','source','reference','contact','sequence','reset']
        for change in changes:
            bad=copy.deepcopy(runs)
            if change=='missing':bad.pop()
            if change=='duplicate':bad[-1]=bad[0]
            if change=='control':bad[0]['trials'][0]['policy_goal'][0]+=.1
            if change=='goal':bad[0]['goals'][0][0]=.3
            if change=='hash':bad[0]['runup_targeting_config_sha256']='0'*64
            if change=='source':bad[0]['runup_targeting_sources_sha256']['tools/runup_metrics.py']='0'*64
            if change=='reference':bad[0]['motion_sources_sha256']['/repo/motions/010.npz']='0'*64
            if change=='contact':bad[0]['contact_evidence_version']=1
            if change=='sequence':bad[0]['sequence_evidence_version']=0
            if change=='reset':bad[0]['perturb_reset']=False
            with self.assertRaises(ValueError,msg=change):api.validate_manifest(c,bad,'f'*64)

    def test_replay_recomputes_contact_goal_mode_and_recovery_instead_of_trusting_records(self):
        c=config_fixture();run=runs_fixture(c)[0];row=api.replay_trial(run,4,trial_steps=200)
        self.assertTrue(row['joint_success']);self.assertAlmostEqual(row['outgoing_speed_mps'],2.)
        self.assertAlmostEqual(row['crossing']['position'][0],0.)
        self.assertFalse(run['records'][4]['success'])
        for change in ['body_ball','nonfoot','support','mode','aim','missing_target']:
            bad=copy.deepcopy(run);bad['records'][4]={'success':True,'contact_time_s':1.}
            if change=='body_ball':bad['traces'][4][300]['robot_ball_contact_force_N']={'right_knee_link':3.}
            if change=='nonfoot':bad['traces'][4][300]['nonfoot_contact_N']=25.
            if change=='support':bad['traces'][4][-1]['support_foot_force_N']={'left':0.,'right':0.}
            if change=='mode':bad['trials'][4]['mode']='knee'
            if change=='aim':bad['trials'][4]['target']['target_x']=.3
            if change=='missing_target':bad['trials'][4].pop('target')
            if change=='missing_target':
                with self.assertRaises(ValueError):api.replay_trial(bad,4,trial_steps=200)
            else:self.assertFalse(api.replay_trial(bad,4,trial_steps=200)['joint_success'],change)

    def test_raw_replay_does_not_require_stored_records_and_modes_are_not_pooled(self):
        c=config_fixture();run=runs_fixture(c)[0];run['records']=[]
        self.assertTrue(api.replay_trial(run,4,trial_steps=200)['joint_success'])
        c['modes']['knee']={'inside':copy.deepcopy(c['modes']['hip']['inside'])}
        c['modes']['knee']['inside']['base_trial']['mode']='knee'
        trials=[api.resolve_runup_targeted_trial(c,m,'inside',x,power=p)
                for m in ['hip','knee'] for p in ['low','medium','high'] for x in [-.3,0.,.3]]
        rows=[]
        for seed in [1811,1812]:
            for t in trials:
                x=t['target']['target_x'];speed=t['target']['speed_mps']
                rows.append(dict(seed=seed,mode=t['mode'],style='inside',power=t['power'],target_x_m=x,
                    outgoing_speed_mps=speed,crossing={'position':[x,-3.7,.11]},joint_success=t['mode']=='hip'))
        report=api.aggregate_results(rows,c)
        self.assertFalse(report['passed']);self.assertTrue(report['combinations']['hip/inside']['accepted'])
        self.assertFalse(report['combinations']['knee/inside']['accepted'])

    def test_short_terminal_is_failure_but_missing_sampling_and_truncated_success_are_rejected(self):
        run=runs_fixture(config_fixture())[0]
        for change in ['truncated','gap','force_sum']:
            bad=copy.deepcopy(run)
            if change=='truncated':bad['traces'][0].pop()
            if change=='gap':bad['traces'][0].pop(100)
            if change=='force_sum':bad['traces'][0][199]['contacts'][0]['force_N']=9.
            with self.assertRaises(ValueError,msg=change):api.replay_trial(bad,0,trial_steps=200)
        run['traces'][0]=run['traces'][0][:400];run['traces'][0][-1]['terminated']=True
        self.assertFalse(api.replay_trial(run,0,trial_steps=200)['joint_success'])

    def test_each_node_and_paired_speed_and_direction_must_pass_not_aggregate_means(self):
        c=config_fixture();result=api.evaluate(c,runs_fixture(c),'f'*64)
        self.assertTrue(result['passed']);self.assertEqual(result['total'],18)
        rows=copy.deepcopy(result['records']);rows[0]['joint_success']=False
        self.assertFalse(api.aggregate_results(rows,c)['passed'])
        rows=copy.deepcopy(result['records'])
        # One seed has only 0.2 separation; opposite seed compensates in the mean.
        for row in rows:
            if row['power']=='medium':row['outgoing_speed_mps']=1.8 if row['seed']==1811 else 2.2
        self.assertFalse(api.aggregate_results(rows,c)['passed'])
        rows=copy.deepcopy(result['records']);rows[0]['crossing']['position'][0]=.05
        self.assertFalse(api.aggregate_results(rows,c)['passed'])
        with self.assertRaises(ValueError):api.aggregate_results(result['records'][:-1],c)
        with self.assertRaises(ValueError):api.aggregate_results(result['records']+[result['records'][0]],c)

    def test_freezes_bounded_velocity_controls_and_rejects_diagnostic_runs(self):
        c=config_fixture();node=c['modes']['hip']['inside']['nodes'][0]
        node['control']['strike_velocity']={'window':[110,120,126,140],
            'velocities':{'right_hip_pitch_joint':-3.}, 'minimum_support_N':80.}
        c['modes']['hip']['inside']['nodes'][1]['control']['coordination']={'start_frame':100, 'times_s':[.12,.27,.7],
            'prepare':{'right_knee_joint':1.1},'strike':{'right_knee_joint':.2},
            'velocity_feedforward':True,'velocity_feedforward_gain':1.2}
        api.validate_runup_target_config(c)
        trial=api.resolve_runup_targeted_trial(c,'hip','inside',-.3,power='low')
        self.assertEqual(trial['strike_velocity']['velocities'],{'right_hip_pitch_joint':-3.})
        for invalid in [-.1,2.1,float('nan')]:
            bad=copy.deepcopy(c);bad['modes']['hip']['inside']['nodes'][1]['control']['coordination']['velocity_feedforward_gain']=invalid
            with self.assertRaises(ValueError):api.validate_runup_target_config(bad)
        bad=copy.deepcopy(c);bad['modes']['hip']['inside']['nodes'][0]['control']['strike_velocity']['velocities']['right_hip_pitch_joint']=13.
        with self.assertRaises(ValueError):api.validate_runup_target_config(bad)
        for window in [[110,120,120,140],[110,120,140]]:
            bad=copy.deepcopy(c);bad['modes']['hip']['inside']['nodes'][0]['control']['strike_velocity']['window']=window
            with self.assertRaises(ValueError):api.validate_runup_target_config(bad)
        bad=copy.deepcopy(c)
        bad['modes']['hip']['inside']['base_trial']['coordination']=copy.deepcopy(c['modes']['hip']['inside']['nodes'][1]['control']['coordination'])
        with self.assertRaises(ValueError):api.validate_runup_target_config(bad)
        runs=runs_fixture(config_fixture());runs[0]['running_diagnostic']=True
        with self.assertRaises(ValueError):api.validate_manifest(config_fixture(),runs,'f'*64)
        self.assertFalse(api.replay_trial(runs[0],4,trial_steps=200)['joint_success'])

    def test_recovery_continuation_allows_only_small_shoulder_roll_offsets(self):
        c=config_fixture();base=c['modes']['hip']['inside']['base_trial']
        base['strike']={'continuation':{'window':[130,140,155,170],
            'offsets':{'left_shoulder_roll_joint':.15}}}
        api.validate_runup_target_config(c)
        trial=api.resolve_runup_targeted_trial(c,'hip','inside',0.,power='medium')
        self.assertEqual(trial['strike']['continuation']['offsets'],{'left_shoulder_roll_joint':.15})
        for change in ['large','absolute','wrong_joint','main_strike']:
            bad=copy.deepcopy(c);strike=bad['modes']['hip']['inside']['base_trial']['strike']
            if change=='large':strike['continuation']['offsets']['left_shoulder_roll_joint']=.251
            if change=='absolute':strike['continuation']['targets']={'left_shoulder_roll_joint':.15}
            if change=='wrong_joint':strike['continuation']['offsets']={'left_shoulder_pitch_joint':.15}
            if change=='main_strike':strike['offsets']={'left_shoulder_roll_joint':.15}
            with self.assertRaises(ValueError,msg=change):api.validate_runup_target_config(bad)

    def test_actual_source_fingerprint_check_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'controller.py').write_text('old')
            config={'source_sha256':{'controller.py':hashlib.sha256(b'old').hexdigest()}}
            api.verify_source_files(config,root)
            (root/'controller.py').write_text('changed')
            with self.assertRaises(ValueError):api.verify_source_files(config,root)

if __name__=='__main__':unittest.main()
