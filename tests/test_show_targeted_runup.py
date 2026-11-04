"""CLI/provenance checks only; no simulator process is launched here."""
import contextlib
import copy
import hashlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE/'tools'),str(HERE),str(Path(__file__).resolve().parent),'/home/you/projects/kinetickick/tools']
try:
    wrapper=importlib.import_module('show_targeted_runup')
except ModuleNotFoundError:
    wrapper=None
from test_runup_targeting import config_fixture
from targeted_shooting import CONTROLLER_SOURCES, UPSTREAM_SOURCE


def snapshot_fixture(c):
    return dict(source_sha256=copy.deepcopy(c['source_sha256']),policy_sha256=c['policy_sha256'],
        asset_sha256=c['asset_sha256'],upstream_sources_sha256=c['upstream_sources_sha256'],
        motion_sources_sha256={'/repo/motions/010.npz':'c'*64})


def result_fixture(c,request,snapshot):
    trials=copy.deepcopy(request['trials']);n=len(trials)
    return dict(seed=request['seed'],trials=trials,points=[t['ball'] for t in trials],
        goals=[t['goal'] for t in trials],policy_goals=[t['policy_goal'] for t in trials],
        traces=[[{'time_s':4.,'terminated':False}] for _ in trials],records=[{} for _ in trials],
        sequence_records=[{} for _ in trials],root_fixed=False,post_initialization_state_writes=False,
        running_diagnostic=False,perturb_reset=request['acceptance'],contact_evidence_version=2,
        sequence_evidence_version=1,goal_width_m=1.6,goal_height_m=1.1,
        policy_sha256=snapshot['policy_sha256'],asset_sha256=snapshot['asset_sha256'],
        controller_sources_sha256={**{p:snapshot['source_sha256'][p] for p in CONTROLLER_SOURCES},
                                   UPSTREAM_SOURCE:snapshot['upstream_sources_sha256']},
        sequence_sources_sha256={p:snapshot['source_sha256']['tools/'+p] for p in
            ['runup_sequence.py','runup_metrics.py','cartesian_strike.py','strike_velocity.py']},
        motion_files=['/repo/motions/010.npz']*n,motion_sources_sha256=snapshot['motion_sources_sha256'])


class TargetedRunupWrapperTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(wrapper,'Targeted runup wrapper is not implemented yet')

    def args(self,*flags):
        return wrapper.make_parser().parse_args(['--config','/tmp/frozen.json','--mode','hip','--style','inside',*flags])

    def request(self,args,c=None):
        return wrapper.build_request(args,c or config_fixture(),root=Path('/repo'),
            manifest_path=Path('/tmp/trials.json'),output_path=Path('/tmp/result.json'),python_executable='python-test')

    def test_explicit_config_and_one_preview_command_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):wrapper.make_parser().parse_args([])
        with self.assertRaises(ValueError):self.request(self.args('--headless'))
        args=self.args('--target-x','0.3','--power','high','--headless')
        request=self.request(args)
        self.assertEqual(len(request['trials']),1)
        self.assertEqual(request['trials'][0]['target'],{'target_x':.3,'speed_mps':2.4})
        cmd=request['command'];self.assertIn('--trials',cmd);self.assertIn('--sequence-evidence',cmd)
        self.assertNotIn('--gui',cmd);self.assertNotIn('--target-x',cmd);self.assertNotIn('--power',cmd)
        self.assertEqual(cmd[0],'python-test');self.assertEqual(cmd[cmd.index('--steps')+1],'200')

    def test_acceptance_is_exact_nine_nodes_with_frozen_seed_and_perturbation(self):
        request=self.request(self.args('--acceptance','--seed','1811'))
        self.assertEqual(len(request['trials']),9);self.assertIn('--perturb-reset',request['command'])
        self.assertNotIn('--gui',request['command'])
        self.assertEqual({(t['power'],t['target']['target_x']) for t in request['trials']},
            {(p,x) for p in ['low','medium','high'] for x in [-.3,0.,.3]})
        for flags in [('--acceptance','--seed','1'),('--acceptance','--seed','1811','--target-x','0'),
                      ('--acceptance','--seed','1811','--video-dir','/tmp/video')]:
            with self.assertRaises(ValueError):self.request(self.args(*flags))

    def test_video_and_gui_commands_do_not_change_frozen_trial(self):
        video=self.request(self.args('--target-x','0','--speed-mps','2','--video-dir','/tmp/frames'))
        gui=self.request(self.args('--target-x','0','--power','medium'))
        self.assertEqual(video['trials'],gui['trials']);self.assertIn('--video-dir',video['command'])
        self.assertNotIn('--gui',video['command']);self.assertIn('--gui',gui['command']);self.assertIn('--hold',gui['command'])

    def test_actual_sources_policy_asset_reference_and_upstream_are_all_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=config_fixture()
            files={name:name.encode() for name in c['source_sha256']}
            files.update({wrapper.POLICY_FILE:b'policy',wrapper.ASSET_FILE:b'asset','motions/010.npz':b'motion',UPSTREAM_SOURCE+'/robot.py':b'upstream'})
            for name,data in files.items():
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            c['source_sha256']={p:hashlib.sha256(files[p]).hexdigest() for p in c['source_sha256']}
            c['policy_sha256']=hashlib.sha256(b'policy').hexdigest();c['asset_sha256']=hashlib.sha256(b'asset').hexdigest()
            c['modes']['hip']['inside']['motion_sha256']=hashlib.sha256(b'motion').hexdigest()
            c['upstream_sources_sha256']=hashlib.sha256(b'robot.py\0upstream\0').hexdigest()
            measured=wrapper.measure_provenance(c,root)
            self.assertEqual(measured['motion_sources_sha256'][str(root/'motions/010.npz')],hashlib.sha256(b'motion').hexdigest())
            for name in ['tools/runup_targeting.py',wrapper.POLICY_FILE,wrapper.ASSET_FILE,'motions/010.npz',UPSTREAM_SOURCE+'/robot.py']:
                (root/name).write_bytes(b'changed')
                with self.assertRaises(ValueError,msg=name):wrapper.measure_provenance(c,root)
                (root/name).write_bytes(files[name])

    def test_child_output_cannot_be_attested_with_different_controls_or_content_hashes(self):
        c=config_fixture();request=self.request(self.args('--target-x','0','--power','medium','--headless'))
        snapshot=snapshot_fixture(c);result=result_fixture(c,request,snapshot)
        wrapper.validate_child_result(result,c,request,snapshot)
        for kind in ['policy','asset','source','upstream','reference','trial','points','seed','diagnostic','short']:
            bad=copy.deepcopy(result)
            if kind in ['policy','asset']:bad[kind+'_sha256']='0'*64
            if kind=='source':bad['sequence_sources_sha256']['runup_metrics.py']='0'*64
            if kind=='upstream':bad['controller_sources_sha256'][UPSTREAM_SOURCE]='0'*64
            if kind=='reference':bad['motion_sources_sha256']['/repo/motions/010.npz']='0'*64
            if kind=='trial':bad['trials'][0]['policy_goal'][0]=.8
            if kind=='points':bad['points'][0][0]=.2
            if kind=='seed':bad['seed']+=1
            if kind=='diagnostic':bad['running_diagnostic']=True
            if kind=='short':bad['traces'][0][-1]['time_s']=3.
            with self.assertRaises(ValueError,msg=kind):wrapper.validate_child_result(bad,c,request,snapshot)

if __name__=='__main__':unittest.main()
