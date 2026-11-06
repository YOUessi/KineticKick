"""Bounded archive selection and honest coordinated-feature ranking."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
SPEC=importlib.util.find_spec('scan_skillcorner_shots')
if SPEC is not None:
    from scan_skillcorner_shots import plan_windows, stream_windows, summarize_coordination, write_bounded


def event(identity='a',frame='401',header='False',player='7'):
    return {'event_id':identity,'event_type':'player_possession','end_type':'shot','match_id':'1925299',
            'player_id':player,'player_name':'Test','period':'1','frame_end':frame,'time_end':'00:40.1',
            'is_header':header,'fully_extrapolated':'False','is_player_possession_end_matched':'True'}


class BoundedShotScan(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC,'SkillCorner shot scanner has not been implemented')

    def test_all_nonheaders_are_planned_and_exclusions_are_explicit(self):
        uncertain=event('c');uncertain['fully_extrapolated']='True';uncertain['is_player_possession_end_matched']='False'
        plans,excluded=plan_windows([event(),event('b',frame='400'),uncertain,event('header',header='True'),event('unknown',header='')])
        self.assertEqual([p['event']['event_id'] for p in plans],['a','b','c'])
        self.assertEqual(plans[0]['expected_frames'],76)
        self.assertEqual(plans[0]['frame_start'],965)
        self.assertEqual(plans[0]['frame_end'],1040)
        self.assertEqual(plans[1]['expected_frames'],75)
        self.assertEqual(len(excluded),2)
        self.assertEqual(len(plans[2]['event_quality_exclusions']),2)

    def test_zip_is_streamed_and_overlapping_shooter_windows_remain_distinct(self):
        plans,_=plan_windows([event('a',frame='400',player='7'),event('b',frame='401',player='8')])
        rows=[{'frame':i,'period':1,'player_data':[{'player_id':7,'joints':None},{'player_id':8,'joints':None}], 'timestamp':None} for i in range(950,1050)]
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'source.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('1925299.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
            windows,stats=stream_windows(archive,plans)
            self.assertEqual(stats['frames_scanned'],100)
            self.assertEqual(len(windows['a']),75)
            self.assertEqual(len(windows['b']),76)
            self.assertTrue(all(len(f['player_data'])==1 and f['player_data'][0]['player_id']==7 for f in windows['a']))
            self.assertTrue(all(len(f['player_data'])==1 and f['player_data'][0]['player_id']==8 for f in windows['b']))
            self.assertFalse((Path(d)/'1925299.jsonl').exists())
            self.assertLessEqual(stats['retained_frame_records'],sum(p['expected_frames'] for p in plans))

    def test_complete_geometry_or_earlier_torso_success_cannot_qualify_anchor(self):
        names=['shoulder_pelvis','trunk_left_thigh','trunk_right_thigh','left_leg_chain','right_leg_chain','left_arm_right_thigh','right_arm_left_thigh','trunk_shoulder_line']
        quality=np.zeros((10,len(names)),dtype=bool)
        quality[:3,:]=True;quality[:,names.index('trunk_shoulder_line')]=True
        arrays={'source_frame_numbers':np.arange(995,1005),'relation_names':np.array(names),
                'relation_valid':np.ones_like(quality),'relation_quality_pass':quality,
                'joint_geometry_valid':np.ones((10,29),bool),'joint_quality_pass':np.ones((10,29),bool),
                'joint_predicted_p90_cm':np.ones((10,29))*10,'player_is_detected':np.ones(10,bool),'ball_is_detected':np.ones(10,bool)}
        summary=summarize_coordination(arrays,1000.)
        self.assertEqual(summary['simultaneous_core_frames'],3)
        self.assertEqual(summary['longest_simultaneous_core_run_frames'],3)
        self.assertFalse(summary['simultaneous_core_at_anchor_bracket'])
        self.assertIn('no simultaneous core support across the event anchor bracket',summary['coordination_screen_exclusions'])
        self.assertEqual(summary['longest_core_run_covering_anchor_frames'],0)
        self.assertEqual(summary['relation_geometry_valid_fraction'],1)
        arrays['relation_quality_pass'][:]=True
        summary=summarize_coordination(arrays,1000.)
        self.assertTrue(summary['simultaneous_core_at_anchor_bracket'])
        self.assertEqual(summary['longest_core_run_covering_anchor_frames'],10)

    def test_total_byte_budget_rejects_before_creating_window(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'existing').write_bytes(b'1234')
            with self.assertRaisesRegex(ValueError,'budget'):
                write_bounded(root/'window.gz',b'abc',root,6)
            self.assertFalse((root/'window.gz').exists())
            write_bounded(root/'window.gz',b'ab',root,6)
            self.assertEqual((root/'window.gz').read_bytes(),b'ab')


if __name__=='__main__': unittest.main()
