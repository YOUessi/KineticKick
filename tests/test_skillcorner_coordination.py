"""Small reconstructed3D contract tests; synthetic checks need no Tang assets."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.find_spec('build_skillcorner_coordination')
if SPEC is not None:
    from build_skillcorner_coordination import build_coordination, load_inputs, verify_sources

SAMPLE = ROOT / 'data/research_human_kicks/skillcorner_probe_20261107'


def fixture():
    points = {'midHip': [0, 0, 0], 'neck': [0, 0, .6],
              'nose': [.05, 0, .8], 'lEye': [.04, .03, .82], 'rEye': [.04, -.03, .82],
              'lEar': [0, .08, .8], 'rEar': [0, -.08, .8]}
    for side, sign in [('l', 1), ('r', -1)]:
        for joint, xyz in {'Hip': [0, .15, 0], 'Knee': [0, .15, -.4],
                           'Ankle': [0, .15, -.9], 'Heel': [-.05, .15, -.95],
                           'BigToe': [.2, .15, -.95], 'SmallToe': [.18, .2, -.95],
                           'Shoulder': [0, .3, .6], 'Elbow': [0, .5, .4],
                           'Wrist': [0, .6, .1], 'Thumb': [.02, .61, .09],
                           'Pinky': [-.02, .62, .08]}.items():
            points[side+joint] = [xyz[0], sign*xyz[1], xyz[2]]
    frames = []
    for i in range(8):
        frames.append({'frame': 1000+i, 'timestamp': f'00:00:{40+i*.04:05.2f}', 'period': 1,
                       'ball_data': {'x': 1+i*.02, 'y': 0, 'z': .1, 'is_detected': True},
                       'possession': {'player_id': None, 'group': 'home team'},
                       'player_data': [{'player_id': 7, 'x': i*.01, 'y': 0, 'is_detected': True,
                                        'joints': {n: {'xyz': (np.array(p)+[i*.01,0,0]).tolist(), 'p90_mae_cm': 1.} for n,p in points.items()}}]})
    match = {'id': 12, 'players': [{'id': 7, 'first_name': 'Synthetic', 'last_name': 'Player', 'team_id': 5, 'number': 1}]}
    event = {'match_id':'12', 'event_id':'8_fixture', 'event_type':'player_possession',
             'player_id':'7', 'frame_start':'400', 'frame_end':'401', 'time_start':'00:40.0',
             'time_end':'00:40.1', 'period':'1', 'phase_index':'1', 'end_type':'shot', 'is_header':'False'}
    source = {'source_class':'synthetic_test_fixture', 'professional_participation_verified':False,
              'source_sha256':'synthetic-no-source-file'}
    return frames, match, event, source


class SkillCornerContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, 'SkillCorner coordination builder has not been implemented')

    def test_metric_units_relative_frame_and_error_units_are_explicit(self):
        f,m,e,s = fixture()
        out,meta = build_coordination(f,m,e,source=s)
        names = list(out['joint_names']); hip = names.index('midHip'); knee = names.index('lKnee')
        np.testing.assert_allclose(out['joint_predicted_p90_m'], .01)
        np.testing.assert_allclose(out['joint_predicted_p90_cm'], 1.)
        np.testing.assert_allclose(out['joints_pelvis_relative_m'][:,hip], 0)
        np.testing.assert_allclose(out['joints_pelvis_relative_m'][:,knee], [[0,.15,-.4]]*len(f))
        np.testing.assert_allclose(out['pelvis_relative_error_budget_m'][:,hip], 0)
        np.testing.assert_allclose(out['pelvis_relative_error_budget_m'][:,knee], .02)
        self.assertAlmostEqual(float(out['length_scale_m']), .9)
        self.assertAlmostEqual(float(out['length_scale_error_budget_m']), .04)
        self.assertNotIn('floor_z_m', out)
        self.assertIn('source axes', meta['coordinate_frame'])
        self.assertFalse(meta['uncertainty']['is_calibrated_confidence_interval'])

    def test_missing_pose_error_and_detection_masks_remain_separate(self):
        f,m,e,s = fixture()
        del f[1]['player_data'][0]['joints']['lWrist']
        f[2]['player_data'][0]['joints']['lAnkle']['p90_mae_cm'] = None
        f[3]['player_data'][0]['is_detected'] = False
        f[4]['player_data'][0]['joints'] = None
        f[5]['ball_data'] = {'x':None,'y':0,'z':None,'is_detected':False}
        out,_ = build_coordination(f,m,e,source=s)
        names=list(out['joint_names']); w=names.index('lWrist'); a=names.index('lAnkle')
        self.assertFalse(out['joint_present'][1,w]); self.assertTrue(np.isnan(out['joints_reconstructed_m'][1,w]).all())
        self.assertTrue(out['joint_geometry_valid'][2,a]); self.assertFalse(out['joint_error_valid'][2,a])
        self.assertTrue(np.isnan(out['pelvis_relative_error_budget_m'][2,a]))
        self.assertTrue(out['joint_geometry_valid'][3].all()); self.assertFalse(out['joint_quality_pass'][3].any())
        self.assertFalse(out['joint_present'][4].any()); self.assertFalse(out['pipeline_joint_interpolated'].any())
        self.assertFalse(out['ball_geometry_valid'][5]); self.assertFalse(out['ball_is_detected'][5])
        self.assertTrue(out['tracking_xy_resampled_10_to_25hz'].all())

    def test_fractional_event_frame_maps_to_time_without_becoming_impact_truth(self):
        f,m,e,s=fixture();out,meta=build_coordination(f,m,e,source=s)
        anchor=meta['event_alignment']
        self.assertEqual(anchor['pose_frame_fractional'],1002.5)
        self.assertEqual(anchor['bracketing_pose_frames'],[1002,1003])
        self.assertAlmostEqual(anchor['event_match_time_s'],40.1)
        self.assertAlmostEqual(anchor['event_clip_time_s'],.1)
        self.assertAlmostEqual(anchor['time_residual_s'],0)
        self.assertIsNone(meta['truth_labels']['contact_time_s'])
        self.assertIsNone(meta['truth_labels']['instep'])
        self.assertIsNone(meta['truth_labels']['kick_strength'])
        self.assertFalse(meta['qualification']['professional_participation']['verified'])
        self.assertTrue(meta['qualification']['shot_event']['verified'])
        self.assertEqual(int(out['possession_is_shooter'].sum()),0)
        bad=copy.deepcopy(e);bad['time_end']='00:41.1'
        with self.assertRaisesRegex(ValueError,'[Ee]vent.*time'):
            build_coordination(f,m,bad,source=s)
        bad=copy.deepcopy(e);bad['period']='2'
        with self.assertRaisesRegex(ValueError,'period'):
            build_coordination(f,m,bad,source=s)

    def test_bad_frame_sequence_ids_and_units_are_rejected(self):
        f,m,e,s=fixture();f[1]['frame']=f[0]['frame']
        with self.assertRaisesRegex(ValueError,'consecutive'):
            build_coordination(f,m,e,source=s)
        f,m,e,s=fixture();e['match_id']='999'
        with self.assertRaisesRegex(ValueError,'match'):
            build_coordination(f,m,e,source=s)
        f,m,e,s=fixture();f[0]['player_data'].append(copy.deepcopy(f[0]['player_data'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            build_coordination(f,m,e,source=s)
        f,m,e,s=fixture();s['coordinate_units']='mm'
        with self.assertRaisesRegex(ValueError,'metres'):
            build_coordination(f,m,e,source=s)

    def test_sensitivity_budget_covers_bounded_geometry_perturbations(self):
        f,m,e,s=fixture();base,_=build_coordination(f,m,e,source=s)
        rng=np.random.default_rng(4);changed=copy.deepcopy(f)
        for frame in changed:
            for point in frame['player_data'][0]['joints'].values():
                delta=rng.normal(size=3);delta=.009*delta/np.linalg.norm(delta)
                point['xyz']=(np.array(point['xyz'])+delta).tolist()
        perturbed,_=build_coordination(changed,m,e,source=s)
        self.assertTrue(np.all(np.abs(base['relation_cosines']-perturbed['relation_cosines']) <= base['relation_cosine_error_budget']+1e-12))
        delta=np.linalg.norm(base['joints_normalized']-perturbed['joints_normalized'],axis=2)
        self.assertTrue(np.all(delta <= base['normalized_position_error_budget']+1e-12))
        inflated=copy.deepcopy(f)
        for frame in inflated:
            for point in frame['player_data'][0]['joints'].values(): point['p90_mae_cm']=40.
        large,_=build_coordination(inflated,m,e,source=s)
        np.testing.assert_allclose(large['relation_cosines'],base['relation_cosines'])
        self.assertTrue(np.all(large['relation_cosine_error_budget'] >= base['relation_cosine_error_budget']))
        self.assertFalse(large['relation_quality_pass'].any())

    def test_relative_geometry_transforms_without_inventing_anatomical_pelvis_axes(self):
        f,m,e,s=fixture();base,_=build_coordination(f,m,e,source=s)
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        transformed=copy.deepcopy(f)
        for frame in transformed:
            for point in frame['player_data'][0]['joints'].values():
                point['xyz']=(1.5*rot@point['xyz']+[10,20,100]).tolist()
                point['p90_mae_cm']*=1.5
        out,_=build_coordination(transformed,m,e,source=s)
        np.testing.assert_allclose(out['joints_normalized'],np.einsum('ij,tkj->tki',rot,base['joints_normalized']),atol=1e-12)
        np.testing.assert_allclose(out['relation_cosines'],base['relation_cosines'],atol=1e-12)
        np.testing.assert_allclose(out['relation_cosine_error_budget'],base['relation_cosine_error_budget'],atol=1e-12)

    def test_pinned_source_hash_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.json';p.write_text('{}')
            cfg={'input_files':{'test.json':{'sha256':hashlib.sha256(b'{}').hexdigest()}}}
            self.assertIn('test.json',verify_sources(Path(d),cfg))
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'hash'):
                verify_sources(Path(d),cfg)

    @unittest.skipUnless(SAMPLE.exists(),'ignored SkillCorner sample not available')
    def test_actual_professional_shot_roundtrip_retains_all_309_frames(self):
        frames,match,event,source=load_inputs(SAMPLE)
        out,meta=build_coordination(frames,match,event,source=source)
        self.assertEqual(out['joints_reconstructed_m'].shape,(309,29,3))
        self.assertEqual(out['source_frame_numbers'][[0,-1]].tolist(),[124487,124795])
        self.assertTrue(out['joint_geometry_valid'].all());self.assertTrue(out['player_is_detected'].all())
        self.assertTrue(out['ball_is_detected'].all())
        self.assertFalse(out['pipeline_joint_interpolated'].any())
        self.assertEqual(meta['event_alignment']['pose_frame_fractional'],124767.5)
        self.assertEqual(meta['event']['phase_index'],'397')
        self.assertTrue(meta['qualification']['professional_participation']['verified'])
        self.assertTrue(meta['qualification']['shot_event']['verified'])
        self.assertIsNone(meta['truth_labels']['contact_time_s'])
        names=list(out['joint_names']);sample=frames[0]['player_data'];p=next(p for p in sample if p['player_id']==965698)
        np.testing.assert_allclose(out['joints_reconstructed_m'][0,names.index('lAnkle')],p['joints']['lAnkle']['xyz'])
        self.assertEqual(out['joint_predicted_p90_cm'][0,names.index('lAnkle')],p['joints']['lAnkle']['p90_mae_cm'])
        buff=io.BytesIO();np.savez_compressed(buff,**out);buff.seek(0)
        with np.load(buff,allow_pickle=False) as loaded:
            for key in loaded.files:
                self.assertNotEqual(loaded[key].dtype,object)
                np.testing.assert_array_equal(loaded[key],out[key])
        json.dumps(meta,allow_nan=False)


if __name__=='__main__':
    unittest.main()
