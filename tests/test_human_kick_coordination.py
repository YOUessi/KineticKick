"""Surface-marker invariants and explicit uncertainty; synthetic tests need no Tang assets."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.find_spec('build_human_kick_coordination')
if SPEC is not None:
    from build_human_kick_coordination import build_coordination, read_cmu_c3d


def fixture(frames=25, fps=20):
    points = {
        'LFWT':[.1,.15,1.], 'RFWT':[.1,-.15,1.],
        'LBWT':[-.1,.15,1.], 'RBWT':[-.1,-.15,1.],
        'CLAV':[.05,0,1.5], 'C7':[-.05,0,1.5],
        'STRN':[.05,0,1.3], 'T10':[-.05,0,1.3],
    }
    for side, sign in [('L',1),('R',-1)]:
        for suffix, xyz in {
            'SHO':[0,.3,1.5], 'ELB':[0,.5,1.3],
            'WRA':[.02,.6,1.1], 'WRB':[-.02,.6,1.1],
            'KNE':[0,.15,.55], 'ANK':[0,.15,.1],
            'HEE':[-.05,.15,.05], 'TOE':[.18,.15,.05],
            'MT5':[.12,.19,.05],
        }.items():
            points[side+suffix]=[xyz[0],sign*xyz[1],xyz[2]]
    names=np.array(list(points))
    xyz=np.repeat(np.array(list(points.values()))[None],frames,axis=0)
    times=np.arange(frames,dtype=float)/fps
    valid=np.ones(xyz.shape[:2],dtype=bool)
    return names,xyz,valid,times


def run_build(names,xyz,valid,times,**kwargs):
    return build_coordination(xyz,names,valid,times,
        source={'source_class':'synthetic_test_fixture','source_sha256':'synthetic-not-a-source-file','clip_id':'synthetic'},**kwargs)


class CoordinationGeometry(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, 'Human coordination builder has not been implemented')

    def test_surface_geometry_scale_and_frame_are_explicit(self):
        names,xyz,valid,t=fixture()
        out,meta=run_build(names,xyz,valid,t)
        np.testing.assert_allclose(out['pelvis_axes_world'],np.broadcast_to(np.eye(3),(len(t),3,3)),atol=1e-12)
        np.testing.assert_allclose(out['length_scale_m'],.9,atol=1e-12)
        thorax=list(out['landmark_names']).index('thorax_center')
        np.testing.assert_allclose(out['landmarks_normalized'][:,thorax],[np.array([0,0,.4/.9])]*len(t),atol=1e-12)
        self.assertFalse(meta['sourcequalification'])
        self.assertIsNone(meta['truth_labels']['contact_time_s'])
        self.assertIn('surface',meta['representation'])
        self.assertTrue(np.all(out['relation_valid']))

    def test_translation_rotation_and_scale_do_not_change_normalized_geometry(self):
        names,xyz,valid,t=fixture()
        xyz[:,list(names).index('RTOE'),0]+=.05*np.sin(4*t)
        angle=.63
        rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rx=np.array([[1,0,0],[0,np.cos(.31),-np.sin(.31)],[0,np.sin(.31),np.cos(.31)]])
        rot=rz@rx
        transformed=1.7*np.einsum('ij,tmj->tmi',rot,xyz)+np.array([2,-3,.7])
        a,_=run_build(names,xyz,valid,t)
        b,_=run_build(names,transformed,valid,t)
        for field in ['landmarks_normalized','landmark_normalized_velocity_per_s','segment_directions_pelvis','relation_cosines']:
            np.testing.assert_allclose(a[field],b[field],atol=2e-12,equal_nan=True)

    def test_short_gap_is_flagged_long_gap_and_edge_are_not_filled(self):
        names,xyz,valid,t=fixture(frames=16)
        toe=list(names).index('LTOE')
        xyz[:,toe,0]=t
        valid[0,toe]=False;valid[4:6,toe]=False;valid[9:13,toe]=False
        xyz[~valid]=np.nan
        out,_=run_build(names,xyz,valid,t,max_gap_frames=2)
        self.assertTrue(out['marker_interpolated'][4:6,toe].all())
        np.testing.assert_allclose(out['markers_m'][4:6,toe,0],t[4:6])
        self.assertFalse(out['source_marker_valid'][4:6,toe].any())
        self.assertFalse(out['marker_valid'][9:13,toe].any())
        self.assertFalse(out['marker_valid'][0,toe])
        landmark=list(out['landmark_names']).index('left_toe')
        self.assertTrue(out['landmark_interpolated'][4:6,landmark].all())
        self.assertFalse(out['landmark_normalized_velocity_valid'][8:14,landmark].any())

    def test_degenerate_pelvis_frame_and_missing_thorax_stay_unavailable(self):
        names,xyz,valid,t=fixture()
        xyz[:,list(names).index('LFWT')]=xyz[:,list(names).index('LBWT')]
        xyz[:,list(names).index('RFWT')]=xyz[:,list(names).index('RBWT')]
        out,_=run_build(names,xyz,valid,t)
        self.assertFalse(out['pelvis_frame_valid'].any())
        self.assertTrue(np.isnan(out['landmarks_normalized']).all())
        names,xyz,valid,t=fixture()
        keep=names!='CLAV'
        out,_=run_build(names[keep],xyz[:,keep],valid[:,keep],t)
        thorax=list(out['landmark_names']).index('thorax_center')
        toe=list(out['landmark_names']).index('left_toe')
        self.assertFalse(out['landmark_valid'][:,thorax].any())
        self.assertTrue(out['landmark_local_valid'][:,toe].all())

    def test_missing_leg_scale_does_not_erase_available_upper_body_geometry(self):
        names,xyz,valid,t=fixture()
        keep=~np.isin(names,['LKNE','RKNE'])
        out,_=run_build(names[keep],xyz[:,keep],valid[:,keep],t)
        thorax=list(out['landmark_names']).index('thorax_center')
        arm=list(out['segment_names']).index('left_upper_arm')
        self.assertFalse(bool(out['length_scale_valid']))
        self.assertTrue(out['landmark_local_valid'][:,thorax].all())
        self.assertTrue(out['segment_valid'][:,arm].all())
        self.assertTrue(np.isnan(out['landmarks_normalized']).all())
        self.assertFalse(out['landmark_normalized_velocity_valid'].any())

    def test_invalid_time_duplicate_names_and_robot_source_are_rejected(self):
        names,xyz,valid,t=fixture()
        with self.assertRaises(ValueError):
            run_build(names,xyz,valid,np.zeros_like(t))
        duplicate=names.copy();duplicate[1]=duplicate[0]
        with self.assertRaises(ValueError):
            run_build(duplicate,xyz,valid,t)
        with self.assertRaises(ValueError):
            build_coordination(xyz,names,valid,t,source={'source_class':'robot_retargeted_reference'})

    def test_foot_peak_does_not_create_contact_or_verified_kick_side(self):
        names,xyz,valid,t=fixture(frames=80,fps=40)
        xyz[:,list(names).index('RTOE'),0]+=.5*np.sin(8*t)
        out,meta=run_build(names,xyz,valid,t)
        self.assertIsNotNone(meta['events']['foot_speed_peak'])
        self.assertIsNone(meta['events']['phases']['strike'])
        self.assertIsNone(meta['events']['ball_marker_centroid_motion_onset'])
        for field in ['contact_time_s','kick_subtype','skill_level','kicking_side']:
            self.assertIsNone(meta['truth_labels'][field])

    def test_ball_motion_candidate_is_independent_and_is_not_ball_center(self):
        names,xyz,valid,t=fixture(frames=80,fps=40)
        ball=np.zeros((len(t),3,3));ball[:,:,2]=.1
        ball[:,0,1]=-.04;ball[:,1,1]=.04;ball[:,2,2]=.17
        ball[:,:,0]=np.maximum(t-.8,0)[:,None]
        _,meta=run_build(names,xyz,valid,t,ball_markers_m=ball,
            ball_marker_names=np.array(['A','B','C']),ball_marker_valid=np.ones(ball.shape[:2],bool))
        event=meta['events']['ball_marker_centroid_motion_onset']
        self.assertIsNotNone(event)
        self.assertGreaterEqual(event['time_s'],.8)
        self.assertLess(event['time_s'],1.0)
        self.assertFalse(event['is_contact_truth'])
        self.assertFalse(event['is_ball_center'])
        self.assertIsNone(meta['events']['foot_speed_peak'])
        self.assertIsNone(meta['events']['phases']['strike'])

    def test_source_zero_residual_flags_and_interpolation_flags_remain_distinct(self):
        names,xyz,valid,t=fixture()
        flags=np.zeros(valid.shape,bool);flags[3,0]=True
        out,meta=run_build(names,xyz,valid,t,source_zero_residual_flag=flags)
        self.assertTrue(out['source_zero_residual_flag'][3,0])
        self.assertFalse(out['marker_interpolated'].any())
        self.assertEqual(meta['quality']['source_zero_residual_samples'],1)


class AcquiredCMUData(unittest.TestCase):
    @unittest.skipUnless(all((ROOT/f'data/research_human_kicks/cmu_subject10/10_{n}.c3d').exists()
                            for n in ['01','02','03','05','06']), 'Requires ignored CMU research captures')
    def test_five_real_clips_produce_valid_masked_features_without_claimed_contact(self):
        self.assertIsNotNone(SPEC,'Human coordination builder has not been implemented')
        expected={'01':801,'02':590,'03':362,'05':436,'06':572}
        for number,count in expected.items():
            with self.subTest(trial=number):
                info,raw=read_cmu_c3d(ROOT/f'data/research_human_kicks/cmu_subject10/10_{number}.c3d')
                out,meta=build_coordination(raw['human_markers_m'],raw['human_marker_names'],raw['human_marker_valid'],raw['time_s'],
                    source={'source_class':info['source_class'],'source_sha256':info['source_sha256'],'clip_id':f'10_{number}'},
                    ball_markers_m=raw['ball_markers_m'],ball_marker_names=raw['ball_marker_names'],ball_marker_valid=raw['ball_marker_valid'],
                    source_zero_residual_flag=raw['human_zero_residual_flag'])
                self.assertEqual(len(out['time_s']),count)
                self.assertTrue(out['pelvis_frame_valid'].any())
                self.assertGreater(float(out['length_scale_m']),.5)
                self.assertLess(float(out['length_scale_m']),1.5)
                self.assertTrue(np.isfinite(out['relation_cosines'][out['relation_valid']]).all())
                self.assertFalse(meta['sourcequalification'])
                self.assertIsNone(meta['truth_labels']['contact_time_s'])
                self.assertIsNone(meta['events']['phases']['strike'])

if __name__=='__main__':
    unittest.main()
