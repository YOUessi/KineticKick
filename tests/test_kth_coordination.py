"""KTH reconstruction contract checks; no GPU or external downloads."""
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
SPEC = importlib.util.find_spec('build_kth_coordination')
if SPEC is not None:
    from build_kth_coordination import build_coordination, load_inputs, verify_sources

SAMPLE = ROOT / 'data/research_human_kicks/kth_player1_sequence2_probe_v1'
NAMES = ['R_ANKLE', 'R_KNEE', 'R_HIP', 'L_HIP', 'L_KNEE', 'L_ANKLE',
         'R_WRIST', 'R_ELBOW', 'R_SHOULDER', 'L_SHOULDER', 'L_ELBOW',
         'L_WRIST', 'BOTTOM_HEAD', 'TOP_HEAD']


def fixture():
    joints = np.array([[0,-.2,-.9], [0,-.2,-.4], [0,-.2,0],
                       [0,.2,0], [0,.2,-.4], [0,.2,-.9],
                       [0,-.7,.1], [0,-.5,.3], [0,-.3,.6],
                       [0,.3,.6], [0,.5,.3], [0,.7,.1],
                       [0,0,.8], [0,0,1.]])
    xyz = np.tile(joints, (5,1,1))
    cameras = np.tile(np.array([[[100,0,0,300], [0,0,-100,200]],
                                [[0,100,0,250], [0,0,-100,200]],
                                [[60,80,0,200], [0,0,-100,200]]]), (5,1,1,1))
    homogeneous = np.concatenate([xyz, np.ones((5,14,1))], axis=-1)
    xy = np.einsum('tcij,tnj->tcni', cameras, homogeneous)
    source = {'source_class':'synthetic_test_fixture', 'professional_match_context_verified':False,
              'goal_sequence_label_verified':False, 'source_hashes':{}}
    return xy, xyz, cameras, source


class KTHContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, 'KTH coordination builder has not been implemented')

    def test_known_projection_units_and_unavailable_foot_thorax_are_explicit(self):
        xy,xyz,c,source=fixture();out,meta=build_coordination(xy,xyz,c,source=source)
        np.testing.assert_array_equal(out['joints_reconstructed_au'],xyz)
        np.testing.assert_allclose(out['reprojection_residual_norm_px'],0,atol=1e-12)
        np.testing.assert_allclose(out['time_s'],np.arange(5)/25)
        np.testing.assert_allclose(out['joints_pelvis_relative_au'],xyz)
        self.assertAlmostEqual(float(out['length_scale_au']),.9)
        l=list(out['landmark_names']); s=list(out['segment_names']); r=list(out['relation_names'])
        self.assertFalse(out['landmark_geometry_valid'][:,l.index('thorax_center')].any())
        self.assertTrue(out['landmark_geometry_valid'][:,l.index('shoulder_midpoint')].all())
        self.assertTrue(out['segment_geometry_valid'][:,s.index('trunk')].all())
        for name in ('left_foot','right_foot'):
            self.assertFalse(out['segment_geometry_valid'][:,s.index(name)].any())
        for name in ('left_shank__left_foot','right_shank__right_foot','left_foot__right_foot'):
            self.assertFalse(out['relation_geometry_valid'][:,r.index(name)].any())
            self.assertTrue(np.isnan(out['relation_cosines'][:,r.index(name)]).all())
        np.testing.assert_allclose(out['relation_cosines'][:,r.index('pelvis_line__shoulder_line')],1)
        self.assertFalse(out['joint_visibility_known'].any())
        self.assertFalse(out['relation_precision_known'].any())
        self.assertNotIn('joints_reconstructed_m',out)
        self.assertIsNone(meta['truth_labels']['contact_time_s'])
        self.assertIsNone(meta['truth_labels']['instep'])
        self.assertIsNone(meta['coordinate_system']['meters_per_unit'])
        self.assertFalse(meta['qualification']['professional_match_context_verified'])

    def test_missing_observation_and_missing_geometry_have_different_masks(self):
        xy,xyz,c,source=fixture();xy[1,0,11]=np.nan;xyz[2,11]=np.nan
        xyz[3,4]=xyz[3,3]  # zero-length left thigh
        xy[4,1,0]+=1000  # large residual remains diagnostic, never deletes geometry
        out,_=build_coordination(xy,xyz,c,source=source)
        r=list(out['relation_names']);s=list(out['segment_names'])
        self.assertTrue(out['joint_geometry_valid'][1,11]);self.assertFalse(out['reprojection_valid'][1,0,11])
        self.assertEqual(out['joint_reprojection_camera_count'][1,11],2)
        self.assertFalse(out['joint_geometry_valid'][2,11])
        self.assertFalse(out['relation_geometry_valid'][2,r.index('left_forearm__right_forearm')])
        self.assertFalse(out['segment_geometry_valid'][3,s.index('left_thigh')])
        self.assertTrue(out['joint_geometry_valid'][4,0])
        self.assertGreater(out['joint_reprojection_rms_px'][4,0],500)
        self.assertFalse(out['pipeline_joint_interpolated'].any())
        self.assertTrue(np.isnan(out['joints_reconstructed_au'][2,11]).all())

    def test_scale_rotation_translation_preserve_relation_and_length_diagnostics(self):
        xy,xyz,c,source=fixture();base,_=build_coordination(xy,xyz,c,source=source)
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        h=np.eye(4);h[:3,:3]=2.5*rot;h[:3,3]=[7,4,2]
        changed=np.einsum('ij,tnj->tni',h[:3,:3],xyz)+h[:3,3]
        cam=np.einsum('tcij,jk->tcik',c,np.linalg.inv(h))
        out,_=build_coordination(xy,changed,cam,source=source)
        np.testing.assert_allclose(out['relation_cosines'],base['relation_cosines'],atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(out['segment_length_relative_deviation'],base['segment_length_relative_deviation'],atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(out['joints_normalized'],np.einsum('ij,tnj->tni',rot,base['joints_normalized']),atol=1e-12)
        np.testing.assert_allclose(out['reprojection_residual_norm_px'],0,atol=1e-12)

    def test_source_hash_mismatch_and_bad_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'positions3d.txt';p.write_bytes(b'1\n')
            cfg={'input_files':{'positions3d.txt':{'size_bytes':2,'sha256':hashlib.sha256(b'1\n').hexdigest()}}}
            self.assertIn('positions3d.txt',verify_sources(Path(d),cfg))
            p.write_bytes(b'2\n')
            with self.assertRaisesRegex(ValueError,'hash'):verify_sources(Path(d),cfg)
        xy,xyz,c,source=fixture()
        with self.assertRaisesRegex(ValueError,'shape'):build_coordination(xy[:-1],xyz,c,source=source)

    @unittest.skipUnless((SAMPLE/'positions3d.txt').exists(),'ignored KTH sample unavailable')
    def test_actual_175_frame_schema_and_lossless_roundtrip(self):
        xy,xyz,c,source=load_inputs(SAMPLE);out,meta=build_coordination(xy,xyz,c,source=source)
        self.assertEqual(xyz.shape,(175,14,3));self.assertEqual(xy.shape,(175,3,14,2));self.assertEqual(c.shape,(175,3,2,4))
        self.assertEqual(out['source_frame_numbers'][[0,-1]].tolist(),[1,175])
        self.assertAlmostEqual(out['time_s'][-1],6.96)
        np.testing.assert_array_equal(out['joints_reconstructed_au'],xyz)
        for t,cam,j in ((0,0,0),(88,1,8),(174,2,13)):
            np.testing.assert_allclose(out['joints_reprojected_px'][t,cam,j],c[t,cam]@np.r_[xyz[t,j],1.])
        self.assertAlmostEqual(meta['diagnostics']['reprojection_rms_px'],10.475061176914755)
        self.assertEqual(out['relation_geometry_valid'].sum(),175*11)
        self.assertFalse(out['relation_precision_known'].any())
        self.assertTrue(meta['qualification']['professional_match_context_verified'])
        self.assertTrue(meta['qualification']['goal_sequence_label_verified'])
        self.assertFalse(meta['qualification']['p2_complete'])
        cmu=json.loads((ROOT/'configs/human_coordination_schema_v1.json').read_text())
        self.assertEqual(list(out['relation_names']),['__'.join(p) for p in cmu['relation_pairs']])
        buf=io.BytesIO();np.savez_compressed(buf,**out);buf.seek(0)
        with np.load(buf,allow_pickle=False) as loaded:
            for k in loaded.files:
                self.assertNotEqual(loaded[k].dtype,object)
                np.testing.assert_array_equal(loaded[k],out[k])
        json.dumps(meta,allow_nan=False)


if __name__=='__main__':
    unittest.main()
