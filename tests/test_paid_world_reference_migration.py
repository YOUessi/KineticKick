"""Regression guards: trajectory-frame semantics and checkpoint input migration."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
SOURCE=ROOT/"experiments/paid29_stage1_contact_phase/seed84_n96_iter200_phase108_148_contactv1/last_model.pt"
MIGRATED=ROOT/"experiments/paid29_root_reference_init/model_154_to166.pt"

@unittest.skipUnless(MOTION.exists(),"PAiD reference material is only present on Tang")
class TestWorldReferenceContract(unittest.TestCase):
    def test_reference_requires_forward_movement(self):
        import numpy as np
        z=np.load(MOTION)
        root=z["body_pos_w"][:,0,:]
        # Source forward is world -Y; cannot call static-in-place marching a kick.
        self.assertLess(float(root[-1,1]-root[0,1]),-.75)
        self.assertGreater(float(abs(root[-1,0]-root[0,0])),.0)
        self.assertEqual(z["joint_pos"].shape[1],29)

    def test_reference_robot_initial_yaw_facing_minus_y(self):
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        q=np.load(MOTION)["body_quat_w"][0,0]
        forward=R.from_quat([q[1],q[2],q[3],q[0]]).apply([1,0,0])
        self.assertLess(float(forward[1]),-.95)

@unittest.skipUnless(SOURCE.exists() and MIGRATED.exists(),
                    "Source and migrated checkpoints only present on Tang")
class TestRootObservationMigration(unittest.TestCase):
    def test_154_to_166_checkpoint_keeps_original_actor_behavior(self):
        import torch
        a=torch.load(SOURCE,map_location="cpu",weights_only=False)
        b=torch.load(MIGRATED,map_location="cpu",weights_only=False)
        lhs=a["model_state_dict"]["actor.0.weight"]
        rhs=b["model_state_dict"]["actor.0.weight"]
        self.assertEqual(list(lhs.shape),[512,154])
        self.assertEqual(list(rhs.shape),[512,166])
        self.assertTrue(torch.equal(lhs,rhs[:,:154]))
        self.assertEqual(float(rhs[:,154:].abs().max()),0.)
        self.assertTrue(torch.equal(
            a["model_state_dict"]["critic.0.weight"],
            b["model_state_dict"]["critic.0.weight"]))
        self.assertEqual(list(b["obs_norm_state_dict"]["_mean"].shape),[1,166])
        self.assertEqual(list(b["obs_norm_state_dict"]["_var"].shape),[1,166])

if __name__=="__main__": unittest.main()
