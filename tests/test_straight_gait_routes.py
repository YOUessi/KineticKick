"""The A→B gait exhibit must never replay winding segments as a straight walk."""
from pathlib import Path
import json,unittest
import numpy as np

ROOT=Path("/home/you/projects/kinetickick")
MANIFEST=ROOT/"reports/straight_gait_reference_manifest.json"

class TestStraightGaitRoutes(unittest.TestCase):
    def test_four_reference_clips_all_head_to_target(self):
        data=json.loads(MANIFEST.read_text())
        self.assertEqual(len(data["clips"]),4)
        for label,s in data["clips"].items():
            self.assertGreater(s["straightness_ratio"],.985,(label,s))
            self.assertGreater(s["forward_facing_mean_cos"],.95,(label,s))
            self.assertLess(s["relative_yaw_std_rad"],.18,(label,s))
            x=np.load(s["file"])["frames"]
            self.assertEqual(x.shape[1],36)
            self.assertAlmostEqual(float(x[0,0]),0.,places=3)
            self.assertAlmostEqual(float(x[-1,0]),s["goal_world_xy"][0],places=3)
            self.assertAlmostEqual(float(x[-1,1]),0.,places=3)
            self.assertTrue(np.all(np.isfinite(x)))
            d=np.diff(x[:,:2],axis=0)
            length=float(np.linalg.norm(d,axis=1).sum())
            net=float(np.linalg.norm(x[-1,:2]-x[0,:2]))
            self.assertGreater(net/length,.985)

    def test_no_autonomous_policy_or_automatic_teleport_is_claimed(self):
        s=(ROOT/"tools/show_g1_straight_walk_reference.py").read_text()
        self.assertIn("one-shot",s)
        self.assertIn('state["completed"]=True',s)
        self.assertIn('state["paused"]=True',s)
        self.assertNotIn("robot.write_root_pose_to_sim",s)
        self.assertNotIn("RslRlVecEnvWrapper",s)
        self.assertTrue(json.loads(MANIFEST.read_text())["source_is_kinematic_not_learned_navigation"])

    def test_no_extra_physics_objects_for_goal(self):
        s=(ROOT/"tools/gait_route_visual.py").read_text()
        self.assertIn("UsdGeom.Cube.Define",s)
        self.assertNotIn("UsdPhysics.CollisionAPI.Apply",s)
        self.assertNotIn("UsdPhysics.RigidBodyAPI.Apply",s)

if __name__=="__main__":
    unittest.main()
