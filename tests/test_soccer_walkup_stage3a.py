"""Regression tests for the actual Stage 3A walk-up on the Tang filesystem.

Tests generated content/positioning without claiming kinematic G1 is physical.
"""
from __future__ import annotations
import json,unittest
from pathlib import Path
import numpy as np

ROOT=Path("/home/you/projects/kinetickick")
SCENE=(ROOT/"tools/soccer_walkup_scene.py").read_text()
SCRIPT=(ROOT/"tools/show_g1_soccer_walkup_reference.py").read_text()
M=json.loads((ROOT/"reports/straight_gait_reference_manifest.json").read_text())["clips"]

class TestWalkup(unittest.TestCase):
    def test_single_physics_ball_and_classic_visual(self):
        self.assertIn('prim_path="/World/MatchFootball"',SCENE)
        self.assertEqual(SCENE.count("RigidObject(RigidObjectCfg("),1)
        self.assertIn('attach_classic_soccer_visual("/World/MatchFootball"',SCENE)
        self.assertIn('"football_count":1',SCENE)
        self.assertIn('"visual_waypoint_spheres":0',SCENE)
        self.assertIn("BALL_MASS=.43",SCENE)
    def test_fixed_ball_ready_goal_spatial_order(self):
        self.assertIn("BALL_XY=(4.80,0.0)",SCENE)
        self.assertIn("READY_XY=(4.20,0.0)",SCENE)
        self.assertIn("GOAL_XY=(8.80,0.0)",SCENE)
    def test_all_four_modes_start_different_distances_but_end_same_ready(self):
        from numpy.linalg import norm
        for mode,ref in M.items():
            x=np.load(ref["file"])["frames"]
            source_dx=float(ref["goal_world_xy"][0])
            start=4.2-source_dx
            self.assertAlmostEqual(float(x[0,0])+start,start,places=3)
            self.assertAlmostEqual(float(x[-1,0])+start,4.2,places=3)
            self.assertAlmostEqual(float(x[-1,1]),0.,places=3)
            self.assertGreater(4.8-(float(x[-1,0])+start),.59)
            self.assertLess(norm(x[-1,:2]-x[0,:2]), source_dx+.02)
    def test_manual_replay_required_and_29dof_kinematic_only(self):
        self.assertIn('state["completed"]=True',SCRIPT)
        self.assertIn('state["paused"]=True',SCRIPT)
        self.assertNotIn("current_t%source_duration",SCRIPT)
        self.assertNotIn("robot.write_root_pose_to_sim",SCRIPT)
        self.assertNotIn("RslRlVecEnvWrapper",SCRIPT)
        self.assertIn("g1_reference_has_physics",SCRIPT)
    def test_goal_has_posts_and_noncolliding_net(self):
        self.assertIn("LeftUpright",SCENE)
        self.assertIn("RightUpright",SCENE)
        self.assertIn("Crossbar",SCENE)
        self.assertIn("NetRenderOnly",SCENE)
        self.assertIn("UsdPhysics.CollisionAPI.Apply",SCENE)

if __name__=="__main__":
    unittest.main()
