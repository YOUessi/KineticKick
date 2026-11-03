"""Regression tests for downloaded 29DoF LAFAN reference and robust G1 A/B audits."""
from pathlib import Path
import sys,unittest,json
import numpy as np

ROOT=Path("/home/you/projects/kinetickick")
sys.path.insert(0,str(ROOT/"tools"))
from g1_retarget_kinematics import (
    DATA,URDF,JOINTS,load_robot_tree,fk_g1_pose,binary_stl_sample
)

class HumanDerivedG1ReferenceTests(unittest.TestCase):
    def test_exact_29_dof_motion_to_urdf_schema(self):
        root,links,joints=load_robot_tree()
        self.assertEqual(root,"pelvis")
        self.assertEqual(len(JOINTS),29)
        self.assertEqual(sum(x.kind!="fixed" for x in joints),29)
        self.assertEqual(len(links),39)
        self.assertEqual(set(JOINTS),set(x.name for x in joints if x.kind!="fixed"))
    def test_five_motions_are_not_empty_and_root_rotations_are_valid(self):
        files=list(DATA.glob("*_selected8s.npz"))
        self.assertEqual(len(files),5)
        for path in files:
            a=np.load(path)["frames"]
            self.assertEqual(a.shape,(240,36))
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertLess(float(np.abs(np.linalg.norm(a[:,3:7],axis=1)-1.).max()),.02)
    def test_urdf_fk_world_foot_positions_are_finite(self):
        root,links,joints=load_robot_tree()
        clip=np.load(DATA/"walk1_subject1_selected8s.npz")["frames"]
        for row in (clip[0],clip[100],clip[200]):
            states=fk_g1_pose(row[:3],row[3:7],row[7:],root,joints)
            for side in ("left","right"):
                pose=states[side+"_ankle_roll_link"]
                self.assertEqual(pose.shape,(4,4))
                self.assertTrue(np.all(np.isfinite(pose)))
                self.assertAlmostEqual(float(np.linalg.det(pose[:3,:3])),1.,places=3)
    def test_binary_stl_wellformed(self):
        f=URDF.parent/"meshes/left_knee_link.STL"
        points=binary_stl_sample(f,max_triangles=40)
        self.assertEqual(points.shape,(120,3))
        self.assertTrue(np.all(np.isfinite(points)))
    def test_paired_velocity_perturbations_differ_across_eval_seeds(self):
        a=json.loads((ROOT/"reports/gait_perturbed_seed42.json").read_text())
        b=json.loads((ROOT/"reports/gait_perturbed_seed45.json").read_text())
        self.assertNotEqual(a["paired_initial_velocity_disturbances_mps"],
                            b["paired_initial_velocity_disturbances_mps"])
        self.assertEqual(len(a["summary"]),9)
        self.assertEqual(len(b["summary"]),9)
        for obj in (a,b):
            self.assertTrue(obj["same_initial_pose"])
            self.assertTrue(obj["scene_friction_randomization_disabled"])
    def test_no_unjustified_candidate_promotion(self):
        a=json.loads((ROOT/"reports/gait_robustness_matrix_20261021.json").read_text())
        self.assertFalse(a["all_trials_failed_if_not_pass"]["V1"])
        self.assertFalse(a["all_trials_failed_if_not_pass"]["V2"])
    def test_visual_reference_does_not_add_physical_actor(self):
        src=(ROOT/"tools/g1_29dof_reference_usd.py").read_text()
        self.assertIn("UsdGeom.Mesh.Define",src)
        self.assertNotIn("UsdPhysics.RigidBodyAPI.Apply",src)
        self.assertNotIn("UsdPhysics.CollisionAPI.Apply",src)

    def test_29dof_tracking_reference_bank_shapes_and_limits(self):
        f=ROOT/"data/human_motion/g1_29dof_reference_bank_v0.npz"
        self.assertTrue(f.exists())
        with np.load(f) as data:
            self.assertEqual(data["q"].shape,(5,240,29))
            self.assertEqual(data["dq"].shape,(5,240,29))
            self.assertEqual(data["root_xyz"].shape,(5,240,3))
            self.assertEqual(data["foot_stance_proxy"].shape,(5,240,2))
            self.assertTrue(np.all(np.isfinite(data["feet_xyz"])))
            self.assertEqual(len(data["joint_names"]),29)
        m=json.loads((ROOT/"reports/g1_29dof_reference_bank_manifest.json").read_text())
        self.assertTrue(m["no_force_contact_ground_truth"])
        self.assertTrue(m["for_29dof_morphology_only"])
        self.assertTrue(m["no_learned_policy"])

if __name__=="__main__":
    unittest.main()
