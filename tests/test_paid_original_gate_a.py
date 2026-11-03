"""Regression tests for KineticKick Gate A's PAiD joint/body reference identity."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
ORIG=ROOT/"configs/paid_original/main.urdf"
LOCAL=ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
MOTION=ROOT/"third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"

@unittest.skipUnless(ORIG.is_file() and LOCAL.is_file() and MOTION.is_file(),
                     "External PAiD motion/URDF are not checked into this workspace")
class TestPAiDOriginalGateA(unittest.TestCase):
    @staticmethod
    def joints(path):
        return {
            j.attrib["name"]: j
            for j in ET.parse(path).getroot().findall("joint")
            if j.attrib["type"] in ("revolute","continuous")
        }

    def test_original_and_local_active_kinematics_match(self):
        a,b=self.joints(ORIG),self.joints(LOCAL)
        self.assertEqual(len(a),29)
        self.assertEqual(set(a),set(b))
        for name in sorted(a):
            for k in ("origin","axis","limit"):
                p,q=a[name].find(k),b[name].find(k)
                self.assertEqual(p.attrib if p is not None else {},
                                 q.attrib if q is not None else {},name+":"+k)

    def test_collision_models_do_differ(self):
        a,b=(ET.parse(p).getroot() for p in (ORIG,LOCAL))
        def contact_geom(root,link):
            for e in root.findall("link"):
                if e.attrib["name"]==link:
                    return e.find("collision/geometry")[0].tag
            raise AssertionError("missing "+link)
        self.assertEqual(contact_geom(a,"left_ankle_roll_link"),"cylinder")
        self.assertEqual(contact_geom(b,"left_ankle_roll_link"),"sphere")

    def test_npz_schema_for_all_soccer_motions(self):
        import numpy as np
        paths=list((ROOT/"third_party/HumanoidSoccer/motions").glob("*/*.npz"))
        self.assertEqual(len(paths),13)
        for p in paths:
            with self.subTest(p=p.name):
                with np.load(p) as data:
                    n=data["joint_pos"].shape[0]
                    self.assertEqual(data["joint_pos"].shape,(n,29))
                    self.assertEqual(data["body_pos_w"].shape,(n,30,3))
                    self.assertEqual(data["body_quat_w"].shape,(n,30,4))

    def test_multiframe_forward_kinematics_reference(self):
        r=subprocess.run([sys.executable,str(ROOT/"tools/audit_paid_urdf_reference.py")],
            capture_output=True,text=True,cwd=ROOT)
        self.assertEqual(r.returncode,0,r.stdout+"\n"+r.stderr)
        report=json.loads((ROOT/"reports/paid_reference_urdf_fk_audit.json").read_text())
        good=report["PAiD_original__isaac_runtime"]
        bad=report["PAiD_original__urdf_xml"]
        self.assertLess(good["matched_mean_position_error_m"],0.00001)
        self.assertLess(good["matched_max_position_error_m"],0.00001)
        self.assertGreater(bad["matched_mean_position_error_m"],0.05,
            "Test must catch accidental use of URDF XML joint order")

if __name__=="__main__":
    unittest.main()
