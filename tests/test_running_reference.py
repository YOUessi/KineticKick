"""Tang-local reference conversion checks; these do not certify physical running."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class RunningReferenceTests(unittest.TestCase):
    def test_conversion_preserves_order_limits_provenance_and_static_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run.npz'
            cmd=[sys.executable,'-B',str(ROOT/'tools/build_running_reference.py'),
                 '--root',str(ROOT),'--output',str(output),'--yaw-rad','-1.5707963267948966']
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            metadata=json.loads(output.with_suffix('.json').read_text())
            arrays=np.load(output)
            self.assertEqual(arrays['joint_pos'].shape,(186,29))
            self.assertEqual(arrays['body_pos_w'].shape,(186,30,3))
            self.assertEqual(len(set(metadata['joint_names'])),29)
            self.assertEqual(metadata['hard_joint_limit_violations_rad'],{})
            self.assertEqual(metadata['seam']['max_q_jump_rad'],0.)
            self.assertEqual(metadata['seam']['max_dq_jump_rad_s'],0.)
            np.testing.assert_allclose(arrays['joint_vel'][-1],0.,atol=1e-6)
            np.testing.assert_allclose(arrays['body_lin_vel_w'][-1],0.)
            np.testing.assert_allclose(np.linalg.norm(arrays['body_quat_w'],axis=-1),1.,atol=1e-6)
            self.assertEqual(metadata['output_sha256'],hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertIn('NOT a running result',metadata['purpose'])
            self.assertIn('penetrate',metadata['contact_feasibility_caveat'])
            again=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertIn('Do not overwrite',again.stderr)
