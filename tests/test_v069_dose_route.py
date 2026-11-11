from pathlib import Path
import ast,importlib.util,json,sys,unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from phase_dose_v069 import curve_contract,DOSE,FIRST,LAST,MARKER_BASE
from phase_coordinated_v063 import phase_trajectory
class TestDoseRouteV069(unittest.TestCase):
    def test_phase_preflight_static(self):
        s=(ROOT/'tools/run_placement_strike_v069.py').read_text()
        tree=ast.parse(s)
        self.assertIn("if cfgdata.get('phase_dose_v069'):",s)
        self.assertIn("from phase_dose_v069 import get_task",s)
        self.assertIn("V069_ACTIVE_PHASE_ROUTE",s)
        self.assertIn("routed_task_class",s)
        self.assertIn("physical_controller_multiplier",s)
    def test_no_silent_illegal_2x(self):
        baseline=torch.tensor(phase_trajectory())
        with self.assertRaises(ValueError):
            curve_contract(baseline*2,'forward')
    def test_slew_unchanged(self):
        from phase_coordinated_v063 import SLEW,AMPLITUDE
        self.assertEqual(SLEW,.006);self.assertEqual(AMPLITUDE,.075)
    def test_dose_within_actuator_envelope(self):
        base=torch.tensor(phase_trajectory())
        y=curve_contract(base,'forward')
        self.assertTrue(torch.allclose(y,base*DOSE))
        self.assertLess(float(y.abs().max()),.075)
        self.assertGreater(float(y.abs().max()),.05)
    def test_reverse_uses_same_amplitude_not_new_phase(self):
        x=torch.tensor(phase_trajectory())
        y=curve_contract(x,'forward')
        z=curve_contract(x,'reverse_window')
        self.assertTrue(torch.equal(z[:FIRST],y[:FIRST]))
        self.assertTrue(torch.equal(z[LAST+1:],y[LAST+1:]))
        self.assertTrue(torch.equal(torch.flip(y[FIRST:LAST+1],[0]),z[FIRST:LAST+1]))
        self.assertTrue(torch.allclose(z.abs().sort(dim=0).values,y.abs().sort(dim=0).values))
    def test_reject_unknown_direction(self):
        with self.assertRaises(ValueError):curve_contract(torch.tensor(phase_trajectory()),'unknown')
    def test_reject_wrong_waveform_shape(self):
        with self.assertRaises(ValueError):curve_contract(torch.zeros((100,29)),'forward')
    def test_reject_nonfinite(self):
        x=torch.tensor(phase_trajectory());x[50,0]=float('nan')
        with self.assertRaises(ValueError):curve_contract(x,'forward')
    def test_preregistered_config_policy_exclusive(self):
        for mode,n in (('forward','forward'),('reverse_window','reversed')):
            c=json.loads((ROOT/f'configs/g1_phase_dose_{n}_v069.json').read_text())
            self.assertEqual(c['phase_dose_v069'],{**MARKER_BASE,'order':mode})
            self.assertNotIn('phase_reversal_v067',c)
            self.assertNotIn('phase_dose_v068',c)
    def test_29dof_and_fail_closed_original_colliders(self):
        from wholebody_safety_interface_v055 import G1_JOINT_ORDER
        from stance_safe_temporal_policy_v067 import stage_29dof_action, RealG1State
        self.assertEqual(len(G1_JOINT_ORDER),29)
        # This is an interface-only fail-closed test, not a physical safety certificate.
        st=RealG1State(tuple(G1_JOINT_ORDER),1.0,np.zeros(29),
            np.zeros(29),np.ones(29)*10,np.stack((-np.ones(29),np.ones(29)),axis=-1),
            150,0,0,0,.1,None,False)
        x=stage_29dof_action(np.zeros(29),np.ones(29)*.03,np.zeros(29),
                             np.ones(29),st)
        self.assertFalse(x.eligible_for_physx_deployment)
        self.assertIn('no_verified_original_GPU_G1_mesh_collider_witness',x.reason)
if __name__=='__main__':unittest.main()
