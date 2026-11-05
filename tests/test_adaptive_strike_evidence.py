import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from adaptive_strike_evidence import verify_candidate_runtime, validate_feedback_trace


class CandidateEvidence(unittest.TestCase):
    def runtime(self):
        base={k:'frozen' for k in ['config_sha256','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','contact_frame_sampler_sha256','termination_policy','contact_geometry']}
        candidate=copy.deepcopy(base)
        candidate.update(stride_shot_source_sha256='runner',adaptive_control={'enabled':True,'source_sha256':{'run_adaptive_stride.py':'runner'}})
        return base,candidate

    def test_ablation_or_changed_runtime_is_not_candidate_c(self):
        base,candidate=self.runtime();sources={'run_adaptive_stride.py':'runner'}
        verify_candidate_runtime(candidate,base,sources)
        candidate['adaptive_control']['enabled']=False
        with self.assertRaises(ValueError):verify_candidate_runtime(candidate,base,sources)
        for key in ['config_sha256','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','termination_policy']:
            _,candidate=self.runtime();candidate[key]='changed'
            with self.subTest(key=key),self.assertRaises(ValueError):verify_candidate_runtime(candidate,base,sources)

    def report(self):
        f=lambda t,latch:dict(time_s=t,phase_s=t,phase_shift_s=0.,placement_xy=[0.,0.],foot_delta_xy=[0.,0.],measured_root_xy=[0.,0.],measured_velocity_xy=[0.,-2.],measured_ball_xy=[0.,-4.],left_support_N=100.,contact_latched=latch)
        return dict(kinematics=[[{'time_s':0.},{'time_s':.02}]],adaptive_control={'feedback':[f(0.,False),f(.02,True)]},traces=[[{'time_s':.015,'robot_ball_contact_force_N':{'right_ankle_roll_link':10.}}]])

    def test_missing_nonfinite_or_premature_latch_evidence_is_rejected(self):
        config={'max_placement_m':.12,'max_phase_shift_s':.12}
        validate_feedback_trace(self.report(),config)
        for mutation in [lambda r:r['adaptive_control'].update(feedback=[]),
                         lambda r:r['adaptive_control']['feedback'].pop(),
                         lambda r:r['adaptive_control']['feedback'][0].update(phase_s=float('nan')),
                         lambda r:r['adaptive_control']['feedback'][0].update(contact_latched=True),
                         lambda r:r['adaptive_control']['feedback'][1].update(foot_delta_xy=[.5,0.])]:
            r=self.report();mutation(r)
            with self.assertRaises(ValueError):validate_feedback_trace(r,config)

if __name__=='__main__':unittest.main()
