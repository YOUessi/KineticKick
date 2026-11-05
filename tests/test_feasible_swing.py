"""A recorded unreachable target must retain feasible progress, not drop to zero."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from placement_strike import BatchedLeg
from cartesian_strike import LegKinematics


def test_recorded_saturated_ankles_keep_bounded_reach_with_foot_orientation():
    urdf=Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf'
    leg=BatchedLeg(urdf,'cpu',.4)
    q0=torch.tensor([[-.88805926,.27332777,.03772300,1.49375308,.44736943,-.18432701]])
    rot=torch.tensor([[[.17862758,.98142204,-.07002113],[-.98339770,.17576936,-.04510106],[-.03195560,.07691491,.99652543]]])
    pos=torch.tensor([[-.17639737,-3.52243543,.76329994]])
    delta=torch.tensor([[.09245697,-.06026735]])
    yaw=torch.tensor([.002730979]);height=torch.tensor([-.015625])
    _,_,old_angle=leg.translate(q0,rot,pos,delta,yaw,height=height)
    assert float(old_angle[0])>.30  # Recorded full request is genuinely infeasible.
    out,error,angle,fraction=leg.translate_feasible(q0.repeat(2,1),rot.repeat(2,1,1),pos.repeat(2,1),
        torch.cat([delta,torch.zeros_like(delta)]),torch.cat([yaw,torch.zeros_like(yaw)]),
        height=torch.cat([height,torch.zeros_like(height)]))
    assert .25<float(fraction[0])<.75
    assert float(error[0])<.025 and float(angle[0])<.12
    assert float((out-q0).abs().max())<=.400001
    assert bool((out>=leg.limits[:,0]-1e-6).all()) and bool((out<=leg.limits[:,1]+1e-6).all())
    torch.testing.assert_close(out[1],q0[0],atol=1e-6,rtol=0)
    # Independent original FK checks actual world displacement/orientation.
    original=LegKinematics(urdf);root=np.eye(4);root[:3,:3]=rot[0].numpy();root[:3,3]=pos[0].numpy()
    before,_=original.forward(q0[0].numpy(),root);after,_=original.forward(out[0].numpy(),root)
    displacement=after[:3,3]-before[:3,3]
    requested=np.array([.09245697,-.06026735,-.015625])
    assert np.dot(displacement,requested)/np.linalg.norm(requested)>.03
    desired=Rotation.from_euler('z',.002730979*float(fraction[0])).as_matrix()@before[:3,:3]
    assert Rotation.from_matrix(desired@after[:3,:3].T).magnitude()<.12
