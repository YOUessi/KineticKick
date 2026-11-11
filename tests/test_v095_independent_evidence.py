"""Independent evidence must fail-closed on fake or incomplete contact data."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v095_independent_evidence as a
def sample(mode):
 return {"device":"cuda:0","capture":mode,"subscribe_phase":"pre_reset",
  "report_mode":"on","layout":"colliding","actual_runtime_frames":180,
  "runtime_z_vz":[[i+1,.4,0.] for i in range(180)],
  "observed_collision_dynamics":True,
  "stage":{"ball_report":True,"box_report":True,"all_colliders":{"ProbeBall":["/World/ProbeBall/shape"],"ProbeBox":["/World/ProbeBox/shape"]}},
  "tensor_contact_points":[],"tensor_pair_witness":False}
def test_native_empty_does_not_prove_no_contact():
 row=dict(device="cuda:0",capture="native",subscribe_phase="pre_reset",
  report_mode="on",layout="colliding",case="synthetic")
 x=a.certify_contact(sample("native"),row)
 assert not x["direct_actor_collider_manifold_certified"]
 assert x["observed_collision"]
def test_synthetic_fake_tensor_contact_rejected():
 z=sample("tensor");z["tensor_pair_witness"]=True
 row=dict(device="cuda:0",capture="tensor",subscribe_phase="pre_reset",
  report_mode="on",layout="colliding",case="synthetic")
 with pytest.raises(ValueError,match="INVALID_TENSOR_WITNESS"):a.certify_contact(z,row)
def test_separated_positive_must_fail():
 z=sample("tensor");z.update(layout="separated",observed_noncollision_dynamics=True,observed_collision_dynamics=False,tensor_contact_counts=1)
 row=dict(device="cuda:0",capture="tensor",subscribe_phase="pre_reset",
  report_mode="on",layout="separated",case="synthetic")
 with pytest.raises(ValueError,match="NEGATIVE_CONTROL_FALSE_POSITIVE"):a.certify_contact(z,row)
def test_off_on_parity_requires_all_180():
 x=sample("native");y=sample("native")
 x["layout"]=y["layout"]="colliding";x["box_position_w"]=y["box_position_w"]=[0,0,0]
 assert a.strict_parity(x,y)["exact_saved_numeric_state"]
 y["runtime_z_vz"][-1][-1]=.01
 assert not a.strict_parity(x,y)["exact_saved_numeric_state"]
