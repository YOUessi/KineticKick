"""Independent fail-closed original G1 tensor pair evidence regression."""
import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v081_tensor_pair import tensor_contacts,direct_tensor_pair_contact,PAIR

def test_missing_or_wrong_path_fails_closed():
    assert not direct_tensor_pair_contact({})
    assert not direct_tensor_pair_contact(dict(target_pair=list(PAIR),direction="foo",
        contacts=[{"point_w":[0,0,0],"normal_w":[1,0,0],"impulse_magnitude":1}]))

@pytest.mark.parametrize("bad",[-1,0,None,float("nan")])
def test_nonpositive_impulse_or_missing(bad):
    assert not direct_tensor_pair_contact(dict(target_pair=list(PAIR),direction="wrist_to_hip",
        contacts=[dict(point_w=[0,0,0],normal_w=[1,0,0],impulse_magnitude=bad)]))

def test_tensor_filter_cannot_forge_two_body_certificate_from_one_event():
    assert not direct_tensor_pair_contact(dict(target_pair=["left_wrist_yaw_link","other"],
        direction="wrist_to_hip",contacts=[{"point_w":[0,0,0],"normal_w":[1,0,0],
                                             "impulse_magnitude":1}]))

def test_well_formed_filtered_witness_is_only_tensor_evidence():
    row=dict(target_pair=list(PAIR),direction="hip_to_wrist",
             contacts=[dict(point_w=[.2,.3,.4],normal_w=[1.,0.,0.],
                            impulse_magnitude=2.,separation_m=-.001)])
    assert direct_tensor_pair_contact(row)

def test_bad_normal_or_point_fails_closed():
    for point,normal in (([0,0,0],[0,0,0]),([math.nan,0,0],[1,0,0]),
                         ([0,0],[1,0,0]),([0,0,0],[1,0])):
        assert not direct_tensor_pair_contact(dict(target_pair=list(PAIR),
            direction="wrist_to_hip",contacts=[dict(point_w=point,normal_w=normal,
                                                     impulse_magnitude=2.)]))

def test_raw_contact_vectors_and_segment():
    np=pytest.importorskip("numpy")
    arrays=(np.array([3.,1.]),np.array([[1,2,3],[4,5,6]]),
            np.array([[0,0,1],[0,1,0]]),np.array([-.01,.02]),
            np.array([[1]]),np.array([[0]]))
    k,events=tensor_contacts(*arrays)
    assert k==1 and len(events)==1 and events[0]["point_w"]==[1,2,3]
    with pytest.raises(ValueError,match="SEGMENT"):
        tensor_contacts(*arrays[:-2],np.array([[3]]),np.array([[0]]))

def test_nonfinite_native_contact_data_is_refused():
    np=pytest.importorskip("numpy")
    with pytest.raises(ValueError,match="NONFINITE"):
        tensor_contacts(np.array([1.]),np.array([[0.,math.nan,0.]]),
                        np.array([[1,0,0]]),np.array([0.]),
                        np.array([[1]]),np.array([[0]]))

def test_frozen_protocol_cannot_infer_native_manifold():
    import json
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/v081_frozen_rigid_pair_tensor.json").read_text())
    assert len(c["trial_order"])*len(c["mode_order"])==6
    assert c["do_not_tune_controllers"] and c["native_collider_manifold_certification"].startswith("UNKNOWN")
    assert c["hard_gates"]["instep_impulse_fraction_min"]==.95
    assert c["hard_gates"]["nonfoot_force_max_N"]==20
