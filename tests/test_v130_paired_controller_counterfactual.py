"""v130 paired, original G1 v54 measured 29DoF/COM/force negatives."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v130_real_com_support_work as main
import v130_paired_controller_counterfactual as paired


def example(n=700,offset_index=280):
    t=np.arange(1,n+1,dtype=float)*.005
    obj={
        "t":t, "q":np.zeros((n,29)), "dq":np.zeros((n,29)),
        "com":np.tile(np.array([0.,0.,.7]),(n,1)),
        "support":np.tile(np.array([120.,80.]),(n,1)),
        "tau":np.ones((n,29)), "impact":1.57,
    }
    other={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in obj.items()}
    other["q"][offset_index:,3]=.003
    other["dq"][offset_index:,3]=.6
    other["com"][offset_index:,0]=.001
    other["support"][offset_index:,0]=130.
    return obj,other


def test_matched_pair_detects_real_change_only_after_change_frame():
    x,y=example()
    c=paired.evaluate_pair(x,y,main.frozen(),"test")
    assert c["first_q_difference_over_1e_9_rad_s"]==pytest.approx(1.405)
    assert c["first_COM_difference_over_1e_7_m_s"]==pytest.approx(1.405)
    assert c["intervals"]["pre_reference_observation"]["max_abs_29q_difference_rad"]==0
    assert c["intervals"]["pre_400ms"]["max_abs_29q_difference_rad"]==pytest.approx(.003)
    assert not c["native_contact_pair_witness"]


def test_mismatched_time_axes_are_rejected():
    x,y=example()
    y["t"][250]+=.03
    with pytest.raises(ValueError,match="SYNCHRONIZED_PHYSICS"):
        paired.evaluate_pair(x,y,main.frozen(),"test")


def test_short_recording_must_not_forge_phases():
    x,y=example(400)
    with pytest.raises(ValueError,match="SYNCHRONIZED_PHYSICS"):
        paired.evaluate_pair(x,y,main.frozen(),"test")


def test_never_compare_fictional_tail_from_early_terminated_case():
    x,y=example(700)
    y={k:(v[:570] if isinstance(v,np.ndarray) else v) for k,v in y.items()}
    c=paired.evaluate_pair(x,y,main.frozen(),"test")
    assert c["shared_original_physical_frames"]==570
    assert c["shared_last_time_s"]==pytest.approx(2.85)
    assert c["intervals"]["observed_late_recovery"]["shared_physical_frames"]==7


def test_work_signed_not_clipped_or_hidden():
    x,y=example()
    y["tau"]=-y["tau"]
    c=paired.evaluate_pair(x,y,main.frozen(),"test")
    assert c["intervals"]["post_1250ms"]["paired_robot_motor_signed_work_delta_J"]<0


def test_actual_original_physics_pairs_stay_uncertified():
    result=paired.paired_audit()
    assert result["new_physx_episodes"]==0
    assert len(result["pairs"])==2
    assert {x["control_comparison"] for x in result["pairs"]}=={
        "frozen_v32_vs_cmu_10_02","frozen_v32_vs_cmu_10_05"}
    for x in result["pairs"]:
        assert x["first_q_difference_over_1e_9_rad_s"]==pytest.approx(1.405)
        assert x["intervals"]["pre_reference_observation"]["max_abs_29q_difference_rad"]==0.0
        assert x["native_contact_pair_witness"] is False
        assert x["physical_rollout_executed_in_v130"] is False
