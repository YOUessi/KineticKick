"""CPU-only accuracy contracts; never count mocked cases as original PhysX."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from audit_massfree_v062 import real_200hz,compare_29dof_sample_alignment,REFERENCES

def sample(t,terminal=False):
    return dict(time_s=t,terminated=terminal,
                joint_vel=[i*.1 for i in range(29)],joint_pos=[i*.01 for i in range(29)])

def obs(t,vel_delta=0,pos_delta=0):
    return dict(t=t,joint_vel_rad_s=[i*.1+vel_delta for i in range(29)],
                joint_pos_rad=[i*.01+pos_delta for i in range(29)])

def test_exact_state_alignment_detects_sim_clock_offset():
    frame=[sample(.005*i) for i in range(1,36)]
    observer=[obs(f["time_s"]+.01) for f in frame]
    x=compare_29dof_sample_alignment(frame,observer)
    assert x["joint_velocity_max_abs_diff"]==0
    assert x["observer_clock_minus_actual_physics_seconds"]==.01
    assert x["observer_time_must_not_be_used_as_measured_state_time"]

@pytest.mark.parametrize("edit",["velocity","position","drop","jitter"])
def test_alignment_fails_closed_on_observer_data_mismatch(edit):
    frame=[sample(.005*i) for i in range(1,36)]
    observer=[obs(f["time_s"]+.01) for f in frame]
    if edit=="velocity":observer[7]["joint_vel_rad_s"][5]+=1
    elif edit=="position":observer[7]["joint_pos_rad"][5]+=.1
    elif edit=="drop":observer.pop()
    else:observer[10]["t"]+=.03
    with pytest.raises(ValueError):compare_29dof_sample_alignment(frame,observer)

def test_non200hz_rejected():
    frames=[sample(.005*i) for i in range(1,36)]
    frames[8]["time_s"]+=.002
    with pytest.raises(ValueError,match="Non-200Hz"):real_200hz(frames)

def test_terminal_microstep_excluded():
    frames=[sample(.005*i) for i in range(1,36)]
    frames.append(sample(.175001,terminal=True))
    x,terminal=real_200hz(frames)
    assert terminal and len(x)==35

def test_v062_frozen_same_backend_reference_matrix():
    assert len(REFERENCES)==5
    assert all("v060_" in v for v in REFERENCES.values())
    assert len([k for k in REFERENCES if "cuda" in k])==3

def test_real_audit_verified_existing_eight_write_once_cases():
    # Runtime-integrated audit verifies original JSONs are present; this test
    # runs only on evidence-rich Tang, not in a clean GitHub checkout.
    root=Path(__file__).resolve().parents[1]
    if not (root/"reports/v062_batch_progress.json").exists():
        pytest.skip("G1 physical raw data is correctly local-only")
    from audit_massfree_v062 import audit
    a=audit(root)
    assert a["real_PhysX_attempts"]==8
    assert a["valid_original_G1"]==5
    assert a["unscorable"]==3
    assert a["strict_PASS_of_valid"]==2
    assert len(a["parity_all_25_original_physics_fields"])==5
    assert all(v["max_abs_numeric_difference"]==0 for v in
               a["parity_all_25_original_physics_fields"].values())
