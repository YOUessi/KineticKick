"""Independent reproducibility assertions about 200Hz v54 observed states."""
import hashlib
from pathlib import Path
import pytest

from v155_real_contact_window_audit import analyze,WINDOW,TERMINAL
from v155_frozen_patch_campaign import run
from v155_plot_patch_and_real_window import build

def test_history_window_has_no_fabricated_physics():
    out=analyze()
    assert out["real_historical_200hz_states_verified"]==6628
    assert out["three_runs_window_states"]==993
    assert out["nominal_strike_anchor_not_contact_truth_s"]==1.570
    assert out["native_left_wrist_left_hip_manifold"]=="UNKNOWN"
    assert out["new_physics_runs"]==0
    assert all(v["actual_window_states"]==331 for v in out["runs"].values())
    assert all(v["foot_sensor_net_not_ground_grf_or_cop"] for v in out["runs"].values())
    assert out["runs"]["cmu_10_05"]["actual_early_terminal_200hz_states"]==81
    assert out["runs"]["cmu_10_05"]["last_complete_full29_t_s"]==5.14
    assert out["runs"]["cmu_10_05"]["named_sensor_5145ms_has_full29_state"] is False

def test_full_freeze_matrix_and_old_physics_provenance():
    out=run()
    assert (out["nominal_count"],out["nominal_passed"])==(48,48)
    assert (out["negative_count"],out["negative_rejected"])==(96,96)
    assert (out["baseline_count"],out["baseline_point_feasible"],out["baseline_patch_rejected"])==(8,8,8)
    assert out["verified_historical_200hz_rows"]==6628
    assert out["new_physx_episodes"]==0
    assert out["real_instep_legal_shot_success_rate"] is None

def test_plot_once_never_overwrites(tmp_path):
    root=Path("/home/you/projects/kinetickick-wt-v155/reports")
    cpu=root/"v155_frozen_patch_cpu_20261114.json"
    history=root/"v155_real_original_200hz_window_20261114.json"
    target=tmp_path/"plot"
    names=build(cpu,history,target)
    old=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names]
    with pytest.raises(FileExistsError,match="WRITE_ONCE"):
        build(cpu,history,target)
    assert old==[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names]
