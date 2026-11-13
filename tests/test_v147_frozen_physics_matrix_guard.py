"""V147 six-case original runner readiness, without starting PhysX."""
import json
from pathlib import Path
import sys

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v147_frozen_physics_matrix_guard as gate
from v147_exact_off_on_parity import frozen


def fake_root(tmp_path):
    (tmp_path/"configs").mkdir()
    (tmp_path/"reports").mkdir()
    (tmp_path/"tools").mkdir()
    rels=["data","experiments","third_party",".pretrained_checkpoints",
          "media","configs/paid_original","configs/paid_g1_main.urdf"]
    for name in rels:
        dst=tmp_path/name
        dst.symlink_to(gate.MAIN/name)
    for name in ("cmu_named_nonfoot_diagnostics_v054.json",
                 "cmu_real_120hz_10_02_physx_zero_v052.json",
                 "cmu_real_120hz_10_02_physx_v052.json",
                 "cmu_real_120hz_10_05_physx_v052.json"):
        (tmp_path/"configs"/name).write_bytes((ROOT/"configs"/name).read_bytes())
    return tmp_path


def test_six_original_policy_pair_matrix_gpu_blocked_actual():
    x=gate.build_manifest(gpu_table="242644, existing pytest, 260 MiB",
                          disk_free_bytes=30*(1024**3))
    assert x["matrix_size"]==6
    assert x["gpu_go_no_go"].startswith("BLOCKED_")
    assert x["actual_new_gpu_physx_runs"]==0
    assert x["true_native_contact_pair_truth"]=="UNKNOWN"
    assert x["native_positive_negative_clock_calibration"] is False
    modes=[r["policy"]+"/"+r["observer_mode"]
           for r in x["future_physics_experiments_not_started"]]
    assert modes==frozen()["expected_future_physx_matrix"]
    assert len({r["output_write_once"] for r in x["future_physics_experiments_not_started"]})==6
    for row in x["future_physics_experiments_not_started"]:
        assert "--eval-seed" in row["command_argv_not_executed"]
        assert "--v146-observer-mode" in row["command_argv_not_executed"]
        assert row["original_frozen_config_sha256"]


def test_zero_gpu_process_only_preflight_not_certified():
    x=gate.build_manifest(gpu_table="",disk_free_bytes=30*1024**3)
    assert x["gpu_go_no_go"].startswith("GPU_PREFLIGHT_ONLY")
    assert not x["auto_execute"]
    assert not x["off_on_real_physx_noninterference_certified"]


def test_low_disk_hard_blocks_even_if_gpu_empty():
    x=gate.build_manifest(gpu_table="",disk_free_bytes=19*1024**3)
    assert x["gpu_go_no_go"].startswith("BLOCKED_V146_DISK_BELOW")
    assert x["actual_new_gpu_physx_runs"]==0


def test_original_sha_drift_halts_future_matrix(tmp_path):
    root=fake_root(tmp_path)
    f=root/"configs/cmu_real_120hz_10_02_physx_v052.json"
    f.write_text(f.read_text()+"\n")
    with pytest.raises(ValueError,match="CONFIG_SHA"):
        gate.build_manifest(gpu_table="",disk_free_bytes=30*1024**3,root=root)


def test_never_overwrites_existing_run_results(tmp_path):
    root=fake_root(tmp_path)
    (root/"reports"/"v147_frozen_v32_off_seed1201_physx.json").write_text("protected")
    with pytest.raises(FileExistsError,match="NO_OVERWRITE"):
        gate.build_manifest(gpu_table="",disk_free_bytes=30*1024**3,root=root)
    assert (root/"reports"/"v147_frozen_v32_off_seed1201_physx.json").read_text()=="protected"


def test_missing_original_immutable_bindings_halts(tmp_path):
    root=fake_root(tmp_path)
    (root/"data").unlink()
    with pytest.raises(RuntimeError,match="ASSETS_NOT_READONLY_BOUND"):
        gate.build_manifest(gpu_table="",disk_free_bytes=30*1024**3,root=root)
