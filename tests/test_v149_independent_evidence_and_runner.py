"""Independent v149 matrix validation, source immutability and negative controls."""
from __future__ import annotations

import copy
import itertools
from functools import lru_cache
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import v149_phase_matrix as matrix
import v149_independent_phase_audit as independent
from v149_native_callback_capture import V149CopyOnlyNativeCapture, frozen
from v146_native_original_bridge import gpu_preflight


@lru_cache(maxsize=1)
def _cached_document():
    return matrix.run_all()

def actual_document():
    return copy.deepcopy(_cached_document())


def test_full_matrix_independent_review_and_historical_sha():
    data = actual_document()
    result = independent.independent_review(data)
    assert result["synthetic_cells_independently_checked"] == 48
    assert result["valid_vec3_copies_preserved"] == 24
    assert result["malformed_scalar_rejections"] == 24
    assert result["historical_sha_verified_200hz_rows_referenced"] == 6628
    assert not result["native_pair_manifold_certified"]
    assert data["new_real_physx_episodes"] == 0


@pytest.mark.parametrize("mutator", (
    lambda x: x["synthetic_cells"].pop(),
    lambda x: x["synthetic_cells"].append(copy.deepcopy(x["synthetic_cells"][0])),
    lambda x: x["synthetic_cells"][0].update({"source_is_runtime":True}),
    lambda x: x["synthetic_cells"][0].update({"same_native_physics_tick_calibrated":True}),
    lambda x: x["synthetic_cells"][0].update({"native_wrist_hip_contact_certified":True}),
    lambda x: x["synthetic_cells"][0].update({"strict_kick_success":True}),
    lambda x: x["synthetic_cells"][0].update({"callback_invocations":2}),
    lambda x: x["synthetic_cells"][0].update({"valid_packets_preserved":3}),
    lambda x: x["synthetic_cells"][0].update({"phase":"INSIDE_SCENE_UPDATE"}),
    lambda x: x["synthetic_cells"][0].update({"schema_eligible_in_fixture":True}),
    lambda x: x["synthetic_cells"][0].update({"invalid_callback_flagged":True}),
    lambda x: x.update({"new_real_physx_episodes":1}),
    lambda x: x.update({"runtime_physx_wrist_hip_pair_truth":"CERTIFIED"}),
    lambda x: x["historical_real_physx_not_rerun"].update({"historical_physics_5ms_frames":6629}),
))
def test_fail_closed_adversarial_matrix_mutations(mutator):
    x = actual_document()
    mutator(x)
    with pytest.raises(ValueError):
        independent.independent_review(x)


def test_original_v080_immutable_and_v149_runner_only_observer_change():
    old=(ROOT/"tools/v146_run_original_with_readonly.py").read_text()
    new=(ROOT/"tools/v149_run_original_anyphase_readonly.py").read_text()
    assert "v149_native_callback_capture" in new
    assert "V149CopyOnlyNativeCapture" in new
    assert "actor.v149.mark_terminal()" in new
    assert "gym.make=make" in new
    assert "base.run=run_after_isaac" in new
    assert "LOCK_EX|fcntl.LOCK_NB" in new
    assert "gpu_preflight()" in new
    assert "native_update(dt)" in new
    assert "self.task.after_physics(dt)" in new
    assert "--v149-observer-mode" in new
    assert old != new
    # Snapshot main v080 original source at the frozen v148 parent.
    import subprocess
    frozen_old=subprocess.check_output([
        "git","-C",str(ROOT),"show",
        "4505e095ec5e13e9226d35adcfb9ba368e053156:tools/v080_run_poll_strike.py"
    ])
    assert frozen_old==(ROOT/"tools/v080_run_poll_strike.py").read_bytes()


def test_original_gpu_mutex_and_min_disk_gate_protect_other_work():
    with pytest.raises(RuntimeError, match="FOREIGN_GPU"):
        gpu_preflight(gpu_table="242644, python, 260 MiB",
                      disk_free_bytes=30*1024**3)
    with pytest.raises(RuntimeError, match="DISK_BELOW"):
        gpu_preflight(gpu_table="",disk_free_bytes=19*1024**3)


def test_independent_review_rejects_false_event_silence_as_physics_negative():
    doc=actual_document()
    c=doc["synthetic_cells"][0]
    assert not c["schema_eligible_in_fixture"]
    assert c["native_wrist_hip_contact_certified"] is False
    assert doc["runtime_physx_wrist_hip_pair_truth"]=="UNKNOWN"
    assert independent.independent_review(doc)["runtime_positive_negative_controls_still_needed"]


def test_isolated_case_survives_callback_after_nominal_termination():
    row=matrix.run_case("cmu_10_05","AFTER_TERMINAL_SCENE_UPDATE",
                        "VECTOR_CONTACT",True)
    assert row["valid_packets_preserved"]==1
    assert row["step_candidates_unattested"]==[1,2]
    assert not row["same_native_physics_tick_calibrated"]


def test_v149_runner_is_exact_observer_only_transform_of_v146():
    a=(ROOT/"tools/v146_run_original_with_readonly.py").read_text()
    b=(ROOT/"tools/v149_run_original_anyphase_readonly.py").read_text()
    transforms={
        "--v146-observer-mode":"--v149-observer-mode",
        "--v146-sidecar":"--v149-sidecar",
        "v146_observer_mode":"v149_observer_mode",
        "v146_sidecar":"v149_sidecar",
        "v146_copy_only_diagnostics":"v149_copy_only_diagnostics",
        "v146_native_original_bridge":"v149_native_callback_capture",
        "OriginalG1ReadOnlyCapture":"V149CopyOnlyNativeCapture",
        "V146_":"V149_",
        "self.v146":"self.v149",
        "actor.v146":"actor.v149",
    }
    for source,target in transforms.items():
        assert source in a
        a=a.replace(source,target)
    a=a.replace("actor.v149.persist_once(a.v149_sidecar)",
                "actor.v149.mark_terminal()\n                actor.v149.persist_once(a.v149_sidecar)")
    assert a==b, "Native runner may change only copied observer/CLI, not controller physics"


def test_v149_plot_is_reproducible_and_write_once(tmp_path):
    from v149_plot_callback_hazard import plot
    source=actual_document()
    output=tmp_path/"synthetic_phase.png"
    info=plot(source,output)
    assert info=={
        "expected_old_bracket_retained_synthetic":6,
        "new_anyphase_copied_synthetic":24,
        "not_native_runtime":True,
    }
    assert output.stat().st_size>2000
    with pytest.raises(FileExistsError):
        plot(source,output)
