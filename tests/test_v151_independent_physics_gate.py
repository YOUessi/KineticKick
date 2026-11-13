"""Independent original-PhysX evidence and report provenance tests for v151."""
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v151_independent_historical_reaudit as audit
import v151_native_campaign as campaign
import v151_native_observability_ledger as native
import v151_off_on_exact_audit as parity
from test_v151_native_campaign_and_adversarial_parity import fake_sidecar


def test_real_v53_v54_source_locked_gate_audit():
    result=audit.independently_reaudit()
    assert result["historical_v53_physx_episodes"]==12
    assert result["historical_v54_physx_episodes"]==3
    assert result["historical_v54_physics_200hz_rows"]==6628
    assert result["excluded_terminal_duplicate_notes"]==1
    assert result["new_gpu_physx_rollouts"]==0
    for policy in ("frozen_v32","cmu_10_02","cmu_10_05"):
        r=result["historical_v53_strict_by_policy"][policy]
        assert (r["success"],r["trials"])==(0,4)
        assert r["failed_gate_counts"].get("legal_right_instep_contact",0)>0


def test_independent_source_audit_authenticates_all_three_original_json():
    for r in audit.independently_reaudit()["historical_v54_rows"]:
        assert len(r["source_sha256"])==64
        assert r["true_left_wrist_hip_native_manifold"]=="UNKNOWN"
        assert r["historical_original_steps"] in (1028,2800)


def test_cpu_only_readout_cannot_self_attest_any_runtime():
    a=fake_sidecar("on")
    b=native.classify_sidecar(a)
    assert not b["isolated_native_positive_control_proven"]
    assert not b["isolated_native_negative_control_proven"]
    assert not b["read_only_off_on_original_state_parity_proven"]


def test_input_parity_matches_v134_independent_exact_equality():
    import v134_native_contact_vector_gate as reference
    from v146_native_original_bridge import PHYSICS_FIELDS
    off=fake_sidecar("off")["original29_5ms_state_rows"]
    on=fake_sidecar("on")["original29_5ms_state_rows"]
    a=reference.exact_off_on_physics_equality(off,on,required_fields=PHYSICS_FIELDS)
    # Our own comparator requires complete historical length; independent method
    # accepts 2-row CPU fixture but equally refuses to certify runtime.
    assert a["exact_common_physics_state_equality"]
    assert a["real_source_provenance_certified"] is False
    on[0]["q"][0]+=0.001
    c=reference.exact_off_on_physics_equality(off,on,required_fields=PHYSICS_FIELDS)
    assert not c["exact_common_physics_state_equality"]


@pytest.mark.parametrize("case",["frozen_v32","cmu_10_02","cmu_10_05"])
def test_v151_has_no_new_real_physx_success_claims(case):
    p=campaign.make_campaign()
    pair=[x for x in p["runs"] if x["policy"]==case]
    assert len(pair)==2
    assert all(x["new_real_physx_episode"] is False for x in pair)
    assert all(x["status"]=="PLANNED_NOT_EXECUTED" for x in pair)
    assert p["strict_instep_physical_success"] is None


def test_source_locked_failure_plots_are_genuine_v53_v54_not_new_physx(tmp_path):
    import v151_plot_real_failure_gates as plots
    result=plots.create(tmp_path)
    assert result["count_matrix"]==[[4,3,2,0,2,1,1],[4,3,2,1,1,0,0],[4,3,2,0,2,1,1]]
    assert result["last_physics_time_s"]==5.14
    assert result["excluded_nonphysics_terminal_notes"]==1
    assert result["new_physx_episodes"]==0
    assert result["native_wrist_hip_pair"]=="UNKNOWN"
    for path in result["paths"]:
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError,match="WRITE_ONCE"):
        plots.create(tmp_path)


def test_historical_audit_never_invents_human_contact_or_pro_style():
    info=audit.independently_reaudit()
    assert info["simultaneous_human_grf_true_instep_supervision"] is False
    assert info["true_contact_collider_pair"]=="UNKNOWN"
    assert info["strict_new_policy_success"] is None
