"""v146 true raw-data independent audit and immutable local source binding."""
from pathlib import Path
import sys
import json
import hashlib
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v146_bind_local_assets as bind
import v146_independent_real_dynamics_check as independent


def test_existing_real_original_independent_all_6628():
    path=ROOT/"reports/v146_original_real_6628_coupled_locked.json"
    assert path.exists()
    result=independent.independent_check(path)
    assert result["independent_actual_original_200hz_rows"]==6628
    assert result["independent_no_gpu_reruns"]
    assert result["no_real_contact_manifold_claimed"]
    assert all(c["whole_body_q_dq_tau_com_ball_foot_net_exact_per_row"]
               for c in result["source_conditions"].values())


def test_real_independent_orig29_qdot_no_overspeed_at_center():
    result=json.loads((ROOT/"reports/v146_independent_real_coupled_final.json").read_text())
    assert result["independent_actual_original_200hz_rows"]==6628
    assert all(c["independent_qdot_exceeding_steps"]==0
               for c in result["source_conditions"].values())
    assert all(c["pre_real_200hz_steps"]==80 and c["post_real_200hz_steps"]==250
               for c in result["source_conditions"].values())


@pytest.mark.parametrize("name",["frozen_v32__cmu_10_02",
                                "frozen_v32__cmu_10_05",
                                "cmu_10_02__cmu_10_05"])
def test_independent_real_first_physics_divergence_at_1p405s(name):
    result=json.loads((ROOT/"reports/v146_independent_real_coupled_final.json").read_text())
    x=result["independent_original_center_pair_differences"][name]
    assert x["first_divergent_true_200hz_sample"]==281
    assert x["first_divergent_time_s"]==1.405


def test_real_physics_figures_are_local_and_source_grounded():
    import subprocess
    for suffix in ("full14_dynamics","last_contact_evidence_gap",
                   "precontact_body_activation"):
        path=ROOT/"reports"/("v146_actual_physx_"+suffix+".png")
        assert path.is_file()
        assert path.stat().st_size>20_000
        r=subprocess.check_output(["git","check-ignore",str(path)],cwd=ROOT,text=True)
        assert str(path.relative_to(ROOT)) in r


def test_original_assets_are_gitignored_links_not_copies():
    result=bind.check_bindings()
    assert result["only_gitignored_paths"]
    assert not result["bound_without_copy_or_delete"]
    assert all(x=="EXISTING_VERIFIED_LINK" for x in result["assets"].values())


def test_bind_without_write_in_tempdir(tmp_path):
    src=tmp_path/"source"
    dst=tmp_path/"target"
    src.mkdir();dst.mkdir()
    (src/"experiments").mkdir()
    (src/"third_party").mkdir()
    (src/"configs").mkdir()
    (dst/"configs").mkdir()
    (src/"experiments"/"unchanged.txt").write_text("DO NOT EDIT")
    check=bind.check_bindings(source=src,target=dst)
    assert check["assets"]["experiments"]=="NEEDS_READONLY_SYMLINK"
    assert not (dst/"experiments").exists()
    first=bind.check_bindings(source=src,target=dst,bind=True)
    assert first["assets"]["experiments"]=="CREATED_GITIGNORED_LINK_ONLY"
    assert (dst/"experiments"/"unchanged.txt").read_text()=="DO NOT EDIT"
    second=bind.check_bindings(source=src,target=dst,bind=True)
    assert second["assets"]["experiments"]=="EXISTING_VERIFIED_LINK"
    assert (src/"experiments"/"unchanged.txt").read_text()=="DO NOT EDIT"


def test_wrong_existing_symlink_is_rejected_without_overwrite(tmp_path):
    src=tmp_path/"source";dst=tmp_path/"target";wrong=tmp_path/"wrong"
    src.mkdir();dst.mkdir();wrong.mkdir()
    (src/"experiments").mkdir()
    (dst/"experiments").symlink_to(wrong,target_is_directory=True)
    with pytest.raises(ValueError,match="NON_ORIGINAL_ASSET_SYMLINK"):
        bind.check_bindings(source=src,target=dst,bind=True)
    assert (dst/"experiments").resolve()==wrong


def test_existing_regular_content_cannot_be_replaced(tmp_path):
    src=tmp_path/"source";dst=tmp_path/"target"
    src.mkdir();dst.mkdir()
    (src/"data").mkdir();(dst/"data").mkdir()
    (dst/"data"/"sentinel").write_text("HOLD")
    with pytest.raises(ValueError,match="LOCAL_PATH_ALREADY_OCCUPIED"):
        bind.check_bindings(source=src,target=dst,bind=True)
    assert (dst/"data"/"sentinel").read_text()=="HOLD"


def test_same_source_and_dest_is_rejected():
    with pytest.raises(ValueError,match="CANNOT_BIND_ORIGINAL"):
        bind.check_bindings(source=ROOT,target=ROOT)


def test_independent_provenance_drift_never_makes_fake_truth(tmp_path):
    original=json.loads((ROOT/"reports/v146_original_real_6628_coupled_locked.json").read_text())
    original["cases"]["cmu_10_05"]["full_real_200hz_coupled_rows"][0]["q29"][0]+=0.000001
    path=tmp_path/"tampered_derived.json"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match="REAL_Q_DQ_MUTATED"):
        independent.independent_check(path)


def test_native_true_contact_and_grf_remain_unknown():
    main=json.loads((ROOT/"reports/v146_original_real_6628_coupled_locked.json").read_text())
    assert main["native_wrist_hip_contact_manifold_truth"]=="UNKNOWN"
    assert main["left_right_net_foot_forces_not_individual_ground_reaction_forces"]
    assert main["new_policy_success_rate"] is None
