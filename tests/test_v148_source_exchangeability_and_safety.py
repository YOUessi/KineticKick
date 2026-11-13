"""Adversarial CPU tests; no new GPU physics or false human force labels."""
import copy
import json
from pathlib import Path

import pytest

from v148_human_anatomical_exchangeability import (
    FREEZE_SHA, frozen, validate_cell, counts_for_label_assignments,
    validated_grid, audit, PERMS, recheck_real_history
)
from v148_independent_peak_recheck import verify_payload
from v148_plot_falsification import create

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"reports/v148_human_source_anatomical_exchangeability_locked.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(INPUT.read_text())


@pytest.fixture
def good():
    return {
        "clip":"10_02","chain":"right","anchor_shift_frames":0,
        "span_frames":2,"median_window_frames":1,"pre_window_seconds":.6,
        "peak_indices":{"pelvis":10,"right_thigh":14,"right_shank":20},
        "lead_frames":[4,6],
        "valid_fractions":{"pelvis":1.0,"right_thigh":1.0,"right_shank":.9},
        "status":"ORDER_MARGIN_PASS"
    }


def test_freeze_is_separate_from_implementation():
    cfg=frozen()
    assert cfg["parent_sha"]=="f4761baefd1cdbcf2ae951ecc97b086139818c44"
    assert FREEZE_SHA=="8da717747c86c77aca0c146c9f6c40c875adec93"
    assert cfg["data_status"]["human_exact_instep_impact"] is False


def test_full_original_720_grid_real_source_complete(payload):
    assert len(payload["source_grid_720_correlated_cells"])==720
    out=validated_grid(payload["source_grid_720_correlated_cells"],frozen())
    assert len(out)==4
    assert all(len(g)==180 for g in out.values())


def test_observed_is_one_label_assignment_not_36_players(payload):
    data=payload["analysis"]
    assert data["original_right_identity_label_score"]==237
    assert data["assignments_with_score_at_least_identity"]==1
    assert len(data["two_clips_36_aggregate_assignment_scores"])==36
    assert data["this_is_not_a_population_p_value"] is True
    assert data["two_clips_one_subject_not_36_independent_observations"] is True


def test_diagnostic_label_reassignment_requires_original_order():
    assert len(PERMS)==6
    assert PERMS[0]==(0,1,2)


def test_source_subgroup_anchor_shift_minus_12_falsifies_both_clips(payload):
    data=payload["analysis"]["by_clip_and_anatomical_side"]
    assert data["right_10_02"]["anchor_shift_control"]["-12"]["passes"]==0
    assert data["right_10_05"]["anchor_shift_control"]["-12"]["passes"]==0


def test_human_data_qualification_gate_remains_closed(payload):
    gate=payload["analysis"]["human_training_evidence_gate"]
    assert gate["qualified_for_human_to_original29_training"] is False
    assert gate["independent_subject_count"]==1
    assert gate["exact_human_right_instep_contact_truth"] is False
    assert gate["synchronized_human_ground_reaction_COP"] is False


def test_historical_real_physx_not_new_results(payload):
    hist=payload["real_physx_provenance"]
    assert hist["historical_original29_real_states_200hz"]==6628
    assert hist["historical_real_physx_episodes"]==3
    assert hist["new_physx_episodes"]==0
    assert hist["new_physx_strict_legal_instep_success_rate"] is None
    assert hist["native_wrist_hip_manifold_truth"]=="UNKNOWN"


def test_positive_original_c3d_cell(good):
    assert validate_cell(good)==((10,14,20),True)
    assert counts_for_label_assignments([good],2)==[1,0,0,0,0,0]


def test_small_positive_gap_rejects(good):
    good["peak_indices"]["right_thigh"]=11
    good["lead_frames"]=[1,9]
    good["status"]="STRICT_ORDER_SMALL_MARGIN"
    assert validate_cell(good)==((10,11,20),False)


@pytest.mark.parametrize("key,bad",[
    ("clip","10_09"),("chain","rightx"),("anchor_shift_frames",7),
    ("anchor_shift_frames",True),
])
def test_reject_unknown_unfrozen_grid_key(good,key,bad):
    good[key]=bad
    with pytest.raises(ValueError):validate_cell(good)


@pytest.mark.parametrize("new_value",[
    None,{"pelvis":10,"right_shank":20},
    {"pelvis":10,"right_thigh":True,"right_shank":20},
    {"pelvis":10,"right_thigh":-1,"right_shank":20},
    {"pelvis":10,"right_thigh":14.2,"right_shank":20},
])
def test_reject_unqualified_original_peaks(good,new_value):
    good["peak_indices"]=new_value
    if new_value is None:good["status"]="ORDER_MARGIN_PASS"
    with pytest.raises(ValueError):validate_cell(good)


@pytest.mark.parametrize("mutation",["lead", "valid", "missing", "status"])
def test_reject_invented_validity_and_margin(good,mutation):
    if mutation=="lead":good["lead_frames"]=[0,0]
    if mutation=="valid":good["valid_fractions"]["pelvis"]=.1
    if mutation=="missing":good["valid_fractions"].pop("pelvis")
    if mutation=="status":good["status"]="REVERSED_OR_TIED"
    with pytest.raises(ValueError):validate_cell(good)


def test_incomplete_source_does_not_gain_a_pass(good):
    good["peak_indices"]=None
    good["status"]="INSUFFICIENT_VALID_MARKERS"
    assert counts_for_label_assignments([good],2)==[0]*6


def test_fail_closed_missing_720_grid(payload):
    with pytest.raises(ValueError,match="INCOMPLETE_GRID"):
        validated_grid(payload["source_grid_720_correlated_cells"][:-1],frozen())


def test_fail_closed_repeated_choice(payload):
    rows=list(payload["source_grid_720_correlated_cells"])
    rows[-1]=rows[0]
    with pytest.raises(ValueError,match="DUPLICATE"):
        validated_grid(rows,frozen())


def test_independent_full_real_source_sha_recheck(payload):
    result=verify_payload(payload)
    assert result["independent_anatomical_label_score"]==237
    assert result["independent_36_label_assignments_at_least_identity"]==1
    assert result["new_physx_episodes"]==0


@pytest.mark.parametrize("mutation",[
    "true_professional","true_human_contact","true_grf","two_subjects",
    "true_motor","raw_grid_truncated","duplicated_grid","faked_margin",
    "false_experimental_pass","native_pair_fabricated","new_physx_faked",
    "wrong_original_physx_hash","wrong_side","wrong_clip"
])
def test_independent_rejects_forged_evidence(payload,mutation):
    obj=copy.deepcopy(payload)
    if mutation=="true_professional":obj["provenance"]["professional_skill_certified"]=True
    elif mutation=="true_human_contact":obj["provenance"]["human_exact_impact_time_measured"]=True
    elif mutation=="true_grf":obj["provenance"]["human_ground_force_plate_available"]=True
    elif mutation=="two_subjects":obj["provenance"]["weak_event_human_subject_count"]=2
    elif mutation=="true_motor":obj["is_offline_research_not_robot_actuation"]=False
    elif mutation=="raw_grid_truncated":obj["source_grid_720_correlated_cells"].pop()
    elif mutation=="duplicated_grid":
        obj["source_grid_720_correlated_cells"][-1]=obj["source_grid_720_correlated_cells"][0]
    elif mutation=="faked_margin":
        obj["source_grid_720_correlated_cells"][0]["status"]="ORDER_MARGIN_PASS"
        obj["source_grid_720_correlated_cells"][0]["peak_indices"]=None
    elif mutation=="false_experimental_pass":
        obj["analysis"]["by_clip_and_anatomical_side"]["right_10_05"]["pass_count"]=180
    elif mutation=="native_pair_fabricated":
        obj["real_physx_provenance"]["native_wrist_hip_manifold_truth"]="CERTIFIED"
    elif mutation=="new_physx_faked":obj["real_physx_provenance"]["new_physx_episodes"]=1
    elif mutation=="wrong_original_physx_hash":
        obj["real_physx_provenance"]["sha256_each_original_case"]["cmu_10_05"]="0"*64
    elif mutation=="wrong_side":obj["source_grid_720_correlated_cells"][0]["chain"]="other"
    elif mutation=="wrong_clip":obj["source_grid_720_correlated_cells"][0]["clip"]="10_99"
    with pytest.raises((ValueError,KeyError,TypeError)):
        verify_payload(obj)


def test_falsification_figure_created_once_with_no_overwrite(payload,tmp_path):
    dest=tmp_path/"source.png"
    create(INPUT,dest)
    assert dest.is_file() and dest.stat().st_size>10000
    with pytest.raises(FileExistsError):
        create(INPUT,dest)


def test_real_history_verifies_source_hash_and_no_physx():
    result=recheck_real_history(frozen())
    assert result["new_physx_episodes"]==0
    assert result["historical_original29_real_states_200hz"]==6628
