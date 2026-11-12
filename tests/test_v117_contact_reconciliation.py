from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v117_contact_evidence_reconciliation import inspect_previous_positive,inspect_g1_observer,audit


def test_real_isolated_positive_and_separated_negative_hash_verified():
    obj=inspect_previous_positive()
    assert obj["certified_isolated_physx_total"]==3
    assert obj["genuine_tensor_contact_events"]==113
    assert len(obj["controls"])==3
    assert not obj["every_pair_native_manifold_certified"]
    assert obj["controls"][1]["tensor_peak_force_N"]>600
    assert obj["controls"][2]["tensor_observed_contact_rows"]==0


def test_immutable_positive_provenance_hash_rejected(tmp_path):
    d=tmp_path/"reports"
    d.mkdir()
    raw={"version":"v0.1.114","actual_certified_physx_runs":3,
         "cases":[{"name":"forged","output_sha":"0"*64,
                   "actual_200hz_steps":180,"tensor_contacts":113,
                   "positive_tensor_force":1,"native_pair_certified":False}]}
    (d/"v114_tensor_positive_batch.audit.json").write_text(json.dumps(raw))
    (d/"v114_forged.json").write_text("{}")
    with pytest.raises(ValueError,match="SOURCE_HASH_MISMATCH"):
        inspect_previous_positive(tmp_path)


def test_true_g1_tensor_not_native_collider_manifold():
    rows=inspect_g1_observer()
    assert rows
    assert {x["episode"] for x in rows}=={"AJ_v32_on","AL_v32_on"}
    assert all(x["both_source_filter_paths_present"] for x in rows)
    assert all(not x["native_identity_and_geometry_certified"] for x in rows)
    assert all(not x["zero_filtered_force_implies_no_collision"] for x in rows)
    out=audit()
    assert not out["native_actor0_actor1_collider0_collider1_contact_geometry_certified"]
    assert not out["learned_motor_action_cleared_by_contact_evidence"]
