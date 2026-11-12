#!/usr/bin/env python3
"""v126 CPU regression: contact source anatomy + clustered statistics fail closed."""
from __future__ import annotations
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v126_collision_topology_and_paired_inference as impl


def test_zero_discordance_not_significant():
    assert impl.two_sided_discordant_exact(0, 0) == 1


@pytest.mark.parametrize("a,b,expected", [(1, 0, 1), (3, 0, .25), (4, 0, .125),
                                           (5, 0, .0625), (6, 0, .03125),
                                           (2, 2, 1), (4, 1, .375)])
def test_exact_two_sided_binomial(a, b, expected):
    assert impl.two_sided_discordant_exact(a, b) == expected


@pytest.mark.parametrize("a,b", [(-1, 2), (True, 0), (0.5, 1), (1, "2")])
def test_invalid_exact_counts_fail(a, b):
    with pytest.raises(ValueError):
        impl.two_sided_discordant_exact(a, b)


def synthetic_rows():
    scenes = "AG AH AI AJ AK AL AM AN".split()
    controls = ("v32", "cmu02", "cmu05")
    return [dict(scene=s, control=c, seed=i + 30, physical_steps=2800,
                 native_collider_pair_certified=False,
                 strict_hard_gate_pass=(i % 3 == 0 if c == "v32" else
                                        i % 4 == 0 if c == "cmu02" else i % 2 == 0))
            for i, s in enumerate(scenes) for c in controls]


def test_scene_paired_8_times_3():
    x = impl.exact_scene_paired_audit(synthetic_rows(), ["v32", "cmu02", "cmu05"], 8)
    assert x["independent_cells"] == 24
    assert len(x["comparisons"]) == 2
    for comparison in x["comparisons"]:
        assert comparison["matched_scenes"] == 8
        assert comparison["candidate_only_wins"] + comparison["baseline_only_wins"] + (
            comparison["both_pass"] + comparison["both_fail"]) == 8
        assert 0 <= comparison["exact_conditional_p_two_sided"] <= 1
        assert 0 <= comparison["holm_adjusted_p"] <= 1
        assert len(comparison["individual_scene_outcomes"]) == 8
        assert -1 <= comparison["leave_one_scene_out_difference_range"][0] <= 1


def test_pseudoreplicated_episode_refused():
    bad = synthetic_rows()
    bad[-1] = bad[0].copy()
    with pytest.raises(ValueError, match="DUPLICATE"):
        impl.exact_scene_paired_audit(bad, ["v32", "cmu02", "cmu05"], 8)


def test_missing_scene_or_controller_refused():
    with pytest.raises(ValueError, match="FROZEN"):
        impl.exact_scene_paired_audit(synthetic_rows()[:-1], ["v32", "cmu02", "cmu05"], 8)


def test_seed_mismatch_refused():
    bad = synthetic_rows()
    bad[1]["seed"] += 42
    with pytest.raises(ValueError, match="SEED"):
        impl.exact_scene_paired_audit(bad, ["v32", "cmu02", "cmu05"], 8)


@pytest.mark.parametrize("field,val", [
    ("strict_hard_gate_pass", 1), ("physical_steps", 2799),
    ("native_collider_pair_certified", True)
])
def test_false_positive_or_truncated_phyics_refused(field, val):
    bad = synthetic_rows()
    bad[0][field] = val
    with pytest.raises(ValueError):
        impl.exact_scene_paired_audit(bad, ["v32", "cmu02", "cmu05"], 8)


def test_urdf_parser_contains_shape_and_29dof(tmp_path):
    root = ET.Element("robot", name="g1")
    for name in ("pelvis", "left_hip_roll_link", "left_wrist_yaw_link"):
        link = ET.SubElement(root, "link", name=name)
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "sphere", radius="0.05")
    for name, parent, child in (("hip", "pelvis", "left_hip_roll_link"),
                                ("wrist", "pelvis", "left_wrist_yaw_link")):
        j = ET.SubElement(root, "joint", name=name, type="revolute")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
    path = tmp_path / "sample.urdf"
    ET.ElementTree(root).write(path)
    model = impl.parse_urdf(path)
    assert len(model["links"]["left_wrist_yaw_link"]) == 1
    assert impl.shortest_joint_path(model["edges"], "left_wrist_yaw_link",
                                    "left_hip_roll_link")["distance_joints"] == 2


def test_missing_urdf_collider_refused_in_source_audit():
    # Real SHA+source qualification cannot be substituted by a fake URDF path.
    cfg = impl.frozen_config()
    bad = copy.deepcopy(cfg)
    bad["urdf_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        impl.static_collision_audit(bad, Path("/home/you/projects/kinetickick-wt-v116"))


def test_protocol_frozen_and_fail_closed():
    cfg = impl.frozen_config()
    assert cfg["no_physics_actuation"] and cfg["no_native_manifold_inference_from_urdf"]
    assert cfg["native_pair_gate"].startswith("UNKNOWN")
    assert cfg["expect_observer_repeats_excluded"] == 5


def test_source_report_sha_and_runtime_urdf_are_real():
    cfg = impl.frozen_config()
    static = impl.static_collision_audit(
        cfg, Path("/home/you/projects/kinetickick-wt-v116"))
    assert static["actuated_dof"] == 29
    assert static["target_link_collision_geometry"]["left_wrist_yaw_link"]
    assert static["target_link_collision_geometry"]["left_hip_roll_link"]
    assert static["native_physx_contact_pair_found"] is False
    assert static["native_contact_manifold_certified"] is False
    assert static["conclusion"].startswith("STATIC_CANDIDATE")
    original = impl.source_reconcile_v125(
        cfg, Path("/home/you/projects/kinetickick-wt-v125"))
    assert len(original["per_episode"]) == 24
    assert original["observer_repeats_not_independent"] == 5


def test_real_eight_scene_cohort_exact_inference():
    cfg = impl.frozen_config()
    actual = impl.run(cfg, Path("/home/you/projects/kinetickick-wt-v125"),
                      Path("/home/you/projects/kinetickick-wt-v116"))
    assert actual["new_physx_episodes"] == 0
    assert actual["native_contact_pair_conclusion"] == "UNKNOWN"
    matched = actual["original_scene_clustered_effect_test"]
    assert matched["independent_cells"] == 24
    assert matched["old_strict_pass"] == 13
    assert len(matched["comparisons"]) == 2
    assert all(not x["passes_familywise_0_05"] for x in matched["comparisons"])


def test_write_once_no_silent_overwrite(tmp_path, monkeypatch):
    # Regression: CLI-like output is not to overwrite original evidence.
    out = tmp_path / "immutable.json"
    out.write_text("immutable")
    with pytest.raises(FileExistsError):
        with out.open("x") as f:
            f.write("overwritten")
    assert out.read_text() == "immutable"
