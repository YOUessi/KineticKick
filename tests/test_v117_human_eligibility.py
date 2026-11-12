from pathlib import Path
import copy
import sys

import ezc3d
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v117_human_marker_temporal_eligibility import (
    REQUIRED,derive_marker_kinematics,inspect)

ROOT=Path(__file__).resolve().parents[1]


def test_real_sources_have_measured_120hz_markers_but_no_force():
    result=inspect()
    assert result["unique_real_c3d_trials"]==6
    assert result["subjects"]==2
    assert result["source_marker_frames"]==3358
    assert result["synchronized_grf_cop_trials"]==0
    assert result["elite_instep_collision_impulses_certified"]==0
    assert result["supervised_real_foot_ball_contact_labels"]==0
    assert len(result["true_kinematic_markers_in_qualification"])==12
    for item in result["per_trial"]:
        assert len(item["source_sha256"])==64
        assert not item["impact_time_known"]
        assert not item["biological_force_cop_or_ball_impulse_available"]
        assert item["fraction_frames_with_all_required_markers"]>.90
        for k,v in item["derived_kinematics"].items():
            assert np.isfinite(v["peak_absolute"])
            assert v["peak_absolute"]>=v["median_absolute"]>=0
            assert 0<=v["time_of_peak_absolute_s"]<=item["n_frames"]/120


def test_injected_unqualified_units_rejected():
    c=ezc3d.c3d(str(ROOT/
        "data/research_human_kicks/cmu_soccer_120hz_v050/raw/10/10_02.c3d"))
    c["parameters"]["POINT"]["UNITS"]["value"]=["cm"]
    with pytest.raises(ValueError,match="C3D_UNITS"):
        derive_marker_kinematics(c)


def test_missing_leg_or_arm_marker_rejected():
    c=ezc3d.c3d(str(ROOT/
        "data/research_human_kicks/cmu_soccer_120hz_v050/raw/10/10_02.c3d"))
    labels=c["parameters"]["POINT"]["LABELS"]["value"]
    labels[labels.index("Justin:RWRA")]="NO_ACTUAL_WRIST_MARKER"
    with pytest.raises(ValueError,match="AMBIGUOUS_OR_MISSING"):
        derive_marker_kinematics(c)
