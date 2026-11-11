import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_v063_post_registered import compare_frames
def test_25_field_original_preaction_comparison_rejects_mismatch():
    x=dict(time_s=.005,root_xyz=[1.,2.,3.],joint_pos=[0.]*29)
    good=compare_frames([x],[dict(x)])
    assert good["matches_at_abs_tol_1e_6"]
    y=dict(x,joint_pos=[1.]+[0.]*28)
    bad=compare_frames([x],[y])
    assert bad["mismatch_count"]>0
def test_comparator_catches_missing_original_physical_field():
    x=dict(time_s=.005,joint_vel=[1.]*29)
    y=dict(time_s=.005)
    assert compare_frames([x],[y])["mismatch_count"]>0
