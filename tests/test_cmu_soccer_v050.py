"""v50 120Hz actual CMU human soccer 3D+ball optical track research tests.

Important: ball WEP triad centroid is NOT calibrated ball center. 2 strong
outgoing-ball tracks are launch onset PROXIES, not validated instep impact.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_cmu_soccer_v050 import (
 valid_xyz,contiguous_high_speed,pose_semantics,parse_amc,LANDMARKS,RELATIONS,RAW)
MANIFEST=ROOT/RAW/"original_source_manifest.json"
STUDY=ROOT/"configs/cmu_soccer_quality_v050.json"
SUMMARY=ROOT/"data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/quality_audit.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_never_fill_marker_dropout_and_never_label_true_impact_from_speed():
    pos=np.zeros((20,3),dtype=float)
    pos[:,0]=np.arange(20)/120.
    pos[5:7]=np.nan
    valid=np.isfinite(pos).all(axis=1)
    speeds,runs=contiguous_high_speed(pos,valid,120,threshold=.5)
    assert np.isnan(speeds[0])
    assert np.isnan(speeds[5]) and np.isnan(speeds[6]) and np.isnan(speeds[7])
    assert sum(len(x) for x in runs)==(20-1-3)
    assert valid_xyz(np.asarray([[1.,2.,np.nan],[1.,1.,1.],[0.,0.,0.]] )).tolist()==[False,True,False]

def test_semantic_human_joint_3d_input_not_29_G1_angles():
    rng=np.random.default_rng(5010)
    h=rng.normal(size=(24,len(LANDMARKS),3))*.4
    out=pose_semantics(h)
    assert out.shape==(24,8)
    assert np.isfinite(out).all()
    assert np.abs(out).max()<=1.
    with pytest.raises(ValueError):
        pose_semantics(h[:,:,:2])

@pytest.mark.skipif(not MANIFEST.exists(),reason="14 original CMU C3D/ASF/AMC files are only on Tang; not in public Git clone")
def test_research_source_manifest_14_original_files_frozen():
    c=json.loads(STUDY.read_text())
    assert c["impact_truth_not_available"] and not c["real_foot_ball_collision_measured_by_force_plate"]
    assert not c["human_higher_proximal_distal_energy_transfer_not_calibrated"] is False
    assert sha(MANIFEST)==c["original_source_manifest_sha256"]
    original=json.loads(MANIFEST.read_text())
    assert len(original["files"])==14
    for e in original["files"]:
        f=ROOT/e["path"]
        assert sha(f)==e["sha256"]
    for p,h in c["source_sha256"].items():
        assert sha(ROOT/p)==h,p

def test_six_original_amc_exactly_same_frames_as_3d_marker_C3D():
    if not SUMMARY.exists():pytest.skip("CMU source not mounted in public clone")
    summary=json.loads(SUMMARY.read_text())
    for r in summary["motions"]:
        amc=ROOT/RAW/"raw"/r["subject"]/(r["id"]+".amc")
        f=parse_amc(amc)
        assert len(f)==r["original_C3D_frames"]==r["original_AMC_frames"]
        assert len(f)>100
        assert f[0]["index"]==1 and f[-1]["index"]==len(f)

@pytest.mark.skipif(not SUMMARY.exists(),reason="Original CMU 120Hz research data stored on Tang")
def test_data_qualification_preserves_partial_ball_visibility_and_impact_uncertainty():
    r=json.loads(SUMMARY.read_text())
    assert r["original_human_soccer_kick_trial_count"]==6
    assert r["original_human_3d_subject_count"]==2
    assert r["original_human_C3D_frame_total"]==3358
    assert r["original_human_pose_prior_pilot_eligible_count"]==6
    assert r["original_fast_ball_marker_motion_sample_count"]==2
    assert r["accurate_verified_human_impact_time_truth_sample_count"]==0
    assert r["accurate_instestep_contact_force_sample_count"]==0
    assert r["elite_professional_biomechanics_groundtruth_count"]==0
    assert r["no_missing_ball_track_interpolation"]
    q={x["id"]:x for x in r["motions"]}
    assert set(q)=={"10_01","10_02","10_03","10_05","10_06","11_01"}
    assert q["10_02"]["inferred_event"]["high_speed_contiguous_raw_120hz_intervals"]>=20
    assert q["10_05"]["inferred_event"]["high_speed_contiguous_raw_120hz_intervals"]>=12
    assert q["10_03"]["inferred_event"]["observed_high_speed_120hz_intervals"]==2
    for trial in r["motions"]:
        assert trial["human_primary_marker_min_valid_fraction"]>=.99
        assert trial["original_3D_rate_hz"]==120.
        assert trial["original_three_ball_markers_all_valid_fraction"]<1.
        assert trial["verified_impact_s"] is None
        assert not trial["real_instestep_force_and_contact_truth_qualified"]
        if trial["inferred_event"]:
            assert trial["inferred_event"].get("verified_true_foot_ball_contact") is False

@pytest.mark.skipif(not SUMMARY.exists(),reason="CMU local original data not available")
def test_real_3d_mocap_physical_scale_and_correct_nan_gap_no_fake_ball_velocity():
    summary=json.loads(SUMMARY.read_text())
    for row in summary["motions"]:
        f=ROOT/"data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050"/(row["id"]+".npz")
        assert sha(f)==row["derived_npz_sha256"]
        with np.load(f,allow_pickle=False) as z:
            n=row["original_C3D_frames"]
            xyz=z["primary_human_3d_m"]
            bc=z["uncalibrated_ball_marker_cluster_centroid_m"]
            ballmask=z["three_ball_marker_valid"]
            ballvel=z["raw_ball_marker_cluster_speed_mps"]
            assert xyz.shape==(n,19,3)
            assert z["marker_derived_human_8_cosine_relations"].shape==(n,8)
            assert z["ASF_AMC_root_UNCALIBRATED_TO_C3D"].shape==(n,6)
            assert bc.shape==(n,3) and ballmask.shape==(n,3)
            assert np.nanmedian(np.linalg.norm(xyz[:,2]-xyz[:,0],axis=-1))>.2
            assert np.all(np.isnan(bc[~ballmask.all(axis=1)]))
            missing=~ballmask.all(axis=1)
            after_invalid=np.flatnonzero(missing[1:]|missing[:-1])+1
            assert np.isnan(ballvel[after_invalid]).all()
            assert np.isfinite(z["time_s"]).all()
            assert z["original_frame_rate_hz"].item()==120.
            assert not z["true_impact_event_available"].item()
