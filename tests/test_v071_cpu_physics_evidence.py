"""v071 CPU-only reproducibility and epistemic safety regression tests."""
import json
import math
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_v071_native_event_ledger import (
    InvalidSource, original_29_velocity_limits, has_direct_cuda_wrist_hip_pair,
    physical_frame_summaries, summarize_case, audit_sources, sha256, verify_manifest,
    v070_completed_snapshot,
)
from audit_v071_v070_launch_stage import audit_inflight,parse_completed_run


def fake_native(time_count=700,contact=1.5,instep_fraction=1.0,velocity=4.0):
    frames=[]
    for i in range(time_count):
        t=round((i+1)*.005,6)
        frames.append(dict(time_s=t,joint_pos=[0.0]*29,joint_vel=[.1]*29,
                           whole_body_com_xyz=[0,0,.65],
                           root_xyz=[0,0,.7],support_foot_force_N={"left":190.,"right":300.},
                           nonfoot_contact_N=0.0,ball_velocity=[velocity,0,0],
                           terminated=False))
    return dict(root_fixed=False,v057_requested_physics_backend="cuda:0",
                joint_names=[f"q{i}" for i in range(29)],
                joint_velocity_limits=[[2.0]*29],traces=[frames],
                records=[dict(contact_time_s=contact,requested_surface_impulse_fraction=instep_fraction,
                              goal=True,recovered=True)],termination_reasons=[[]],
                stride_shot_success=True)


def cfg():
    return dict(dt_s=.005,contact_pre_s=.4,contact_post_s=1.25,
                full_recovery_s=14.,instep_fraction_min=.95,
                speed_proxy_min_mps=3.5)


def test_batched_original_29_qdot_limit():
    a=fake_native()
    assert original_29_velocity_limits(a)==[2.0]*29
    a["joint_velocity_limits"]=[2.]*28
    with pytest.raises(InvalidSource):
        original_29_velocity_limits(a)


def test_window_preserved_when_early_censored():
    r=summarize_case(fake_native(),"synthetic",cfg())
    assert r["actual_window_native_frames"]>=330
    assert r["contact_minus_0p4_to_plus_1p25_fully_observed"]
    assert not r["strict_all_gates_pass"]
    assert "real_full_14s" in r["gates_failed"]


def test_early_nonimpact_window_absent():
    r=summarize_case(fake_native(time_count=329,contact=None),"synthetic",cfg())
    assert not r["contact_minus_0p4_to_plus_1p25_fully_observed"]
    assert r["actual_window_native_frames"]==0
    assert not r["strict_all_gates_pass"]


def test_95_percent_instep_strict_gate():
    s=fake_native(instep_fraction=.729834707982299)
    a=summarize_case(s,"historical_nonstandard",cfg())
    assert "measured_instep_fraction_ge_95pct" in a["gates_failed"]
    s["records"][0]["requested_surface_impulse_fraction"]=.95
    b=summarize_case(s,"threshold",cfg())
    assert "measured_instep_fraction_ge_95pct" not in b["gates_failed"]


def test_original_qdot_limit_not_pose_cost():
    a=fake_native()
    a["traces"][0][200]["joint_vel"][3]=2.1
    r=summarize_case(a,"overlimit",cfg())
    assert r["all_original_29_qdot_max_ratio"]==pytest.approx(1.05)
    assert "original_29_qdot_within_limit" in r["gates_failed"]


def test_no_native_pair_from_net_force_or_link_origin():
    a=fake_native()
    a["contacts"]=[dict(body0="left_wrist_yaw_link",body1="left_hip_roll_link",net_force_N=25.890153)]
    a["equal_opposite_contact_sensor_net_forces"]=[25.890153,-25.890153]
    assert has_direct_cuda_wrist_hip_pair(a) is False


def test_bare_invented_collider_without_manifold_rejected():
    a=fake_native()
    a["native_physics_contact_pairs"]=[dict(
        backend_physx_cuda_original_g1=True,
        body0_collider_path="/World/G1/left_wrist_yaw_link",
        body1_collider_path="/World/G1/left_hip_roll_link")]
    assert has_direct_cuda_wrist_hip_pair(a) is False


def test_paired_manifold_schema_is_only_a_prerequisite_not_verified_provenance():
    a=fake_native()
    pair=dict(backend_physx_cuda_original_g1=True,
              body0_collider_path="/World/G1/left_wrist_yaw_link/collision",
              body1_collider_path="/World/G1/left_hip_roll_link/collision",
              contact_point_world=[.2,.1,.6],normal_world=[0,0,1],
              normal_impulse_Ns=.08)
    a["native_physics_contact_pairs"]=[pair]
    assert has_direct_cuda_wrist_hip_pair(a) is True
    pair["normal_impulse_Ns"]=0
    assert has_direct_cuda_wrist_hip_pair(a) is False
    # This validates data schema only. Proving hardware-native instrumentation
    # requires trusted producer code/SHA plus read-only 25-field PhysX parity.


def test_bad_sample_clock_not_silently_interpolated():
    a=fake_native()
    a["traces"][0][300]["time_s"]+=.002
    with pytest.raises(InvalidSource,match="CLOCK"):
        physical_frame_summaries(a)


def test_missing_real_com_causes_failure():
    a=fake_native()
    del a["traces"][0][7]["whole_body_com_xyz"]
    with pytest.raises(InvalidSource,match="MISSING_original_com"):
        physical_frame_summaries(a)


def test_source_hash_drift_error(tmp_path):
    c={"v67_raw_sha256":{f"v067_new_stance_{z}_{p}.json":"0"*64 for z in "abc" for p in ("v32","synth","reverse")}}
    p=tmp_path/"v067_new_stance_a_v32.json"
    p.write_text("{}")
    with pytest.raises(InvalidSource,match="ORIGINAL_G1_RAW_DRIFT"):
        verify_manifest(tmp_path,c)


def test_v070_hang_stage_not_physical_failure(tmp_path):
    log=tmp_path/"v070_trial_a_v32.log"
    log.write_text("V058_BACKEND_ARG_FORWARDED cuda:0\n[INFO][AppLauncher]\nLoading user config located at:\nGLFW initialization failed\n")
    x=dict(id="trial_a_v32",status="NO_G1_OUTPUT",return_code=124,wall_s=240.4,
           log_sha256=sha256(log))
    run=parse_completed_run(x,tmp_path)
    assert run["classification"]=="ISAAC_KIT_BOOT_OR_SCENE_INIT_NO_G1_EPISODE"
    assert not run["is_certified_scored_G1_29DoF"]
    assert run["markers"]["seen_gui_context_failure"]
    log.write_text(log.read_text()+"unregistered edit")
    with pytest.raises(InvalidSource,match="MUTATED_COMPLETED_V070_LOG"):
        parse_completed_run(x,tmp_path)


def test_v070_snapshot_never_claims_completed_while_busy(tmp_path):
    log=tmp_path/"v070_trial_a_v32.log";log.write_text("[INFO][AppLauncher]\n")
    run=dict(id="trial_a_v32",status="NO_G1_OUTPUT",return_code=124,wall_s=240.4,
             log_sha256=sha256(log))
    p=tmp_path/"v070_batch_progress.json"
    p.write_text(json.dumps(dict(version="v0.1.70",runs=[run])))
    summary=audit_inflight(p)
    assert summary["v070_completed_rows_snapshot"]==1
    assert not summary["v070_completed_all"]
    assert summary["v070_completed_rows_without_scorable_original_29DoF"]==1


def test_tampered_or_ambiguous_progress_snapshot(tmp_path):
    p=tmp_path/"v070_batch_progress.json"
    p.write_text(json.dumps(dict(version="v0.1.70",runs=[{"id":"trial_a_v32"}]*2)))
    with pytest.raises(InvalidSource,match="DUPLICATE"):
        audit_inflight(p)


def test_independent_historical_9_episode_censored_ledger_if_source_present():
    original=Path("/home/you/projects/kinetickick-wt-v067/reports")
    if not (original/"v067_batch_progress.json").exists():
        pytest.skip("historical physical raw is intentionally stored on Tang only")
    c=json.loads((ROOT/"configs/g1_physx_readonly_evidence_ledger_v071.json").read_text())
    r=audit_sources(original,c,sha256(ROOT/"configs/g1_physx_readonly_evidence_ledger_v071.json"))
    assert r["source_native_physics_episodes"]==9
    assert r["newly_run_physics_episodes"]==0
    assert r["strict_pass_count"]==4
    assert r["independently_verified_native_gpu_wrist_hip_contact_manifolds"]==0
    data={x["id"]:x for x in r["cases"]}
    assert data["new_stance_b_synth"]["instep_fraction"]==pytest.approx(.729834707982299)
    assert data["new_stance_c_v32"]["actual_contact_s"] is None
    assert data["new_stance_c_v32"]["physical_native_200hz_rows"]==329
    assert data["new_stance_b_v32"]["physical_native_200hz_rows"]==741
    assert not data["new_stance_b_v32"]["full_14_second_native_recovery_observed"]
    assert data["new_stance_a_v32"]["physical_native_200hz_rows"]==2800
