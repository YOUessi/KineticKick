"""v146 original-runner opt-in observer and genuine SHA-locked v54 physics audit.

GPU is intentionally never started. Synthetic callback witnesses are NOT
original PhysX contact pair truths.
"""
import copy
import json
import math
import subprocess
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import v146_native_original_bridge as live
import v146_real_200hz_coupled_audit as audit
from test_v142_native_contact_and_human_qualification import state, owner_map
from test_v142_runtime_bridge import real_shape_header, real_shape_point


class FakeOriginalSampler:
    def __init__(self):
        self.calls=[]
    def sample(self, step):
        self.calls.append(step)
        return state(step)


def shape_stage():
    names = {
        1:"/World/Robot/left_wrist_yaw_link",
        2:"/World/Robot/left_hip_roll_link",
        3:"/World/Robot/left_wrist_yaw_link/collision",
        4:"/World/Robot/left_hip_roll_link/collision",
    }
    return (42,owner_map(),lambda key:names[key],
            {"actual_usd_read_only_inspection":False,
             "test_fixture_not_runtime":True})


def session(mode="on", stage_reader=shape_stage, subscribe=None):
    s=live.OriginalG1ReadOnlyCapture(
        mode=mode,sampler=FakeOriginalSampler(),
        stage_reader=stage_reader,
        native_subscribe=subscribe if subscribe is not None else lambda cb:object())
    s.attach()
    return s


def sample_step(s, n):
    s.before_native_update(.005)
    s.after_native_update(.005)
    assert s.rows[-1]["step"] == n


def test_frozen_source_is_independent_original_v145_parent():
    f=live.frozen()
    assert f["parent_sha"]=="b8146bb4c6f4e06284def2bfdfd20671ebfa96f4"
    assert f["original_runner"]=="tools/v080_run_poll_strike.py"
    assert f["full_horizon_steps"]==2800
    assert f["original_dof"]==29
    assert f["callback_clock_source"].startswith("POST_SCENE_UPDATE_INTERVAL")
    assert len(f["frozen_live_paired_matrix"])==6
    assert not f["human_data_synchronised_grf_or_true_instep_certified"]


def test_original_v080_runner_is_byte_exact_parent_not_mutated():
    origin=subprocess.check_output(["git","-C",str(ROOT),"show",
                         "b8146bb4c6f4e06284def2bfdfd20671ebfa96f4:tools/v080_run_poll_strike.py"])
    assert origin==(ROOT/"tools/v080_run_poll_strike.py").read_bytes()
    v146=(ROOT/"tools/v146_run_original_with_readonly.py").read_text()
    for needle in ("--v146-observer-mode","--v146-sidecar",
                   "before_native_update(dt)", "after_native_update(dt)",
                   "gpu_preflight()", "LOCK_EX|fcntl.LOCK_NB", "persist_once"):
        assert needle in v146


@pytest.mark.parametrize("free,gputab,error",[
    (19*1024**3, "", "DISK_BELOW_20"),
    (30*1024**3, "242644, python, 260 MiB", "FOREIGN_GPU"),
    (30*1024**3, "12, another python, 520 MiB", "FOREIGN_GPU"),
])
def test_gpu_and_disk_gate_blocks_unsafe_original_physics(free,gputab,error):
    with pytest.raises(RuntimeError,match=error):
        live.gpu_preflight(gpu_table=gputab,disk_free_bytes=free)


def test_gpu_idle_check_is_only_preflight_not_reservation():
    out=live.gpu_preflight(gpu_table="",disk_free_bytes=30*1024**3)
    assert out["gpu_is_idle_at_preflight"]
    assert out["gpu_preflight_is_not_exclusive_mutex"]


def test_off_mode_never_inspects_stage_or_subscribes():
    def fail():
        raise AssertionError("OFF observer may not access USD stage")
    s=session("off",stage_reader=fail,subscribe=fail)
    assert s.subscription is None
    sample_step(s,1)
    sample_step(s,2)
    r=s.review()
    assert r["original_29dof_completed_5ms_rows"]==2
    assert r["copied_native_callback_packets"]==0
    assert r["left_wrist_hip_native_physx_truth"]=="UNKNOWN"
    assert not r["same_native_physics_tick_calibrated"]


def test_on_mode_multiple_callbacks_one_original5ms_state():
    callback={}
    def subscribe(cb):
        callback["cb"]=cb
        return object()
    s=session("on",subscribe=subscribe)
    s.before_native_update(.005)
    header=real_shape_header()
    points=real_shape_point()
    callback["cb"](header,points)
    callback["cb"](header,points)
    points[0]["impulse"][0]=14.5
    s.after_native_update(.005)
    sample_step(s,2)
    assert s.sampler.calls==[1,2]
    r=s.review()
    assert r["original_29dof_completed_5ms_rows"]==2
    assert r["copied_native_callback_packets"]==2
    assert r["copied_manifold_points"]==2
    assert len(r["wrist_hip_interval_candidates_not_truth"])==2
    assert abs(r["wrist_hip_interval_candidates_not_truth"][0]["total_vector_magnitude_Ns"]-1.25)<1e-10
    assert not r["same_native_physics_tick_calibrated"]
    assert not r["off_on_physical_equality_certified"]
    assert r["new_physx_episode_success"] is None
    assert not r["motor_execution_authorized"]


def test_on_off_injected_state_same_without_source_mutation():
    off=session("off")
    on=session("on")
    for step in range(1,21):
        sample_step(off,step)
        sample_step(on,step)
    assert off.rows==on.rows
    assert off.review()["original_full_state_checksum"]==on.review()["original_full_state_checksum"]
    # equality of CPU fixtures DOES NOT certify noninterference in PhysX
    assert not off.review()["off_on_physical_equality_certified"]


def test_missing_stage_does_not_affect_physical_actions_but_invalidates_observer():
    def missing():
        raise ValueError("NO_STAGE")
    s=session("on",stage_reader=missing)
    assert s.failed and "NO_STAGE" in s.errors[0]
    s.before_native_update(.005)
    s.after_native_update(.005)
    assert s.review()["readout_incomplete"]
    assert not hasattr(s,"apply_action")
    assert not hasattr(s,"set_joint_target")


def test_callback_outside_scene_update_is_rejected():
    callback={}
    s=session("on",subscribe=lambda cb:callback.setdefault("cb",cb) or object())
    callback["cb"](real_shape_header(),real_shape_point())
    sample_step(s,1)
    assert s.review()["copied_native_callback_packets"]==0
    assert s.review()["readout_incomplete"]


def test_wrong_contact_shape_fails_closed_without_physics_exception():
    cb={}
    s=session("on",subscribe=lambda fn:cb.setdefault("callback",fn) or object())
    s.before_native_update(.005)
    p=real_shape_point()
    p[0]["impulse"]=4.2
    cb["callback"](real_shape_header(),p)
    s.after_native_update(.005)
    assert s.review()["readout_incomplete"]
    assert "UNTRUSTED_NATIVE_CALLBACK" in s.errors[0]


@pytest.mark.parametrize("dt",[-1.,0.,.02,True,float("nan")])
def test_non200hz_dt_fails_closed(dt):
    s=session("off")
    s.before_native_update(dt)
    assert s.failed


def test_write_once_and_state_provenance_not_physx_trust(tmp_path):
    s=session("off")
    sample_step(s,1)
    file=tmp_path/"copy.json"
    digest=s.persist_once(file)
    assert len(digest)==64
    result=json.loads(file.read_text())
    assert result["original29_5ms_state_rows"][0]["step"]==1
    assert result["summary"]["left_wrist_hip_native_physx_truth"]=="UNKNOWN"
    with pytest.raises(FileExistsError):
        s.persist_once(file)


def test_callback_cap_is_hard_limit_and_rejects(caplog):
    s=live.OriginalG1ReadOnlyCapture(
        mode="on",sampler=FakeOriginalSampler(),max_packets=1,
        stage_reader=shape_stage,native_subscribe=lambda cb:object())
    s.attach()
    s.before_native_update(.005)
    s.on_callback(real_shape_header(),real_shape_point())
    s.on_callback(real_shape_header(),real_shape_point())
    assert s.failed
    assert "MEMORY_CAP" in s.errors[-1]


@pytest.mark.parametrize("wrong",[
    {"q":[0.]*28}, {"dq":[float("nan")]*29},
    {"step":True}, {"com_world":[0.,0.]},
])
def test_original29_incomplete_mock_state_rejected(wrong):
    bad=state()
    bad.update(wrong)
    from v142_native_pair_ledger import verify_original_state
    with pytest.raises((ValueError,TypeError)):
        verify_original_state(bad)


@pytest.fixture(scope="module")
def real_sources():
    return audit.verified_sources()


@pytest.fixture(scope="module")
def real_result():
    return audit.audit_all()


def test_original_v54_source_actual_sha_not_filenames(real_sources):
    assert set(real_sources)=={"frozen_v32","cmu_10_02","cmu_10_05"}
    for path in real_sources.values():
        assert path.is_file()
        assert path.stat().st_size>10_000_000


def test_original_actual_6628_and_terminal_duplicate_not_new_tick(real_result):
    assert real_result["historical_5ms_frames"]==6628
    assert real_result["excluded_terminal_duplicate_rows"]==1
    assert real_result["new_physx_episodes"]==0
    assert real_result["native_wrist_hip_contact_manifold_truth"]=="UNKNOWN"
    assert real_result["wholebody_arm_leg_contact_distance_missing_in_original_5ms_trace"]


@pytest.mark.parametrize("name,count,end",[
    ("frozen_v32",2800,14.),("cmu_10_02",2800,14.),("cmu_10_05",1028,5.14)])
def test_actual_original_all29_and_5ms_strict(real_result,name,count,end):
    d=real_result["cases"][name]
    rows=d["full_real_200hz_coupled_rows"]
    assert len(rows)==count
    assert d["last_actual_5ms_state_time_s"]==end
    for i in (0,len(rows)//2,len(rows)-1):
        r=rows[i]
        assert r["physics_step"]==i+1
        assert abs(r["time_s"]-.005*(i+1))<1e-6
        for field in ("q29","qdot29","sim_torque29","qdot_ratio29"):
            assert len(r[field])==29
        assert r["arm_hip_true_collider_pair"] is None
        assert r["net_foot_contact_not_isolated_grf"]


def test_actual_frozen_contact_windows_pre_excludes_contact(real_result):
    for row in real_result["cases"].values():
        w=row["window"]
        assert w["before_steps"]==80
        assert w["after_steps"]==250
        assert abs(w["first_before_time_s"]-1.17)<1e-6
        assert abs(w["last_before_time_s"]-1.565)<1e-6
        assert abs(w["first_after_time_s"]-1.575)<1e-6
        assert abs(w["last_after_time_s"]-2.82)<1e-6
        assert w["pre_or_post_foot_force_is_net_not_grf"]


def test_real_first_wrist_hip_impulse_not_certified(real_result):
    short=real_result["cases"]["cmu_10_05"]
    assert abs(short["peak_nonfoot_net_N"]-25.890153884887695)<1e-6
    assert short["last_actual_5ms_state_time_s"]==5.14
    assert short["recovery_if_missing_14s"]=="UNKNOWN"
    assert not short["real_14s_coverage"]
    assert short["left_wrist_hip_native_pair"]=="UNKNOWN"
    assert not short["v54_legacy_center_record_success_flag"]
    assert not short["new_full_original29_instep_physics_gate_certified"]
    assert real_result["cases"]["frozen_v32"]["v54_legacy_center_record_success_flag"]
    assert real_result["cases"]["cmu_10_02"]["v54_legacy_center_stride_shot_success_flag"]


def test_original_200hz_window_too_short_refused():
    rows=[{"time_s":.005*(i+1),"qdot_ratio29":[0.]*29,
           "nonfoot_net_contact_N":0.,"left_foot_net_contact_N":0.,
           "right_foot_net_contact_N":0.} for i in range(400)]
    with pytest.raises(ValueError,match="INSUFFICIENT_REAL"):
        audit.windows(rows,1.57)


def test_real_historical_interpolicy_independent_numerics(real_result):
    pair=real_result["interpolicy_original_center_ball_comparison"]
    assert len(pair)==3
    for item in pair.values():
        assert item["shared_original_physics_steps"] in (1028,2800)
        assert 0 <= item["pre_contact_1p57_q_identical_steps"] <=314
        assert not item["same_ball_and_no_policy_ablations_certified"]
        assert item["pre_contact_after_first_divergence_shared_steps"]==33
        assert set(item["pre_contact_original29_body_group_differences"])=={"legs","waist","arms"}


def test_real_source_mutation_and_invalid_5ms_not_inferred(real_sources):
    path=real_sources["cmu_10_05"]
    d=json.loads(path.read_text())
    original=d["traces"][0]
    a,b=audit.split_real_5ms_rows(original,1028)
    assert len(b)==1
    wrong=copy.deepcopy(original[:1028])
    wrong[2]["time_s"]=wrong[1]["time_s"]
    with pytest.raises(ValueError,match="DROPPED_OR_DUPLICATE"):
        audit.split_real_5ms_rows(wrong,1028)
    wrong=copy.deepcopy(original)
    wrong[-1]["time_s"]=5.145
    with pytest.raises(ValueError,match="NOT_KNOWN"):
        audit.split_real_5ms_rows(wrong,1028)


def test_real_source_status_cannot_be_promoted_to_new_physx(real_result):
    assert real_result["new_policy_success_rate"] is None
    assert not real_result["professional_human_impact_or_grf_supervision_certified"]
    assert all(not c["new_full_original29_instep_physics_gate_certified"]
               for c in real_result["cases"].values())
