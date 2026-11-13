"""v142 adversarial CPU contracts. Fake packets NEVER establish live native pair."""
import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v142_native_pair_ledger as v142
import v142_human_data_qualification as humans
import v142_batch_readiness as batch
import v134_native_contact_vector_gate as v134


def state(step=1):
    return {
        "step": step,
        "q": [0.0]*29, "dq": [0.0]*29, "motor_torque": [0.0]*29,
        "base_pose": {"position": [0.,0.,1.], "quaternion":[1.,0.,0.,0.]},
        "base_twist": {"linear": [0.,0.,0.], "angular":[0.,0.,0.]},
        "com_world": [0.,0.,.5],
        "left_foot_net_force": [0.,0.,100.], "right_foot_net_force":[0.,0.,100.],
        "nonfoot_net_force": [0.,0.,0.],
        "ball_pose": [0.,0.,0.], "ball_twist":[0.,0.,0.],
    }


def packet(step=1, impulse=(1., 0., 0.), event="CONTACT_FOUND",
           actor=(1,2), collider=(3,4), points=True):
    p = [] if not points else [{
        "position": [0.,0.,0.],
        "normal": [1.,0.,0.],
        "impulse": list(impulse),
        "separation": -.01,
        "face_index0": 3,
        "face_index1": 4,
    }]
    h = [{"stage_id": 42, "type": event,
          "actor0": actor[0], "actor1": actor[1],
          "collider0":collider[0], "collider1":collider[1],
          "contact_data_offset": 0, "num_contact_data":len(p)}]
    names = {1:"/World/Robot/left_wrist_yaw_link",
             2:"/World/Robot/left_hip_roll_link",
             3:"/World/Robot/left_wrist_yaw_link/collision",
             4:"/World/Robot/left_hip_roll_link/collision",
             5:"/World/Robot/torso_link",
             6:"/World/Robot/torso_link/collision"}
    return v134.copy_contact_vector_frame(
        h,p,decode_path=lambda k:names[k],
        step_hint=step,expected_stage_id=42)


def owner_map():
    return {"/World/Robot/left_wrist_yaw_link/collision":
                "/World/Robot/left_wrist_yaw_link",
            "/World/Robot/left_hip_roll_link/collision":
                "/World/Robot/left_hip_roll_link",
            "/World/Robot/torso_link/collision":
                "/World/Robot/torso_link"}


def review(p=None, st=None, owners=None):
    return v142.packet_review(p if p is not None else packet(),
                             st if st is not None else state(),
                             owners if owners is not None else owner_map(),
                             expected_stage_id=42)


def test_protocol_frozen():
    c=v142.frozen()
    assert c["original_cases"] == {"frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}
    assert len(c["physical_fields"]) == 12


def test_historical_sha_locked_real_physics():
    x=v142.historical_recheck()
    assert x["historical_physics_5ms_frames"]==6628
    assert x["cmu_10_05_last_actual_q_s"]==5.14
    assert x["cmu_10_05_sensor_event_s"]==5.145
    assert x["native_physx_pair_manifold_truth"]=="UNKNOWN"
    assert x["new_physx_episodes"]==0


def test_actual_two_cmu_c3d_eligibility():
    x=humans.review_assets()
    assert x["clip_count"]==2
    assert x["distinct_subject_count"]==1
    assert x["subject_disjoint_train_test_possible_with_these_two_clips"] is False
    assert x["synchronized_real_ground_reaction_force_samples"]==0
    assert x["measured_human_impact_events"]==0
    assert x["can_truthfully_train_end_to_end_powerful_instep_kick"] is False
    for clip in ("10_02","10_05"):
        row=x["clips"][clip]
        assert row["rate_hz"]==120
        assert row["analog_channel_count"]==0
        assert row["frames"]>400
        assert len(row["c3d_sha256"])==64


@pytest.mark.parametrize("clip",["10_02","10_05"])
def test_actual_c3d_header_independently_recognized(clip):
    row=humans.metadata_c3d(
        humans.DATA_ROOT/"raw/10"/(clip+".c3d"))
    assert row["no_analog_channels_means_no_direct_C3D_forceplate_recording"]
    assert row["true_human_ball_contact_from_this_file"] is False


def test_human_scope_reject_undeclared():
    with pytest.raises(ValueError,match="UNDECLARED"):
        humans.review_assets(clips=("10_03",))


def test_source_manifest_unverified_upstream_signature():
    x=humans.review_assets()
    assert x["source_upstream_checksum_attested"] is False
    assert x["source_transport_TLS_chain_verified"] is False


def test_realistic_fake_contact_is_candidate_only():
    x=review()
    assert len(x["left_wrist_hip_colliding_pair_candidates"])==1
    assert x["left_wrist_hip_colliding_pair_candidates"][0]["contact_points"]==1
    assert x["runtime_native_pair_certified"] is False
    assert x["strict_kick_pass"] is False


@pytest.mark.parametrize("mutation,reason",[
    (lambda x: x.update(source_claim="REAL_PHYSX_CERTIFIED"),"SPOOFED_NATIVE"),
    (lambda x: x.update(native_pair_certified=True),"SELF_ATTESTED"),
    (lambda x: x.update(motor_execution_authorized=True),"SELF_ATTESTED"),
    (lambda x: x.update(all_contact_data_consumed=False),"UNCOVERED"),
    (lambda x: x.update(stage_id=43),"DIFFERENT_NATIVE"),
    (lambda x: x.update(step_hint=2),"OFF_BY_ONE"),
    (lambda x: x.update(time_hint_s=.01),"WRONG_REAL"),
    (lambda x: x.update(total_points=2),"COUNTS_INCONSISTENT"),
    (lambda x: x.update(total_headers=2),"COUNTS_INCONSISTENT"),
    (lambda x: x.update(headers="fake"),"MALFORMED"),
])
def test_adversarial_packets_fail_closed(mutation,reason):
    p=packet()
    mutation(p)
    with pytest.raises(ValueError,match=reason):
        review(p=p)


@pytest.mark.parametrize("field",[
    "q","dq","motor_torque","base_pose","base_twist","com_world",
    "left_foot_net_force","right_foot_net_force","nonfoot_net_force",
    "ball_pose","ball_twist","step"
])
def test_missing_any_real_state_field_rejected(field):
    x=state()
    del x[field]
    with pytest.raises(ValueError,match="INCOMPLETE_REAL"):
        review(st=x)


def test_wrong_29dof_dimensions_rejected():
    x=state()
    x["q"]=[0.]*28
    with pytest.raises(ValueError,match="INVALID_ORIGINAL29"):
        review(st=x)


def test_wrong_actor_owner_pair_rejected_without_fake_positive():
    p=packet(actor=(1,5))
    x=review(p=p)
    assert not x["left_wrist_hip_colliding_pair_candidates"]
    assert x["rejected_headers_by_reason"]["REJECT_ACTOR_OWNER_MISMATCH"]==1


def test_missing_owner_map_rejected():
    with pytest.raises(ValueError,match="RUNTIME_STAGE_OWNER_MAP"):
        review(owners={})


def test_partial_owner_mapping_rejects():
    x=review(owners={list(owner_map())[0]:list(owner_map().values())[0]})
    assert not x["left_wrist_hip_colliding_pair_candidates"]


def test_empty_manifold_cannot_prove_no_collision():
    x=review(p=packet(points=False))
    assert not x["left_wrist_hip_colliding_pair_candidates"]
    assert x["rejected_headers_by_reason"]["NO_MANIFOLD_POINTS"]==1


def test_zero_impulse_cannot_claim_confirmed_contact():
    x=review(p=packet(impulse=(0.,0.,0.)))
    assert x["rejected_headers_by_reason"]["ZERO_VECTOR_IMPULSE"]==1


def test_unrecognized_event_is_not_active_contact():
    x=review(p=packet(event="UNDEFINED"))
    assert x["rejected_headers_by_reason"]["NOT_ACTIVE_CONTACT_EVENT"]==1


def test_unknown_torso_event_not_wrist_hip():
    x=review(p=packet(actor=(1,5),collider=(3,6)))
    assert x["native_callback_headers"]==1
    assert not x["left_wrist_hip_colliding_pair_candidates"]


def test_timeline_multiple_callbacks_one_physics_step():
    p1=packet(step=1)
    p2=packet(step=1)
    x=v142.review_timeline([p1,p2],[state(1),state(2)],owner_map(),stage_id=42)
    assert x["actual_input_physics_step_count"]==2
    assert x["callback_packet_count"]==2
    assert x["target_native_contact_candidates"]==2
    assert x["no_callback_does_not_prove_no_contact"]


@pytest.mark.parametrize("rows",[
    [state(1), state(1)],
    [state(2)],
    [state(1), state(3)],
    []
])
def test_timeline_reject_missing_physical_steps(rows):
    with pytest.raises(ValueError,match="MISSING_OR_DUPLICATE"):
        v142.review_timeline([],rows,owner_map(),stage_id=42)


def test_timeline_reject_early_or_late_callback():
    with pytest.raises(ValueError,match="CALLBACK_NOT_ATTACHED"):
        v142.review_timeline([packet(step=3)],[state(1),state(2)],owner_map(),stage_id=42)


def test_exact_off_on_real_field_comparator():
    a=[state(1),state(2)]
    b=copy.deepcopy(a)
    x=v142.paired_exact_cpu_review(a,b)
    assert x["exact_common_physics_state_equality"]
    assert x["real_source_provenance_certified"] is False
    b[1]["dq"][28]=1e-12
    y=v142.paired_exact_cpu_review(a,b)
    assert not y["exact_common_physics_state_equality"]
    assert y["mismatch_frame_indices"]==[1]


def test_strict_12_physics_fields_cannot_be_silently_dropped():
    a=[state()]
    b=[state()]
    del b[0]["nonfoot_net_force"]
    with pytest.raises(ValueError,match="PHYSICAL_FIELD_SET"):
        v142.paired_exact_cpu_review(a,b)


@pytest.mark.parametrize("free,gpu,dirty,lock,expected",[
    (30,[{"pid":242644,"executable":"pytest","used_memory":"260 MiB"}],False,True,"GPU_OCCUPIED"),
    (19,[],False,True,"DISK_BELOW"),
    (30,[],True,True,"SOURCE_DIRTY"),
    (30,[],False,False,"ORIGINAL_GPU_INTERLOCK"),
])
def test_gpu_and_source_protection(free,gpu,dirty,lock,expected):
    r=batch.readiness(free_gib=free,gpu_rows=gpu,
                      branch="version/v0.1.142",git_dirty=dirty,lock_exists=lock)
    assert any(expected in x for x in r["blockers"])
    assert r["new_strategy_14s_kick_success_rate"] is None
    assert not r["can_start_real_native_now"]
    assert r["original_G1_physx_runs_executed_by_this_tool"]==0
    assert len(r["original_G1_paired_jobs"])==6


def test_even_unoccupied_gpu_does_not_authorize_unqualified_physical_exec():
    r=batch.readiness(free_gib=30,gpu_rows=[],branch="version/v0.1.142",
                      git_dirty=False,lock_exists=True)
    assert not r["blockers"]
    assert r["would_pass_minimal_resource_preflight"]
    assert not r["can_start_real_native_now"]


def test_wrong_branch_rejected():
    with pytest.raises(ValueError,match="WRONG_NATIVE_EVIDENCE_BRANCH"):
        batch.readiness(free_gib=30,gpu_rows=[],branch="version/v0.1.141",
                        git_dirty=False,lock_exists=True)


def test_evidence_write_once_without_clobber(tmp_path):
    p=tmp_path/"manifest.json"
    h=v142.safe_write_once(p,{"native_pair_certified":False})
    assert len(h)==64
    with pytest.raises(FileExistsError):
        v142.safe_write_once(p,{"native_pair_certified":True})
    assert json.loads(p.read_text())["native_pair_certified"] is False


def test_write_once_symlink_is_rejected(tmp_path):
    target=tmp_path/"target.json"
    target.write_text("{}")
    link=tmp_path/"link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError,match="UNSAFE_EVIDENCE"):
        v142.safe_write_once(link,{"x":1})
    assert target.read_text()=="{}"
