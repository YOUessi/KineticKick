"""V151 frozen G1 native contact observability & strict parity adversarial tests."""
import copy
import json
from pathlib import Path
import sys

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v151_native_campaign as campaign
import v151_native_observability_ledger as ledger
import v151_off_on_exact_audit as parity
from v146_native_original_bridge import json_sha
from test_v142_native_contact_and_human_qualification import state


def fake_sidecar(mode="on",n=2):
    rows=[state(i+1) for i in range(n)]
    return {
        "summary":{
            "mode":mode,"original_29dof_completed_5ms_rows":n,
            "original_full_state_checksum":json_sha(rows),
            "left_wrist_hip_native_physx_truth":"UNKNOWN",
            "same_native_physics_tick_calibrated":False,
            "off_on_physical_equality_certified":False,
            "runtime_positive_negative_controls_passed":False,
            "readout_incomplete":False,
            "copied_native_callback_packets":0,
            "existing_report_api_census":{
                "pair_report_schema_eligible":True,
                "no_events_means_no_contact":False,
                "runtime_positive_negative_controls_passed":False}},
        "original29_5ms_state_rows":rows,
        "raw_callback_copies":[],
        "collider_to_rigid_owner":{
            "/World/G1/left_wrist_yaw_link/collision":"/World/G1/left_wrist_yaw_link",
            "/World/G1/left_hip_roll_link/collision":"/World/G1/left_hip_roll_link"}}


def fake_packet(phase="INSIDE_SCENE_UPDATE",active=True):
    return {
        "source_claim":"CPU_OR_LIVE_CALLBACK_UNATTESTED",
        "native_pair_certified":False,"motor_execution_authorized":False,
        "all_contact_data_consumed":True,
        "v149_arrival_phase":phase,
        "clock_basis":"SCENE_UPDATE_ADJACENCY_ONLY",
        "v149_untrusted_step_hint_not_native_clock":True,
        "v149_adjacent_step_candidates_unattested":[1,2],
        "headers":[{
            "event_type":"CONTACT_FOUND" if active else "CONTACT_LOST",
            "collider_paths":[
                "/World/G1/left_wrist_yaw_link/collision",
                "/World/G1/left_hip_roll_link/collision"],
            "manifold_points":[{"impulse_world_vector_Ns":[.2,0,0]}]
                    if active else []}]}


@pytest.fixture(autouse=True)
def tiny_history(monkeypatch):
    real=campaign.frozen
    assert real()["history_expected_rows"]=={
        "frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}
    monkeypatch.setattr(parity,"frozen",lambda:{"history_expected_rows":{
        "frozen_v32":2,"cmu_10_02":2,"cmu_10_05":2}})


def test_frozen_parent_and_protocol_code_not_drift():
    data=campaign.frozen()
    assert data["parent_sha"]=="9d9575662db2b6434f30b35c93a457a889425f58"
    assert data["strict_kick_gates"]["wholebody_dof"]==29
    assert data["native_positive_negative_controls_required"]
    assert data["live_trial_count"]==6


def test_real_original_v54_source_shas_and_actual_seed_difference():
    inventory=campaign.source_inventory()
    assert len(inventory)==3
    for row in inventory:
        assert len(row["config_sha256"])==64
        assert row["config"]=="configs/"+row["config"].split("/")[-1]
        data=json.loads((campaign.SOURCE_ROOT/row["config"]).read_text())
        assert data["seed"]==2601 # in config, original PhysX episode eval seed=1201


def test_real_original_v54_historical_6628_locked_inventory():
    from v146_real_200hz_coupled_audit import verified_sources,split_real_5ms_rows
    sources=verified_sources()
    assert set(sources)=={"frozen_v32","cmu_10_02","cmu_10_05"}
    expected=campaign.frozen()["history_expected_rows"]
    total=0
    for name,path in sources.items():
        data=json.loads(path.read_text())
        rows,terminal=split_real_5ms_rows(data["traces"][0],expected[name])
        assert len(rows)==expected[name]
        assert len(terminal)==int(name=="cmu_10_05")
        total+=len(rows)
    assert total==6628


def test_manifest_exact_six_and_correct_original_runner():
    p=campaign.make_campaign()
    assert p["run_count"]==6 and p["physically_executed"]==0
    assert [x["name"] for x in p["runs"]]==[
        "frozen_v32_off","frozen_v32_on","cmu_10_02_off","cmu_10_02_on",
        "cmu_10_05_off","cmu_10_05_on"]
    for row in p["runs"]:
        cmd=row["argv"]
        assert cmd[0]=="/home/you/conda_env/env_isaaclab/bin/python"
        assert cmd.count("--v149-observer-mode")==1
        assert cmd[cmd.index("--v149-observer-mode")+1]==row["mode"]
        assert cmd[cmd.index("--eval-seed")+1]=="1201"
        assert cmd[cmd.index("--ball-offset")+1:cmd.index("--ball-offset")+3]==["0","0"]
        assert cmd[cmd.index("--velocity-delta")+1]=="0"
        assert Path(row["sidecar"]).name!=Path(row["output"]).name
    assert p["left_wrist_hip_manifold"]=="UNKNOWN"


@pytest.mark.parametrize("free_gib,gpu,lock,reason",[
    (19.8,"",True,"DISK_BELOW"),
    (30.,"42, python, 260 MiB",True,"FOREIGN_GPU_CONTEXT"),
    (30.,"",False,"MUTEX_MISSING"),
])
def test_preflight_denies_unsafe_gpu(free_gib,gpu,lock,reason):
    with pytest.raises(RuntimeError,match=reason):
        campaign.safety_preflight(gpu_table=gpu,
            disk_bytes=int(free_gib*1024**3),lock_exists=lock)


def test_preflight_idle_does_not_reserve_gpu():
    a=campaign.safety_preflight(gpu_table="",disk_bytes=30*1024**3,lock_exists=True)
    assert a["eligible_only_at_this_instant"]
    assert a["shared_mutex_must_be_locked_by_runner"]


def test_plan_paths_unique_and_immutable():
    p=campaign.make_campaign()
    assert campaign.available_output_paths(p)
    p["runs"][1]["output"]=p["runs"][0]["output"]
    with pytest.raises(ValueError,match="DUPLICATE_OUTPUT"):
        campaign.available_output_paths(p)


def test_existing_protected_path_rejected(tmp_path):
    p=campaign.make_campaign()
    p["runs"][0]["log"]=str(tmp_path/"exists.log")
    Path(p["runs"][0]["log"]).write_text("kept")
    with pytest.raises(FileExistsError,match="OUTPUT_EXISTS"):
        campaign.available_output_paths(p)
    assert Path(p["runs"][0]["log"]).read_text()=="kept"


def test_never_gpu_execute_if_preflight_blocked(monkeypatch):
    monkeypatch.setattr(campaign,"safety_preflight",lambda: (_ for _ in ()).throw(
        RuntimeError("V151_FOREIGN_GPU_CONTEXT_NO_PHYSICS")))
    def should_not_check_output(_):
        raise AssertionError("no writes when GPU busy")
    monkeypatch.setattr(campaign,"available_output_paths",should_not_check_output)
    with pytest.raises(RuntimeError,match="FOREIGN_GPU_CONTEXT"):
        campaign.execute_sequential()


def test_full_original_29_dof_input_parity_exact():
    a=parity.compare_inputs(fake_sidecar("off"),fake_sidecar("on"),policy="frozen_v32")
    assert a["input_level_all_common_original29_physics_equal"]
    assert a["recorded_input_frames_each"]==2
    assert a["first_divergence"] is None
    assert a["pair_native_physx_truth"]=="UNKNOWN"
    assert not a["read_only_nonintrusion_certified_from_cpu_inputs"]


@pytest.mark.parametrize("field,mutate",[
    ("q",lambda row:row["q"].__setitem__(0,row["q"][0]+.002)),
    ("dq",lambda row:row["dq"].__setitem__(28,row["dq"][28]+.05)),
    ("motor_torque",lambda row:row["motor_torque"].__setitem__(3,row["motor_torque"][3]+.1)),
    ("base_pose",lambda row:row["base_pose"]["position"].__setitem__(0,row["base_pose"]["position"][0]+.1)),
    ("base_twist",lambda row:row["base_twist"]["angular"].__setitem__(2,row["base_twist"]["angular"][2]+.1)),
    ("com_world",lambda row:row["com_world"].__setitem__(1,row["com_world"][1]+.01)),
    ("left_foot_net_force",lambda row:row["left_foot_net_force"].__setitem__(2,row["left_foot_net_force"][2]+2)),
    ("right_foot_net_force",lambda row:row["right_foot_net_force"].__setitem__(1,row["right_foot_net_force"][1]+2)),
    ("nonfoot_net_force",lambda row:row["nonfoot_net_force"].__setitem__(0,row["nonfoot_net_force"][0]+2)),
    ("ball_pose",lambda row:row["ball_pose"].__setitem__(2,row["ball_pose"][2]+.05)),
    ("ball_twist",lambda row:row["ball_twist"].__setitem__(0,row["ball_twist"][0]+1)),
])
def test_every_full_physics_field_can_falsify_off_on_equality(field,mutate):
    off=fake_sidecar("off")
    on=fake_sidecar("on")
    mutate(on["original29_5ms_state_rows"][1])
    on["summary"]["original_full_state_checksum"]=json_sha(on["original29_5ms_state_rows"])
    r=parity.compare_inputs(off,on,policy="frozen_v32")
    assert not r["input_level_all_common_original29_physics_equal"]
    assert r["first_divergence"]["step"]==2
    assert r["first_divergence"]["field"].startswith(field)
    assert r["mismatching_frames"]==1


@pytest.mark.parametrize("tamper",["mode","checksum","short","counter","step","claim","extra","nan"])
def test_parity_refuses_missing_spoofed_or_corrupt_state(tamper):
    off=fake_sidecar("off")
    on=fake_sidecar("on")
    if tamper=="mode":on["summary"]["mode"]="off"
    elif tamper=="checksum":on["summary"]["original_full_state_checksum"]="0"*64
    elif tamper=="short":on["original29_5ms_state_rows"].pop()
    elif tamper=="counter":on["summary"]["original_29dof_completed_5ms_rows"]=3
    elif tamper=="step":
        on["original29_5ms_state_rows"][1]["step"]=1
        on["summary"]["original_full_state_checksum"]=json_sha(on["original29_5ms_state_rows"])
    elif tamper=="claim":on["summary"]["left_wrist_hip_native_physx_truth"]="PROVEN"
    elif tamper=="extra":
        on["original29_5ms_state_rows"][1]["fake"]=5
        on["summary"]["original_full_state_checksum"]=json_sha(on["original29_5ms_state_rows"])
    elif tamper=="nan":
        on["original29_5ms_state_rows"][1]["q"][0]=float("nan")
        # The source checksum itself rejects NaNs; never forge a valid digest.
        with pytest.raises(ValueError):
            json_sha(on["original29_5ms_state_rows"])
    with pytest.raises(ValueError):
        parity.compare_inputs(off,on,policy="frozen_v32")


@pytest.mark.parametrize("mode",["off","on"])
def test_nonfinite_or_early_termination_never_invent_14s_recovery(mode):
    row=fake_sidecar(mode)
    assert len(parity.validate_sidecar(row,mode=mode,expected_rows=2))==2
    with pytest.raises(ValueError,match="TRUNCATED"):
        parity.validate_sidecar(row,mode=mode,expected_rows=2800)


@pytest.mark.parametrize("policy",["frozen_v32","cmu_10_02","cmu_10_05"])
@pytest.mark.parametrize("schema", [False,True])
@pytest.mark.parametrize("phase",ledger.PHASES)
@pytest.mark.parametrize("active",[False,True])
def test_48_source_only_contact_matrix_never_certifies_true_manifold(policy,schema,phase,active):
    row=fake_sidecar()
    row["summary"]["existing_report_api_census"]["pair_report_schema_eligible"]=schema
    pkt=fake_packet(phase,active)
    row["raw_callback_copies"]=[pkt]
    row["summary"]["copied_native_callback_packets"]=1
    outcome=ledger.classify_sidecar(row)
    assert outcome["source_kind"]=="UNATTESTED_SIDECAR_INPUT"
    assert outcome["left_wrist_hip_true_manifold"]=="UNKNOWN"
    assert outcome["absence_of_wrist_hip_contact_proven"] is False
    assert outcome["exact_native_physx_step_proven"] is False
    assert outcome["legal_instep_shot_certified"] is False
    assert outcome["wrist_hip_candidate_count_input"]==int(active)
    if not schema:
        assert outcome["status"]=="REPORT_SCHEMA_UNOBSERVABLE"
    elif active:
        assert outcome["status"]=="NATIVE_PAIR_CANDIDATES_UNTIMED_UNATTESTED"
    else:
        assert outcome["status"]=="OTHER_CALLBACKS_TARGET_ABSENCE_UNKNOWN"


def test_zero_callback_does_not_prove_absence():
    value=ledger.classify_sidecar(fake_sidecar())
    assert value["status"]=="ZERO_CALLBACKS_CONTACT_ABSENCE_UNKNOWN"
    assert not value["absence_of_wrist_hip_contact_proven"]


def test_incomplete_callback_capture_refuses_negative_claim():
    a=fake_sidecar()
    a["summary"]["readout_incomplete"]=True
    a["raw_callback_copies"]=[fake_packet()]
    a["summary"]["copied_native_callback_packets"]=1
    out=ledger.classify_sidecar(a)
    assert out["status"]=="INCOMPLETE_CAPTURE_NO_NEGATIVE_CLAIM"
    assert not out["absence_of_wrist_hip_contact_proven"]


@pytest.mark.parametrize("tamper",[
    "native_pair_certified","source_claim","clock_basis",
    "v149_adjacent_step_candidates_unattested",
    "v149_arrival_phase","v149_untrusted_step_hint_not_native_clock",
    "motor_execution_authorized","all_contact_data_consumed",
])
def test_packet_adversarial_source_and_clock_spoof(tamper):
    packet=fake_packet()
    replacements={
        "native_pair_certified":True,"source_claim":"REAL_PHYSX_VERIFIED",
        "clock_basis":"NATIVE_PHYSICS_TICK_CERTIFIED",
        "v149_adjacent_step_candidates_unattested":[False],
        "v149_arrival_phase":"REAL_TIME_CONFIRMED",
        "v149_untrusted_step_hint_not_native_clock":False,
        "motor_execution_authorized":True,"all_contact_data_consumed":False}
    packet[tamper]=replacements[tamper]
    data=fake_sidecar()
    data["raw_callback_copies"]=[packet]
    data["summary"]["copied_native_callback_packets"]=1
    with pytest.raises(ValueError):
        ledger.classify_sidecar(data)


@pytest.mark.parametrize("value",[[.1],[.1,0,None],[float("nan"),0,0],
                                    [True,0,0]])
def test_malformed_native_impulse_is_not_contact(value):
    p=fake_packet()
    p["headers"][0]["manifold_points"][0]["impulse_world_vector_Ns"]=value
    with pytest.raises(ValueError):
        ledger.candidate_count([p],fake_sidecar()["collider_to_rigid_owner"])


def test_wrong_owner_map_and_noncontact_event_refused():
    fake=fake_sidecar()
    assert len(ledger.candidate_count([fake_packet()],fake["collider_to_rigid_owner"]))==1
    owner=fake["collider_to_rigid_owner"]
    owner["/World/G1/left_hip_roll_link/collision"]="/World/G1/left_elbow_link"
    assert ledger.candidate_count([fake_packet()],owner)==[]
    assert ledger.candidate_count([fake_packet(active=False)],fake["collider_to_rigid_owner"])==[]


@pytest.mark.parametrize("key,value",[
    ("same_native_physics_tick_calibrated",True),
    ("off_on_physical_equality_certified",True),
    ("runtime_positive_negative_controls_passed",True),
    ("left_wrist_hip_native_physx_truth","NO_CONTACT"),
])
def test_laundered_physx_provenance_rejected(key,value):
    s=fake_sidecar()
    s["summary"][key]=value
    with pytest.raises(ValueError):
        ledger.classify_sidecar(s)


def test_source_only_census_does_not_prove_runtime_usd():
    result=ledger.source_reporting_gap()
    sources=result["source_review"]
    assert sources["tools/soccer_world.py"]["has_contact_report_apply"]
    assert sources["tools/v149_run_original_anyphase_readonly.py"]["has_contact_report_apply"] is False
    assert result["report_api_eligibility_for_wrist_hip_runtime"]=="UNKNOWN"
    assert result["no_runtime_physx_started"]
