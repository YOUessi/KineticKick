#!/usr/bin/env python3
"""V151 fail-closed native wrist--hip observability and contact decision ledger.

A packet from native subscriber is evidence only after provenance calibration;
sidecar data alone is not a runtime certificate. Missing API/callback is NOT
physical absence. Does not apply USD schemas or start GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from v142_native_pair_ledger import safe_write_once
from v151_native_campaign import frozen

TARGET=frozenset(("left_wrist_yaw_link","left_hip_roll_link"))
PHASES=("BEFORE_FIRST_SCENE_UPDATE","INSIDE_SCENE_UPDATE",
        "BETWEEN_SCENE_UPDATES","AFTER_TERMINAL_SCENE_UPDATE")


def _paths_are_target(header,owners):
    colliders=header.get("collider_paths")
    if not isinstance(colliders,list) or len(colliders)!=2:
        return False
    paths=[owners.get(path) for path in colliders]
    return (all(isinstance(s,str) and s.startswith("/") for s in paths)
            and paths[0]!=paths[1]
            and frozenset(x.rsplit("/",1)[-1] for x in paths)==TARGET)


def candidate_count(packets,owner_map):
    if not isinstance(packets,list) or not isinstance(owner_map,dict):
        raise ValueError("V151_MALFORMED_NATIVE_SIDE_CAR")
    unique=[]
    for item in packets:
        if (item.get("source_claim")!="CPU_OR_LIVE_CALLBACK_UNATTESTED"
            or item.get("native_pair_certified") is not False
            or item.get("all_contact_data_consumed") is not True
            or item.get("motor_execution_authorized") is not False):
            raise ValueError("V151_SPOOFED_NATIVE_ATTESTATION")
        if item.get("v149_arrival_phase") not in PHASES:
            raise ValueError("V151_INVALID_NATIVE_EVENT_ARRIVAL_PHASE")
        if item.get("clock_basis")!="SCENE_UPDATE_ADJACENCY_ONLY":
            raise ValueError("V151_FALSE_CALIBRATED_NATIVE_CLOCK")
        if item.get("v149_untrusted_step_hint_not_native_clock") is not True:
            raise ValueError("V151_FALSE_NATIVE_STEP_ATTESTATION")
        candidates=item.get("v149_adjacent_step_candidates_unattested")
        if (not isinstance(candidates,list) or not candidates
            or any(type(v) is not int or v < 1 for v in candidates)):
            raise ValueError("V151_INVALID_UNATTESTED_STEP_CANDIDATES")
        for head in item.get("headers",[]):
            if (head.get("event_type") not in ("CONTACT_FOUND","CONTACT_PERSISTS")
                or not _paths_are_target(head,owner_map)):
                continue
            positive=False
            for point in head.get("manifold_points",[]):
                impulse=point.get("impulse_world_vector_Ns")
                if not isinstance(impulse,list) or len(impulse)!=3:
                    raise ValueError("V151_BAD_PHYSX_VECTOR_IMPULSE")
                if any(isinstance(v,bool) or not isinstance(v,(int,float))
                       or not (float("-inf") < v < float("inf")) for v in impulse):
                    raise ValueError("V151_NONFINITE_PHYSX_IMPULSE")
                if sum(v*v for v in impulse)>0:
                    positive=True
            if positive:
                unique.append({
                    "collider_paths":list(head["collider_paths"]),
                    "rigid_owners":[owner_map[x] for x in head["collider_paths"]],
                    "untrusted_step_candidates":candidates,
                    "arrival_phase":item["v149_arrival_phase"],
                    "native_pair_certified":False})
    return unique


def classify_sidecar(data, *, expect_mode="on"):
    if not isinstance(data,dict):
        raise ValueError("V151_SIDECAR_NOT_OBJECT")
    summary=data.get("summary")
    if not isinstance(summary,dict) or summary.get("mode")!=expect_mode:
        raise ValueError("V151_OBSERVER_MODE_MISMATCH")
    for k in ("same_native_physics_tick_calibrated",
              "off_on_physical_equality_certified",
              "runtime_positive_negative_controls_passed"):
        if summary.get(k) is not False:
            raise ValueError("V151_UNSUPPORTED_LIVE_AUTHORIZATION:" + k)
    if summary.get("left_wrist_hip_native_physx_truth")!="UNKNOWN":
        raise ValueError("V151_FALSE_NATIVE_CONTACT_CERTIFICATION")
    schema=summary.get("existing_report_api_census")
    if not isinstance(schema,dict):
        raise ValueError("V151_REPORT_SCHEMA_CENSUS_MISSING")
    for key in ("pair_report_schema_eligible","no_events_means_no_contact",
                "runtime_positive_negative_controls_passed"):
        if type(schema.get(key)) is not bool:
            raise ValueError("V151_INVALID_CENSUS_BOOL:"+key)
    if schema["no_events_means_no_contact"] or schema["runtime_positive_negative_controls_passed"]:
        raise ValueError("V151_CENSUS_SELF_ATTESTATION")
    packets=data.get("raw_callback_copies")
    if not isinstance(packets,list) or type(summary.get("copied_native_callback_packets")) is not int:
        raise ValueError("V151_UNVERIFIED_CALLBACK_COUNT")
    if len(packets)!=summary["copied_native_callback_packets"]:
        raise ValueError("V151_CALLBACKS_NOT_FULLY_PRESERVED")
    if expect_mode=="off" and packets:
        raise ValueError("V151_OFF_MODE_HAS_CALLBACKS")
    witnesses=candidate_count(packets,data.get("collider_to_rigid_owner",{}))
    if expect_mode=="off":
        status="OFF_NOT_OBSERVING"
    elif summary.get("readout_incomplete"):
        status="INCOMPLETE_CAPTURE_NO_NEGATIVE_CLAIM"
    elif not schema["pair_report_schema_eligible"]:
        status="REPORT_SCHEMA_UNOBSERVABLE"
    elif witnesses:
        status="NATIVE_PAIR_CANDIDATES_UNTIMED_UNATTESTED"
    elif not packets:
        status="ZERO_CALLBACKS_CONTACT_ABSENCE_UNKNOWN"
    else:
        status="OTHER_CALLBACKS_TARGET_ABSENCE_UNKNOWN"
    return dict(
        source_kind="UNATTESTED_SIDECAR_INPUT",
        status=status,
        complete_29dof_state_rows_input=summary.get("original_29dof_completed_5ms_rows"),
        report_schema_eligible_input=schema["pair_report_schema_eligible"],
        callback_copies_input=len(packets),
        wrist_hip_candidate_count_input=len(witnesses),
        wrist_hip_candidate_examples=witnesses[:10],
        exact_native_physx_step_proven=False,
        isolated_native_positive_control_proven=False,
        isolated_native_negative_control_proven=False,
        read_only_off_on_original_state_parity_proven=False,
        left_wrist_hip_true_manifold="UNKNOWN",
        absence_of_wrist_hip_contact_proven=False,
        legal_instep_shot_certified=False,
        human_kinetic_chain_learned=False,
    )


def source_reporting_gap():
    """Static, verifiable *source only* scan, never claim live USD state."""
    root=Path(__file__).resolve().parents[1]
    paths=[
        "tools/soccer_world.py", # shoe patch: original actor does not imply wrist
        "tools/v149_run_original_anyphase_readonly.py",
        "tools/v149_native_callback_capture.py",
        "tools/v096_contact_probe.py", # separately instrumented diagnostic
    ]
    facts={}
    for rel in paths:
        text=(root/rel).read_text()
        facts[rel]={
            "file_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "has_contact_report_apply": "PhysxContactReportAPI.Apply(" in text,
            "has_report_schema_read_only": "HasAPI(PhysxSchema.PhysxContactReportAPI)" in text,
            "source_read_does_not_prove_runtime_usd":True,
        }
    return dict(source_review=facts,
                original_runner_not_authorized_to_apply_contact_api=True,
                report_api_eligibility_for_wrist_hip_runtime="UNKNOWN",
                no_runtime_physx_started=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv)
    report=source_reporting_gap()
    if a.sidecar:
        if a.sidecar.is_symlink():
            raise ValueError("V151_SYMLINK_SIDECAR")
        report["input_ledger"]=classify_sidecar(
            json.loads(a.sidecar.read_text()))
    else:
        report["input_ledger"]=None
    report["frozen_protocol"]="v0.1.151"
    report["frozen_sha"]=frozen()["parent_sha"]
    h=safe_write_once(a.output,report)
    print(json.dumps(dict(output=str(a.output),sha256=h,
                          physx_started=0,contact_truth="UNKNOWN")))


if __name__=="__main__":
    main()
