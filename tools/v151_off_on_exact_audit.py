#!/usr/bin/env python3
"""V151 strict OFF/ON full G1 sidecar parity: input-level equality, not native proof.

Preserves all first and per-field deviations, with canonical JSON and physical
field checking. Does NOT equate a copied event with a native PhysX event.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from v146_native_original_bridge import PHYSICS_FIELDS, json_sha
from v142_native_pair_ledger import verify_original_state, safe_write_once
from v151_native_campaign import frozen

MEASURED = tuple(sorted(PHYSICS_FIELDS))
VECTOR_FIELDS = {"q":29,"dq":29,"motor_torque":29,
                 "com_world":3,"left_foot_net_force":3,
                 "right_foot_net_force":3,"nonfoot_net_force":3,
                 "ball_pose":3,"ball_twist":3}
POSE_FIELDS={"base_pose":{"position":3,"quaternion":4},
             "base_twist":{"linear":3,"angular":3}}


def _flatten(row):
    if set(row)!=set(PHYSICS_FIELDS):
        raise ValueError("V151_MISSING_OR_EXTRA_FULL_STATE_FIELD")
    result={"step":row["step"]}
    for field,n in VECTOR_FIELDS.items():
        value=row[field]
        if len(value)!=n:
            raise ValueError("V151_BAD_VECTOR_LENGTH_" + field)
        for i,x in enumerate(value):
            result[field+"["+str(i)+"]"]=float(x)
    for field, subfields in POSE_FIELDS.items():
        obj=row[field]
        if set(obj)!=set(subfields):
            raise ValueError("V151_BAD_BASE_COMPONENTS")
        for tag,n in subfields.items():
            if len(obj[tag])!=n:
                raise ValueError("V151_BAD_BASE_VECTOR_" + tag)
            for i,x in enumerate(obj[tag]):
                result[field+"."+tag+"["+str(i)+"]"]=float(x)
    if not all(math.isfinite(float(x)) for x in result.values()):
        raise ValueError("V151_NONFINITE_PHYSICS_COMPONENT")
    return result


def validate_sidecar(data, *, mode, expected_rows):
    if not isinstance(data,dict) or not isinstance(data.get("summary"),dict):
        raise ValueError("V151_MISSING_SIDECAR_SUMMARY")
    summary=data["summary"]
    if mode not in ("off","on") or summary.get("mode")!=mode:
        raise ValueError("V151_WRONG_OBSERVER_MODE")
    rows=data.get("original29_5ms_state_rows")
    if not isinstance(rows,list) or len(rows)!=expected_rows:
        raise ValueError("V151_TRUNCATED_OR_UNEXPECTED_PHYSICS_ROWS")
    if summary.get("original_29dof_completed_5ms_rows")!=len(rows):
        raise ValueError("V151_STATE_COUNT_MISMATCH")
    if summary.get("original_full_state_checksum")!=json_sha(rows):
        raise ValueError("V151_SIDECAR_FULL_STATE_CHECKSUM_MISMATCH")
    if mode=="off" and (data.get("raw_callback_copies") or
                        summary.get("copied_native_callback_packets")!=0):
        raise ValueError("V151_OFF_MODE_REPORTED_CALLBACKS")
    if summary.get("left_wrist_hip_native_physx_truth")!="UNKNOWN":
        raise ValueError("V151_UNATTESTED_CONTACT_SELF_CERTIFICATION")
    if summary.get("same_native_physics_tick_calibrated") is not False:
        raise ValueError("V151_UNSUPPORTED_NATIVE_TICK_CERTIFICATION")
    if summary.get("off_on_physical_equality_certified") is not False:
        raise ValueError("V151_UNSUPPORTED_PARITY_SELF_CERTIFICATION")
    for i,row in enumerate(rows,1):
        verify_original_state(row)
        if row["step"]!=i:
            raise ValueError("V151_DUPLICATE_OR_SKIPPED_PHYSICS_TICK")
        _flatten(row)
    return rows


def compare_inputs(off,on, *, policy):
    expected=frozen()["history_expected_rows"].get(policy)
    if expected is None:
        raise ValueError("V151_UNKNOWN_FROZEN_POLICY")
    off_rows=validate_sidecar(off,mode="off",expected_rows=expected)
    on_rows=validate_sidecar(on,mode="on",expected_rows=expected)
    field_max={}
    mismatch_rows=0
    mismatch_components=0
    first=None
    for i,(offrow,onrow) in enumerate(zip(off_rows,on_rows),1):
        a,b=_flatten(offrow),_flatten(onrow)
        if set(a)!=set(b):
            raise ValueError("V151_PARITY_DIFFERENT_FIELD_SET")
        changed=False
        for field in sorted(a):
            if a[field]!=b[field]:
                difference=abs(a[field]-b[field])
                if not changed:
                    mismatch_rows+=1
                    changed=True
                mismatch_components+=1
                field_max[field]=max(field_max.get(field,0.),difference)
                if first is None:
                    first={"step":i,"time_s":round(i*.005,9),
                           "field":field,"off":a[field],"on":b[field],
                           "abs_delta":difference}
    equality=mismatch_components==0
    return {
        "policy":policy,"recorded_input_frames_each":expected,
        "full_state_variables_checked_per_frame":len(_flatten(off_rows[0]))-1,
        "input_level_all_common_original29_physics_equal":equality,
        "first_divergence":first,"mismatching_frames":mismatch_rows,
        "mismatching_values":mismatch_components,
        "max_abs_delta_per_component":field_max,
        "off_full_state_sha256":json_sha(off_rows),
        "on_full_state_sha256":json_sha(on_rows),
        "on_native_callbacks_copied":len(on.get("raw_callback_copies",[])),
        "off_native_callbacks_copied":0,
        "read_only_nonintrusion_certified_from_cpu_inputs":False,
        "pair_native_physx_truth":"UNKNOWN",
        "event_absence_implies_no_contact":False,
        "strict_kick_success":False,
        "full_14s_recovery_confirmed":False,
        "historical_short_run_does_not_prove_fall":policy=="cmu_10_05",
        "physical_pairing_provenance":"INPUT_SIDECARS_ONLY_NOT_RUNTIME_ATTESTATION",
    }


def load_and_compare(off_path,on_path,*,policy):
    for path in (off_path,on_path):
        if Path(path).is_symlink():
            raise ValueError("V151_SYMLINK_EVIDENCE_REJECTED")
    off=json.loads(Path(off_path).read_text())
    on=json.loads(Path(on_path).read_text())
    return compare_inputs(off,on,policy=policy)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off",type=Path,required=True)
    ap.add_argument("--on",type=Path,required=True)
    ap.add_argument("--policy",required=True,choices=list(frozen()["policies"]))
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv)
    value=load_and_compare(a.off,a.on,policy=a.policy)
    digest=safe_write_once(a.output,value)
    print(json.dumps(dict(policy=a.policy,all_equal=value[
        "input_level_all_common_original29_physics_equal"],output=str(a.output),
        sha256=digest,miss_rows=value["mismatching_frames"])))


if __name__=="__main__":
    main()
