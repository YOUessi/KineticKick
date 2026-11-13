#!/usr/bin/env python3
"""Independent pure-Python recheck of v148 result; no primary audit calls.

Validates raw C3D-derived peak triplets saved by primary analysis and original
PhysX source hashes against original SHA-qualified files. Not a contact claim.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import permutations,product
import hashlib
import json
import math
from pathlib import Path

from v146_real_200hz_coupled_audit import verified_sources
from v142_native_pair_ledger import safe_write_once

PERM=list(permutations((0,1,2)))
SHIFTS=(-12,-6,0,6,12)


def verify_payload(payload):
    if payload.get("version")!="v0.1.148" or payload.get("source_is_sha_checked_original_C3D") is not True:
        raise ValueError("INDEPENDENT_SOURCE_PROVENANCE_FALSE")
    if payload.get("no_new_training_or_power_claim") is not True or payload.get("is_offline_research_not_robot_actuation") is not True:
        raise ValueError("INDEPENDENT_UNTRUSTED_MOTOR_CLAIMS")
    if payload["provenance"]["weak_event_human_subject_count"]!=1:
        raise ValueError("INDEPENDENT_CROSS_SUBJECT_CLAIM_FALSE")
    if any(payload["provenance"][k] is not False for k in (
        "human_exact_impact_time_measured","human_ground_force_plate_available",
        "professional_skill_certified")):
        raise ValueError("INDEPENDENT_FABRICATED_HUMAN_TRUTH")
    rows=payload["source_grid_720_correlated_cells"]
    if len(rows)!=720:raise ValueError("INDEPENDENT_RAW_GRID_COUNT_DRIFT")
    seen=set()
    groups=defaultdict(list)
    for row in rows:
        clip,side=row["clip"],row["chain"]
        if clip not in ("10_02","10_05") or side not in ("right","left"):
            raise ValueError("INDEPENDENT_CLIP_SIDE_CHANGED")
        if row["anchor_shift_frames"] not in SHIFTS:
            raise ValueError("INDEPENDENT_ANCHOR_SHIFT_CHANGED")
        key=(clip,side,row["span_frames"],row["median_window_frames"],
             row["pre_window_seconds"],row["anchor_shift_frames"])
        if key in seen:raise ValueError("INDEPENDENT_DUPLICATE_REGISTERED_CELL")
        seen.add(key)
        raw=row["peak_indices"]
        if raw is None:
            if row["status"]=="ORDER_MARGIN_PASS":
                raise ValueError("INDEPENDENT_INVALID_PASS_WITHOUT_PEAKS")
            values=None
        else:
            names=["pelvis",side+"_thigh",side+"_shank"]
            if set(raw)!=set(names):raise ValueError("INDEPENDENT_UNVERIFIED_MARKERS")
            values=[raw[key] for key in names]
            if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in values):
                raise ValueError("INDEPENDENT_INVALID_PEAK_FRAME")
            margin=min(values[1]-values[0],values[2]-values[1])>=2
            if (row["status"]=="ORDER_MARGIN_PASS")!=margin:
                raise ValueError("INDEPENDENT_SOURCE_LABEL_FALSIFIED")
        groups[(clip,side)].append((row["anchor_shift_frames"],values))
    if len(groups)!=4 or any(len(v)!=180 for v in groups.values()):
        raise ValueError("INDEPENDENT_GROUP_DENOMINATOR_DRIFT")
    results={}
    for clip in ("10_02","10_05"):
        for side in ("right","left"):
            entries=groups[(clip,side)]
            score=sum(v is not None and v[1]-v[0]>=2 and v[2]-v[1]>=2
                      for _,v in entries)
            shifts={}
            for shift in SHIFTS:
                subset=[v for t,v in entries if t==shift]
                if len(subset)!=36:raise ValueError("INDEPENDENT_SHIFT_DENOMINATOR")
                shifts[str(shift)]=sum(v is not None and v[1]-v[0]>=2 and v[2]-v[1]>=2 for v in subset)
            reported=payload["analysis"]["by_clip_and_anatomical_side"][side+"_"+clip]
            if score!=reported["pass_count"] or any(
                shifts[k]!=v["passes"] for k,v in reported["anchor_shift_control"].items()):
                raise ValueError("INDEPENDENT_PAIRED_SCORE_MISMATCH")
            results[side+"_"+clip]={"pass":score,"shift_passes":shifts}
    per_clip={}
    for clip in ("10_02","10_05"):
        samples=[v for _,v in groups[(clip,"right")]]
        per_clip[clip]=[sum(v is not None and
                            v[j]-v[i]>=2 and v[k]-v[j]>=2 for v in samples)
                        for i,j,k in PERM]
        if per_clip[clip]!=payload["analysis"]["per_clip_six_label_assignment_passes"][clip]:
            raise ValueError("INDEPENDENT_EXCHANGEABILITY_RECHECK_FAILED")
    scores=[a+b for a,b in product(per_clip["10_02"],per_clip["10_05"])]
    identity=scores[0]
    if (len(scores)!=36 or
        identity!=payload["analysis"]["original_right_identity_label_score"] or
        sum(score>=identity for score in scores)!=payload["analysis"]["assignments_with_score_at_least_identity"] or
        [min(scores),max(scores)]!=payload["analysis"]["range_of_label_reassignment_scores"]):
        raise ValueError("INDEPENDENT_36_ASSIGNMENTS_DISAGREE")
    gate=payload["analysis"]["human_training_evidence_gate"]
    if gate["qualified_for_human_to_original29_training"] is not False:
        raise ValueError("INDEPENDENT_FALSE_TRAINING_ADMISSION")
    if payload["analysis"]["this_is_not_a_population_p_value"] is not True:
        raise ValueError("INDEPENDENT_FALSE_POPULATION_INFERENCE")
    sources=verified_sources()
    expected=payload["real_physx_provenance"]["sha256_each_original_case"]
    if set(sources)!=set(expected):
        raise ValueError("INDEPENDENT_ORIGINAL_PHYSX_SOURCE_MISMATCH")
    for name,path in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[name]:
            raise ValueError("INDEPENDENT_ORIGINAL_PHYSX_HASH_DRIFT")
    if (payload["real_physx_provenance"]["historical_original29_real_states_200hz"]!=6628 or
        payload["real_physx_provenance"]["new_physx_episodes"]!=0 or
        payload["real_physx_provenance"]["native_wrist_hip_manifold_truth"]!="UNKNOWN"):
        raise ValueError("INDEPENDENT_FALSE_NEW_PHYSICAL_CLAIM")
    return {
        "provenance":"original_C3D_grid_and_original_PhysX_SHA_cross_checked",
        "independent_anatomical_label_score":identity,
        "independent_36_label_assignments_at_least_identity":sum(score>=identity for score in scores),
        "independent_36_score_range":[min(scores),max(scores)],
        "per_side":results,
        "human_training_eligible":False,
        "population_p_value_not_estimated":True,
        "new_physx_episodes":0,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    result=verify_payload(json.loads(a.source.read_text()))
    digest=safe_write_once(a.output,result)
    print(json.dumps({"sha256":digest,**result},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
