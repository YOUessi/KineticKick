#!/usr/bin/env python3
"""Source-SHA-locked human-kinematic timing falsification, never force/impact truth.

Two same-subject 120Hz CMU clips. Whole-clip anatomical-label permutations
preserve correlation of the 180 overlapping analysis choices. This is a
diagnostic of phase-order specificity, NOT population inference and NOT an RL
policy, professional kicking supervision, or PhysX evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

import v144_human_temporal_source_and_physical_audit as human
import v145_phase_order_robustness as source
from v142_native_pair_ledger import safe_write_once
from v146_real_200hz_coupled_audit import verified_sources, split_real_5ms_rows

ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path("configs/v148_frozen_human_chain_exchangeability.json")
FREEZE_SHA="8da717747c86c77aca0c146c9f6c40c875adec93"
TRIPLE=("pelvis","right_thigh","right_shank")
SHIFTS=(-12,-6,0,6,12)
PERMS=tuple(itertools.permutations(range(3)))
PASS="ORDER_MARGIN_PASS"


def frozen():
    locked=subprocess.check_output(["git","-C",str(ROOT),"show",FREEZE_SHA+":"+str(CONFIG)])
    if (ROOT/CONFIG).read_bytes()!=locked:
        raise ValueError("V148_FROZEN_PROTOCOL_DRIFT")
    cfg=json.loads(locked)
    if (cfg.get("version")!="v0.1.148" or
        cfg.get("parent_sha")!="f4761baefd1cdbcf2ae951ecc97b086139818c44" or
        cfg.get("grid_rows")!=720 or cfg.get("per_chain_clip")!=180 or
        cfg.get("per_clip_anchor_shift_rows")!=36 or
        cfg.get("anchor_shifts_frames")!=list(SHIFTS) or
        cfg.get("strict_minimum_lead_frames")!=2 or
        cfg.get("rate_hz")!=120 or
        cfg.get("exact_two_clip_label_assignments")!=36 or
        cfg.get("segment_label_permutations_per_clip")!=6 or
        cfg.get("pass_rate_floor_each_clip")!=.8 or
        cfg.get("pass_rate_floor_each_anchor_shift")!=.6 or
        cfg.get("required_independent_human_subjects_for_train_eval")!=3 or
        cfg.get("expected_original_historical_physx_rows")!=6628 or
        cfg.get("expected_original_historical_physx_episodes")!=3 or
        any(v is not True for v in cfg["privacy_and_provenance"].values()) or
        any(v is not False for k,v in cfg["data_status"].items() if k!="human_motion_timing_is_weak_proxy_only") or
        cfg["data_status"]["human_motion_timing_is_weak_proxy_only"] is not True):
        raise ValueError("V148_UNFROZEN_ASSUMPTIONS")
    return cfg


def _strict_number(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(float(x))


def validate_cell(row, *, min_margin=2):
    """Validate exact source grid status against unmodified, non-imputed peaks.

    Return (peaks, ordered) where peaks is None if the source itself had an
    incomplete triplet. Never rescue a missing event by relabelling it.
    """
    if row.get("chain") not in ("right","left") or row.get("clip") not in ("10_02","10_05"):
        raise ValueError("V148_CELL_SOURCE_OR_CHAIN_CHANGED")
    if row.get("anchor_shift_frames") not in SHIFTS or isinstance(row.get("anchor_shift_frames"),bool):
        raise ValueError("V148_CELL_ANCHOR_SHIFT_DRIFT")
    peaks=row.get("peak_indices")
    if peaks is None:
        if row.get("status")==PASS:
            raise ValueError("V148_PASS_WITHOUT_ACTUAL_MARKERS")
        return None, False
    expected=("pelvis",row["chain"]+"_thigh",row["chain"]+"_shank")
    if not isinstance(peaks,dict) or set(peaks)!=set(expected):
        raise ValueError("V148_INCOMPLETE_OR_UNTRUSTED_PEAK_TRIPLE")
    values=tuple(peaks[k] for k in expected)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in values):
        raise ValueError("V148_BAD_PEAK_FRAME")
    leads=(values[1]-values[0],values[2]-values[1])
    if row.get("lead_frames")!=list(leads):
        raise ValueError("V148_SOURCE_LEAD_FRAMES_CHANGED")
    valid=row.get("valid_fractions")
    if not isinstance(valid,dict) or set(valid)!=set(expected) or not all(
        _strict_number(valid[k]) and .75<=valid[k]<=1.0 for k in expected):
        raise ValueError("V148_INVALID_SOURCE_MARKER_COVERAGE")
    actually_pass=min(leads)>=min_margin
    if (row.get("status")==PASS)!=actually_pass:
        raise ValueError("V148_SOURCE_RESULT_CONTRADICTION")
    return values, actually_pass


def analysis_key(r):
    return (r["clip"],r["chain"],r["span_frames"],r["median_window_frames"],
            r["pre_window_seconds"],r["anchor_shift_frames"])


def validated_grid(rows,cfg):
    if len(rows)!=cfg["grid_rows"]:raise ValueError("V148_INCOMPLETE_GRID")
    index={}
    for row in rows:
        k=analysis_key(row)
        if k in index:raise ValueError("V148_DUPLICATE_ANALYSIS_CHOICE")
        index[k]=row
        validate_cell(row,min_margin=cfg["strict_minimum_lead_frames"])
    expected={(clip,side) for clip in cfg["clip_ids"] for side in ("right","left")}
    actual={(k[0],k[1]) for k in index}
    if expected!=actual:raise ValueError("V148_SOURCE_SIDE_CLIP_DENOMINATOR")
    out=defaultdict(list)
    for key,row in index.items():out[key[:2]].append(row)
    for key,cells in out.items():
        if len(cells)!=180 or Counter(r["anchor_shift_frames"] for r in cells)!={x:36 for x in SHIFTS}:
            raise ValueError("V148_MISSING_REGISTERED_GRID_CELL")
    for clip in cfg["clip_ids"]:
        right={analysis_key(r)[2:] for r in out[(clip,"right")]}
        left={analysis_key(r)[2:] for r in out[(clip,"left")]}
        if right!=left:raise ValueError("V148_CONTRALATERAL_PAIRING_DRIFT")
    return out


def counts_for_label_assignments(rows, margin):
    """Only permute labels, never rotate raw signals, windows or missing markers."""
    ready=[validate_cell(r,min_margin=margin)[0] for r in rows]
    return [sum(v is not None and
                v[p[1]]-v[p[0]]>=margin and v[p[2]]-v[p[1]]>=margin
                for v in ready) for p in PERMS]


def lead_hist(rows,margin):
    gaps=[validate_cell(r,min_margin=margin)[0] for r in rows]
    gaps=[(v[1]-v[0],v[2]-v[1]) for v in gaps if v is not None]
    if not gaps:return {"complete_triplets":0,"lead1_frames":None,"lead2_frames":None}
    a=np.asarray(gaps,float)
    return {"complete_triplets":len(gaps),
            "gap_pelvis_to_thigh_frames_10_50_90_percentile":np.percentile(a[:,0],[10,50,90]).tolist(),
            "gap_thigh_to_shank_frames_10_50_90_percentile":np.percentile(a[:,1],[10,50,90]).tolist(),
            "unique_observed_peak_pairs":len(set(map(tuple,gaps)))}


def audit(rows, quality, cfg):
    grouped=validated_grid(rows,cfg)
    subjects={quality[clip]["subject"] for clip in cfg["clip_ids"]}
    if len(subjects)!=1 or subjects!={"10"}:
        raise ValueError("V148_UNREGISTERED_NEW_SUBJECT_OR_SOURCE")
    by_side={}
    for clip in cfg["clip_ids"]:
        for side in ("right","left"):
            cell=grouped[(clip,side)]
            pass_n=sum(r["status"]==PASS for r in cell)
            per_shift={}
            for shift in SHIFTS:
                subset=[r for r in cell if r["anchor_shift_frames"]==shift]
                per_shift[str(shift)]={"passes":sum(r["status"]==PASS for r in subset),
                                       "denominator":36,
                                       "pass_fraction":sum(r["status"]==PASS for r in subset)/36}
            by_side[side+"_"+clip]={
                "pass_count":pass_n,"registered_analysis_choices":180,
                "pass_fraction_of_correlated_choices":pass_n/180,
                "complete_triplets":sum(r["peak_indices"] is not None for r in cell),
                "anchor_shift_control":per_shift,
                "lead_gap_summary_on_complete_triplets":lead_hist(cell,cfg["strict_minimum_lead_frames"]),
            }
    permutations_by_clip={
        clip:counts_for_label_assignments(grouped[(clip,"right")],cfg["strict_minimum_lead_frames"])
        for clip in cfg["clip_ids"]}
    for clip in cfg["clip_ids"]:
        if permutations_by_clip[clip][0]!=by_side["right_"+clip]["pass_count"]:
            raise ValueError("V148_IDENTITY_LABEL_SCORE_DISAGREEMENT")
    identity_score=sum(permutations_by_clip[clip][0] for clip in cfg["clip_ids"])
    diagnostic=[]
    for first in range(6):
        for second in range(6):
            diagnostic.append({"clip1_assignment":first,"clip2_assignment":second,
                               "score":permutations_by_clip[cfg["clip_ids"][0]][first]+
                                       permutations_by_clip[cfg["clip_ids"][1]][second]})
    scores=[d["score"] for d in diagnostic]
    # No population p-values: the two original clips belong to one subject and
    # label-exchangeability is only an exploratory counterfactual diagnostic.
    gate_by_clip={clip:by_side["right_"+clip]["pass_fraction_of_correlated_choices"]>=cfg["pass_rate_floor_each_clip"]
                  for clip in cfg["clip_ids"]}
    gate_by_shift={clip:all(v["pass_fraction"]>=cfg["pass_rate_floor_each_anchor_shift"]
                           for v in by_side["right_"+clip]["anchor_shift_control"].values())
                   for clip in cfg["clip_ids"]}
    gate={
        "pass_rate_each_clip_at_least_80pct":gate_by_clip,
        "each_shift_in_each_clip_at_least_60pct":gate_by_shift,
        "independent_subject_count":len(subjects),
        "three_independent_contact_annotated_subjects":len(subjects)>=cfg["required_independent_human_subjects_for_train_eval"],
        "exact_human_right_instep_contact_truth":cfg["data_status"]["human_exact_instep_impact"],
        "synchronized_human_ground_reaction_COP":cfg["data_status"]["human_synchronized_grf_cop"],
        "verified_professional_kicking_skill":cfg["data_status"]["elite_professional_certified"],
        "subject_disjoint_impact_test":cfg["data_status"]["subject_disjoint_contact_eval"],
    }
    gate["qualified_for_human_to_original29_training"]=(
        all(gate_by_clip.values()) and all(gate_by_shift.values()) and
        gate["three_independent_contact_annotated_subjects"] and
        gate["exact_human_right_instep_contact_truth"] and
        gate["synchronized_human_ground_reaction_COP"] and
        gate["verified_professional_kicking_skill"] and
        gate["subject_disjoint_impact_test"])
    return {
        "per_clip_anatomical_label_assignment_order":[list(p) for p in PERMS],
        "per_clip_six_label_assignment_passes":permutations_by_clip,
        "two_clips_36_aggregate_assignment_scores":diagnostic,
        "original_right_identity_label_score":identity_score,
        "assignments_with_score_at_least_identity":sum(v>=identity_score for v in scores),
        "range_of_label_reassignment_scores":[min(scores),max(scores)],
        "this_is_not_a_population_p_value":True,
        "two_clips_one_subject_not_36_independent_observations":True,
        "by_clip_and_anatomical_side":by_side,
        "human_training_evidence_gate":gate
    }


def recheck_real_history(cfg):
    sources=verified_sources()
    expected={"frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}
    if set(sources)!=set(expected):
        raise ValueError("V148_ORIGINAL_REAL_CASES_CHANGED")
    total=0
    shas={}
    for key, path in sources.items():
        raw=path.read_bytes()
        history=json.loads(raw)
        rows, notes=split_real_5ms_rows(history["traces"][0],expected[key])
        if key!="cmu_10_05" and notes:
            raise ValueError("V148_FALSE_TERMINAL_ANNOTATION")
        total+=len(rows)
        shas[key]=hashlib.sha256(raw).hexdigest()
    if total!=cfg["expected_original_historical_physx_rows"]:
        raise ValueError("V148_REAL_PHYSX_DENOMINATOR_CHANGED")
    return {"sha256_each_original_case":shas,"historical_real_physx_episodes":len(sources),
            "historical_original29_real_states_200hz":total,"new_physx_episodes":0,
            "new_physx_strict_legal_instep_success_rate":None,
            "native_wrist_hip_manifold_truth":"UNKNOWN",
            "off_on_live_physx_noninterference":"NOT_TESTED"}


def run(path):
    cfg=frozen()
    v145_cfg=source.frozen()
    human_cfg=human.frozen()
    quality,manifest=human.fetch_qualified_sources(human_cfg)
    if set(cfg["clip_ids"])!=set(v145_cfg["weak_anchor_clips"]):
        raise ValueError("V148_QUALIFIED_WEAK_EVENT_CLIPS_CHANGED")
    rows=source.compute_grid(v145_cfg,quality)
    analysis=audit(rows,quality,cfg)
    provenance={
        "source_original_clip_count":len(quality),
        "source_original_frame_count":sum(r["original_C3D_frames"] for r in quality.values()),
        "weak_event_clips":list(cfg["clip_ids"]),
        "weak_event_human_subject_count":len({quality[k]["subject"] for k in cfg["clip_ids"]}),
        "human_exact_impact_time_measured":False,
        "human_ground_force_plate_available":False,
        "professional_skill_certified":False,
    }
    if provenance["source_original_clip_count"]!=6 or provenance["source_original_frame_count"]!=3358:
        raise ValueError("V148_HUMAN_RAW_DENOMINATOR_DRIFT")
    report={
        "version":cfg["version"],"protocol_commit_sha":FREEZE_SHA,
        "source_is_sha_checked_original_C3D":True,
        "provenance":provenance,
        "original_source_grid_rows":len(rows),
        "analysis":analysis,
        "real_physx_provenance":recheck_real_history(cfg),
        "no_new_training_or_power_claim":True,
        "is_offline_research_not_robot_actuation":True,
        "source_grid_720_correlated_cells":rows,
    }
    digest=safe_write_once(path,report)
    return report,digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result,digest=run(args.output)
    print(json.dumps({
        "output":str(args.output),"sha256":digest,
        "identity_right_total":result["analysis"]["original_right_identity_label_score"],
        "label_assignment_rank":result["analysis"]["assignments_with_score_at_least_identity"],
        "label_assignment_score_range":result["analysis"]["range_of_label_reassignment_scores"],
        "by_side":{k:v["pass_count"] for k,v in result["analysis"]["by_clip_and_anatomical_side"].items()},
        "human_training_eligible":result["analysis"]["human_training_evidence_gate"]["qualified_for_human_to_original29_training"],
        "historical_physx_states":result["real_physx_provenance"]["historical_original29_real_states_200hz"],
        "new_physx_episodes":0},indent=2))


if __name__=="__main__":main()
