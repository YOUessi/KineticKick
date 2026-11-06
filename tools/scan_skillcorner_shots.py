"""Bounded scan of one licensed SkillCorner match; never relax feature gates.

Streams the pinned ZIP member, retaining only +/-1.5 s shooter windows. Original
compressed data remains local. No whole-match JSONL, network, training or GPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from build_skillcorner_coordination import CONFIG_PATH, ROOT, build_coordination, verify_sources

MATCH_ID = 1925299
ARCHIVE_SHA256 = '0415665098d594fb5681815000f7a6312037788a3237910a9f41bb8de5c336a0'
ARCHIVE_BYTES = 623408231
BYTE_BUDGET = 750000000
CORE_GROUPS = {
    'pelvis_shoulder': ['shoulder_pelvis'],
    'trunk_leg': ['trunk_left_thigh', 'trunk_right_thigh'],
    'leg_chain': ['left_leg_chain', 'right_leg_chain'],
    'upper_arm_leg': ['left_arm_right_thigh', 'right_arm_left_thigh'],
}
CORE_JOINTS = ['midHip','neck','lShoulder','rShoulder','lElbow','rElbow',
               'lHip','rHip','lKnee','rKnee','lAnkle','rAnkle']


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def write_bounded(path, blob, root, budget=BYTE_BUDGET):
    """Count archive plus every output, rejecting growth before a write."""
    path, root = Path(path), Path(root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output must stay inside budget directory')
    prior = path.stat().st_size if path.exists() else 0
    if directory_bytes(root)-prior+len(blob) > budget:
        raise ValueError('Total archive/window byte budget would be exceeded')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(blob)
    return {'path':str(path.relative_to(root)), 'size_bytes':len(blob), 'sha256':hashlib.sha256(blob).hexdigest()}


def plan_windows(events):
    plans, excluded = [], []
    for event in events:
        if event.get('event_type') != 'player_possession' or event.get('end_type') != 'shot':
            continue
        if int(event['match_id']) != MATCH_ID:
            raise ValueError('Only match 1925299 is authorized for this scan')
        if event.get('is_header') != 'False':
            excluded.append({'event':event, 'reason':'header' if event.get('is_header') == 'True' else 'header status unknown'})
            continue
        center = int(event['frame_end'])*2.5
        start, end = math.ceil(center-37.5), math.floor(center+37.5)
        reasons = []
        if event.get('fully_extrapolated') != 'False':
            reasons.append('event fully extrapolated or extrapolation status unknown')
        if event.get('is_player_possession_end_matched') != 'True':
            reasons.append('event end not matched or matching status unknown')
        plans.append({'event':event, 'pose_anchor':center, 'frame_start':start, 'frame_end':end,
                      'expected_frames':end-start+1, 'event_quality_exclusions':reasons})
    if len({p['event']['event_id'] for p in plans}) != len(plans):
        raise ValueError('Duplicate event id in planned windows')
    return plans, excluded


def stream_windows(archive, plans, progress=None):
    """Memory bounded by at most the planned window records, not match length."""
    windows = {p['event']['event_id']:[] for p in plans}
    count, previous, first = 0, None, None
    with zipfile.ZipFile(archive) as z:
        member = z.getinfo('1925299.jsonl')
        with z.open(member) as handle:
            for line in handle:
                frame = json.loads(line)
                number = frame['frame']
                if not isinstance(number,int) or (previous is not None and number != previous+1):
                    raise ValueError('Full-match frames are not consecutive integers')
                if first is None:
                    first = number
                previous = number
                count += 1
                for plan in plans:
                    if plan['frame_start'] <= number <= plan['frame_end']:
                        player_id = int(plan['event']['player_id'])
                        # Retain raw shared frame fields; only other players are removed.
                        selected = dict(frame)
                        selected['player_data'] = [p for p in frame['player_data'] if p['player_id'] == player_id]
                        windows[plan['event']['event_id']].append(selected)
                if progress is not None and count % 25000 == 0:
                    progress(count)
        stats = {'frames_scanned':count,'first_frame':first,'last_frame':previous,
                 'jsonl_member':member.filename,'uncompressed_member_bytes':member.file_size,
                 'retained_frame_records':sum(len(w) for w in windows.values()),
                 'max_planned_frame_records':sum(p['expected_frames'] for p in plans),
                 'other_zip_members_not_extracted':[i.filename for i in z.infolist() if i.filename != member.filename],
                 'whole_jsonl_written_to_disk':False,'other_player_data_retained':False}
    if stats['retained_frame_records'] > stats['max_planned_frame_records']:
        raise ValueError('Window memory bound exceeded')
    return windows, stats


def _runs(mask):
    indices = np.flatnonzero(mask)
    return [] if not len(indices) else np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)


def _quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return None if not finite.size else dict(zip(('min','median','p95','max'),np.quantile(finite,[0,.5,.95,1]).tolist()))


def summarize_coordination(arrays, anchor):
    frames = arrays['source_frame_numbers']
    relation_names = list(arrays['relation_names'])
    quality = arrays['relation_quality_pass']
    groups = {}
    for name, alternatives in CORE_GROUPS.items():
        indices = [relation_names.index(n) for n in alternatives if n in relation_names]
        groups[name] = quality[:,indices].any(axis=1) if indices else np.zeros(len(frames),bool)
    simultaneous = np.stack(list(groups.values()),axis=1).all(axis=1)
    runs = _runs(simultaneous)
    covering = [r for r in runs if frames[r[0]] <= math.floor(anchor) and frames[r[-1]] >= math.ceil(anchor)]
    near = np.abs(frames-anchor) <= 12.5
    group_near_counts = {name:int((mask & near).sum()) for name,mask in groups.items()}
    joint_names = list(arrays.get('joint_names',[]))
    core_ids = [joint_names.index(n) for n in CORE_JOINTS if n in joint_names]
    errors = arrays['joint_predicted_p90_cm'][:,core_ids] if core_ids else arrays['joint_predicted_p90_cm']
    return {
        'frames':len(frames),'joint_geometry_valid_fraction':float(arrays['joint_geometry_valid'].mean()),
        'joint_quality_pass_fraction':float(arrays['joint_quality_pass'].mean()),
        'player_detected_fraction':float(arrays['player_is_detected'].mean()),
        'ball_detected_fraction':float(arrays['ball_is_detected'].mean()),
        'relation_geometry_valid_fraction':float(arrays['relation_valid'].mean()),
        'all_joint_predicted_p90_cm':_quantiles(arrays['joint_predicted_p90_cm']),
        'core_joint_predicted_p90_cm':_quantiles(errors),
        'relation_quality_frames':{str(n):int(quality[:,i].sum()) for i,n in enumerate(relation_names)},
        'core_groups_quality_frames':{name:int(mask.sum()) for name,mask in groups.items()},
        'core_groups_quality_frames_near_anchor_0_5s':group_near_counts,
        'simultaneous_core_frames':int(simultaneous.sum()),
        'simultaneous_core_frames_near_anchor_0_5s':int((simultaneous & near).sum()),
        'minimum_group_frames_near_anchor_0_5s':min(group_near_counts.values()),
        'summed_group_frames_near_anchor_0_5s':sum(group_near_counts.values()),
        'longest_simultaneous_core_run_frames':max((len(r) for r in runs),default=0),
        'longest_core_run_covering_anchor_frames':max((len(r) for r in covering),default=0),
        'simultaneous_core_at_anchor_bracket':bool(covering),
        'coordination_screen_exclusions': (
            ([] if covering else ['no simultaneous core support across the event anchor bracket'])
            + ['no quality-pass frames for core group '+name for name,mask in groups.items() if not mask.any()]),
        'simultaneous_core_runs':[{'start_frame':int(frames[r[0]]),'end_frame':int(frames[r[-1]]),
                                   'frames':len(r),'elapsed_s':(len(r)-1)/25} for r in runs],
    }


def _rank(row):
    q = row.get('quality') or {}
    errors = q.get('core_joint_predicted_p90_cm')
    return (not row['exclusions'], q.get('longest_core_run_covering_anchor_frames',0),
            q.get('simultaneous_core_frames_near_anchor_0_5s',0),q.get('minimum_group_frames_near_anchor_0_5s',0),
            q.get('summed_group_frames_near_anchor_0_5s',0),q.get('joint_geometry_valid_fraction',0),
            -errors['p95'] if errors else -float('inf'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,default=ROOT/'data/research_human_kicks/skillcorner_match1925299_scan_v1')
    parser.add_argument('--source-dir',type=Path,default=ROOT/'data/research_human_kicks/skillcorner_probe_20261107')
    args = parser.parse_args()
    work = args.work_dir
    archive = work/'1925299.jsonl.zip'
    if archive.stat().st_size != ARCHIVE_BYTES or _sha(archive) != ARCHIVE_SHA256:
        raise ValueError('Pinned complete archive size/hash mismatch')
    if directory_bytes(work) > BYTE_BUDGET:
        raise ValueError('Existing experiment exceeds authorized byte budget')
    config = json.loads(CONFIG_PATH.read_text())
    verify_sources(args.source_dir,config)
    match = json.loads((args.source_dir/'1925299_match.json').read_text())
    with (args.source_dir/'1925299_dynamic_events.csv').open() as handle:
        plans, excluded = plan_windows(list(csv.DictReader(handle)))
    plan_record = {'match_id':MATCH_ID,'half_window_s':1.5,'nonheader_shots':plans,'excluded_header_or_unknown_shots':excluded,
                   'quality_policy_unchanged':config['quality_policy'],'core_groups':CORE_GROUPS}
    artifacts = [write_bounded(work/'shot_plan.json',_json_bytes(plan_record),work)]
    windows, stream_stats = stream_windows(archive,plans,progress=lambda n:print(json.dumps({'frames_streamed':n}),flush=True))
    results = []
    for plan in plans:
        event = plan['event']; identity = event['event_id']; rows = windows[identity]
        stem = f"{identity}_{event['player_id']}"
        raw = ''.join(json.dumps(f,separators=(',',':'),ensure_ascii=False)+'\n' for f in rows).encode()
        raw_artifact = write_bounded(work/'windows'/(stem+'.jsonl.gz'),gzip.compress(raw,mtime=0),work)
        artifacts.append(raw_artifact)
        result = {'event':event,'planned_frame_start':plan['frame_start'],'planned_frame_end':plan['frame_end'],
                  'expected_frames':plan['expected_frames'],'observed_frames':len(rows),'raw_window':raw_artifact,
                  'exclusions':list(plan['event_quality_exclusions']), 'quality':None,
                  'qualification_unknown':['instep','kick_side','maximal_strength','precise_contact','independent_ground_truth'],
                  'manual_video_adjudication_performed':False}
        if len(rows) != plan['expected_frames']:
            result['exclusions'].append('incomplete frame coverage')
        player = next((p for p in match['players'] if p['id']==int(event['player_id'])),None)
        source = {'source_class':'broadcast_video_pose_estimate','coordinate_units':'m','source_sha256':ARCHIVE_SHA256,
                  'archive_bytes':ARCHIVE_BYTES,'source_window_sha256':raw_artifact['sha256'],
                  'hf_revision':config['hf_revision'],'github_revision':config['github_revision'],
                  'professional_participation_verified':player is not None,
                  'professional_participation_evidence':{'scope':'Professional match roster; framewise actor ID remains vendor-estimated',
                    'player_id':event['player_id'],'player_name':event['player_name'],'official_match_report':config['professional_participation_evidence']['official_club_report']},
                  'data_license':config['data_license'],'other_players_removed_from_window':True,
                  'event_quality_exclusions':plan['event_quality_exclusions']}
        try:
            arrays, meta = build_coordination(rows,match,event,source=source)
        except (ValueError,TypeError,KeyError) as exc:
            result['exclusions'].append('adapter rejected window: '+str(exc))
        else:
            result['quality'] = summarize_coordination(arrays,plan['pose_anchor'])
            result['coordination_candidate_exclusions'] = (
                result['exclusions'] + result['quality']['coordination_screen_exclusions'])
            buffer = io.BytesIO(); np.savez_compressed(buffer,**arrays)
            npz_artifact = write_bounded(work/'coordination'/(stem+'.npz'),buffer.getvalue(),work)
            meta['output_npz_sha256'] = npz_artifact['sha256']
            meta['scan_quality'] = result['quality']
            meta_artifact = write_bounded(work/'coordination'/(stem+'.json'),_json_bytes(meta),work)
            artifacts.extend([npz_artifact,meta_artifact])
            result['coordination_outputs'] = [npz_artifact,meta_artifact]
            if not arrays['joint_geometry_valid'].all():
                result['quality_warnings'] = ['Incomplete joint coverage; missing values preserved with masks']
        results.append(result)
    ranked = sorted(results,key=_rank,reverse=True)
    for i,row in enumerate(ranked): row['rank'] = i+1
    usable = [r for r in ranked if not r['exclusions'] and r['quality'] and r['quality']['simultaneous_core_at_anchor_bracket']]
    report = {'schema_version':1,'match_id':MATCH_ID,'single_match_only':True,'second_match_downloaded':False,
              'archive':{'path':archive.name,'size_bytes':ARCHIVE_BYTES,'sha256':ARCHIVE_SHA256},
              'config_sha256':_sha(CONFIG_PATH),'builder_sha256':_sha(ROOT/'tools/build_skillcorner_coordination.py'),
              'scanner_sha256':_sha(Path(__file__)),'source_event_sha256':_sha(args.source_dir/'1925299_dynamic_events.csv'),
              'stream':stream_stats,'nonheader_windows':len(plans),'excluded_headers_or_unknowns':excluded,
              'quality_policy_unchanged':config['quality_policy'],'core_groups':CORE_GROUPS,
              'ranking':'Non-excluded event, longest all-core run covering both anchor-bracket frames, simultaneous core near +/-0.5s, minimum group coverage, sum of core-group coverage, geometry coverage, then lower core-joint error p95. No thresholds changed.',
              'core_group_interpretation':'Each group is OR over listed alternatives; all groups must pass simultaneously. This checks coarse upper-arm/leg/pelvis/trunk coupling, not full anatomical truth or foot contact.',
              'windows_with_core_screen_at_anchor':[r['event']['event_id'] for r in usable],
              'conclusion':'No non-excluded window supports simultaneous core coordination at the shot anchor under unchanged gates.' if not usable else 'At least one window supports the provisional core screen at the event anchor; inspect listed contiguous duration and unknown qualifications before use.',
              'p2_complete':False,'ranked_windows':ranked,'outputs':artifacts,
              'byte_budget':BYTE_BUDGET,'bytes_before_final_report':directory_bytes(work)}
    report_artifact = write_bounded(work/'scan_report.json',_json_bytes(report),work)
    print(json.dumps({'report':str(work/'scan_report.json'),'archive_bytes':ARCHIVE_BYTES,'total_bytes':directory_bytes(work),
                      'remaining_budget_bytes':BYTE_BUDGET-directory_bytes(work),'frames_scanned':stream_stats['frames_scanned'],
                      'nonheader_windows':len(plans),'core_screen_at_anchor':report['windows_with_core_screen_at_anchor'],
                      'top_five':[{'event':r['event']['event_id'],'player':r['event']['player_name'],'exclusions':r['exclusions'],'quality':r['quality']} for r in ranked[:5]]},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
