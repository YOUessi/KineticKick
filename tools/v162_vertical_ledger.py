"""Summarize all retained near-vertical CPU development and full-horizon evidence."""
import argparse
import json
from pathlib import Path

from v161_evaluate import digest
from v162_vertical_audit import audit


def ledger(folder):
    population={}
    for file in sorted(folder.glob('*/all.jsonl')):
        with file.open() as lines:
            rows=[json.loads(line) for line in lines]
        population[file.parent.name]=dict(count=len(rows), loaded_right_contact=sum(r.get('first') is not None for r in rows),
                          source_sha256=digest(file),partial_serialization_interruption=file.parent.name in {'closed','pose','bridge'})
    episodes={}
    for file in sorted(folder.glob('*/tracking_result.json')):
        trial=file.parent
        score_file=trial/'vertical_offline_score.json'
        if not score_file.exists():
            result=audit(trial)
            with score_file.open('x') as output:
                json.dump(result,output,indent=2,allow_nan=False);output.write('\n')
        else:result=json.loads(score_file.read_text())
        pose=result['vertical_instep']
        episodes[trial.name]=dict(v161=result['v161_pass'],both_instep=result['both_corrected_witness_pose_checks'],
                                  surface_fractions=result['all_source_patch_fractions'],vertical=pose['pass'],joint=result['pass_combined'],
                                  first_and_weighted=pose.get('primary_post_step'),sensitivity=pose.get('pre_step_sensitivity'),
                                  clearance_m=pose.get('minimum_padded_clearance_m'),outgoing_mps=result['speed_xy_mps'],
                                  stable_s=result['terminal_standing_s'],follow=result['follow']['passed'],failures=result['v161_failures'],
                                  input_sha256=result['input_sha256'],score_sha256=digest(score_file))
    done={k:v for k,v in population.items() if not v['partial_serialization_interruption']}
    return dict(search_work=population,completed_search_rows=sum(x['count'] for x in done.values()),
                interrupted_search_rows=sum(x['count'] for x in population.values() if x['partial_serialization_interruption']),
                interrupted_trajectory_note='3 additional miss trajectories triggered serialization failure; completed after conversion fix under the same seeded schedule. Partial rows overlap complete generations and do not add independent conditions.',
                full_physical_trials=len(episodes),single_centre_trials=sum(n.startswith('full_') for n in episodes),
                independent_trials=sum(n.startswith('heldout_') for n in episodes),strict_vertical_full_success=sum(r['joint'] for r in episodes.values()),
                full_episode_rows=episodes)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--folder',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    a=cli.parse_args();r=ledger(a.folder)
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print({k:v for k,v in r.items() if k not in ('full_episode_rows','search_work')})
