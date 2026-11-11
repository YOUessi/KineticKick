#!/usr/bin/env python3
"""Fail-closed factual source triage. No network download or GT fabrication."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/soccer_source_qualification_v067.json'
def qualify(c):
    required=c['required_true_fields']
    entries=c['sources']
    if len(entries)<4 or len({e['id'] for e in entries})!=len(entries):
        raise ValueError('incomplete or duplicate source manifest')
    out=[]
    for e in entries:
        for field in required:
            if field not in e or e[field] not in (True,False,None):
                raise ValueError('unknown/tristate evidence field must be explicit: '+e['id']+' '+field)
        accepted=all(e[k] is True for k in required)
        rejected=[k for k in required if e[k] is not True]
        out.append({'id':e['id'],'eligible_for_professional_instep_contact_learning':accepted,
                    'unverified_or_false_requirements':rejected,
                    'qualification':e['qualification'],'local_asset_verified':e['local_asset_verified']})
    return {'version':c['version'],'sources_reviewed':len(out),
            'eligible':sum(x['eligible_for_professional_instep_contact_learning'] for x in out),
            'entries':out}
def main():
    result=qualify(json.loads(CONFIG.read_text()))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['eligible']>0:raise RuntimeError('Material data eligibility changed: require independent manual scientific review before use')
if __name__=='__main__':main()
