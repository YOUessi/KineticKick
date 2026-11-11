#!/usr/bin/env python3
"""Postregistered, evidence-preserving repair for recorder's terminal marker row.

Some censored original 200Hz episodes append a duplicate state at t+1us solely
to carry "terminated=True". The raw JSON must remain untouched. Count it as a
terminal annotation, not a real extra 200Hz simulation step.
"""
from __future__ import annotations
import v085_evidence as original

def terminal_only_extra_step(rows):
    if len(rows)<2:return False
    prev,last=rows[-2:]
    if last.get("terminated") is not True or prev.get("terminated") is not False:
        return False
    if not (0<float(last["time_s"])-float(prev["time_s"])<=0.00001):
        return False
    if set(prev)!=set(last):return False
    for key in prev:
        if key not in ("time_s","terminated") and prev[key]!=last[key]:
            return False
    return True

def audit_episode(raw,expected):
    rows=raw["traces"][0]
    if not terminal_only_extra_step(rows):
        result=original._unpatched_audit_episode(raw,expected)
        result["raw_recorder_rows"]=len(rows)
        result["pure_termination_marker_rows"]=0
        return result
    clean=dict(raw)
    clean["traces"]=[rows[:-1]]
    evidence=original._unpatched_audit_episode(clean,expected)
    evidence["raw_recorder_rows"]=len(rows)
    evidence["pure_termination_marker_rows"]=1
    evidence["raw_terminal_marker_time_s"]=rows[-1]["time_s"]
    evidence["nonfoot_net_force_peak_N"]=round(
        max(evidence["nonfoot_net_force_peak_N"],float(rows[-1]["nonfoot_contact_N"])),5)
    evidence["recorder_terminal_marker_did_not_advance_physics"]=True
    evidence["termination_marker_not_imputed_as_extra_physical_step"]=True
    return evidence

original._unpatched_audit_episode=original.audit_episode
original.audit_episode=audit_episode
if __name__=="__main__":
    original.summarize()
