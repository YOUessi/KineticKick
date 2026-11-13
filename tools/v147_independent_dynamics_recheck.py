#!/usr/bin/env python3
"""v147 independent, Python-loop-based falsification of vectorized real dynamics.

Fresh SHA-locked source reads. Deliberately never imports v147's vectorized
summarize_segment or its preprocessing. Entire input is historic actual PhysX.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from v146_real_200hz_coupled_audit import verified_sources
from v147_exact_off_on_parity import frozen
from v142_native_pair_ledger import safe_write_once

def manual_check(snapshot):
    cfg=frozen()
    if snapshot["frozen_protocol_sha"] != (
            "3ecf094137d9f1d82e94a9dba3b9bf7d215c97c8"):
        raise ValueError("V147_WRONG_PROTOCOL_FOR_INDEPENDENT_REVIEW")
    inputs=verified_sources()
    checked=0
    max_error=0.
    for name, path in inputs.items():
        case=snapshot["cases"][name]
        original=json.loads(Path(path).read_text())
        records=original["traces"][0][:cfg["original_v54_expected_rows"][name]]
        names=original["joint_names"]
        if len(names)!=29 or len(records)!=case["actual_original29_200hz_rows"]:
            raise ValueError("V147_INDEPENDENT_SOURCE_LENGTH_WRONG")
        cuts={
            "pre":cfg["precontact_window"],
            "post":cfg["postcontact_window"],
            "late_failed_time_support_net":cfg["failed_event_window"],
            "full_real_trace":[.005,len(records)*.005],
        }
        for label, (lower,upper) in cuts.items():
            rows=[r for r in records if lower-1e-9<=r["time_s"]<=upper+1e-9]
            claim=case["all_window_metrics"][label]
            if len(rows)!=claim["sample_count"]:
                raise ValueError("V147_INDEPENDENT_ROW_COUNTS_CHANGED")
            for prefix, predicate in (
                ("left_leg", lambda n:n.startswith("left_") and
                 any(x in n for x in ("hip_","knee_","ankle_"))),
                ("right_leg",lambda n:n.startswith("right_") and
                 any(x in n for x in ("hip_","knee_","ankle_"))),
                ("waist",lambda n:n.startswith("waist_")),
                ("left_arm",lambda n:n.startswith("left_") and
                 any(x in n for x in ("shoulder_","elbow_","wrist_"))),
                ("right_arm",lambda n:n.startswith("right_") and
                 any(x in n for x in ("shoulder_","elbow_","wrist_"))),
            ):
                idx=[i for i,n in enumerate(names) if predicate(n)]
                manual=math.fsum(.005*abs(float(r["joint_vel"][j])
                    *float(r["joint_torque_estimate"][j]))
                    for r in rows for j in idx)
                result=claim["group_absolute_sim_torque_speed_proxy_integrals_J"][prefix]
                delta=abs(manual-result)
                max_error=max(max_error,delta)
                if not math.isclose(manual,result,rel_tol=1e-11,abs_tol=1e-9):
                    raise ValueError("V147_INDEPENDENT_GROUP_TORQUE_POWER_CONFLICT")
            all_proxy=math.fsum(.005*abs(float(r["joint_vel"][j])
                      *float(r["joint_torque_estimate"][j]))
                      for r in rows for j in range(29))
            if not math.isclose(all_proxy,
                    claim["sim_torque_abs_joint_power_proxy_integral_J"],
                    rel_tol=1e-11,abs_tol=1e-9):
                raise ValueError("V147_INDEPENDENT_TOTAL_TORQUE_POWER_CONFLICT")
            peak=max(r["nonfoot_contact_N"] for r in rows)
            if not math.isclose(float(peak),
                    claim["max_nonfoot_net_contact_N"],abs_tol=1e-9):
                raise ValueError("V147_INDEPENDENT_NONFOOT_PEAK_CONFLICT")
            checked+=len(rows)
    return {
        "independently_reviewed_policy_windows":12,
        "independently_reviewed_real_rows_across_overlapping_windows":checked,
        "max_absolute_group_sim_power_integral_recheck_difference_J":max_error,
        "new_physx_episodes":0,
        "native_collider_pair_truth":"UNKNOWN",
        "real_runtime_off_on_equality":None,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    output=manual_check(json.loads(args.input.read_text()))
    digest=safe_write_once(args.output,output)
    print(json.dumps({"sha256":digest,**output}))

if __name__=="__main__":
    main()
