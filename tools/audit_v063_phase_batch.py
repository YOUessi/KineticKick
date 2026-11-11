#!/usr/bin/env python3
"""Independent paired 200Hz G1 physics audit; no re-labeling of censored runs."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_v063_phase_batch import sha, check_real, config, PROTOCOL, PROGRESS
from audit_v059_observer_parity import compare_raw
def examine():
    c=config();p=json.loads(PROGRESS.read_text())
    assert p["protocol_sha256"]==sha(PROTOCOL)
    valid={};rows=[]
    for case,meta in zip(c["cases"],p["runs"]):
        assert case["id"]==meta["id"]
        path=ROOT/case["output"]
        assert meta["log_sha256"]==sha(path.with_suffix(".log"))
        if meta["status"]=="VALID_G1":
            assert path.exists() and sha(path)==meta["raw_sha256"]
            r=json.loads(path.read_text())
            assert check_real(case,r,path.with_suffix(".log").read_text(errors="replace"),meta["return_code"],c)["status"]=="VALID_G1"
            valid[case["id"]]=r
            trace=r["traces"][0]
            q=np.array([x["joint_vel"] for x in trace],float)
            limits=np.array(r["joint_velocity_limits"],float) if "joint_velocity_limits" in r else None
            z=r.get("phase_v063_summary",{})
            rows.append(dict(id=case["id"],outcome="PASS" if r["stride_shot_success"] else "FAIL",
                seed=case["seed"],trace_rows=len(trace),end_time_s=trace[-1]["time_s"],
                peak_qdot_abs_rad_s=float(np.max(np.abs(q))),
                actual_max_coordination_delta_rad=z.get("actual_max_joint_delta_rad",0.),
                number_of_20ms_nonzero_intent_steps=z.get("actual_nonzero_action_steps",0),
                main_termination=r["termination_reasons"],reference_limit_claim="physical gate unchanged"))
        else:
            rows.append(dict(id=case["id"],outcome="UNSCORABLE",reason=meta.get("reason")))
    groups=[]
    for root in ("center","hold_a","hold_b","hold_c"):
        b=valid.get(root+"_v32");a=valid.get(root+"_synth")
        if not (a and b):
            groups.append(dict(group=root,classification="UNSCORABLE",reason="Missing one real G1 trajectory"));continue
        be=b["traces"][0];ae=a["traces"][0]
        # Pre-action equivalence verifies original initialization and inherited
        # v32 dynamics; 0.5s is strictly earlier than new .86s control onset.
        from audit_v059_observer_parity import compare_raw
        pre=compare_raw(be[:100],ae[:100])
        # The whole traces are expected to diverge; no false physics parity.
        z=a["phase_v063_summary"]
        effective=z["actual_nonzero_action_steps"]>0 and z["actual_max_joint_delta_rad"]>.006
        groups.append(dict(group=root,baseline_pass=b["stride_shot_success"],
            candidate_pass=a["stride_shot_success"],candidate_action_effective=effective,
            max_applied_joint_delta_rad=z["actual_max_joint_delta_rad"],
            candidate_nonzero_steps=z["actual_nonzero_action_steps"],
            preaction_0_to_0p5s_physics_parity=pre))
    b=valid.get("center_v32")
    center_history=None
    if b:
        old=json.loads((ROOT/"reports/v062_cuda_v32_safe_full.json").read_text())
        center_history=compare_raw(b["traces"][0],old["traces"][0])
    return dict(version="v0.1.63",hypothesis="SYNTHETIC fullbody temporal hypothesis only, not learned human strike",
        protocol_sha256=sha(PROTOCOL),attempted=len(rows),scorable=sum(z["outcome"]!="UNSCORABLE" for z in rows),
        strict_pass=sum(z["outcome"]=="PASS" for z in rows),
        comparison_count=len(groups),full_contact_manifold_left_wrist_hip_unproven=True,
        professional_human_mocap_contact_and_GRF_unavailable=True,
        independent_groups=groups,rows=rows,center_old_run_25field_replay=center_history)
def main():
    data=examine()
    dest=ROOT/"reports/v063_independent_audit.json"
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:data[k] for k in ("attempted","scorable","strict_pass")}))
    for r in data["independent_groups"]:
        print(r["group"],r.get("baseline_pass"),r.get("candidate_pass"),
              r.get("candidate_action_effective"),r.get("max_applied_joint_delta_rad"))
if __name__=="__main__":main()
