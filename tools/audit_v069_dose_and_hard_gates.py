#!/usr/bin/env python3
"""Independent SHA-locked, 200Hz all-29DoF PhysX falsification and hard-gate audit.

Read-only post-physics consumer. Raw episodes are write-once and all results
(including uncensored negative cases and actual dose activation) are retained.
No velocity/impulse threshold relaxation, no synthetic post-termination frames.
"""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_v069_dose_batch import config,sha,PROGRESS,PROTOCOL
from audit_v067_hard_gates import physically_scored
from audit_v060_observer_factorial import assert_complete_physics_fields
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

GROUPS={
    "support_leg":("left_hip_pitch_joint","left_hip_roll_joint","left_knee_joint","left_ankle_pitch_joint"),
    "swing_leg":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"),
    "torso":("waist_yaw_joint","waist_pitch_joint","waist_roll_joint"),
    "arms":("left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_elbow_joint","right_elbow_joint")
}
def observed_200hz(tr):
    # Upstream occasionally appends a terminal +1us snapshot. It is a real
    # terminal state but NOT an extra 5ms full physics sample.
    t=np.asarray([r["time_s"] for r in tr],float)
    if len(t)<100 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("Invalid native G1 physics time")
    dt=np.diff(t)
    if abs(dt[-1]-.000001)<5e-6:
        t=t[:-1];tr=tr[:-1];dt=np.diff(t)
    if not np.allclose(dt,.005,atol=2e-5,rtol=0):
        raise ValueError("Fake or non-200Hz physical samples")
    if len(t)>2800:raise ValueError("Unexpected physics longer than 14 seconds")
    return tr,t

def windows(t,contact):
    if contact is None or not math.isfinite(float(contact)):
        return dict(contact_s=None,pre_0p4_complete=False,post_1p25_complete=False)
    tc=float(contact)
    pre=t[(t>=tc-.4001)&(t<=tc+.0001)]
    post=t[(t>=tc-.0001)&(t<=tc+1.2501)]
    return dict(contact_s=tc,
        pre_0p4_complete=bool(len(pre)>=79 and pre[0]<=tc-.395),
        post_1p25_complete=bool(len(post)>=249 and post[-1]>=tc+1.245),
        observed_pre_count=int(len(pre)),observed_post_count=int(len(post)),
        last_true_physics_s=float(t[-1]))

def block_summary(tr,t,tc,start,end):
    if tc is None:return None
    select=np.where((t>=tc+start-1e-6)&(t<=tc+end+1e-6))[0]
    if not len(select):return None
    com=np.asarray([tr[i]["whole_body_com_xyz"] for i in select],float)
    foot=np.asarray([[tr[i]["support_foot_force_N"]["left"],tr[i]["support_foot_force_N"]["right"]] for i in select],float)
    q=np.asarray([tr[i]["joint_pos"] for i in select],float)
    nonfoot=np.asarray([tr[i]["nonfoot_contact_N"] for i in select],float)
    return dict(actual_frames=len(select),com_min_z_m=float(com[:,2].min()),
                com_max_z_m=float(com[:,2].max()),
                support_left_mean_N=float(foot[:,0].mean()),
                support_right_mean_N=float(foot[:,1].mean()),
                observed_max_nonfoot_N=float(nonfoot.max()),
                q_pos_peak_abs_rad=float(np.abs(q).max()))
def physical_features(raw):
    tr=raw["traces"][0]
    assert_complete_physics_fields(tr)
    tr,t=observed_200hz(tr)
    q=np.asarray([r["joint_pos"] for r in tr],float)
    dq=np.asarray([r["joint_vel"] for r in tr],float)
    if q.shape!=(len(t),29) or dq.shape!=(len(t),29):raise ValueError("29DoF physical data missing")
    lim=np.asarray(raw["joint_velocity_limits"],float).reshape(-1)
    if lim.shape!=(29,) or (lim<=0).any():raise ValueError("Original G1 limits invalid")
    peaks=np.max(np.abs(dq)/lim,axis=0)
    tc=raw.get("records",[{}])[0].get("contact_time_s")
    w=windows(t,tc)
    spans={}
    for group,names in GROUPS.items():
        ids=[G1_JOINT_ORDER.index(x) for x in names]
        active=np.where((t>=.86)&(t<=1.56))[0]
        if len(active):
            speed=np.max(np.abs(dq[active][:,ids]),axis=1)
            ix=active[int(np.argmax(speed))]
            spans[group]=dict(max_angular_speed_rad_s=float(speed.max()),
                              native_time_of_max_s=float(t[ix]))
    obs=raw.get("safe_body_v062")
    distance=None
    if isinstance(obs,dict) and obs.get("mode")=="safe_full":
        if len(obs.get("rows",[]))!=len(raw["traces"][0]):
            raise ValueError("Observer full-physics row mismatch")
        d=[r["left_wrist_hip_link_origin_distance_m"] for r in obs["rows"]]
        distance=float(min(d))
    return dict(true_native_frames=len(tr),last_native_s=float(t[-1]),
        full_original_14s=bool(t[-1]>=13.995),
        strict_29_qdot_max_ratio=float(peaks.max()),
        worst_original_joint=G1_JOINT_ORDER[int(peaks.argmax())],
        timing=w,pre_contact_0p4s_observed=block_summary(tr,t,tc,-.4,0),
        post_contact_1p25s_observed=block_summary(tr,t,tc,0,1.25),
        coupling_temporal_group_qdot_peaks=spans,
        wrist_hip_link_origin_distance_min_m_PROXY_not_collider_clearance=distance)

def audit():
    c=config();p=json.loads(PROGRESS.read_text())
    if p["protocol_sha256"]!=sha(PROTOCOL) or len(p["runs"])!=12:
        raise ValueError("Incomplete 12-case frozen protocol evidence")
    results=[];groups={}
    for case,entry in zip(c["cases"],p["runs"]):
        if case["id"]!=entry["id"]:raise ValueError("Frozen case order changed")
        path=ROOT/case["output"];log=path.with_suffix(".log")
        if not log.exists() or sha(log)!=entry["log_sha256"]:
            raise RuntimeError("Original physical log drift "+case["id"])
        if not path.exists() or sha(path)!=entry.get("raw_sha256"):
            results.append(dict(id=case["id"],policy=case["policy"],status="NO_ORIGINAL_G1_OUTPUT",hard_gate_pass=False))
            continue
        raw=json.loads(path.read_text())
        if (raw.get("root_fixed") is not False or raw.get("seed")!=case["seed"]
             or tuple(raw.get("joint_names",()))!=G1_JOINT_ORDER):
            raise RuntimeError("Not original original free-root G1 29DoF")
        txt=log.read_text(errors="replace")
        if "V058_NATIVE_RETURN cuda:0" not in txt or "V058_BACKEND_ARG_FORWARDED cuda:0" not in txt or "Environment device    : cuda:0" not in txt:
            raise RuntimeError("Cannot prove native CUDA/PhysX execution")
        certified=(entry["status"]=="VALID_G1" and
                   raw.get("policy_sha256")==c["policy_sha256"][c["policies"][case["policy"]]]
                   and type(raw.get("stride_shot_success")) is bool)
        hard=physically_scored(raw,certified)
        kin=physical_features(raw)
        if not kin["timing"]["post_1p25_complete"]:
            hard["hard_gate_pass"]=False
            hard["failed_gates"].append("real_postimpact_1p25s_required")
        dose=raw.get("phase_v069_metadata")
        if case["policy"] in ("dose","dose_reverse") and not dose:
            hard["hard_gate_pass"]=False
            hard["failed_gates"].append("no_verified_dose_runtime_route")
        row=dict(id=case["id"],group=case["ball_case"],seed=case["seed"],
                 policy=case["policy"],source_sha256=entry["raw_sha256"],
                 wrapper_source_certified=certified,original_wrapper_result=raw.get("stride_shot_success"),
                 hard_gate_pass=hard["hard_gate_pass"],hard_gates=hard,
                 actual_dose_route=dose,phase_summary=raw.get("phase_v063_summary"),
                 actual_native_kinetics_200hz=kin)
        results.append(row)
        groups.setdefault(case["ball_case"],{})[case["policy"]]=(row,raw)
    contrasts={}
    for group,policies in groups.items():
        if set(policies)!={"v32","low","dose","dose_reverse"}:
            contrasts[group]={"error":"MISSING_FROZEN_POLICY_OR_SOURCE"}
            continue
        a=policies["v32"][1]["traces"][0]
        groupc={}
        for pol in ("low","dose","dose_reverse"):
            b=policies[pol][1]["traces"][0]
            ta=np.asarray([v["time_s"] for v in a],float)
            tb=np.asarray([v["time_s"] for v in b],float)
            n=min(len(ta),len(tb))
            if not np.allclose(ta[:min(n,100)],tb[:min(n,100)],atol=1e-6):
                raise ValueError("Initial original physical clock context mismatch")
            qa=np.asarray([x["joint_pos"] for x in a[:n]],float)
            qb=np.asarray([x["joint_pos"] for x in b[:n]],float)
            early=(ta[:n]<=.84)
            active=(ta[:n]>=.86)&(ta[:n]<=1.70)
            groupc[pol]=dict(real_common_frames=int(n),
                preimpact_initial_state_max_abs_delta_rad=float(np.abs(qa[early]-qb[early]).max()) if early.any() else None,
                observed_preimpact_after_phase_max_abs_delta_rad=float(np.abs(qa[active]-qb[active]).max()) if active.any() else None,
                actually_measurable_nonzero_difference=bool(active.any() and np.abs(qa[active]-qb[active]).max()>1e-7))
        contrasts[group]=groupc
    return dict(version="v0.1.69",protocol_sha256=sha(PROTOCOL),
                source_G1_GPU_PhysX_real_attempts=len(results),
                original_G1_wrapper_certified=sum(x.get("wrapper_source_certified",False) for x in results),
                original_legacy_wrapper_positive=sum(x.get("original_wrapper_result") is True for x in results),
                independently_verified_all_hard_gate_PASS=sum(x["hard_gate_pass"] for x in results),
                no_GPU_native_manifold_or_human_instep_truth_certified=True,
                frozen_bounded_synthetic_dose_1p25_with_reverse_negative_control=True,
                physical_episode_data=results,paired_29dof_real_kinetics=contrasts)
def main():
    data=audit()
    out=ROOT/"reports/v069_independent_hard_gate_and_kinetics.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("V069_NATIVE",data["source_G1_GPU_PhysX_real_attempts"],
          "CERTIFIED",data["original_G1_wrapper_certified"],
          "ORIGINAL_WRAPPER_POSITIVE",data["original_legacy_wrapper_positive"],
          "STRICT_ALL_GATE",data["independently_verified_all_hard_gate_PASS"],flush=True)
    for r in data["physical_episode_data"]:
        k=r.get("actual_native_kinetics_200hz",{})
        print(r["id"],"STRICT",r["hard_gate_pass"],
              "frames",k.get("true_native_frames"),
              "dose",r.get("actual_dose_route",{}).get("actual_nonzero_action_steps") if r.get("actual_dose_route") else None,
              "failed",r.get("hard_gates",{}).get("failed_gates"),flush=True)
if __name__=="__main__":main()
