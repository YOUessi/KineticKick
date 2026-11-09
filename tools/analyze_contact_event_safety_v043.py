#!/usr/bin/env python3
"""v43 offline qdot risk contract and event-aligned full-body safety visualization.

This is retrospective diagnostics on the ALREADY INSPECTED original v42
development data. Does not train/suggest a safe control policy. The actual
unsafe labels come from original 2800 x 29 PhysX joint velocities, not action
reference bounds or a learned surrogate score.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/contact_event_safety_v043.json"
DATA="data/research_human_kicks/contact_event_safety_v043/event_aligned_qdot_dynamics.npz"
PROV="data/research_human_kicks/contact_event_safety_v043/event_aligned_audit.json"
OUT="data/research_human_kicks/contact_event_safety_v043/retrospective_qdot_guard_audit.json"
FIG="media/v043_contact_event_delayed_joint_speed_risk.png"

def sha(p):
 h=hashlib.sha256()
 with (ROOT/p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def first_real_overrun(source):
 r=json.loads((ROOT/source).read_text())
 original=np.asarray([x["joint_vel"] for x in r["traces"][0]],dtype=float)
 times=np.asarray([x["time_s"] for x in r["traces"][0]],dtype=float)
 limits=np.asarray(r["joint_velocity_limits"][0],dtype=float)
 if original.shape!=(2800,29) or times.shape!=(2800,) or limits.shape!=(29,):
  raise ValueError("Full 14s original PhysX required")
 ratio=np.abs(original)/limits
 over=ratio>1.+1e-6/np.max(limits)
 # This matches the original strict qdot limit observation: abs_vel>limit+1e-6.
 over=np.abs(original)>limits[None]+1e-6
 bad=np.argwhere(over)
 if not len(bad):return None
 step,idx=map(int,bad[0])
 record=dict(actual_5ms_event_time_s=float(times[step]),joint=r["joint_names"][idx],
      actual_joint_speed_rad_s=float(abs(original[step,idx])),
      original_joint_speed_limit_rad_s=float(limits[idx]),
      ratio=float(ratio[step,idx]))
 record["relative_time_to_ball_contact_s"]=record["actual_5ms_event_time_s"]-float(r["records"][0]["contact_time_s"])
 return record

def stats(pred,actual):
 tp=sum(bool(p) and bool(a) for p,a in zip(pred,actual))
 fp=sum(bool(p) and not bool(a) for p,a in zip(pred,actual))
 fn=sum(not bool(p) and bool(a) for p,a in zip(pred,actual))
 tn=sum(not bool(p) and not bool(a) for p,a in zip(pred,actual))
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,
   recall=None if tp+fn==0 else tp/(tp+fn),
   precision=None if tp+fp==0 else tp/(tp+fp),
   specificity=None if tn+fp==0 else tn/(tn+fp))

def run():
 config=json.loads((ROOT/STUDY).read_text())
 provenance=json.loads((ROOT/PROV).read_text())
 if provenance["v43_study_sha256"]!=sha(STUDY) or provenance["output_sha256"]!=sha(DATA):
  raise ValueError("Source/provenance SHA mismatch")
 for f,digest in config["source_sha256"].items():
  if sha(f)!=digest:raise ValueError("Original source drift "+f)
 with np.load(ROOT/DATA,allow_pickle=False) as z:
  relative=z["relative_contact_s"].astype(float)
  normalized=z["qdot_ratio"].astype(float)
  support=z["support"].astype(float)
  q=z["q"].astype(float)
  limits=z["original_joint_velocity_limits_rad_s"].astype(float)
  case=z["source_case"].tolist()
  ctrl=z["controller"].tolist()
  names=z["joint_names"].tolist()
 if normalized.shape!=(12,251,29) or not np.all(np.isfinite(normalized)):
  raise ValueError("Invalid 5-ms event aligned qdot shapes")
 if case!=[r["id"] for r in provenance["events"]] or ctrl!=[r["controller"] for r in provenance["events"]]:
  raise ValueError("Broken original physx pairing order")
 if len({x for x in case})!=6:
  raise ValueError("Not exactly six original trial condition pairs")
 if (ROOT/OUT).exists():raise FileExistsError("Do not overwrite evidence")
 results=[]
 for i,old in enumerate(provenance["events"]):
  signal=normalized[i]
  before=(relative>=-.25)&(relative<0)
  during=(relative>=0)&(relative<=1.)
  post_first_quarter=(relative>=0)&(relative<=.25)
  current_pre_peak=float(signal[before].max())
  first=first_real_overrun(old["source_path"])
  if (first is None)!=(old["full_episode_actual_qdot_limit_exceedances"]==0):
   raise ValueError("Original event and new strict full-sample overrun disagree")
  if first is not None and first["relative_time_to_ball_contact_s"]<=-.25:
   raise ValueError("Unexpected 5ms overrun before declared pre-window")
  item=dict(
    id=old["id"],controller=old["controller"],
    preserved_original_physx_sha256=old["source_sha256"],
    original_strict_success=old["strict_functional_pass"],
    first_original_full_episode_qdot_overrun=first,
    pred_pre_0to_minus250ms_guard09=current_pre_peak>=config["precontact_warn_ratio"],
    pre_0to_minus250ms_actual_max_qdot_ratio=current_pre_peak,
    first_250ms_post_impact_actual_max_qdot_ratio=float(signal[post_first_quarter].max()),
    entire_contact_to_plus1s_actual_max_qdot_ratio=float(signal[during].max()),
    source_post1s_max_qdot_ratio=old["contact_relative_0to1s_qdot_max_ratio"],
    source_full14s_max_qdot_ratio=old["full_episode_worst_qdot_ratio"],
    full14s_speed_limit_violation=(first is not None),
    contact_plus_1s_scope_misses_hazard=(first is not None and
        (first["relative_time_to_ball_contact_s"]>1.0)),
    estimated_stance_load_peak_after_impact_N=float(support[i,during,0].max()),
    whole_body_velocity_proxy_warning_only=True)
  results.append(item)
 predicted=[r["pred_pre_0to_minus250ms_guard09"] for r in results]
 truth=[r["full14s_speed_limit_violation"] for r in results]
 confusion=stats(predicted,truth)
 qsafe=sum(not x["pred_pre_0to_minus250ms_guard09"] for x in results)
 failures_late=[(x["id"],x["controller"],x["first_original_full_episode_qdot_overrun"])
                for x in results if x["full14s_speed_limit_violation"]]
 report=dict(
   version="0.1.43",type="Read-only original PhysX contact-event qdot safety evidence",
   study_sha256=sha(STUDY),data_sha256=sha(DATA),input_manifest_sha256=sha(PROV),
   source_parent_commit="ad7bd18",frozen_input_cases=len({x["id"] for x in results}),
   controller_episode_count=len(results),event_binned_steps=251,
   all_run_source_hashes_verified=True,
   threshold09_is_a_retrospective_heuristic_not_a_proven_barrier_certificate=True,
   pre_impact_qdot_ratio_warning_confusion_vs_any_full14s_actual_limit_exceedance=confusion,
   actual_entire_episode_overspeed_count=sum(truth),
   pre_quiet_but_future_overrun_count=sum(not p and v for p,v in zip(predicted,truth)),
   contact_to_plus1s_window_missed_overruns=sum(x["contact_plus_1s_scope_misses_hazard"] for x in results),
   original_true_overruns=failures_late,
   guarantee_of_future_safe_motion=False,
   data_is_new_to_v043_model=False,
   risk_data_is_independent_test=False,
   no_control_or_simulator_changes=True,
   no_new_physical_runs=True,
   research_interpretation="The original precontact qdot envelope alone missed actual delayed joint speed exceedances; body-wide impact/recovery coupling must be treated by a predictive contact-aware multi-stage controller, not a simple instantaneous ankle speed threshold.",
   limitations=[
    "This is a descriptive already-known v42 development set, not a fresh evaluation population",
    "Each 5ms sample from one 14-second episode is correlated; do not misinterpret N=12*251 samples as N=3012 independent experiments",
    "The original qdot warning threshold 0.90 is heuristic (not hardware-certified)",
    "The 0.25s precontact window does not include original gait startup qdot",
    "One important speed exceedance occurs later than +1.0s, so full-episode hazard analysis cannot stop at the contact window",
    "All PD power is a clipped motor simulation estimate, not calibrated human biomechanical energy transfer",
    "No G1 controller is trained or modified by this retrospective study"
   ],cases=results)
 out=ROOT/OUT
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 # Paired event-aligned max qdot ratio curves: all 6 original conditions,
 # no reference or event-time re-warp favoring a particular controller.
 fig,axes=plt.subplots(3,2,figsize=(12.3,9))
 for j,ax in enumerate(axes.ravel()):
  idx=[i for i,case_name in enumerate(case) if case_name==f"trial{j+1:02d}"]
  if len(idx)!=2:raise ValueError("Broken original source pairing")
  for i in idx:
   peak=normalized[i].max(axis=-1)
   ax.plot(relative,peak,label="Frozen v32" if ctrl[i]=="original_v32" else "Style v41",lw=1.4)
  ax.axhline(1.,linestyle="--",linewidth=1.1,label="Original joint speed limit")
  ax.axvline(0,linestyle=":",linewidth=1.)
  ax.set_title(f"Original trial{j+1:02d} — contact-aligned actual joint qdot ratio")
  ax.set_xlabel("Seconds relative to actual ball contact")
  ax.set_ylabel("Max measured |joint speed| / original qdot limit")
  ax.set_ylim(0,max(1.37,float(normalized[idx].max())+.07))
  ax.grid(alpha=.25)
 axes[0,0].legend(fontsize=8)
 fig.suptitle("KineticKick v0.1.43 | Measured G1 whole-body delayed joint-speed risk after impact")
 fig.tight_layout(rect=(0,.02,1,.96))
 fig.text(.015,.006,"Only six known original physical ball conditions paired across frozen v32/style v41; 5-ms PhysX samples and first foot-ball contact observed. Not statistical generalization, a human power model or a certified predictive safety barrier.",fontsize=8)
 outfig=ROOT/"media/v043_contact_event_delayed_joint_speed_risk.png"
 outfig.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(outfig,dpi=165)
 plt.close(fig)
 print("V043_RETROSPECTIVE_QDOT_HAZARD",json.dumps({
    "confusion":confusion,"violating_episodes":len(failures_late),
    "quiet_before_contact_but_unsafe_later":report["pre_quiet_but_future_overrun_count"],
    "overrun_more_than_one_second_after_contact":report["contact_to_plus1s_window_missed_overruns"],
    "figure":str(outfig),"analysis_sha256":sha(OUT)},ensure_ascii=False),flush=True)

if __name__=="__main__":run()
