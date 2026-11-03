"""Summarize repeated paired-disturbance multi-speed G1 gait tests.

Evaluation robustness probes != independent training seeds.
Guardrail conjunctively considers speed, knee, trunk, foot slip, falls.
"""
import json,statistics
from pathlib import Path

ROOT=Path("/home/you/projects/kinetickick")
FILES=[ROOT/f"reports/gait_perturbed_seed{s}.json" for s in (42,45)]
MODELS=["OFFICIAL","V1","V2"]
SPEEDS=[.25,.5,.8]
KEYS=[
  "mean_abs_velocity_error_mps",
  "left_stance_knee_abs_median_rad",
  "right_stance_knee_abs_median_rad",
  "base_tilt_p95_deg",
  "torso_link_tilt_p95_deg",
  "left_stance_foot_link_xy_speed_p90_mps",
  "right_stance_foot_link_xy_speed_p90_mps",
  "no_support_fraction_45N",
  "episode_resets",
]
def main():
    experiment=[]
    for path in FILES:
        obj=json.loads(path.read_text())
        assert obj["same_scene"] and obj["paired_initial_velocity_disturbances_mps"]
        experiment.extend(obj["summary"])
    data={}
    for model in MODELS:
        data[model]={}
        for speed in SPEEDS:
            rec=[x for x in experiment if x["model"]==model and x["speed_command_mps"]==speed]
            if len(rec)!=2: raise ValueError((model,speed,len(rec)))
            item={}
            for key in KEYS:
                vals=[r[key] for r in rec]
                item[key]=round(statistics.mean(vals),5) if all(v is not None for v in vals) else None
            data[model][str(speed)]=item
    checks={}
    for model in ("V1","V2"):
        checks[model]={}
        for speed in SPEEDS:
            baseline=data["OFFICIAL"][str(speed)]
            trial=data[model][str(speed)]
            knee0=sum(baseline[f"{s}_stance_knee_abs_median_rad"] for s in ("left","right"))/2
            knee=sum(trial[f"{s}_stance_knee_abs_median_rad"] for s in ("left","right"))/2
            knee_gain=knee0-knee
            speed_ok=(trial["mean_abs_velocity_error_mps"]<=baseline["mean_abs_velocity_error_mps"]+.035)
            base_ok=(trial["base_tilt_p95_deg"]<=baseline["base_tilt_p95_deg"]+1.5)
            torso_ok=(trial["torso_link_tilt_p95_deg"]<=baseline["torso_link_tilt_p95_deg"]+1.5)
            slip_ok=all(trial[f"{s}_stance_foot_link_xy_speed_p90_mps"]<=
                        baseline[f"{s}_stance_foot_link_xy_speed_p90_mps"]+.025 for s in ("left","right"))
            stable=trial["episode_resets"]==0
            checks[model][str(speed)]={
                 "knee_abs_median_reduction_rad":round(knee_gain,4),
                 "speed_tracking_sanity":bool(speed_ok),"base_posture_sanity":bool(base_ok),
                 "torso_posture_sanity":bool(torso_ok),"foot_link_slip_sanity":bool(slip_ok),
                 "no_resets":bool(stable),
                 "all_constraints_pass":bool(speed_ok and base_ok and torso_ok and slip_ok and stable and knee_gain>.10)
            }
    selected={model:all(v["all_constraints_pass"] for v in checks[model].values())
              for model in ("V1","V2")}
    result={
       "source_repeated_perturbed_eval_files":[str(x) for x in FILES],
       "training_seeds":"V1 and V2 both trained with seed40 only",
       "evaluation_seeds":[42,45],
       "scenarios":"2 paired random initial velocity perturbations × 3 speeds, 9 robots per simulation",
       "true_human_motion_benchmark_complete":False,
       "all_trials_failed_if_not_pass":selected,
       "averaged":data,
       "per_speed_guardrails":checks,
       "thresholds":{
          "knee_reduction_min_rad":.10,
          "max_tracking_error_increase_mps":.035,
          "max_base_torso_tilt_p95_increase_deg":1.5,
          "max_foot_link_stance_slip_p90_increase_mps":.025,
          "no_env_resets":True
       }
    }
    target=ROOT/"reports/gait_robustness_matrix_20261021.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    header=["# G1 多速度成对扰动评测（2026-10-21）","",
            "严格限制：V1/V2均为单训练种子seed40；评测采用seed42/45的成对小扰动，并不是多个独立训练种子或人类自然度证据。",
            "",
            "| Command speed | Policy | Knee support abs rad (L/R) | Velocity MAE m/s | Base tilt P95 deg | Foot link slip P90 m/s (L/R) | All guardrails |",
            "|---:|---|---:|---:|---:|---:|:---:|"]
    for speed in SPEEDS:
        for model in MODELS:
            m=data[model][str(speed)]
            ok="Baseline" if model=="OFFICIAL" else ("Yes" if checks[model][str(speed)]["all_constraints_pass"] else "No")
            header.append("| %.2f | %s | %.3f / %.3f | %.3f | %.2f | %.3f / %.3f | %s |"%(
                speed,model,
                m["left_stance_knee_abs_median_rad"],m["right_stance_knee_abs_median_rad"],
                m["mean_abs_velocity_error_mps"],m["base_tilt_p95_deg"],
                m["left_stance_foot_link_xy_speed_p90_mps"],
                m["right_stance_foot_link_xy_speed_p90_mps"],ok))
    header.extend(["",
      "指标限制：脚滑使用踝滚动刚体承重时XY速度代理，不是鞋底接触点；角度系仿真原始关节坐标。",
      "目标是改善真人步态整体协调，不能仅凭膝变直宣布自然步态已解决。",
      "结果不包括转身/过渡、人类动作盲评、真实机器人硬件。"])
    md=target.with_suffix(".md")
    md.write_text("\n".join(header)+"\n")
    print("ROBUSTNESS_SUMMARY",json.dumps(selected),flush=True)
    for s in SPEEDS:
        print("ROBUSTNESS_SPEED",s,[(model,checks[model][str(s)]["all_constraints_pass"]) for model in ("V1","V2")],flush=True)
    print("ROBUSTNESS_FILES",target,md,flush=True)
if __name__=="__main__":main()
