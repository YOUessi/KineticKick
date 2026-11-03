"""Compare OFFICIAL vs preliminary biomechanics-guided G1 gait policy.

Descriptive evidence only: single seed, human ratings unavailable.
"""
from pathlib import Path
import json,csv,math,hashlib,sys

ROOT=Path("/home/you/projects/kinetickick")
A=ROOT/"reports/human_walk_baseline_v050.json"
B=ROOT/"reports/human_walk_pilot_v050.json"
OUTPUT=ROOT/"reports/gait_ablation_comparison.json"
FIELDS=[
  ("root_forward_velocity_mae_mps","speed_tracking_mae_mps",False),
  ("root_forward_speed_mean_mps","mean_forward_speed_mps",None),
  ("root_height_median_m","root_height_median_m",None),
  ("torso_upright_tilt_deg_p95","torso_tilt_p95_deg",False),
  ("left_knee_abs_stance_median_rad","left_stance_knee_abs_median_rad",False),
  ("right_knee_abs_stance_median_rad","right_stance_knee_abs_median_rad",False),
  ("left_knee_abs_swing_median_rad","left_swing_knee_abs_median_rad",None),
  ("right_knee_abs_swing_median_rad","right_swing_knee_abs_median_rad",None),
  ("left_foot_link_slip_speed_p90_mps","left_stance_foot_link_slip_p90",False),
  ("right_foot_link_slip_speed_p90_mps","right_stance_foot_link_slip_p90",False),
  ("no_support_fraction","threshold_45N_no_support_fraction",False),
  ("left_right_hip_pitch_correlation","hip_anti_phase_correlation",None),
  ("observed_episode_resets","episode_resets",False),
]
def main():
    if not B.exists():
        print("GAIT_A_B_UNAVAILABLE","trained model evaluation missing",flush=True)
        return
    base=json.loads(A.read_text())
    trial=json.loads(B.read_text())
    compare={}
    for src,label,lower in FIELDS:
        a=base.get(src);b=trial.get(src)
        compare[label]={
          "baseline":a,"candidate":b,
          "delta":None if (a is None or b is None) else b-a,
          "lower_desirable_if_other_goals_preserved":lower
        }
    min_new=trial.get("root_height_min_m")
    fall=trial.get("observed_episode_resets",0)
    tracking=trial.get("root_forward_velocity_mae_mps",1e9)
    knee=sum(trial.get(k,1e9) for k in (
        "left_knee_abs_stance_median_rad","right_knee_abs_stance_median_rad"))/2
    knee_base=sum(base[k] for k in (
        "left_knee_abs_stance_median_rad","right_knee_abs_stance_median_rad"))/2
    # Interpreting the knee rad numbers requires verified robot joint conventions.
    low_risk=(fall==0 and tracking <= base["root_forward_velocity_mae_mps"]+.05
              and min_new is not None and min_new >.46)
    # The first pilot improved knees but made trunk tilt and foot slip WORSE.
    # Never promote a gait simply because it passes speed/fall checks.
    slip_ok=all(trial[f"{side}_foot_link_slip_speed_p90_mps"]
                <=base[f"{side}_foot_link_slip_speed_p90_mps"]+.025
                for side in ("left","right"))
    tilt_ok=(trial["torso_upright_tilt_deg_p95"]
             <=base["torso_upright_tilt_deg_p95"]+1.5)
    changes={
        "preliminary_policy_passes_stability_speed_sanity":bool(low_risk),
        "preliminary_policy_passes_full_gait_safety_guardrails":bool(low_risk and slip_ok and tilt_ok),
        "foot_link_slip_guardrail_passes":bool(slip_ok),
        "trunk_tilt_guardrail_passes":bool(tilt_ok),
        "stance_knee_abs_change_rad":knee-knee_base,
        "stance_knee_cost_reduced":knee<knee_base,
        "claim_humanlike_motion_improved":False,
        "reason":"No human reference, no multi-seed, no blinded perceptual ratings; short fine-tune.",
    }
    output={
      "baseline":str(A),"pilot":str(B),"same_eval_seed":base["seed"]==trial["seed"],
      "same_command_speed":base["command_forward_mps"]==trial["command_forward_mps"],
      "metrics":compare,"sanity":changes,
      "next_required":["multi-seed train/eval","multi-speed held-out testing",
                       "matched mocap reference","controlled observer ratings"],
    }
    OUTPUT.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print("GAIT_ABLATION_RESULT",json.dumps(output["sanity"]),flush=True)
    for key,val in compare.items():
        if val["delta"] is not None:
            print("GAIT_METRIC",key,"base",round(val["baseline"],4),
                  "pilot",round(val["candidate"],4),
                  "delta",round(val["delta"],4))
if __name__=="__main__":main()
