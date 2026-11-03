"""Immutable-ish stage-1 evidence rollup (no unverified success claims).

Reads experiments already completed on Tang, computes short quantitative
summaries and hashes source scripts + NVIDIA checkpoint for traceability.
No rendering / GPU required.
"""
from __future__ import annotations
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path("/home/you/projects/kinetickick")
R=ROOT/"reports"

def load_json(basename):
    path=R/basename
    if not path.exists():return None
    return json.loads(path.read_text())

def sha(name):
    p=ROOT/name
    return {"path":name,"bytes":p.stat().st_size,
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} if p.exists() else None

def foot_stats(csv_name):
    p=R/csv_name
    if not p.exists():return None
    rows=list(csv.DictReader(p.open()))
    if not rows:return None
    episodes=sum(int(float(q["reset"])) for q in rows)
    touches=sum(float(q["ball_foot_contact_N"])>2. for q in rows)
    def error(q):
        return float(q["left_support_foot_error_m"])
    slip=[float(q["left_support_foot_slip_mps"]) for q in rows]
    ready=[q for q in rows if q["phase"]=="preplant_hold"
           and error(q)<.2
           and float(q["left_ground_force_n"])>30
           and float(q["left_support_foot_slip_mps"])<.08
           and float(q["projected_gravity_z"])<-.9
           and float(q["root_z"])>.48]
    return {
      "steps":len(rows),
      "min_left_foot_target_error_m":round(min(map(error,rows)),4),
      "final_left_foot_target_error_m":round(error(rows[-1]),4),
      "median_foot_slip_mps":round(sorted(slip)[len(slip)//2],3),
      "number_of_frames_satisfying_all_support_constraints":len(ready),
      "episode_reset_count":episodes,
      "unexpected_ball_contact_frames":int(touches),
      "note":"Support-ready frames not necessarily consecutive; need 15 consecutive for policy transition."
    }

def main():
    sources={
      "walking":load_json("pretrained_g1_walk_vx045.json"),
      "goal_approach_no_ball":load_json("goal_approach_right06.json"),
      "goal_approach_with_ball":load_json("soccer_scene_approach_left04.json"),
      "preplant_24cm_stop":load_json("support_foot_plant_right06.json"),
      "preplant_8cm_stop":load_json("support_foot_plant_precise06.json"),
    }
    tables={
      "preplant_24cm_stop":foot_stats("support_foot_plant_right06.csv"),
      "preplant_8cm_stop":foot_stats("support_foot_plant_precise06.csv"),
    }
    hashes=[
        sha("tools/goal_conditioned_planner.py"),
        sha("tools/eval_pretrained_g1_walk.py"),
        sha("tools/eval_goal_guided_approach.py"),
        sha("tools/eval_soccer_scene_approach.py"),
        sha("tools/eval_support_foot_plant.py"),
        sha("tools/eval_support_foot_plant_precise.py"),
        sha(".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"),
    ]
    summary={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "project":"KineticKick",
        "root":"Tang /home/you/projects/kinetickick",
        "experiment_summaries":sources,
        "foot_placement_comparisons":tables,
        "model_and_script_hashes":[x for x in hashes if x],
        "observations_not_established":[
            "Foot-ball natural kick under free-root balance",
            "Long-term G1 locomotion robustness",
            "Precision support-foot planting across varied goals",
            "Human video to natural motor coordination policy",
            "Onboard RGB goal detection",
            "Live IsaacLab video of running-up AND kicking using a single learned policy"
        ],
    }
    outfile=R/"stage1_evidence_rollup.json"
    outfile.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=["# Stage 1 — Free-root G1 walking and soccer positioning",
           "",
           "These are observed runs, not a trained natural humanoid kicking controller.",
           ""]
    for key,part in sources.items():
        lines.append(f"## {key}")
        if part is None:
            lines.append("Not completed or no saved summary yet.")
        else:
            for k in ("duration_s","min_root_height_m","resets","reset_count",
                      "displacement_x_m","final_distance_to_approach_xy_m",
                      "final_heading_error_deg","min_support_left_foot_target_error_m",
                      "support_left_foot_stationary_verified_step",
                      "max_foot_ball_contact_force_N"):
                if k in part:
                    lines.append(f"- {k}: {part[k]}")
        lines.append("")
    for key,comp in tables.items():
        lines.extend([f"## {key} — foot constraints",
                      "Not an achieved planned foot plant unless 15 consecutive frames satisfy all criteria."])
        if comp:
            for k,v in comp.items():
                lines.append(f"- {k}: {v}")
        else:lines.append("Pending.")
        lines.append("")
    lines.extend(["## Next gate","Learn support-foot placement/hold with explicit left-foot support contact",
                 "and measurable naturalness, before inserting an aggressive shot into the walking policy.",
                 "Never treat a fixed-root shot demo as free-root humanoid control."])
    (R/"stage1_evidence_rollup.md").write_text("\n".join(lines)+"\n")
    print("EVIDENCE_ROLLUP_DONE",str(outfile),flush=True)
    print("COMPLETED_FIELDS",[(k,v is not None) for k,v in sources.items()],flush=True)

if __name__=="__main__":main()
