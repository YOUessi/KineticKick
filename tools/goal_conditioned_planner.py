"""KineticKick goal-conditioned approach / planting / contact plan.

CPU-only mathematical planner. This is NOT a learned controller, a model of
human motion, or robot joint commands. It supplies geometrically coherent
high-level targets and phase labels to a future G1 RL locomotion/tracking policy.

Coordinate system: +X forward in world; +Y robot-left; Z up.
Contact velocity is requested relative to ball->goal (goal center target).
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import argparse,json,math
from pathlib import Path

Vec2=tuple[float,float]
EPS=1e-6

def add(a:Vec2,b:Vec2)->Vec2:
    return a[0]+b[0],a[1]+b[1]
def mul(k:float,a:Vec2)->Vec2:
    return k*a[0],k*a[1]
def sub(a:Vec2,b:Vec2)->Vec2:
    return a[0]-b[0],a[1]-b[1]
def length(a:Vec2)->float:
    return math.hypot(a[0],a[1])
def unit(v:Vec2)->Vec2:
    n=length(v)
    if n<EPS: raise ValueError("ball and goal must differ in horizontal position")
    return v[0]/n,v[1]/n
def left(fwd:Vec2)->Vec2:
    return -fwd[1],fwd[0]
def pcast(xy,field:str)->Vec2:
    v=tuple(float(z) for z in xy)
    if len(v)!=2 or not all(math.isfinite(z) for z in v):
        raise ValueError(f"{field} must be two finite real values")
    return v

@dataclass(frozen=True)
class Phase:
    name:str
    min_duration_s:float
    exit_event:str
    note:str

PHASES=[
    Phase("approach",0.2,"robot_within_preplant_zone",
          "walking policy: move toward a goal-aligned setup location"),
    Phase("plant",0.25,"support_foot_stable",
          "left/right support-foot placement and COM over support polygon"),
    Phase("backswing",0.22,"swing_leg_at_ready_phase",
          "swing foot behind football and counterbalance arms"),
    Phase("strike",0.18,"actual_ball_contact_detected",
          "phase-conditioned hip -> knee -> ankle with selected foot surface"),
    Phase("follow_through",0.35,"swing_foot_passes_ball",
          "continue swing instead of stopping instantly at first impact"),
    Phase("recovery",0.45,"stable_double_support",
          "recover stable stance without falling or premature root reset"),
]

def goal_conditioned_plan(*,ball_xy,goal_xy,robot_xy,foot="right",
                          surface="instep_top",ball_radius=.11,
                          target_ball_speed_mps=3.0):
    b=pcast(ball_xy,"ball_xy")
    g=pcast(goal_xy,"goal_xy")
    r=pcast(robot_xy,"robot_xy")
    if foot not in ("left","right"):raise ValueError("foot must be left or right")
    if surface not in ("instep_top","medial","lateral"):
        raise ValueError("surface must be instep_top, medial or lateral")
    if not (.04<ball_radius<.18):raise ValueError("ball radius out of prototype range")
    if not (0.5<=target_ball_speed_mps<=8.0):
        raise ValueError("target_ball_speed_mps must be in [0.5,8.0]")
    f=unit(sub(g,b))
    l=left(f)
    # Left support foot for right-foot kick, and vice versa.
    support_sign=+1 if foot=="right" else -1
    support_foot=add(add(b,mul(-.14,f)),mul(.16*support_sign,l))
    approach=add(add(b,mul(-.78,f)),mul(.04*support_sign,l))
    preplant=add(add(b,mul(-.33,f)),mul(-.015*support_sign,l))
    swing_start=add(add(b,mul(-.36,f)),mul(-.07*support_sign,l))
    impact=add(b,mul(-ball_radius,f))
    follow=add(add(b,mul(.25,f)),mul(-.045*support_sign,l))
    heading=math.atan2(f[1],f[0])
    init_distance=length(sub(approach,r))
    return {
        "prototype":"G1_goal_conditioned_whole_body_skeleton",
        "trajectory_source":"geometric_planning_NOT_trained_control",
        "robot_start_world_xy":list(r),
        "ball_world_xy":list(b),
        "goal_world_xy":list(g),
        "shot_direction_world":[round(q,6) for q in f],
        "shot_heading_rad":round(heading,6),
        "shot_heading_deg":round(math.degrees(heading),3),
        "kick_foot":foot,
        "support_foot":"left" if foot=="right" else "right",
        "contact_surface":surface,
        "target_ball_speed_mps":target_ball_speed_mps,
        "approach_xy":list(approach),
        "preplant_base_xy":list(preplant),
        "support_foot_plant_xy":list(support_foot),
        "swing_foot_start_xy":list(swing_start),
        "impact_target_xy":list(impact),
        "follow_through_xy":list(follow),
        "initial_approach_distance_m":round(init_distance,4),
        "nominal_approach_time_s":round(max(0.4,init_distance/.45),2),
        "phases":[asdict(q) for q in PHASES],
        "guardrails":{
            "root_must_be_unfixed":True,
            "require_left_right_ground_contact_events":True,
            "require_body_upright_before_strike":True,
            "require_stable_support_foot_before_strike":True,
            "require_robot_angular_velocity_not_excessive":True,
            "no_shot_inference_from_ball_post_contact_frames":True,
            "camera_goal_detector":"not_implemented",
        },
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--robot",type=float,nargs=2,default=(-1.1,0.0))
    p.add_argument("--ball",type=float,nargs=2,default=(0.0,0.0))
    p.add_argument("--goal",type=float,nargs=2,default=(3.2,.4))
    p.add_argument("--surface",choices=("instep_top","medial","lateral"),default="instep_top")
    p.add_argument("--power",type=float,default=3.0)
    p.add_argument("--out",type=str)
    opts=p.parse_args()
    result=goal_conditioned_plan(ball_xy=opts.ball,goal_xy=opts.goal,
              robot_xy=opts.robot,surface=opts.surface,
              target_ball_speed_mps=opts.power)
    output=Path(opts.out) if opts.out else None
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
