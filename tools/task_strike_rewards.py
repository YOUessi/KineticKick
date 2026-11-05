"""Training-only task completion objective; independent physical gates unchanged."""
import torch


def task_focused_value(base,elapsed,scored,quality,goal_event):
    quality=quality.clamp(0.,1.).square()
    late_weight=torch.where(scored,quality,torch.full_like(quality,.05))
    weight=torch.where(elapsed<=125.,torch.ones_like(quality),late_weight)
    # Never attenuate termination penalties. A goal is a one-step event.
    return base.clamp_max(0.)+base.clamp_min(0.)*weight+1000.*goal_event*quality


def task_focused_reward(env):
    from closed_loop_strike import get_task
    task=get_task(env);event=task.score_event.clone();base=task.reward()
    return task_focused_value(base,task.elapsed,task.scored,task.contact_quality,event)


def missed_strike(env):
    from closed_loop_strike import get_task
    task=get_task(env)
    return (task.elapsed>125.)&~task.contact


def missed_goal(env):
    from closed_loop_strike import get_task
    task=get_task(env)
    return (task.elapsed>200.)&~task.scored
