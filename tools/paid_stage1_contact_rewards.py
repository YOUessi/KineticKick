"""KineticKick experimental PAiD contact-phase tracking reward terms.

These terms do not directly move a physical body. They return shaped rewards
from reference relative-body goals, actual PhysX articulation, and foot force
sensors. Added to a distinct Stage-I ablation only; untouched PAiD baseline
remains preserved.
"""
from __future__ import annotations

import torch


def torso_height_tracking_exp(env, command_name: str, std: float = 0.12):
    """Reward keeping physical torso at the desired reference vertical height."""
    cmd = env.command_manager.get_term(command_name)
    dz = cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2]
    return torch.exp(-torch.square(dz / std))


def lifted_foot_z_tracking_exp(
    env, command_name: str, std: float = 0.09, min_lift: float = 0.05
):
    """Apply extra foot-z tracking when the reference lifts one foot.

    A foot is considered lifted when its link origin is above the other foot's
    reference link origin by at least min_lift. This is a kinematic proxy, not
    measured contact intent or a newly labeled human contact event.
    """
    cmd = env.command_manager.get_term(command_name)
    body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    idx = [cmd.cfg.body_names.index(n) for n in body_names]
    ref = cmd.body_pos_relative_w[:, idx, 2]
    current = cmd.robot_body_pos_w[:, idx, 2]
    reference_lift = ref - ref.min(dim=1, keepdim=True).values
    swing = (reference_lift > min_lift).float()
    per_foot_reward = torch.exp(-torch.square((current - ref) / std))
    return (swing * per_foot_reward).sum(dim=1) / swing.sum(dim=1).clamp(min=1.0)


def support_contact_stability_exp(
    env, command_name: str, target_normal_force: float = 35.0
):
    """Support load agreement, coarse contact proxy; no fabricated contact labels.

    Uses the foot with the lower reference link height as likely support;
    returns fraction of support feet whose measured vertical normal force
    exceeds threshold. Not used for final soccer naturalness claim.
    """
    cmd = env.command_manager.get_term(command_name)
    names = ("left_ankle_roll_link", "right_ankle_roll_link")
    idx = [cmd.cfg.body_names.index(n) for n in names]
    foot_z = cmd.body_pos_relative_w[:, idx, 2]
    expected_stance = (foot_z - foot_z.min(dim=1, keepdim=True).values) < 0.03
    sensor = env.scene["contact_forces"]
    sensor_idx = [sensor.body_names.index(n) for n in names]
    force_z = sensor.data.net_forces_w[:, sensor_idx, 2]
    observed_loaded = force_z > target_normal_force
    return (
        (expected_stance & observed_loaded).float().sum(dim=1)
        / expected_stance.float().sum(dim=1).clamp(min=1.0)
    )
