"""SE(2) remapping of the existing 176-value running reference observation.

Only reference cues are remapped. Measured state, joint references and contact
labels remain unchanged. Call after the original placement anchor offset.
"""
import torch


def transform_running_reference(observations, transform, root_position, root_rotation,
                                anchor_position, anchor_rotation, active):
    if observations.shape != (len(root_position), 176):
        raise ValueError('Original 176-value running observation required')
    changed = active & (transform.alpha > 0.)
    if not bool(changed.any()):
        return observations
    out = observations.clone()
    for start in (154, 157):
        local = observations[:, start:start+3]
        world = root_position + (root_rotation @ local[..., None]).squeeze(-1)
        moved = transform.transform_positions(world)
        out[:, start:start+3] = (root_rotation.transpose(-1, -2) @ (moved-root_position)[..., None]).squeeze(-1)
    anchor_world = anchor_position + (anchor_rotation @ observations[:, 160:163, None]).squeeze(-1)
    out[:, 160:163] = (anchor_rotation.transpose(-1, -2) @
                      (transform.transform_positions(anchor_world)-anchor_position)[..., None]).squeeze(-1)
    # The original six numbers store the first two columns of the relative
    # rotation matrix, row-major. Their world transform requires torso rotation.
    relative_columns = observations[:, 163:169].reshape(-1, 3, 2)
    out[:, 163:169] = (anchor_rotation.transpose(-1, -2) @ transform.rotation @
                      anchor_rotation @ relative_columns).reshape(-1, 6)
    out[:, 61:64] = transform.transform_vectors(observations[:, 61:64])
    out[:, 63] += transform.yaw_rate
    return torch.where(changed[:, None], out, observations)
