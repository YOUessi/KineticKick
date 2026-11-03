#!/usr/bin/env python3
"""Create a new *isolated* PAiD PPO checkpoint with global-reference actor inputs.

The original 154-dim actor features remain in the exact same order; 12 extra
features are appended (reference anchor position 3, reference orientation 6,
observed base linear velocity 3). New input weights start at zero to preserve
original policy output on day zero and become trainable. Critic is unchanged.
Do not rewrite the source checkpoint.
"""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path("/home/you/projects/kinetickick")
source=ROOT/"experiments/paid29_stage1_contact_phase/seed84_n96_iter200_phase108_148_contactv1/last_model.pt"
dest=ROOT/"experiments/paid29_root_reference_init/model_154_to166.pt"
assert source.is_file()
if dest.exists():
    raise FileExistsError(f"Refusing to overwrite: {dest}")
dest.parent.mkdir(parents=True,exist_ok=True)
ckpt=torch.load(source,map_location="cpu",weights_only=False)
weights=ckpt["model_state_dict"]
name="actor.0.weight"
w=weights[name]
assert w.shape==(512,154),w.shape
weights[name]=torch.cat([w,torch.zeros((512,12),dtype=w.dtype)],dim=1)
obs=ckpt["obs_norm_state_dict"]
assert obs["_mean"].shape==(1,154)
for k, fill in (("_mean",0.0),("_var",1.0),("_std",1.0)):
    orig=obs[k]
    obs[k]=torch.cat([orig,torch.full((1,12),fill,dtype=orig.dtype)],dim=1)
assert obs["_mean"].shape==(1,166)
# The old AdamW moments for 512x154 are incompatible with 512x166.
# Do not accidentally restore optimizer tensors of mismatched shape.
ckpt["optimizer_state_dict"]={}
ckpt["infos"]={"migration":"append_12_actor_features_only; optimizer reset; no learned result",
              "source":str(source),
              "actor_features":["original_154",
                 "motion_anchor_pos_b_3","motion_anchor_ori_b_6","base_lin_vel_3"]}
torch.save(ckpt,dest)
manifest={
    "source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
    "dest":str(dest),"dest_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
    "old_observation_dims":154,"new_observation_dims":166,
    "appended_terms":["motion_anchor_pos_b 3","motion_anchor_ori_b 6","base_lin_vel 3"],
    "actor_new_input_weights":"zero-initialized",
    "optimizer":"reinitialized on load; optimizer_state_dict cleared",
    "claim":"Checkpoint adaptation only, not a trained better model."
}
(dest.parent/"migration_manifest.json").write_text(json.dumps(manifest,indent=2))
print("ROOT_OBS_CHECKPOINT_PREPARED",json.dumps(manifest),flush=True)
