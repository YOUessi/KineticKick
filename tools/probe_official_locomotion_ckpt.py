"""Check official published G1 gait checkpoint without changing Isaac Lab packages.

Prints whether NVIDIA's published RSL-RL pretrained policy can be fetched for
the installed Isaac Lab v2.1.x task. Downloads only when available.
"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app

from isaaclab.utils.pretrained_checkpoint import (
    get_published_pretrained_checkpoint,
    get_published_pretrained_checkpoint_path,
)
from pathlib import Path
try:
    task="Isaac-Velocity-Flat-G1-v0"
    print("CHECKPOINT_REMOTE_SOURCE",get_published_pretrained_checkpoint_path("rsl_rl",task),flush=True)
    path=get_published_pretrained_checkpoint("rsl_rl",task)
    print("CHECKPOINT_FETCH_RESULT",path,flush=True)
    if path:
        p=Path(path)
        print("CHECKPOINT_BYTES",p.stat().st_size,flush=True)
finally:
    app.close()
