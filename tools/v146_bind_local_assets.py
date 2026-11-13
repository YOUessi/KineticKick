#!/usr/bin/env python3
"""Bind ORIGINAL Tang-local immutable media/model inputs to v146 worktree.

Creates only symlinks under existing gitignored locations, never copies,
deletes or changes original CMU, C3D, ONNX, NPZ, images or checkpoints. Safe
default is --check; explicit --bind is for actual future G1 physics runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SOURCE = Path("/home/you/projects/kinetickick")
TARGET = Path(__file__).resolve().parents[1]
ALLOWED = (
    "third_party", ".pretrained_checkpoints", "experiments", "data",
    "media", "configs/paid_original", "configs/paid_g1_main.urdf",
)


def check_bindings(*, source=SOURCE, target=TARGET, bind=False):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("CANNOT_BIND_ORIGINAL_TO_ITSELF")
    if not source.is_dir() or not target.is_dir():
        raise ValueError("ORIGINAL_OR_WORKTREE_MISSING")
    out = {}
    for rel in ALLOWED:
        src = source / rel
        dst = target / rel
        if not src.exists():
            out[rel] = "NO_ORIGINAL_ASSET_DO_NOT_FABRICATE"
            continue
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError("NON_ORIGINAL_ASSET_SYMLINK:" + rel)
            out[rel] = "EXISTING_VERIFIED_LINK"
        elif dst.exists():
            raise ValueError("LOCAL_PATH_ALREADY_OCCUPIED_NO_OVERWRITE:" + rel)
        elif bind:
            if not dst.parent.is_dir():
                raise ValueError("TARGET_PARENT_NOT_EXISTS:" + rel)
            dst.symlink_to(src, target_is_directory=src.is_dir())
            out[rel] = "CREATED_GITIGNORED_LINK_ONLY"
        else:
            out[rel] = "NEEDS_READONLY_SYMLINK"
    return {"source":str(source), "worktree":str(target),
            "only_gitignored_paths":True, "bound_without_copy_or_delete":bind,
            "assets":out}


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group()
    group.add_argument("--bind",action="store_true")
    group.add_argument("--check",action="store_true")
    args=p.parse_args()
    print(json.dumps(check_bindings(bind=args.bind),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
