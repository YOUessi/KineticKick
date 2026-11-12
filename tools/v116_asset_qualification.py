#!/usr/bin/env python3
"""Fail-closed, read-only original-G1 asset preflight before GPU physics.

Checks actual ignored recovery model and human reference hashes, not just tracked code.
Never creates, rewrites or deletes any original asset.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_asset(root: Path, relative: str, expected: str, label: str) -> dict:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not relative:
        raise ValueError(f"UNSAFE_ASSET_PATH:{label}")
    path = root / part
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_G1_ASSET:{label}:{relative}")
    actual = digest(path)
    if actual != expected or len(expected) != 64:
        raise ValueError(f"G1_ASSET_SHA_MISMATCH:{label}:{relative}")
    return dict(label=label, relative_path=relative, sha256=actual, bytes=path.stat().st_size)


def validate_controller_assets(root: Path, rel_config: str) -> list[dict]:
    c = json.loads((root / rel_config).read_text())
    recovery = c.get("post_shot_recovery", {})
    if not recovery.get("checkpoint") or not recovery.get("checkpoint_sha256"):
        raise ValueError("RECOVERY_MODEL_UNQUALIFIED")
    out = [validate_asset(root, recovery["checkpoint"],
                          recovery["checkpoint_sha256"], "recovery_checkpoint")]
    path = c.get("soccer_checkpoint")
    want = c.get("frozen_files_sha256", {}).get(path)
    if not path or not want:
        raise ValueError("SOCCER_POLICY_SHA_UNQUALIFIED")
    out.append(validate_asset(root, path, want, "soccer_checkpoint"))
    human = c.get("human_coordination_reference", {})
    if human.get("enabled"):
        if not human.get("trajectory") or not human.get("trajectory_sha256"):
            raise ValueError("HUMAN_REFERENCE_SHA_UNQUALIFIED")
        out.append(validate_asset(root, human["trajectory"],
                                  human["trajectory_sha256"], "kinematic_reference_only"))
    return out


def assert_all_assets(root: Path, matrix: dict) -> dict:
    # Every original controller must be independently qualified before any child is launched.
    manifest = {}
    for label, rel in matrix["control_configs"].items():
        manifest[label] = validate_controller_assets(root, rel)
    for label in manifest:
        assert any(x["label"] == "recovery_checkpoint" for x in manifest[label])
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "configs/v116_frozen_matrix.json").read_text())
    print(json.dumps(assert_all_assets(root, c), indent=2))
