#!/usr/bin/env python3
"""Make two source-faithful v126 research failure images from immutable JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def draw(raw: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(raw.read_text())
    if data["native_contact_pair_conclusion"] != "UNKNOWN":
        raise ValueError("COLLISION_TRUTH_CLAIM_NOT_ALLOWED")
    pairs = data["original_scene_clustered_effect_test"]["comparisons"]
    if len(pairs) != 2:
        raise ValueError("NO_TWO_PREREGISTERED_COMPARISONS")
    scenes = [x["scene"] for x in pairs[0]["individual_scene_outcomes"]]
    baseline = [x["baseline_pass"] for x in pairs[0]["individual_scene_outcomes"]]
    controllers = [
        ("v32", baseline),
        ("CMU10_02", [x["candidate_pass"] for x in pairs[0]["individual_scene_outcomes"]]),
        ("CMU10_05", [x["candidate_pass"] for x in pairs[1]["individual_scene_outcomes"]]),
    ]
    # Heatmap is observed success only; no simulated/guessed behavior.
    fig, ax = plt.subplots(figsize=(11, 3.5))
    matrix = [[int(v) for v in values] for _, values in controllers]
    ax.imshow(matrix, vmin=0, vmax=1, cmap="Greys", interpolation="nearest", aspect="auto")
    for i, row in enumerate(matrix):
        for j, state in enumerate(row):
            ax.text(j, i, "PASS" if state else "FAIL", va="center", ha="center",
                    color="white" if state else "black", fontsize=9)
    ax.set_xticks(range(len(scenes)), scenes)
    ax.set_yticks(range(3), [name for name, _ in controllers])
    ax.set_title("Original G1 200Hz PhysX: 8 scene-paired outcomes (observer repeats excluded)")
    ax.set_xlabel("Frozen ball position / seed (independent scene is unit)")
    fig.tight_layout()
    target = out_dir / "v126_real_scene_paired_strict_gates.png"
    if target.exists():
        raise FileExistsError(str(target))
    fig.savefig(target, dpi=160)
    plt.close(fig)

    anatomy = data["source_urdf_contact_topology"]
    path = anatomy["topological_connection"]["links"]
    if path[0] != "left_wrist_yaw_link" or path[-1] != "left_hip_roll_link":
        raise ValueError("UNQUALIFIED_ANATOMIC_PATH")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(range(len(path)), [0] * len(path), color="#808080", lw=2)
    for i, link in enumerate(path):
        ax.scatter(i, 0, s=140, facecolor="#dddddd", edgecolor="black", zorder=3)
        ax.text(i, 0.10 if i % 2 == 0 else -0.13, link.replace("_link", ""),
                rotation=55, va="bottom" if i % 2 == 0 else "top",
                ha="left", fontsize=8)
    ax.set_xlim(-.5, len(path))
    ax.set_ylim(-.8, .8)
    ax.axis("off")
    ax.set_title("Original 29DoF G1 URDF joint graph: wrist–hip candidate (NOT a PhysX contact witness)")
    fig.tight_layout()
    target2 = out_dir / "v126_original_G1_wrist_hip_static_topology_NOT_CONTACT.png"
    if target2.exists():
        raise FileExistsError(str(target2))
    fig.savefig(target2, dpi=160)
    plt.close(fig)
    return [str(target), str(target2)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    opts = p.parse_args()
    print("v126_real_figures", draw(opts.audit, opts.out))
