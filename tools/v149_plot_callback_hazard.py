#!/usr/bin/env python3
"""Source-only callback arrival censoring plot. No GPU physics evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from v149_independent_phase_audit import independent_review, PHASES


def plot(document, output):
    independent_review(document)
    path = Path(output)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError("V149_PLOT_CANNOT_OVERWRITE")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [c for c in document["synthetic_cells"]
             if c["header_kind"] == "VECTOR_CONTACT"]
    by_phase = Counter(c["phase"] for c in valid if c["valid_packets_preserved"])
    x = list(range(len(PHASES)))
    new = [by_phase[p] for p in PHASES]
    old = [by_phase[p] if p == "INSIDE_SCENE_UPDATE" else 0 for p in PHASES]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i-.18 for i in x], old, width=.36, label="v146 bracket-only expected retention")
    ax.bar([i+.18 for i in x], new, width=.36, label="v149 anyphase copy-only retention")
    ax.set_xticks(x, ["Before first", "Inside update", "Between updates", "After terminal"])
    ax.set_ylabel("Copied valid Vec3 callback fixtures (synthetic)")
    ax.set_ylim(0, 7)
    ax.set_title("CPU phase-order falsification, NOT true PhysX contacts")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return {"expected_old_bracket_retained_synthetic": sum(old),
            "new_anyphase_copied_synthetic": sum(new),
            "not_native_runtime": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(plot(json.loads(args.input.read_text()),args.output)))


if __name__=="__main__":
    main()
