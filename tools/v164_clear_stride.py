"""Scale extra source velocity by measured speed shortfall before calling GMT."""

import argparse
from pathlib import Path

import numpy as np
import v164_support_track as tracking


class SpeedMatchedSupport(tracking.Support):
    """Remove extra stride demand once the pelvis attains the source speed."""

    def __init__(self, ref, policy, cfg, signals):
        super().__init__(ref, policy, cfg.copy(), signals)
        self.max_stride_gain = float(cfg["inverse_stride"])
        signals["effective_stride_gain"] = []

    def __call__(self, observation):
        requested = self.v[self.ctx["frame"], :2]
        measured = self.ctx["root_velocity"][:2]
        speed_sq = requested @ requested
        progress = float(measured @ requested) / max(1e-8, speed_sq)
        gain = self.max_stride_gain * np.clip(1.0 - progress, 0.0, 1.0)
        self.cfg["inverse_stride"] = gain
        self.signals["effective_stride_gain"].append(gain)
        return super().__call__(observation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_known_args()[0].output
    tracking.Support = SpeedMatchedSupport
    tracking.main()
    tracking.b.save_json(
        output / "stride_feedback.json",
        {"stride_feedback_sha256": tracking.b.digest(Path(__file__))},
    )
