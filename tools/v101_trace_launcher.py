#!/usr/bin/env python3
"""Diagnose original unchanged G1/PhysX launch without patching the controller.

Periodic Python stacks are diagnostics, NEVER evidence of a successful physical step.
"""
import argparse
import faulthandler
import runpy
import sys
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval-s", type=int, default=15)
    p.add_argument("target")
    p.add_argument("target_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not (5 <= a.interval_s <= 60):
        raise ValueError("Trace interval outside approved range")
    target = Path(a.target).resolve()
    if not target.is_file() or target.name not in (
        "v080_run_poll_strike.py", "v098_run_pair_strike.py",
        "v094_box_pair_probe.py"
    ):
        raise ValueError("Only frozen original runners are allowed")
    extra = a.target_args[1:] if a.target_args[:1] == ["--"] else a.target_args
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(a.interval_s, repeat=True, file=sys.stderr)
    print("V101_INSTRUMENTED_LAUNCH", target.name, "args", extra, flush=True)
    sys.argv = [str(target), *extra]
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        faulthandler.cancel_dump_traceback_later()
        print("V101_INSTRUMENTED_EXIT", target.name, flush=True)

if __name__ == "__main__":
    main()
