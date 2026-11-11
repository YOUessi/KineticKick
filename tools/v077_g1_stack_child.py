#!/usr/bin/env python3
"""Bounded diagnostic trampoline, NOT a validated read-only PhysX observer.

Installs only a signal-triggered CPython traceback dump before executing the
frozen v76 original G1 control entry point. Do not use its trajectories for
off/on no-perturbation claims: even this trampoline can change timings.
"""
from __future__ import annotations
import faulthandler
from pathlib import Path
import runpy
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "tools/v076_run_placement_strike.py"


def main() -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    print("V077_DIAG_SIGNAL_REGISTERED", flush=True)
    sys.argv = [str(ENTRY), *sys.argv[1:]]
    runpy.run_path(str(ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
