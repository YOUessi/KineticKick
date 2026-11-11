#!/usr/bin/env python3
"""One-shot Isaac/Kit lifecycle probe. A diagnostic, NEVER a G1 trial."""
from __future__ import annotations
import argparse
import faulthandler
import gc
import signal
import sys
import time

def mark(event: str) -> None:
    print(f"V074_LIFECYCLE {time.monotonic():.6f} {event}", flush=True)

def run(variant: str) -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    faulthandler.dump_traceback_later(12, repeat=True, file=sys.stderr)
    mark("APP_LAUNCH_ENTER")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app
    mark("APP_LAUNCH_RETURN")
    try:
        mark("SIM_IMPORT_ENTER")
        from isaaclab.sim import SimulationCfg, SimulationContext
        mark("SIM_IMPORT_RETURN")
        mark("SIM_CONSTRUCT_ENTER")
        sim = SimulationContext(SimulationCfg(dt=0.02))
        mark("SIM_CONSTRUCT_RETURN")
        mark("SIM_RESET_ENTER")
        sim.reset()
        mark("SIM_RESET_RETURN")
        for i in range(8):
            sim.step()
            mark(f"SIM_STEP_{i+1}_RETURN")
        mark("EIGHT_REAL_SIM_STEPS_DONE_NOT_G1")
        if variant == "cleanup":
            mark("PRE_CLOSE_SIM_STOP_ENTER")
            sim.stop()
            mark("PRE_CLOSE_SIM_STOP_RETURN")
            mark("PRE_CLOSE_CLEAR_INSTANCE_ENTER")
            SimulationContext.clear_instance()
            mark("PRE_CLOSE_CLEAR_INSTANCE_RETURN")
            del sim
            gc.collect()
            mark("PRE_CLOSE_GARBAGE_COLLECT_RETURN")
    finally:
        mark("APP_CLOSE_ENTER")
        app.close()
        mark("APP_CLOSE_RETURN")
        faulthandler.cancel_dump_traceback_later()
    mark("PYTHON_MAIN_RETURN")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=("baseline", "cleanup"))
    run(p.parse_args().variant)
