#!/usr/bin/env python3
"""Pre-registered shutdown-only STOP callback guard. Not a G1 experiment."""
import argparse
import faulthandler
import signal
import sys
import time

def mark(s):
    print("V075_SHUTDOWN_TRACE", "%.6f" % time.monotonic(), s, flush=True)

def run(variant):
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(15, repeat=True)
    mark("APP_LAUNCH_ENTER")
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    mark("APP_LAUNCH_RETURN")
    sim=None
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext
        mark("SIM_CONSTRUCT_ENTER")
        sim=SimulationContext(SimulationCfg(dt=0.02))
        mark("SIM_RESET_ENTER")
        sim.reset()
        mark("SIM_RESET_RETURN")
        for i in range(8):
            sim.step()
            mark("SIM_STEP_%d_RETURN" % (i+1))
        mark("EIGHT_REAL_SIM_STEPS_DONE_NOT_G1")
    finally:
        if sim is not None:
            # ONLY after the last physical integration and before any STOP event.
            mark("POST_PHYSICS_SHUTDOWN_GUARD_ENTER")
            sim._disable_app_control_on_stop_handle=True
            mark("POST_PHYSICS_SHUTDOWN_GUARD_SET_TRUE")
            if variant=="post_physics_guarded_stop":
                mark("SIM_STOP_ENTER")
                sim.stop()
                mark("SIM_STOP_RETURN")
        mark("APP_CLOSE_ENTER")
        app.close()
        mark("APP_CLOSE_RETURN")
        faulthandler.cancel_dump_traceback_later()
    mark("PYTHON_MAIN_RETURN")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--variant",required=True,
                    choices=("post_physics_guarded_close","post_physics_guarded_stop"))
    run(ap.parse_args().variant)
