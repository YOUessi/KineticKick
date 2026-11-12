#!/usr/bin/env python3
"""Bounded failure-only original G1 Python exception attribution.

This instrument changes execution timing; NEVER report OFF/ON physics parity.
When an original preclose exception already exists, abort this child instead of
entering the known blocking SimulationApp.close callback. No controller changes.
"""
import argparse,faulthandler,os,runpy,sys,traceback
from pathlib import Path
NAME="v076_eval_guarded.py"
FINAL_LINES=(183,192)
_seen=set()
def report_exception(label,exc):
    print("V105_"+label+" "+type(exc).__name__+": "+str(exc),file=sys.stderr,flush=True)
def global_trace(frame,event,arg):
    if event=="call" and Path(frame.f_code.co_filename).name==NAME and frame.f_code.co_name=="main":
        return local_trace
    return None
def local_trace(frame,event,arg):
    if event=="exception":
        typ,exc,tb=arg
        key=(frame.f_lineno,type(exc).__name__,str(exc)[:120])
        if key not in _seen:
            _seen.add(key);report_exception("MAIN_EXCEPTION_PROPAGATED",exc)
    if event=="line" and frame.f_lineno in FINAL_LINES:
        typ,exc,tb=sys.exc_info()
        print("V105_ORIGINAL_MAIN_FINALLY line="+str(frame.f_lineno)+
          " active="+str(exc is not None),flush=True)
        if exc is not None:
            report_exception("ORIGINAL_PRE_CLOSE_CAUSE",exc)
            if tb is not None:
                traceback.print_exception(typ,exc,tb,file=sys.stderr)
                sys.stderr.flush()
            if frame.f_lineno==192:
                print("V105_ABORT_BLOCKING_TEARDOWN_ONLY_ON_EXISTING_EXCEPTION",flush=True)
                sys.stdout.flush();sys.stderr.flush()
                os._exit(78)
    return local_trace
def main():
    p=argparse.ArgumentParser();p.add_argument("target");p.add_argument("args",nargs=argparse.REMAINDER)
    a=p.parse_args();target=Path(a.target).resolve()
    if target.name!="v080_run_poll_strike.py" or not target.is_file():
        raise ValueError("NOT_FROZEN_ORIGINAL_V080")
    rest=a.args[1:] if a.args[:1]==["--"] else a.args
    sys.argv=[str(target),*rest]
    faulthandler.enable(file=sys.stderr,all_threads=True)
    faulthandler.dump_traceback_later(12,repeat=True,file=sys.stderr)
    print("V105_ORIGINAL_EXCEPTION_TRACER_ENABLED",flush=True)
    sys.settrace(global_trace)
    try:runpy.run_path(str(target),run_name="__main__")
    finally:
        sys.settrace(None)
        faulthandler.cancel_dump_traceback_later()
        print("V105_TRACER_EXIT",flush=True)
if __name__=="__main__":main()
