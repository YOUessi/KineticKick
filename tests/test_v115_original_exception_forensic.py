import difflib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def test_diagnostic_wrapper_does_not_change_original_physics_except_exception_print():
    original=(R/"tools/v076_eval_guarded.py").read_text()
    diagnostic=(R/"tools/v115_eval_guarded_diagnostic.py").read_text()
    anchor="    finally:\n        if held_env is not None:"
    replacement="""    except BaseException as _v115_original_error:
        import traceback as _v115_traceback
        print("V115_UNMASKED_PRE_SHUTDOWN_ORIGINAL_EXCEPTION",
              type(_v115_original_error).__name__,repr(_v115_original_error),flush=True)
        _v115_traceback.print_exc()
        raise
    finally:
        if held_env is not None:"""
    assert diagnostic==original.replace(anchor,replacement)
    assert original.count(anchor)==1
    poll=(R/"tools/v080_run_poll_strike.py").read_text()
    assert (R/"tools/v115_diagnostic_poll_strike.py").read_text()==poll.replace(
       "tools/v076_eval_guarded.py","tools/v115_eval_guarded_diagnostic.py")
def test_protocol_diagnostic_unscored():
    text=(R/"docs/V115_ORIGINAL_EXCEPTION_CAPTURE_FROZEN.md").read_text()
    assert "diagnostic-only" in text
    assert "not certified" in text
