from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
freeze=json.loads((ROOT/"configs/v076_frozen_g1_native_shutdown.json").read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_original_sha_lock():
 for name,sha in freeze["immutable_inputs_sha256"].items():
  assert digest(ROOT/name)==sha
def test_shutdown_guard_only_after_physics():
 a=(ROOT/"tools/eval_free_shooting.py").read_text()
 b=(ROOT/"tools/v076_eval_guarded.py").read_text()
 original="        if held_env is not None:\n            held_env.close()\n        app.close()"
 replacement=("        if held_env is not None:\n"
 "            from isaaclab.sim import SimulationContext\n"
 "            context=SimulationContext.instance()\n"
 "            if context is None: raise RuntimeError('V076_NO_SIM_CONTEXT_AFTER_PHYSICS')\n"
 "            print('V076_AFTER_PHYSICS_PRE_SHUTDOWN_GUARD_SET',flush=True)\n"
 "            context._disable_app_control_on_stop_handle=True\n"
 "            print('V076_POST_PHYSICS_STOP_CALLBACK_GUARD_TRUE',flush=True)\n"
 "            held_env.close()\n"
 "        app.close()")
 assert a.count(original)==1 and b==a.replace(original,replacement)
def test_runner_only_changes_eval_path():
 a=(ROOT/"tools/run_placement_strike_v056.py").read_text()
 b=(ROOT/"tools/v076_run_placement_strike.py").read_text()
 assert a.count("ROOT/'tools/eval_free_shooting.py'")==2
 assert b==a.replace("ROOT/'tools/eval_free_shooting.py'","ROOT/'tools/v076_eval_guarded.py'")
def test_no_contact_schema_mutation():
 b=(ROOT/"tools/v076_eval_guarded.py").read_text()
 assert "PhysxContactReportAPI" not in b
def test_nonfoot_shot_gate_still_present():
 b=(ROOT/"tools/v076_run_placement_strike.py").read_text()
 assert "physical_nonfoot" in b and "from contact_frame_sampler import ContactFrameSampler" in b
def test_freeze_has_three_pairs_and_lock():
 assert len(freeze["physical_trials_order"])==3
 assert len(freeze["per_trial"])==2
 assert "kinetickick_physics_gpu.lock" in freeze["gpu_lock"]
