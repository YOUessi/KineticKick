import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v100_gpu_batch as m
def test_frozen_matrix():
 c=json.loads((ROOT/"configs/v100_frozen_matrix.json").read_text())
 assert c["base_committed_sha"]=="70e3f2a7b743c7c6c04df54c981b4313726bb603" and c["version"]=="v0.1.100"
 assert len(c["cases"])==12 and len({x['name'] for x in c['cases']})==12
 assert [x["scene"] for x in c["cases"]]==["L"]*6+["M"]*3+["N"]*3
 assert [x["observer"] for x in c["cases"]][:6]==[False,True]*3
 assert c["resources"]["min_free_gib"]>=20 and c["resources"]["per_child_timeout_s"]==110
 assert c["previous_v099_unscored_attempt"]["raw_physics_episode_certified"] is False
 assert all(x["seed"] in (1601,1607,1613) for x in c["cases"])
def test_safety_and_wrappers():
 c=json.loads((ROOT/"configs/v100_frozen_matrix.json").read_text())
 assert c["control_wrapper_off"]=="tools/v080_run_poll_strike.py"
 assert c["control_wrapper_on"]=="tools/v098_run_pair_strike.py"
 assert c["gates"]["instep_impulse_min"]==.95 and c["gates"]["max_nonfoot_N_exclusive"]==20
 assert c["gates"]["full_recovery_s"]==14 and c["gates"]["all29_speed_bounds"]
 assert c["safety"]["original_controllers_unchanged"] and c["safety"]["no_native_manifold_claim"]
 assert "tools/v080_run_poll_strike.py" in m.FROZEN
def test_raw_physical_parity_full_recursive():
 a={"q":[1.,2.],"dq":[4.],"support":{"left":10.}}
 assert m._max_abs(a,a)==0
 assert m._max_abs(a,{**a,"q":[1.,3.]})==1.
 assert m._max_abs(a,{**a,"extra":3})==float("inf")
