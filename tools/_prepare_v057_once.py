from pathlib import Path
import hashlib, json
p=Path("tools/eval_free_shooting.py")
s=p.read_text()
old='cfg.sim.device = "cuda:0"'
assert s.count(old)==1
needle='    parser.add_argument("--seed", type=int, default=601)'
assert s.count(needle)==1
s=s.replace(old,'cfg.sim.device = args.sim_device')
s=s.replace(needle,needle+'\n    parser.add_argument("--sim-device", choices=("cpu","cuda:0"), default="cuda:0")')
Path("tools/eval_free_shooting_backend_v057.py").write_text("# v057 diagnostic copy: only simulation backend selected; no original policy/controller changes.\n"+s)
s=Path("tools/run_placement_strike_v056.py").read_text()
assert s.count("tools/eval_free_shooting.py")==2
s=s.replace("tools/eval_free_shooting.py","tools/eval_free_shooting_backend_v057.py")
s=s.replace("    a=p.parse_args()", '    p.add_argument("--sim-device",choices=("cpu","cuda:0"),default="cuda:0")\n    a=p.parse_args()')
s=s.replace('if a.perturb_reset_v056:sys.argv+=["--perturb-reset"]','sys.argv += ["--sim-device",a.sim_device]\n        if a.perturb_reset_v056:sys.argv+=["--perturb-reset"]')
s=s.replace("            result.update(running_evidence=run,stride_shot_evidence=stride,","            result.update(v057_requested_physics_backend=a.sim_device,v057_diagnostic_backend_only=True,running_evidence=run,stride_shot_evidence=stride,")
Path("tools/run_placement_strike_v057.py").write_text("# v057 G1 backend diagnostic variant. Original controller, contact gates and physics dt unchanged.\n"+s)
v56=json.loads(Path("configs/physx_contact_safety_v056.json").read_text())
new={"version":"v0.1.57","parent_branch":"version/v0.1.56",
"parent_commit":"4714393ed447ef569bc1ebe757f6665929e24a32",
"claim":"CPU-GPU separate-backend G1 qualification; no CPU-to-GPU physical equivalence assumed",
"preregistered":"6 G1 episodes (3 frozen controllers x 2 backends), read-only subscriber and whole-body observer; 4 toy controls (positive/negative x CPU/GPU). Original 200Hz, 14s capped at early termination. No training or tuning.",
"python":"/home/you/conda_env/env_isaaclab/bin/python",
"minimum_free_gib":20,
"source_hashes":{},
"policies":v56["policies"],"cases":[],"toy_cases":[]}
sources=["tools/eval_free_shooting.py","tools/eval_free_shooting_backend_v057.py",
"tools/run_placement_strike_v056.py","tools/run_placement_strike_v057.py",
"tools/native_contact_reader_v056.py","tools/physx_pair_dynamics_v055.py",
"tools/probe_physx_pair_isolated_v056.py","tools/closed_loop_acceptance.py",
"tools/running_balance.py","tools/audit_real_batch_v056.py"]+list(v56["policies"].values())
for rel in sources:
    new["source_hashes"][rel]=hashlib.sha256(Path(rel).read_bytes()).hexdigest()
for device in ["cpu","cuda:0"]:
    for name in ["frozen_v32","cmu_10_02","cmu_10_05"]:
        dev=device.replace(":","_")
        new["cases"].append({"id":f"center_{dev}_{name}","backend":device,
            "policy":name,"seed":1201,"ball_offset_xy_m":[0,0],
            "velocity_y_delta_mps":0,"perturb_reset":False,
            "output":f"reports/v057_center_{dev}_{name}.json",
            "gpu_reference":f"reports/v056_center_native_{name}.json",
            "group":"backend_qualification_not_training"})
for device in ["cpu","cuda:0"]:
    for state in ["collide","separated"]:
        dev=device.replace(":","_")
        new["toy_cases"].append({"id":f"toy_{dev}_{state}",
            "backend":device,"control":state,
            "output":f"reports/v057_toy_{dev}_{state}.json"})
Path("configs/physx_g1_backend_pair_v057.json").write_text(json.dumps(new,indent=2)+"\n")
print("FROZEN_DESIGN",len(new["cases"]),len(new["toy_cases"]))
print("SOURCE_CONFIG_SHA",hashlib.sha256(Path("configs/physx_g1_backend_pair_v057.json").read_bytes()).hexdigest())