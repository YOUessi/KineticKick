#!/usr/bin/env python3
"""Generate on-Tang CPU observer-parity and genuine contact-pair failure plots."""
import json
from pathlib import Path
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def load(s):
    return json.loads((ROOT/f"reports/v059_{s}.json").read_text())
off=load("cpu_v32_off_s1201")
full=load("cpu_v32_full_s1201")
pair=full["physx_native_report_v056"]
assert off["v057_requested_physics_backend"]=="cpu"
assert full["v057_requested_physics_backend"]=="cpu"
assert len(off["joint_names"])==len(full["joint_names"])==29
assert full["stride_shot_success"] is False
series={}
for name,r in (("off",off),("read_only_on",full)):
    f=r["traces"][0]
    limit=r["joint_velocity_limits"][0]
    series[name]={
        "t":[x["time_s"] for x in f],
        "qdot":[max(abs(v)/lim for v,lim in zip(x["joint_vel"],limit)) for x in f],
        "net":[x["nonfoot_contact_N"] for x in f],
    }
out=ROOT/"media"
out.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(10,4.5))
for label,x in series.items():
    plt.plot(x["t"],x["qdot"],lw=1.15,alpha=.80,label=f"CPU native observer {label}")
plt.axhline(1,linestyle="--",label="29DoF qdot original limit")
plt.axvline(1.58,linestyle=":",label="first ball contact 1.58s")
plt.xlabel("CPU physical simulation time (s)")
plt.ylabel("Maximum original 29DoF |qdot| / joint velocity limit")
plt.title("G1 CPU v32: 29DoF measured velocity; 333 rows incl. terminal marker")
plt.legend();plt.tight_layout()
p1=out/"v059_cpu_observer_qdot_parity.png"
plt.savefig(p1,dpi=155);plt.close()

plt.figure(figsize=(10,4.5))
for label,x in series.items():
    plt.plot(x["t"],x["net"],lw=1.1,alpha=.75,label=f"CPU native observer {label} net nonfoot")
n=0
for event in pair["saved_interesting_events"]:
    if event.get("candidate_pelvis_right_hip") and event.get("point_sample"):
        plt.axvline(event["time_s"],linestyle=":",alpha=.55,
                    label="native G1 pelvis/right-hip point" if not n else None)
        n+=len(event["point_sample"])
plt.axvline(1.58,linestyle="--",label="ball impact")
plt.title("Original G1 CPU: native pelvis–right hip point evidence near nonfoot stop")
plt.xlabel("CPU physical simulation time (s)")
plt.ylabel("Nonfoot body net contact N; NOT pair-resolved force")
plt.legend();plt.tight_layout()
p2=out/"v059_cpu_g1_native_hip_pair_nonfoot.png"
plt.savefig(p2,dpi=155);plt.close()
print("V059_PLOTS",p1,p2,"actual_native_point_count",n)
