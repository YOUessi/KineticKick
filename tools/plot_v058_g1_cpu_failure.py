#!/usr/bin/env python3
"""Render measured 200Hz support, nonfoot and actuator-limit evidence; CPU G1 only."""
import json
from pathlib import Path
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/"reports/v058_v058_cpu_frozen_v32.json").read_text())
assert data["v057_requested_physics_backend"]=="cpu"
frames=data["traces"][0]
t=[x["time_s"] for x in frames]
contact=data["records"][0]["contact_time_s"]
end=t[-1]
if data["stride_shot_success"] is True:raise RuntimeError("Expected actual failure evidence")
out=ROOT/"media";out.mkdir(exist_ok=True)
plt.figure(figsize=(10,4.8))
for foot in ("left","right"):
    plt.plot(t,[x["support_foot_force_N"][foot] for x in frames],label=foot+" support normal (N)",lw=1.4)
plt.plot(t,[x["nonfoot_contact_N"] for x in frames],label="nonfoot net sensor (N)",lw=1.3)
plt.axvline(contact,linestyle="--",label="ball contact",linewidth=1)
plt.xlabel("200 Hz CPU G1 simulated time (s)")
plt.ylabel("Observed force (N); body net force is not pair attribution")
plt.title(f"v058 CPU original29DoF | early nonfoot stop at {end:.3f}s")
plt.legend(loc="upper left");plt.tight_layout()
a=out/"v058_cpu_g1_support_nonfoot.png"
plt.savefig(a,dpi=155);plt.close()
limits=data["joint_velocity_limits"][0]
ratio=[max(abs(v)/lim for v,lim in zip(x["joint_vel"],limits)) for x in frames]
plt.figure(figsize=(10,4))
plt.plot(t,ratio,label="max original 29 joint |qdot| / original limit",lw=1.5)
plt.axhline(1,linestyle="--",label="original joint limit",linewidth=1)
plt.axvline(contact,linestyle=":",label="ball contact")
plt.xlabel("200 Hz CPU G1 simulated time (s)")
plt.ylabel("Maximum normalized measured joint speed")
plt.title("v058 CPU G1: all original 29DoF qdot; censored at physical stop")
plt.legend();plt.tight_layout()
b=out/"v058_cpu_g1_original_qdot_ratio.png"
plt.savefig(b,dpi=155);plt.close()
print("VISUAL_EVIDENCE",a,b,"frames",len(t),"maximum_qdot_ratio",max(ratio))
