#!/usr/bin/env python3
"""Fail-closed retrospective audit of frozen v057 backend-qualification attempts.

The source-hash-locked v057 experiment must NEVER be edited or retroactively
rescored as physical success. An exit code of zero without recorded G1 JSON
does not mean simulation completed. Toy contact probes are not G1 evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RE_DEVICE = re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(protocol: dict, progress: dict, root: Path) -> dict:
    cases = protocol["cases"] + protocol["toy_cases"]
    if len(cases) != 10 or len({c["id"] for c in cases}) != 10:
        raise ValueError("Expected ten unique preregistered cases")
    if len(progress["runs"]) != 10 or [r["id"] for r in progress["runs"]] != [c["id"] for c in cases]:
        raise ValueError("Incomplete/unordered protocol; cannot issue final findings")
    result = []
    for case, row in zip(cases, progress["runs"]):
        dest = root / case["output"]
        log = dest.with_suffix(".log")
        if not log.exists() or sha(log) != row["log_sha256"]:
            raise ValueError("Missing or drifted original log: " + case["id"])
        if dest.exists() != bool(row.get("report_sha256")):
            raise ValueError("Report/progress disagreement: " + case["id"])
        if dest.exists() and sha(dest) != row["report_sha256"]:
            raise ValueError("Drifted original report: " + case["id"])
        devices = RE_DEVICE.findall(log.read_text(errors="replace"))
        observed = devices[0] if devices else None
        is_g1 = case in protocol["cases"]
        verified = is_g1 and dest.exists() and row.get("status") == "PHYSICAL_VALIDATED" and observed == case["backend"] and row["exit_code"] == 0
        issue = []
        if observed != case["backend"]:
            issue.append("backend_not_proven" if observed is None else "backend_mismatch")
        if is_g1 and not dest.exists():
            issue.append("no_G1_200Hz_output_even_if_exit_zero")
        if row["exit_code"] != 0:
            issue.append("abnormal_exit_or_shutdown_timeout")
        if not is_g1:
            issue.append("isolated_two_box_toy_not_G1")
        result.append(dict(id=case["id"],kind="G1" if is_g1 else "toy",
                           planned_device=case["backend"],observed_device=observed,
                           child_exit_code=row["exit_code"],raw_report=dest.exists(),
                           verified_g1=bool(verified),issues=issue,
                           native_contact_events=row.get("contact_events"),filtered_force_peak_N=row.get("peak_N")))
    return dict(version="v0.1.57",original_protocol_sha256=progress["protocol_sha256"],
                registered_G1=sum(x["kind"]=="G1" for x in result),
                validated_G1=sum(x["verified_g1"] for x in result),
                verified_strict_successes=sum(x["verified_g1"] and next(r for r in progress["runs"] if r["id"]==x["id"]).get("strict_success") is True for x in result),
                success_rate="NOT_ESTIMABLE_WITH_ZERO_VALID_G1",
                registered_toy=sum(x["kind"]=="toy" for x in result),
                cpu_G1_backend_mismatch=sum(x["kind"]=="G1" and x["planned_device"]=="cpu" and x["observed_device"]=="cuda:0" for x in result),
                toy_cpu_contact_certified=any(x["id"]=="toy_cpu_collide" and x["native_contact_events"] and x["native_contact_events"]>0 for x in result),
                g1_wrist_hip_pair_certified=False,details=result)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    root=a.root.resolve()
    protocol_path=root/"configs/physx_g1_backend_pair_v057.json"
    lock=json.loads((root/"reports/v057_protocol_lock.json").read_text())
    if sha(protocol_path)!=lock["protocol_sha256"]:
        raise RuntimeError("Protocol drift")
    protocol=json.loads(protocol_path.read_text())
    progress=json.loads((root/"reports/v057_batch_progress.json").read_text())
    if progress["protocol_sha256"]!=lock["protocol_sha256"]:
        raise RuntimeError("Progress drift")
    report=audit(protocol,progress,root)
    with a.output.open("x") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))

if __name__=="__main__":main()
