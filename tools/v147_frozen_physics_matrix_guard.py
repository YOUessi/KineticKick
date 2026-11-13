#!/usr/bin/env python3
"""v147 immutable six-run original-G1 matrix readiness, NO physics launch.

Reports SHA/provenance, output reservations and GPU/disk safety. Does not
acquire the GPU, create output files, execute IsaacLab or modify USD stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v147_exact_off_on_parity import frozen, FROZEN_SHA
from v146_native_original_bridge import gpu_preflight
from v142_native_pair_ledger import safe_write_once

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/you/projects/kinetickick")
V54 = ROOT/"configs/cmu_named_nonfoot_diagnostics_v054.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(*, gpu_table=None, disk_free_bytes=None, root=ROOT):
    cfg=frozen()
    root=Path(root)
    old=json.loads((root/"configs/cmu_named_nonfoot_diagnostics_v054.json").read_text())
    base=old["original_three_center_policies"]
    cases=list(cfg["original_v54_expected_rows"])
    if list(base)!=cases:
        raise ValueError("V147_ORIGINAL_THREE_POLICY_ORDER_CHANGED")
    expected=cfg["expected_future_physx_matrix"]
    planned=[]
    for name in cases:
        src=root/base[name]["config"]
        if not src.is_file() or sha(src)!=old["immutable_sha256"][base[name]["config"]]:
            raise ValueError("V147_OLD_POLICY_CONFIG_SHA_MISMATCH:"+name)
        for mode in ("off","on"):
            rowid=name+"/"+mode
            if rowid!=expected[len(planned)]:
                raise ValueError("V147_FROZEN_SIX_CASE_MATRIX_CHANGED")
            out=root/"reports"/f"v147_{name}_{mode}_seed1201_physx.json"
            side=root/"reports"/f"v147_{name}_{mode}_seed1201_sidecar.json"
            if out.exists() or side.exists() or out.is_symlink() or side.is_symlink():
                raise FileExistsError("V147_RUN_OUTPUT_ALREADY_EXISTS_NO_OVERWRITE:"+rowid)
            cmd=[
                "/home/you/conda_env/env_isaaclab/bin/python",
                str(root/"tools/v146_run_original_with_readonly.py"),
                "--config",str(src), "--output",str(out),
                "--eval-seed","1201", "--contact-diagnostics",
                "--v146-observer-mode",mode,
                "--v146-sidecar",str(side),
            ]
            planned.append({
                "policy":name, "observer_mode":mode,
                "original_frozen_config_sha256":sha(src),
                "output_write_once":str(out),
                "sidecar_write_once":str(side),
                "command_argv_not_executed":cmd,
                "original_expected_full_state_limit":cfg["original_v54_expected_rows"][name],
                "strict_outcome_must_be_recomputed_from_raw_physx":True,
            })
    if len(planned)!=6:
        raise ValueError("V147_SIX_CASE_MATRIX_INCOMPLETE")
    asset_items=("data","experiments","third_party",".pretrained_checkpoints",
                 "media","configs/paid_original","configs/paid_g1_main.urdf")
    assets={rel:(root/rel).is_symlink() and
            (root/rel).resolve()==(MAIN/rel).resolve()
            for rel in asset_items}
    if not all(assets.values()):
        raise RuntimeError("V147_ORIGINAL_ASSETS_NOT_READONLY_BOUND")
    try:
        gpu_status=gpu_preflight(gpu_table=gpu_table,disk_free_bytes=disk_free_bytes)
        status="GPU_PREFLIGHT_ONLY_NOT_MUTEX_OR_BACKEND_CALIBRATION"
    except RuntimeError as exc:
        status="BLOCKED_"+str(exc)
        gpu_status=None
    return {
        "version":"v0.1.147","frozen_protocol_sha":FROZEN_SHA,
        "original_policy_configs_sha_verified":True,
        "read_only_original_asset_links":assets,
        "matrix_size":len(planned),
        "future_physics_experiments_not_started":planned,
        "gpu_go_no_go":status,
        "gpu_preflight_if_idle":gpu_status,
        "actual_new_gpu_physx_runs":0,
        "true_native_contact_pair_truth":"UNKNOWN",
        "native_positive_negative_clock_calibration":False,
        "off_on_real_physx_noninterference_certified":False,
        "auto_execute":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=build_manifest()
    digest=safe_write_once(args.output,result)
    print(json.dumps({
        "manifest":str(args.output),"sha256":digest,
        "cases":result["matrix_size"],"status":result["gpu_go_no_go"],
        "new_physx":result["actual_new_gpu_physx_runs"],
    },ensure_ascii=False))


if __name__=="__main__":
    main()
