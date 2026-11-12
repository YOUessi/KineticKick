#!/usr/bin/env python3
"""Plot exact sampled v54 original G1 event windows; no interpolated 5.145s q."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v128_v54_event_geometry as old
import v133_native_contact_frame as contract


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-prefix",type=Path,default=
                    contract.ROOT/"reports/v133_real_G1_200hz")
    args=ap.parse_args()
    frozen=old.frozen_protocol()
    cfg=contract.frozen()
    plots={"qdot":[],"nonfoot":[],"support":[]}
    for case in cfg["original_v54_cases"]:
        filename="v054_"+case+"_contact_diagnostics.json"
        path=Path(frozen["data_root"])/filename
        old.require_sha(path,frozen["input_sha256"][filename])
        raw=json.loads(path.read_text())
        rows=cfg["original_v54_cases"][case]+(case=="cmu_10_05")
        x=old.validate_rows(raw,rows,.005)
        choose=(x["t"]>=5.065)&(x["t"]<=5.14+1e-7)
        times=x["t"][choose]
        ratio=np.max(np.abs(x["dq"][choose])/x["limits"][None,:],axis=1)
        plots["qdot"].append((case,times,ratio))
        plots["nonfoot"].append((case,times,x["nonfoot"][choose]))
        plots["support"].append((case,times,x["support"][choose,0]))
    outputs=[]
    for key,title,ylab,threshold in (
        ("qdot","Original G1 all-29 qdot speed-limit ratio",
         "max(abs(qdot_j)/original_limit_j)",1.),
        ("nonfoot","Named-body net-force proxy, NOT a collider pair",
         "nonfoot force (N)",20.),
        ("support","Original left-foot support-force proxy",
         "left foot reported net-force proxy (N)",None)):
        out=Path(str(args.output_prefix)+"_"+key+".png")
        if out.exists():
            raise FileExistsError(out)
        fig,ax=plt.subplots(figsize=(8,4))
        for case,t,y in plots[key]:
            ax.plot(t,y,marker=".",markersize=3,label=case)
        if threshold is not None:
            ax.axhline(threshold,color="black",linestyle=":",linewidth=.8,
                       label="frozen gate, not runtime contact truth")
        ax.axvline(cfg["original_named_event_s"],color="black",
                   linewidth=.8,linestyle="--",label="named sensor stamp 5.145s")
        ax.set_title(title)
        ax.set_xlabel("original physical sample time (s); no extrapolated q at 5.145s")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out,dpi=135)
        plt.close(fig)
        outputs.append(str(out))
    print("\n".join(outputs))


if __name__=="__main__":
    main()
