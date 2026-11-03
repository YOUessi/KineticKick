"""Safely fetch G1 29DoF URDF + meshes from the LAFAN1 G1 dataset mirror.

Works on Tang where huggingface.co is blocked and hf-mirror.com is reachable.
Downloads ONLY public robot_description/g1 subtree, validates expected file sizes,
preserves relative mesh paths, and NEVER publishes or overwrites the 37DoF robot.
Source: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
"""
from __future__ import annotations
import concurrent.futures,hashlib,json,subprocess
from pathlib import Path
from urllib.parse import quote

BASE=Path("/home/you/projects/kinetickick")
TREE=BASE/"reports/dataset_discovery/g1_robot_description_tree.json"
OUTPUT=BASE/"data/assets/g1_29dof_lafan1"
HOST="https://hf-mirror.com/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/"
def download(item):
    rel=item["path"]
    if not rel.startswith("robot_description/g1/") or item.get("type")!="file":
        raise ValueError(rel)
    target=OUTPUT/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    size=int(item["size"])
    if target.exists() and target.stat().st_size==size:
        return rel,True,"already_verified"
    tmp=target.with_suffix(target.suffix+".part")
    url=HOST+quote(rel,safe="/")
    result=subprocess.run([
        "curl","-fLsS","--retry","2","--retry-delay","2",
        "--connect-timeout","8","--max-time","70",
        "--output",str(tmp),url],capture_output=True,text=True,timeout=76)
    if result.returncode!=0:
        return rel,False,("curl_exit_"+str(result.returncode)+" "+result.stderr[-180:])
    if tmp.stat().st_size!=size:
        return rel,False,f"size_mismatch expected={size} got={tmp.stat().st_size}"
    tmp.replace(target)
    return rel,True,"downloaded"

def main():
    tree=json.loads(TREE.read_text())
    wanted=[x for x in tree if x.get("type")=="file"
            and x["path"].startswith("robot_description/g1/")
            and (x["path"].lower().endswith(".stl") or x["path"].endswith(".urdf"))]
    print("G1_29DOF_ASSET_MANIFEST","count",len(wanted),
          "expected_MB",round(sum(int(x["size"]) for x in wanted)/1e6,2),flush=True)
    successes=[]
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,x) for x in wanted]
        for f in concurrent.futures.as_completed(futures):
            rel,ok,msg=f.result()
            if ok:successes.append(rel)
            else:failures.append((rel,msg))
            print("G1_29DOF_FILE","OK" if ok else "FAILED",rel,msg,flush=True)
    for tail in ["LICENSE","README.md"]:
        path=OUTPUT/tail
        result=subprocess.run(
            ["curl","-fLsS","--connect-timeout","8","--max-time","25",
             "-o",str(path),HOST+tail],timeout=30,capture_output=True)
        print("G1_29DOF_LICENSE_SOURCE",tail,result.returncode,flush=True)
    out={
        "source":HOST,"source_dataset":"lvhaidong/LAFAN1_Retargeting_Dataset",
        "robot_model":"Unitree G1 29DoF rev1_0",
        "files_expected":len(wanted),
        "files_verified":len(successes),
        "files_failed":failures,
        "local_root":str(OUTPUT),
        "not_suitable_to_directly_swap_with_isaaclab_37dof":True,
        "urdf":str(OUTPUT/"robot_description/g1/g1_29dof_rev_1_0.urdf")
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"asset_fetch_manifest.json").write_text(json.dumps(out,indent=2))
    print("G1_29DOF_ASSET_FETCH_DONE",json.dumps(out),flush=True)

if __name__=="__main__":main()
