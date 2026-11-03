#!/usr/bin/env python3
"""Prepare PAiD's exact upstream URDF, changing only package:// mesh URI strings."""
from pathlib import Path
import hashlib
import json
import re

root=Path(__file__).resolve().parents[1]
source=root/"third_party/HumanoidSoccer/source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf"
mesh_root=root/"third_party/HumanoidSoccer/source/whole_body_tracking/soccer/assets/unitree_description/meshes/g1"
out=root/"configs/paid_original/paid_main_resolved.urdf"
assert source.exists()
text=source.read_text()
old=r"package://unitree_description/meshes/g1/([^\"<]+)"
found=re.findall(old,text)
assert found and all((mesh_root/name).exists() for name in found), "Original STL assets missing"
modified=re.sub(old,lambda m: str(mesh_root/m.group(1)),text)
assert re.sub(re.escape(str(mesh_root))+r"/([^\"<]+)",lambda m: "package://unitree_description/meshes/g1/"+m.group(1),modified)==text
out.write_text(modified)
manifest={
 "source":str(source),
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "modified":str(out),
 "modified_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
 "change":"Only replace package://unitree_description/meshes/g1/filename with absolute mesh path",
 "mesh_reference_count":len(found), "unique_mesh_count":len(set(found)),
 "geometry_dynamics_joint_data_unchanged":True
}
(root/"reports/paid_original_runtime_urdf_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
