"""Read-only human-robot event-phase qualification: reject synthetic truths."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v092_motion_qualification",ROOT/"tools/v092_motion_qualification.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_speed_peak_returns_real_index():
 import numpy as np
 t=np.arange(100)/100
 x=np.zeros(100);x[49]=2.
 r=m.speed_peak(x,t,.5)
 assert r["relative_to_candidate_s"]==-.01
def test_speed_peak_rejects_missing():
 import numpy as np
 assert m.speed_peak(np.full(100,float("nan")),np.arange(100)/100,.5) is None
def test_invalid_no_foot_impulse_from_c3d():
 import numpy as np
 t=np.arange(100)/100
 r=m.speed_peak(np.ones(100),t,.5)
 assert "speed_proxy_mps" in r and "measured_impulse" not in r
def test_human_original_6_signed_hashes():
 import json
 p=Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050/raw")
 out=ROOT/"reports/v092_human_robot_geom_retry1.audit.json"
 if not out.exists():return
 report=json.loads(out.read_text())
 assert report["qualified_human_professional_impact_examples"]==0
 assert len(report["human"])==6
 assert all((p/z["file"]).exists() and m.sha(p/z["file"])==z["sha256"] for z in report["human"])
 assert report["human_source_clip_count"]==6
 assert report["original_G1_hard_gate_success"]==0
 assert report["original_G1_truncated_full_window"]==2
def test_subject_disjoint_is_not_two_groups_with_5_1_suitable_for_generalization():
 import json
 out=ROOT/"reports/v092_human_robot_geom_retry1.audit.json"
 if not out.exists():return
 a=json.loads(out.read_text())
 assert a["source_subject_group_counts"]=={"10":5,"11":1}
 assert not a["comparative_naturalness_quantitative_valid"]
