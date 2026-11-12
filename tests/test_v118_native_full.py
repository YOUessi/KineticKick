from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v118_native_full_probe import native_positive

def test_native_complete_identity_and_geometry():
    item=dict(actors=["/World/ProbeBall","/World/ProbeBox"],
              colliders=["/World/ProbeBall/mesh","/World/ProbeBox/mesh"],
              points=[dict(position=[0.,0.,0.2],normal=[0.,0.,1.],
                           impulse=[0.,0.,2.])])
    assert native_positive(item)
    assert not native_positive({**item,"points":[]})
    assert not native_positive({**item,"colliders":["/World/ProbeBall/mesh","/World/Floor"]})
    assert not native_positive({**item,"points":[dict(position=[0.,0.,0.2],
                                      normal=[0.,0.,1.],impulse=[0.,0.,0.])]})

def test_native_matrix_and_source_are_frozen():
    from v118_native_full_batch import CASES,PROBE,FROZEN
    assert CASES==(("positive_off","off","pose","colliding"),
                  ("positive_on","on","both","colliding"),
                  ("separated_on","on","both","separated"))
    assert PROBE.name=="v118_native_full_probe.py"
    assert "tests/test_v118_native_full.py" in FROZEN
