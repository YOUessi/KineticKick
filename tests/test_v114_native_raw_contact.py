"""Never promote one body or zero impulse into native collision truth."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from v114_native_raw_contact_probe import raw_native_pair,unpack_xyz
def e():
 return dict(body0='/World/ProbeBall',body1='/World/ProbeBox',position=[0.,0.,.2],
             normal=[0.,0.,1.],impulse=[0.,0.,.5])
def test_true_named_body_pair_requires_contact_vector():
 assert raw_native_pair(e())
def test_self_identity_not_a_pair():
 d=e();d['body1']='/World/ProbeBall'
 assert not raw_native_pair(d)
def test_near_zero_impulse_rejected():
 d=e();d['impulse']=[0.,0.,0.]
 assert not raw_native_pair(d)
def test_missing_normal_not_certified():
 d=e();d['normal']=None
 assert not raw_native_pair(d)
def test_only_body_pairs_not_full_collider_manifold():
 assert 'native_actor_collider_shape_pair_certified=False' in (ROOT/'tools/v114_native_raw_contact_probe.py').read_text()
