"""Adversarial contact-graph falsification: symbolic, not actual PhysX contact."""
import copy
from collections import Counter
import math
import pytest
from v153_contact_graph_identifiability import (
    WRIST, HIP, graph_edges, create_case, matrix, net_forces, vector,
    directions, magnitude, frozen,
)
from v153_independent_graph_audit import audit_matrix

ORIGINAL=[21.220844268798828,14.821828842163086,.5377691388130188]
TOPOLOGIES=("direct","two_to_third","two_to_world","split_paths")

@pytest.mark.parametrize("i",range(24))
@pytest.mark.parametrize("scale",(.25,1.,2.))
@pytest.mark.parametrize("topology",TOPOLOGIES)
def test_full_frozen_adversarial_matrix(i,scale,topology):
    d=directions(ORIGINAL)[i]
    f=[v*magnitude(ORIGINAL)*scale for v in d]
    edges=graph_edges(topology,f)
    case=create_case(topology,f)
    net=net_forces(edges)
    assert len(case["net_wrist_N"])==3
    assert all(abs(a-b)<1e-10 for a,b in zip(net[WRIST],f))
    assert all(abs(a+b)<1e-10 for a,b in zip(net[HIP],f))
    assert case["has_wrist_hip_direct_edge"] is (topology=="direct")
    assert case["all_contacts_are_counterfactual_not_runtime_ones"]
    assert max(abs(v) for v in net.get("unspecified_third_body",[0.,0.,0.]))<1e-10

def test_full_matrix_and_independent_incidence():
    m=matrix(ORIGINAL)
    assert len(m["rows"])==288
    assert all(v==72 for v in Counter(row["hypothesis"] for row in m["rows"]).values())
    source={"observed_wrist_net_force_N":ORIGINAL}
    result=audit_matrix(m,source)
    assert result["direction_scale_groups_with_4_indistinguishable_nets"]==72
    assert result["true_native_wrist_hip_collider_pair"]=="UNKNOWN"

@pytest.mark.parametrize("bad",(
    None,[],[1.,2.],[1.,2.,3.,4.],[True,1.,2.],
    [float("nan"),1.,2.],[float("inf"),1.,2.],"1,2,3"
))
def test_bad_vector_rejected(bad):
    with pytest.raises(ValueError):
        vector(bad)

def test_zero_force_rejected():
    with pytest.raises(ValueError,match="ZERO"):
        create_case("direct",[0,0,0])

def test_unknown_contact_topology_rejected():
    with pytest.raises(ValueError):
        create_case("invented",[1,0,0])

def test_direct_vs_nondirect_same_net_not_same_truth():
    f=ORIGINAL
    direct=create_case("direct",f)
    other=create_case("two_to_third",f)
    assert direct["net_wrist_N"]==other["net_wrist_N"]
    assert direct["net_hip_N"]==other["net_hip_N"]
    assert direct["has_wrist_hip_direct_edge"]
    assert not other["has_wrist_hip_direct_edge"]

def test_incidence_detects_false_native_certification():
    m=matrix(ORIGINAL)
    m["all_graphs_are_physically_certified"]=True
    with pytest.raises(ValueError,match="FALSE_RUNTIME"):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_incidence_detects_false_edge_claim():
    m=matrix(ORIGINAL)
    m["rows"][0]["has_wrist_hip_direct_edge"]=False
    with pytest.raises(ValueError,match="FALSE_DIRECT_EDGE"):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_incidence_detects_topology_spoof():
    m=matrix(ORIGINAL)
    m["rows"][1]["symbolic_edges_not_physx_manifold"][0]["b"]=HIP
    with pytest.raises(ValueError,match="WRONG_CONTACT_GRAPH"):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_incidence_detects_force_tamper():
    m=matrix(ORIGINAL)
    m["rows"][0]["symbolic_edges_not_physx_manifold"][0]["force_on_a_N"][2]+=.2
    with pytest.raises(ValueError,match="FORCE_MISMATCH"):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_incidence_detects_fake_observation():
    m=matrix(ORIGINAL)
    m["rows"][0]["observation_wrist_N"][1]+=.1
    with pytest.raises(ValueError):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_incidence_detects_duplicate():
    m=matrix(ORIGINAL)
    m["rows"][1]["hypothesis"]=m["rows"][0]["hypothesis"]
    with pytest.raises(ValueError):
        audit_matrix(m,{"observed_wrist_net_force_N":ORIGINAL})

def test_duplicate_self_edge_not_allowed():
    with pytest.raises(ValueError,match="SELF_EDGE"):
        net_forces([{"a":WRIST,"b":WRIST,"force_on_a_N":[1.,0.,0.]}])

def test_frozen_protocol():
    assert frozen()["matrix"]["total"]==288
