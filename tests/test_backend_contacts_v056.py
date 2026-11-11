"""Fail-closed CPU/GPU PhysX reporter backend qualification, no Isaac imports."""
from copy import deepcopy
import pytest

from tools.qualify_backend_contacts_v056 import (
    identify_probe, identify_g1, qualify,
)


def toy(collision=True, native=True):
    point = {"position": "(0,0,0)", "normal": "(0,0,1)",
             "impulse": "(0,0,0.5)"}
    return {
        "schema": "v056 isolation and signed negative", "not_G1_evidence": True,
        "sim_dt_s": .005, "sensor_body_count": 1, "sensor_filter_count": 1,
        "errors": [], "contact_tensor_errors": [],
        "filtered_pair_peak_N": 650. if collision else 0.,
        "callbacks": 3 if collision and native else 0,
        "polled_headers": 3 if collision and native else 0,
        "raw_events": [{
            "paths": ["/World/Probe/A", "/World/Probe/B",
                      "/World/Probe/A/geometry/mesh", "/World/Probe/B/geometry/mesh"],
            "count": 1, "contacts": [point]
        }] if collision and native else [],
        "total_probe_events": 3 if collision and native else 0,
        "native_pair_proven": bool(collision and native),
    }


def native_g1(calls=0):
    return {
        "root_fixed": False, "joint_names": list(range(29)),
        "physx_native_report_v056": {
            "errors": [], "dropped_events": 0,
            "callback_calls": calls, "native_headers": calls,
            "wrist_hip_direct_path_events": 0, "rigid_pair_certified": False,
            "report_api_present_on_original_bodies": {
                "/World/envs/env_0/Robot/left_wrist_yaw_link": True,
                "/World/envs/env_0/Robot/left_hip_roll_link": True},
        },
    }


def example():
    probes = {
        "gpu_collide": identify_probe(toy(True, False)),
        "gpu_separated": identify_probe(toy(False, False)),
        "cpu_collide": identify_probe(toy(True, True)),
        "cpu_separated": identify_probe(toy(False, True)),
    }
    g1 = {key: identify_g1(native_g1()) for key in (
        "frozen_v32", "cmu_10_02", "cmu_10_05")}
    return probes, g1


def test_positive_toy_native_actor_collider_point_proven_only_on_cpu():
    yes = identify_probe(toy(True, True))
    no = identify_probe(toy(True, False))
    assert yes["actor_and_collider_paths_plus_manifold_present"]
    assert yes["native_pair_proven_in_toy_only"]
    assert no["filtered_pair_peak_N"] > 20
    assert not no["native_pair_proven_in_toy_only"]


def test_malformed_toy_actor_or_collider_paths_never_prove_truth():
    wrong = toy(True, True)
    wrong["raw_events"][0]["paths"][3] = "/World/Probe/Unrelated/shape"
    assert not identify_probe(wrong)["native_pair_proven_in_toy_only"]


def test_missing_200hz_or_sensor_instrumentation_rejected():
    wrong = toy()
    wrong["sim_dt_s"] = .01
    with pytest.raises(ValueError, match="200Hz"):
        identify_probe(wrong)


def test_g1_zero_native_calls_is_unknown_even_with_contact_report_schema():
    row = identify_g1(native_g1())
    assert row["report_api_present_on_observed_links"]
    assert not row["contact_reporting_operational_observed"]
    assert not row["pair_manifold_proven"]
    assert row["absence_of_events_proves_no_self_collision"] is False


def test_positive_toy_does_not_certify_gpu_original_g1():
    probes, g1 = example()
    verdict = qualify(probes, g1)
    assert verdict["gpu_reporter_false_negative_for_verified_toy_contact"]
    assert verdict["cpu_toy_pair_paths_manifold_positive"]
    assert verdict["cpu_negative_control_clear"]
    assert not verdict["original_gpu_G1_pair_proven"]
    assert verdict["G1_collision_pair_inference"] == "UNDETERMINED_NOT_A_NEGATIVE"


def test_negative_control_with_real_filtered_contact_rejected():
    probes, g1 = example()
    probes["cpu_separated"]["filtered_pair_peak_N"] = 30.
    with pytest.raises(ValueError, match="Negative control"):
        qualify(probes, g1)


def test_missing_policy_or_probe_is_not_silent_success():
    probes, g1 = example()
    del g1["cmu_10_05"]
    with pytest.raises(ValueError, match="all 3"):
        qualify(probes, g1)
