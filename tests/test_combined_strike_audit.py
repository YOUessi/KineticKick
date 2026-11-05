"""Real old combined failure remains auditable without a per-case v23 run."""
from pathlib import Path
import hashlib,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_placement_strike import verify
ROOT=Path(__file__).resolve().parents[1]


def test_declared_combination_uses_original_initial_state_and_requested_delta():
    reference=json.loads((ROOT/'reports/v023_baseline_center.json').read_text())
    result=json.loads((ROOT/'reports/v026_final_combined_left_slower.json').read_text())
    config=json.loads((ROOT/'configs/state_conditioned_strike_v1.json').read_text())
    digest=hashlib.sha256((ROOT/'configs/state_conditioned_strike_v1.json').read_bytes()).hexdigest()
    case=dict(id='combined_left_slower',ball_offset_xy_m=[-.025,0.],root_velocity_y_delta_mps=.10)
    velocity=reference['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    sources=dict(result['joint_strike']['source_sha256'])
    verify(result,reference,case,config,digest,sources,velocity)
    result['joint_strike']['initial_root_velocity'][1]+=.01
    with pytest.raises(ValueError,match='initial velocity'):
        verify(result,reference,case,config,digest,sources,velocity)
