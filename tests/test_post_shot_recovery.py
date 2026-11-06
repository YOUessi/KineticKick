"""CPU checks of the residual boundary; the frozen expert is a counted stub."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


@pytest.fixture
def recovery(monkeypatch):
    module = importlib.import_module('post_shot_recovery')
    names = json.loads((ROOT / 'configs/state_conditioned_strike_v6.json').read_text())['source_metadata']['joint_names']

    def initialize(self, env, config):
        self.env = env; self.cfg = config; self.n = env.num_envs; self.device = 'cpu'
        n = self.n; self.elapsed = torch.zeros(n); self.frame = torch.full((n,), 100.)
        self.step_velocity = torch.zeros(n, 29); self.contact = torch.ones(n, dtype=torch.bool)
        self.expert = SimpleNamespace(start=torch.full((n,), -1.), plan={
            'strike': {'window': [65, 75, 82, 91]},
            'strike_velocity': {'window': [74, 77, 78, 79]},
            'stride': {'window_s': [1.42, 1.52, 1.64, 1.78]}})
        self.landing = SimpleNamespace(separated_s=torch.full((n,), .015))
        self.feet = [0, 1]; self.shank_geometry = None; self.debug = []
        quat = torch.zeros(n, 3, 4); quat[..., 0] = 1.
        data = SimpleNamespace(joint_pos=torch.zeros(n, 29), joint_vel=torch.zeros(n, 29),
            default_joint_pos=torch.zeros(n, 29), joint_vel_limits=torch.full((n, 29), 20.),
            soft_joint_pos_limits=torch.tensor([-1., 1.]).expand(n, 29, 2),
            root_pos_w=torch.tensor([0., 0., .8]).repeat(n, 1), root_quat_w=quat[:, 0],
            projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(n, 1),
            root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
            body_pos_w=torch.zeros(n, 3, 3), body_quat_w=quat,
            body_link_lin_vel_w=torch.zeros(n, 3, 3), body_ang_vel_w=torch.zeros(n, 3, 3))
        self.robot = SimpleNamespace(data=data, joint_names=names)
        self.fake_base = torch.linspace(-2., 2., 29).repeat(n, 1)
        self.base_calls = 0

    def commands(self, actions=None):
        assert actions is None
        self.base_calls += 1; self.elapsed += 1
        return self.fake_base

    def reset(self, ids):
        self.elapsed[ids] = 0.; self.frame[ids] = 0.
        self.expert.start[ids] = -1.; self.contact[ids] = False

    monkeypatch.setattr(module.PlacementStrikeTask, '__init__', initialize)
    monkeypatch.setattr(module.PlacementStrikeTask, 'commands', commands)
    monkeypatch.setattr(module.PlacementStrikeTask, 'reset_fields', reset)
    monkeypatch.setattr(module.PlacementStrikeTask, 'after_physics', lambda self, dt: None)
    import running_balance
    monkeypatch.setattr(running_balance, 'ground_force', lambda env: env.support)

    def make(n=1, extra=None, before_termination_manager=False):
        sensor = SimpleNamespace(body_names=['left_ankle_roll_link', 'right_ankle_roll_link', 'pelvis'],
            data=SimpleNamespace(net_forces_w=torch.zeros(n, 3, 3)))
        class Scene(dict):
            env_origins = torch.zeros(n, 3)
        scene = Scene(contact_forces=sensor)
        env = SimpleNamespace(num_envs=n, step_dt=.02, physics_dt=.005, scene=scene,
            support=torch.zeros(n, 2), termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
            action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(_scale=.5, _offset=0.)))
        config = {'post_shot_recovery': dict(module.DEFAULT_RECOVERY_CONFIG)}
        config['post_shot_recovery'].update(extra or {})
        if before_termination_manager:
            # ObservationManager invokes task_features while constructing its
            # terms; ManagerBasedRLEnv creates TerminationManager afterwards.
            del env.termination_manager
            env.cfg = SimpleNamespace(joint_strike=config)
            return env
        return module.PostShotRecoveryTask(env, config)
    return module, make


def test_protected_prefix_and_each_activation_gate(recovery):
    _, make = recovery; task = make(6)
    task.contact[0] = False; task.landing.separated_s[1] = .01
    task.frame[2] = 89.; task.step_velocity[3, 0] = .01; task.expert.start[4] = 0.
    base = task.fake_base.clone()
    out = task.commands(torch.full((6, 29), 100.))
    assert torch.equal(out, base)  # Trigger sample also has exactly zero delta.
    assert task.activation_time.tolist() == [-1., -1., -1., -1., -1., 0.]
    assert task.base_calls == 1
    task.commands(torch.full((6, 29), 100.))
    assert torch.equal(task.actual_delta[:5], torch.zeros(5, 29))
    assert bool((task.actual_delta[5] > 0.).all())


def test_zero_probe_is_exact_and_no_new_soft_target_projection(recovery):
    _, make = recovery; task = make()
    for _ in range(10):
        assert torch.equal(task.commands(None), task.fake_base)
    for _ in range(8): task.commands(torch.full((1, 29), 100.))
    assert task.actual_delta.abs().max() <= .120001
    # The original target is already beyond its soft interval. Outward bounded
    # residuals are allowed; neither baseline nor final target is projected.
    assert task.last_output[0, -1] > task.fake_base[0, -1]
    assert task.robot.data.soft_joint_pos_limits[0, -1, 1] == 1.


def test_rate_bound_fade_to_exact_zero_and_local_reset(recovery):
    _, make = recovery; task = make(2)
    previous = torch.zeros(2, 29)
    for i in range(24):
        raw = torch.full((2, 29), 100. if i < 15 else -100.)
        task.commands(raw)
        assert (task.actual_delta - previous).abs().max() <= .020001
        assert task.actual_delta.abs().max() <= .120001
        previous = task.actual_delta.clone()
    task.expert.start[:] = task.elapsed
    for _ in range(12):
        task.commands(torch.randn(2, 29) * 100.)
        assert (task.actual_delta - previous).abs().max() <= .020001
        previous = task.actual_delta.clone()
    assert torch.equal(task.actual_delta, torch.zeros(2, 29))
    assert torch.equal(task.last_output, task.fake_base)
    other_time = task.activation_time[1].clone()
    task.reset_fields(torch.tensor([0]))
    assert task.activation_time[0] == -1 and task.activation_time[1] == other_time
    assert torch.equal(task.recent_baseline_action[0], torch.zeros(29))


def test_feature_contract_uses_link_origin_and_lowest_surface_velocity(recovery):
    module, make = recovery; task = make()
    task._foot_capsules = [(torch.tensor([[.1, 0., 0.], [.1, 0., .1]]), torch.tensor([.05, .05]))] * 2
    task.robot.data.body_pos_w[:, :2, 2] = .3
    task.robot.data.body_ang_vel_w[:, :2, 1] = 2.
    feature = task.recovery_features(); state = task.recovery_state()
    assert feature.shape == (1, module.FEATURE_WIDTH) == (1, 154)
    assert sum(width for _, width in module.FEATURE_LAYOUT) == 154
    torch.testing.assert_close(state['capsule_height_m'], torch.tensor([[.25, .25]]))
    torch.testing.assert_close(state['capsule_lowest_point_velocity_w'], torch.tensor([[[-.1, 0., -.2], [-.1, 0., -.2]]]))
    assert torch.equal(state['foot_link_velocity_w'], torch.zeros(1, 2, 3))


def test_five_ms_peaks_are_consumed_once_and_terminal_cost_is_once(recovery, monkeypatch):
    _, make = recovery; task = make(); task.commands(torch.zeros(1, 29))
    def objective(state):
        rate = -state['substep_qd_ratio_peak']
        terminal = -100. * state['terminated'].float()
        return rate + terminal, {'quality': rate, 'terminal_cost': terminal}
    monkeypatch.setattr(task, '_objective', objective)
    for ratio, force in [(0.5, 0.), (2., 30.), (.5, 0.), (.5, 0.)]:
        task.robot.data.joint_vel[:] = ratio * 20.
        task.env.scene['contact_forces'].data.net_forces_w[:, 2, 2] = force
        task.after_physics(.005)
    state = task.recovery_state()
    assert state['substep_qd_ratio_peak'].item() == 2.
    assert state['substep_nonfoot_force_peak'].item() == 30.
    task.env.termination_manager.terminated[:] = True
    torch.testing.assert_close(task.reward(), torch.tensor([-5000.875]))
    assert task.last_recovery_reward_components['terminal_rate'].item() == -5000.
    assert task.recovery_state()['substep_qd_ratio_peak'].item() == 0.
    assert task.reward().item() == 0.  # Repeated consumption cannot pay twice.


def test_invalid_actions_and_checkpoint_digest_fail_explicitly(recovery, tmp_path):
    module, make = recovery; task = make()
    with pytest.raises(ValueError): task.commands(torch.zeros(1, 30))
    with pytest.raises(ValueError): task.commands(torch.full((1, 29), float('nan')))
    checkpoint = tmp_path / 'actor.pt'; checkpoint.write_bytes(b'not a trusted checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        make(extra={'checkpoint': str(checkpoint), 'checkpoint_sha256': '0' * 64})
    actor = module.PostShotRecoveryActor(module.FEATURE_WIDTH, module.FEATURE_WIDTH, 29)
    assert torch.equal(actor.act_inference(torch.randn(2, module.FEATURE_WIDTH)), torch.zeros(2, 29))
    with pytest.raises(ValueError): module.PostShotRecoveryActor(module.FEATURE_WIDTH, module.FEATURE_WIDTH, 30)


def test_internal_inference_uses_same_precommand_features_as_training(recovery):
    module, make = recovery; evaluated = make(); trained = make()
    seen = []
    def infer(features):
        seen.append(features.clone())
        return features[:, -29:] + .4
    evaluated.recovery_actor = SimpleNamespace(act_inference=infer)
    for _ in range(4):
        before = trained.recovery_features().clone()
        expected = trained.commands(before[:, -29:] + .4)
        actual = evaluated.commands(None)
        assert torch.equal(seen[-1], before)
        assert torch.equal(actual, expected)
    assert torch.equal(evaluated.recent_baseline_action, evaluated.fake_base)


def test_real_objective_remains_active_after_residual_fade(recovery):
    _, make = recovery; task = make(); task.commands(torch.zeros(1, 29))
    task.fade_start_time[:] = 0.; task._recovery_time_s[:] = .22
    state = task.recovery_state()
    assert state['post_active'].item() and not state['residual_active'].item()
    task.env.support[:] = 100.
    for _ in range(4): task.after_physics(.005)
    result = task.reward()
    assert result.shape == (1,) and torch.isfinite(result).all()
    assert task.last_recovery_reward_components['sampled_duration_s'].item() == pytest.approx(.02)


def test_zero_dt_forwarding_is_not_a_quality_sample(recovery, monkeypatch):
    module, make = recovery; task = make(); forwarded = []
    monkeypatch.setattr(module.PlacementStrikeTask, 'after_physics', lambda self, dt: forwarded.append(dt))
    task.after_physics(0.)
    assert forwarded == [0.]
    assert task._quality_time_s.item() == task._recovery_time_s.item() == 0.
    assert task._physics_steps.item() == 0
    for invalid in (-.005, .02, float('nan')):
        with pytest.raises(ValueError): task.after_physics(invalid)


def test_initial_observation_constructs_task_before_termination_manager(recovery):
    module, make = recovery
    env = make(2, before_termination_manager=True)
    assert not hasattr(env, '_post_shot_recovery_task')
    # Exercise the actual lazy constructor entry point used to determine the
    # observation shape, without inventing an already initialized done manager.
    features = module.task_features(env)
    task = module.get_task(env)
    assert features.shape == (2, module.FEATURE_WIDTH) and torch.isfinite(features).all()
    assert torch.equal(task.recovery_state()['terminated'], torch.zeros(2, dtype=torch.bool))
    assert not hasattr(env, 'termination_manager')  # Read-only fallback.
    env.termination_manager = SimpleNamespace(terminated=torch.tensor([True, False]))
    assert task.recovery_state()['terminated'] is env.termination_manager.terminated
    env.termination_manager.terminated[:] = torch.tensor([False, True])
    assert task.recovery_state()['terminated'].tolist() == [False, True]
