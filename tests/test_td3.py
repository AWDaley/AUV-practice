"""Numerical and optimizer checks for the independently implemented TD3 learner."""

import math

import numpy as np
import pytest
import torch

from auv_sim.td3 import ReplayBuffer, TD3, TD3Config


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_buffer(seed=19, count=64):
    buffer = ReplayBuffer(3, 2, capacity=128, seed=seed)
    rng = np.random.default_rng(31)
    for index in range(count):
        obs = rng.normal(size=3).astype(np.float32)
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        next_obs = obs + rng.normal(scale=0.1, size=3)
        reward = -float(np.square(obs).sum() + 0.2 * np.square(action).sum())
        buffer.add(obs, action, reward, next_obs, index % 11 == 0)
    return buffer


def parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(before, module):
    return any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


def constant_critics(module, first, second):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.q1_net[-1].bias.fill_(first)
        module.q2_net[-1].bias.fill_(second)


def test_replay_wrap_shapes_and_true_termination_masks():
    buffer = ReplayBuffer(3, 2, capacity=2, seed=3)
    for index, terminal in ((1, False), (2, True), (3, False)):
        buffer.add([index] * 3, [0.25, -0.25], float(index), [index + 1] * 3, terminal)
    assert len(buffer) == 2
    obs, actions, rewards, next_obs, not_done = buffer.sample(64)
    assert [tuple(value.shape) for value in (obs, actions, rewards, next_obs, not_done)] == [
        (64, 3), (64, 2), (64, 1), (64, 3), (64, 1),
    ]
    assert all(value.dtype == torch.float32 for value in (obs, actions, rewards, next_obs, not_done))
    assert set(obs[:, 0].tolist()) == {2.0, 3.0}
    assert torch.equal(rewards[:, 0], obs[:, 0])
    assert torch.equal(next_obs, obs + 1)
    assert torch.all(not_done[obs[:, 0] == 2] == 0)
    assert torch.all(not_done[obs[:, 0] == 3] == 1)


def test_actual_learning_and_delayed_actor_target_updates():
    config = TD3Config(hidden_dim=32, batch_size=16, tau=0.2, policy_delay=2)
    agent = TD3(3, 2, config)
    buffer = make_buffer()
    actor_before, critic_before = parameters(agent.actor), parameters(agent.critic)
    actor_target_before = parameters(agent.actor_target)
    critic_target_before = parameters(agent.critic_target)

    first = agent.update(buffer)
    assert first["updates"] == 1 and first["actor_updates"] == 0
    assert first["actor_loss"] is None
    assert changed(critic_before, agent.critic)
    assert not changed(actor_before, agent.actor)
    assert not changed(actor_target_before, agent.actor_target)
    assert not changed(critic_target_before, agent.critic_target)

    second = agent.update(buffer)
    assert second["updates"] == 2 and second["actor_updates"] == 1
    assert second["actor_loss"] is not None
    assert changed(actor_before, agent.actor)
    for old, online, target in zip(actor_target_before, agent.actor.parameters(), agent.actor_target.parameters()):
        torch.testing.assert_close(target, (1 - config.tau) * old + config.tau * online)
    for old, online, target in zip(critic_target_before, agent.critic.parameters(), agent.critic_target.parameters()):
        torch.testing.assert_close(target, (1 - config.tau) * old + config.tau * online)
    assert all(math.isfinite(float(value)) for value in second.values())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in agent.actor_target.parameters())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in agent.critic_target.parameters())


@pytest.mark.parametrize("terminated, expected_loss, expected_error", [(False, 2.005, 1.0), (True, 52.0, 5.0)])
def test_terminal_mask_changes_the_actual_critic_learning_target(terminated, expected_loss, expected_error):
    agent = TD3(3, 2, TD3Config(hidden_dim=16, batch_size=1))
    constant_critics(agent.critic, 2.0, 4.0)
    constant_critics(agent.critic_target, 8.0, 5.0)
    buffer = ReplayBuffer(3, 2, capacity=1)
    buffer.add([0, 0, 0], [0, 0], -2.0, [1, 1, 1], terminated)
    result = agent.update(buffer)
    # Nonterminal target = -2 + .99 * min(8, 5) = 2.95;
    # terminal target = -2. Both cases run the real Adam critic update.
    assert result["critic_loss"] == pytest.approx(expected_loss, abs=1e-5)
    assert result["td_error"] == pytest.approx(expected_error)
    assert result["q1"] == 2.0 and result["q2"] == 4.0
    first_bias = agent.critic.q1_net[-1].bias.item()
    assert first_bias < 2 if terminated else first_bias > 2
    assert agent.critic.q2_net[-1].bias.item() < 4


def test_actions_and_target_smoothing_are_bounded_and_clipped_in_order():
    agent = TD3(3, 2, TD3Config(hidden_dim=16, policy_noise=10.0, noise_clip=0.25))
    with torch.no_grad():
        for parameter in agent.actor_target.parameters():
            parameter.zero_()
        agent.actor_target.net[-2].bias.copy_(torch.atanh(torch.tensor([0.9, -0.9])))
    observations = torch.zeros((64, 3))
    check_rng = torch.Generator().set_state(agent._noise_rng.get_state())
    raw_noise = torch.randn((64, 2), generator=check_rng) * agent.config.policy_noise
    bounded_noise = raw_noise.clamp(-0.25, 0.25)
    expected = (agent.actor_target(observations) + bounded_noise).clamp(-1, 1)
    actual = agent._target_actions(observations)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (raw_noise.abs() > 0.25).any()
    assert actual.max().item() == 1.0 and actual.min().item() == -1.0
    assert not actual.requires_grad
    for observation in ([0, 0, 0], [1000, -1000, 1000]):
        action = agent.select_action(observation)
        assert action.shape == (2,) and action.dtype == np.float32
        assert np.isfinite(action).all() and (np.abs(action) <= 1).all()


def test_actor_receives_gradients_through_the_critic_action_input():
    agent = TD3(3, 2, TD3Config(hidden_dim=32, batch_size=16, policy_delay=1))
    # Construct real ReLU critics with Q1 = a[0] and Q2 = -a[0] near
    # a = 0. A zero-reward terminal sample leaves the critics unchanged.
    # Maximizing Q1 must move the actor's first action in the positive direction;
    # using Q2, a minimum, or the wrong loss sign would fail this check.
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        for parameter in agent.critic.parameters():
            parameter.zero_()
        for branch, slope in ((agent.critic.q1_net, 1.0), (agent.critic.q2_net, -1.0)):
            branch[0].weight[0, 3] = slope
            branch[0].bias[0] = 1.0
            branch[2].weight[0, 0] = 1.0
            branch[4].weight[0, 0] = 1.0
            branch[4].bias[0] = -1.0
    buffer = ReplayBuffer(3, 2, capacity=1)
    buffer.add([0, 0, 0], [0, 0], 0.0, [0, 0, 0], True)
    before = parameters(agent.actor)
    metrics = agent.update(buffer)
    assert metrics["critic_loss"] == 0.0
    assert metrics["actor_loss"] is not None
    assert changed(before, agent.actor)
    assert agent.select_action([0, 0, 0])[0] > 0
    gradients = [parameter.grad for parameter in agent.actor.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert all(parameter.requires_grad for parameter in agent.critic.parameters())


def test_safe_checkpoint_restores_predictions_counts_and_next_update(tmp_path):
    config = TD3Config(hidden_dim=32, batch_size=16)
    original = TD3(3, 2, config, seed=5)
    buffer = make_buffer()
    for _ in range(3):
        original.update(buffer)
    checkpoint = tmp_path / "learner.pt"
    torch.save(original.state_dict(), checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = TD3(3, 2, config, seed=99)
    restored.load_state_dict(payload)
    observation, action = [0.1, -0.5, 0.7], [0.2, -0.3]
    np.testing.assert_array_equal(original.select_action(observation), restored.select_action(observation))
    assert original.action_values(observation, action) == restored.action_values(observation, action)
    assert restored.updates == 3 and restored.actor_updates == 1
    # Fresh, identically seeded buffers supply the same next batch. This checks
    # optimizer state, target lag and the noise generator, not only inference.
    assert original.update(make_buffer(seed=101)) == restored.update(make_buffer(seed=101))
    for name in ("actor", "critic", "actor_target", "critic_target"):
        for first, second in zip(getattr(original, name).parameters(), getattr(restored, name).parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_rejects_bad_input_and_nonfinite_training_values():
    buffer = ReplayBuffer(3, 2)
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)
    with pytest.raises(ValueError, match="finite"):
        buffer.add([0, 0, 0], [0, 0], np.nan, [0, 0, 0], False)
    with pytest.raises(ValueError, match="normalized"):
        buffer.add([0, 0, 0], [2, 0], 0, [0, 0, 0], False)
    with pytest.raises(ValueError, match="shape"):
        buffer.add([0, 0], [0, 0], 0, [0, 0, 0], False)
    agent = TD3(3, 2, TD3Config(hidden_dim=16, batch_size=8))
    with torch.no_grad():
        agent.critic.q1_net[-1].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="critic loss"):
        agent.update(make_buffer())
    assert agent.updates == 0


@pytest.mark.parametrize("options", [{"discount": float("nan")}, {"tau": 1.1}, {"policy_noise": -1}, {"batch_size": 0}])
def test_config_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        TD3Config(**options)
