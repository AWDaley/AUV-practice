import json
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from auv_sim.environment import AUVEnv, EnvConfig


def make_env(**kwargs):
    return AUVEnv(EnvConfig(scenario="open", current_strength=0, course_variety=False, **kwargs))


def test_seeded_resets_and_rollouts_are_reproducible():
    first = AUVEnv(EnvConfig(scenario="reef", current_strength=0.2))
    second = AUVEnv(EnvConfig(scenario="reef", current_strength=0.2))
    obs1, info1 = first.reset(seed=123)
    obs2, info2 = second.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    assert info1 == info2
    assert first.snapshot() == second.snapshot()
    for action in ([0.8, -0.1], [0.9, 0.2], [-0.4, 0.8]):
        result1 = first.step(action)
        result2 = second.step(action)
        np.testing.assert_array_equal(result1[0], result2[0])
        assert result1[1:] == result2[1:]
    first.reset(seed=124)
    assert first.snapshot()["goal"] != second.snapshot()["goal"]
    obs3, _ = first.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs3)


def test_swept_collision_catches_an_obstacle_even_when_endpoint_is_clear():
    env = make_env(dt=2.0)
    env.position = np.array([2.0, 5.0])
    env.goal = np.array([18.0, 5.0])
    env.heading = 0.0
    env.speed = 1.8
    env.obstacles = np.array([[3.5, 5.0, 0.35]])
    obs, _, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated
    assert info["outcome"] == "collision"
    assert env.position[0] == pytest.approx(2.9)
    assert info["min_clearance"] == pytest.approx(0.0, abs=1e-9)
    assert env.observation_space.contains(obs)


def test_sweep_detects_tangent_contact():
    env = make_env(dt=2.0)
    env.position = np.array([2.0, 5.0])
    env.heading = 0.0
    env.speed = 1.8
    env.obstacles = np.array([[3.5, 5.6, 0.35]])
    _, _, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated
    assert info["outcome"] == "collision"
    assert env.position[0] == pytest.approx(3.5, abs=1e-6)


@pytest.mark.parametrize("position,heading,axis,contact", [
    ([19.0, 10.0], 0.0, 0, 19.75),
    ([1.0, 10.0], math.pi, 0, 0.25),
    ([10.0, 19.0], math.pi / 2, 1, 19.75),
    ([10.0, 1.0], -math.pi / 2, 1, 0.25),
])
def test_wall_collision_keeps_entire_vehicle_in_bounds(position, heading, axis, contact):
    env = make_env(dt=1.0)
    env.position = np.array(position, dtype=float)
    env.heading = heading
    env.speed = 1.8
    _, _, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated
    assert info["outcome"] == "collision"
    assert env.position[axis] == pytest.approx(contact)
    assert np.all(env.position >= 0.25)
    assert np.all(env.position <= 19.75)


def test_timeout_is_truncation_but_success_at_horizon_is_termination():
    env = make_env(max_steps=1)
    _, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert truncated and not terminated and info["outcome"] == "timeout"
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    env.reset()
    env.goal = env.position + np.array([0.5, 0.0])
    _, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and not truncated and info["outcome"] == "success"
    assert info["reward_terms"]["goal_bonus"] == 100
    assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_first_physical_event_wins_over_a_later_event_in_same_step():
    env = make_env(dt=4.0)
    env.position = np.array([2.0, 5.0])
    env.heading = 0.0
    env.speed = 1.8
    env.goal = np.array([4.0, 5.0])
    env.obstacles = np.array([[7.0, 5.0, 0.4]])
    _, _, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated and info["outcome"] == "success"
    assert env.position[0] == pytest.approx(3.25)


@pytest.mark.parametrize("scenario", ["open", "reef", "channel"])
def test_observations_stay_finite_normalized_and_snapshot_is_json(scenario):
    env = AUVEnv(EnvConfig(scenario=scenario, current_strength=0.4, max_steps=100))
    obs, _ = env.reset(seed=31)
    assert obs.shape == (21,) and obs.dtype == np.float32
    for _ in range(100):
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs))
        snapshot = env.snapshot()
        json.dumps(snapshot, allow_nan=False)
        assert len(snapshot["sonar"]) == 16
        assert -math.pi <= snapshot["vehicle"]["heading"] <= math.pi
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert env.observation_space.contains(obs)
    assert set(env.snapshot()) == {
        "world", "vehicle", "goal", "obstacles", "sonar", "current", "trail",
        "episode_step", "episode_return", "last_reward", "distance_to_goal",
        "min_clearance", "outcome", "reward_terms",
    }


def test_body_frame_features_and_sonar_geometry():
    env = make_env()
    env.position = np.array([10.0, 10.0])
    env.heading = math.pi / 2
    env.goal = np.array([10.0, 14.0])
    env.current = np.array([0.1, 0.2])
    env.obstacles = np.array([[10.0, 13.0, 0.5]])
    obs = env._observation()
    assert obs[0] == pytest.approx(2.5 / 6)
    assert obs[16] == pytest.approx(4 / math.hypot(20, 20))
    assert obs[17] == pytest.approx(0, abs=1e-7)
    assert obs[19] == pytest.approx(0.5)
    assert obs[20] == pytest.approx(-0.25)
    ray = env.snapshot()["sonar"][0]
    assert ray["end_x"] == pytest.approx(10)
    assert ray["end_y"] == pytest.approx(12.5)


@pytest.mark.parametrize("action", [[np.nan, 0], [0, np.inf], [0], [[0, 0]]])
def test_invalid_action_is_rejected_without_advancing(action):
    env = make_env()
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.snapshot() == before


@pytest.mark.parametrize("kwargs", [
    {"scenario": "missing"}, {"current_strength": float("nan")},
    {"current_strength": 0.41}, {"current_strength": -0.1},
    {"dt": 0}, {"dt": float("inf")}, {"max_steps": 0}, {"max_steps": 1.2},
    {"seed": -1},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_no_reverse_and_trail_memory_is_bounded():
    env = make_env(max_steps=700)
    for _ in range(650):
        obs, _, terminated, truncated, _ = env.step([-1.0, 1.0])
        assert env.speed == 0
        assert env.observation_space.contains(obs)
        assert not terminated and not truncated
    assert len(env.snapshot()["trail"]) == 600


@pytest.mark.parametrize("scenario", ["open", "reef", "channel"])
def test_explicit_map_aware_baseline_can_complete_presets(scenario):
    env = AUVEnv(EnvConfig(scenario=scenario, current_strength=0.08))
    env.reset(seed=7)
    for _ in range(env.config.max_steps):
        action = env.heuristic_action()
        assert env.action_space.contains(action)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["outcome"] == "success"


def test_gymnasium_environment_contract():
    check_env(make_env(), skip_render_check=True)
