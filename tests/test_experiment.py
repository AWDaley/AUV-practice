import json
import time

import numpy as np
import pytest
import torch

from auv_sim.experiment import Experiment
from auv_sim.td3 import TD3Config


@pytest.fixture
def lab(tmp_path):
    instance = Experiment(tmp_path, scenario="open", current_strength=0,
                          td3_config=TD3Config(hidden_dim=32, batch_size=8, buffer_capacity=128))
    instance.warmup_steps = 8
    yield instance
    instance.close()


def finish(lab):
    state = lab.wait(timeout=15)
    assert lab.worker is None or not lab.worker.is_alive(), state["message"]
    assert state["mode"] != "error", state["message"]
    return state


def test_preview_never_learns(lab):
    before = {k: v.clone() for k, v in lab.agent.actor.state_dict().items()}
    for _ in range(5):
        state = lab.control("step", "baseline")
    assert state["env"]["episode_step"] == 5
    assert state["training"]["steps"] == 0
    assert len(lab.replay) == 0
    assert all(torch.equal(v, before[k]) for k, v in lab.agent.actor.state_dict().items())


def test_training_really_updates_and_budget_is_additional(lab):
    before = {k: v.clone() for k, v in lab.agent.actor.state_dict().items()}
    lab.start_training(32, 0.1)
    state = finish(lab)
    assert state["training"]["steps"] == 32
    assert state["training"]["updates"] == 24
    assert state["training"]["actor_updates"] == 12
    assert len(lab.replay) == 32
    assert any(not torch.equal(v, before[k]) for k, v in lab.agent.actor.state_dict().items())
    lab.start_training(4, 0.1)
    assert finish(lab)["training"]["steps"] == 36
    json.dumps(lab.state(), allow_nan=False)


def test_evaluation_is_reproducible_and_does_not_change_training(lab):
    lab.start_training(12)
    finish(lab)
    before = {k: v.clone() for k, v in lab.agent.actor.state_dict().items()}
    counts = (len(lab.replay), lab.training["steps"], lab.training["updates"])
    lab.start_evaluation(2, "random")
    first = finish(lab)["evaluation"]
    lab.start_evaluation(2, "random")
    second = finish(lab)["evaluation"]
    assert first == second
    assert counts == (len(lab.replay), lab.training["steps"], lab.training["updates"])
    assert all(torch.equal(v, before[k]) for k, v in lab.agent.actor.state_dict().items())


def test_checkpoint_restores_actor_optimizer_counters_but_not_replay(lab):
    lab.start_training(24)
    finish(lab)
    obs = lab.observation.copy()
    expected = lab.agent.select_action(obs)
    state = lab.save()
    name = state["checkpoints"][0]["name"]
    count = state["training"]["updates"]
    lab.start_training(16)
    finish(lab)
    state = lab.load(name)
    np.testing.assert_array_equal(lab.agent.select_action(obs), expected)
    assert state["training"]["updates"] == count
    assert state["training"]["buffer_size"] == 0
    assert len(lab.replay) == 0
    assert "not exact" in state["message"]
    lab.start_training(12)
    assert finish(lab)["training"]["updates"] > count


def test_invalid_checkpoint_name_does_not_discard_experiment(lab):
    before = lab.agent
    with pytest.raises(ValueError):
        lab.load("../outside.pt")
    assert lab.agent is before


def test_incomplete_checkpoint_does_not_replace_live_policy(lab):
    state = lab.save()
    name = state["checkpoints"][0]["name"]
    path = lab.data_dir / name
    payload = torch.load(path, weights_only=True)
    del payload["history"]
    torch.save(payload, path)
    actor, environment, replay = lab.agent, lab.env, lab.replay
    with pytest.raises((KeyError, ValueError)):
        lab.load(name)
    assert lab.agent is actor
    assert lab.env is environment
    assert lab.replay is replay


def test_pause_stops_worker_and_new_experiment_clears_learning(lab):
    lab.start_training(100_000)
    lab.control("pause", "td3")
    assert lab.worker is None
    count = lab.training["steps"]
    time.sleep(0.02)
    assert lab.training["steps"] == count
    state = lab.reset(seed=12, scenario="reef", current_strength=0.1)
    assert state["training"]["steps"] == 0
    assert len(lab.replay) == 0
    assert not state["policy_ready"]


def test_export_is_real_json(lab, tmp_path):
    lab.export(tmp_path / "results.json")
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["config"]["scenario"] == "open"
    assert payload["training"]["steps"] == 0
