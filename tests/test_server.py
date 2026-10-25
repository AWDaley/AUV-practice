from fastapi.testclient import TestClient
import pytest

from auv_sim.experiment import Experiment
from auv_sim.server import create_app
from auv_sim.td3 import TD3Config


@pytest.fixture
def client(tmp_path):
    lab = Experiment(tmp_path, td3_config=TD3Config(hidden_dim=32, batch_size=8, buffer_capacity=128))
    lab.warmup_steps = 8
    with TestClient(create_app(experiment=lab)) as client:
        yield client


def test_health_static_and_state(client):
    assert client.get("/health").json()["algorithm"] == "TD3"
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    state = client.get("/api/state").json()
    assert len(state["env"]["sonar"]) == 16
    assert state["mode"] == "idle"


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/train", {"steps": -5}),
    ("/api/train", {"steps": 100, "exploration_noise": 2}),
    ("/api/reset", {"scenario": "missing"}),
    ("/api/reset", {"current_strength": 2}),
    ("/api/control", {"command": "delete", "controller": "td3"}),
    ("/api/evaluate", {"episodes": 0}),
])
def test_invalid_inputs_are_rejected(client, endpoint, payload):
    assert client.post(endpoint, json=payload).status_code == 422


def test_step_save_load_and_result_endpoints(client):
    state = client.post("/api/control", json={"command": "step"}).json()
    assert state["env"]["episode_step"] == 1
    saved = client.post("/api/save", json={}).json()
    name = saved["checkpoints"][0]["name"]
    restored = client.post("/api/load", json={"name": name})
    assert restored.status_code == 200
    assert restored.json()["env"]["episode_step"] == 0
    assert client.post("/api/load", json={"name": "../../outside.pt"}).status_code == 400
    assert "history" in client.get("/api/results").json()
