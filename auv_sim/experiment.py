"""Own one experiment and serialize mutations between the UI and worker.

Preview, training, and evaluation share a policy, but only training writes to
replay or changes weights. Evaluation uses an independent environment and seed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any

import numpy as np
import torch

from .environment import AUVEnv, EnvConfig
from .td3 import ReplayBuffer, TD3, TD3Config


class Experiment:
    """A single-user local lab. Commands halt the previous worker before acting."""

    warmup_steps = 1_000

    def __init__(self, data_dir: Path | str = "runs", *, seed: int = 7,
                 scenario: str = "reef", current_strength: float = 0.08,
                 td3_config: TD3Config | None = None) -> None:
        torch.set_num_threads(1)
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Keep the real, measured example available without changing the fresh actor.
        examples = Path(__file__).parent / "assets"
        for source in examples.glob("td3-demo-reef-50000.*"):
            destination = self.data_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        self.lock = threading.RLock()
        self.command_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.td3_config = td3_config or TD3Config()
        self.config = {"seed": seed, "scenario": scenario, "current_strength": current_strength}
        self._initialize()

    def _initialize(self) -> None:
        self.env = AUVEnv(EnvConfig(scenario=self.config["scenario"],
                                    current_strength=self.config["current_strength"]))
        self.observation, _ = self.env.reset(seed=self.config["seed"])
        self.agent = TD3(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                         self.td3_config, seed=self.config["seed"])
        self.replay = ReplayBuffer(self.env.observation_space.shape[0],
                                   self.env.action_space.shape[0],
                                   capacity=self.td3_config.buffer_capacity,
                                   seed=self.config["seed"] + 2)
        self.rng = np.random.default_rng(self.config["seed"] + 1)
        self.history: deque[dict[str, Any]] = deque(maxlen=500)
        self.evaluation: dict[str, Any] | None = None
        self.training: dict[str, Any] = {
            "steps": 0, "total_steps": 0, "updates": 0, "actor_updates": 0,
            "episodes": 0, "buffer_size": 0, "exploration_noise": 0.15,
            "actor_loss": None, "critic_loss": None, "q1": None, "q2": None,
            "td_error": None, "steps_per_second": 0.0, "phase": "Not trained",
        }
        self.action = np.zeros(2, dtype=np.float32)
        self.mode = "idle"
        self.controller = "baseline"
        self.message = "Play the baseline, load the trained example from Checkpoints, or train a fresh TD3 policy."
        self.env_role = "preview"
        self.episode_done = False
        self._evaluation_env: AUVEnv | None = None
        self._evaluation_action = np.zeros(2, dtype=np.float32)

    def _halt(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=10)
            if self.worker.is_alive():
                raise RuntimeError("The current update is still finishing. Try again shortly.")
        self.worker = None
        with self.lock:
            if self.mode not in {"idle", "error"}:
                self.mode = "paused"
            self.training["steps_per_second"] = 0.0
            self._evaluation_env = None

    def close(self) -> None:
        with self.command_lock:
            self._halt()

    def reset(self, *, seed: int, scenario: str, current_strength: float) -> dict:
        with self.command_lock:
            # Validate before interrupting or discarding a working experiment.
            EnvConfig(scenario=scenario, current_strength=current_strength)
            if not isinstance(seed, int) or not 0 <= seed < 2**31:
                raise ValueError("Seed must be an integer in [0, 2147483647].")
            self._halt()
            with self.lock:
                self.config = {"seed": seed, "scenario": scenario,
                               "current_strength": current_strength}
                self._initialize()
            return self.state()

    def _start_worker(self, target) -> None:
        self.stop_event = threading.Event()
        stop = self.stop_event

        def guarded() -> None:
            try:
                target(stop)
            except Exception as exc:
                with self.lock:
                    self.mode = "error"
                    self.message = f"{type(exc).__name__}: {exc}"
                    self.training["steps_per_second"] = 0.0
                stop.set()

        self.worker = threading.Thread(target=guarded, daemon=True, name="auv-simulation")
        self.worker.start()

    def _choose(self, controller: str, env: AUVEnv, observation: np.ndarray) -> np.ndarray:
        if controller == "baseline":
            return env.heuristic_action()
        if controller == "random":
            return self.rng.uniform(-1, 1, size=2).astype(np.float32)
        if controller == "td3":
            return self.agent.select_action(observation)
        raise ValueError("Unknown controller.")

    def _preview_tick(self) -> None:
        if self.episode_done:
            self.observation, _ = self.env.reset()
            self.episode_done = False
        self.action = self._choose(self.controller, self.env, self.observation)
        self.observation, _, terminated, truncated, _ = self.env.step(self.action)
        self.episode_done = terminated or truncated

    def control(self, command: str, controller: str = "baseline", speed: float = 1) -> dict:
        if controller not in {"baseline", "td3", "random"}:
            raise ValueError("Unknown controller.")
        if command not in {"play", "pause", "step"}:
            raise ValueError("Unknown command.")
        if not np.isfinite(speed) or not 0 < speed <= 32:
            raise ValueError("Playback speed must be positive and at most 32.")
        with self.command_lock:
            self._halt()
            with self.lock:
                self.controller = controller
                if command == "pause":
                    self.mode = "paused"
                    self.message = "Paused. Training weights and replay are preserved."
                    return self.state()
                if self.env_role != "preview":
                    self.observation, _ = self.env.reset()
                    self.episode_done = False
                self.env_role = "preview"
                if command == "step":
                    self._preview_tick()
                    self.mode = "paused"
                    self.message = "Advanced one simulation step. No learning update was performed."
                    return self.state()
                self.mode = "preview"
                self.message = ("Map-aware A* baseline: hand-written, with access to the obstacle map."
                                if controller == "baseline" else
                                "Running TD3 without exploration noise; weights are fixed."
                                if controller == "td3" else "Running random commands; no learning.")

            def run(stop: threading.Event) -> None:
                while not stop.is_set():
                    started = time.monotonic()
                    with self.lock:
                        self._preview_tick()
                        done = self.episode_done
                    # Keep terminal outcomes visible before starting another episode.
                    delay = 0.65 if done else max(0, self.env.config.dt / speed - (time.monotonic() - started))
                    stop.wait(delay)

            self._start_worker(run)
            return self.state()

    def start_training(self, steps: int = 10_000, exploration_noise: float = 0.15) -> dict:
        if not isinstance(steps, int) or not 1 <= steps <= 1_000_000:
            raise ValueError("Training budget must be 1 to 1,000,000 additional steps.")
        if not np.isfinite(exploration_noise) or not 0 <= exploration_noise <= 1:
            raise ValueError("Exploration noise must be in [0, 1].")
        with self.command_lock:
            self._halt()
            with self.lock:
                if self.env_role != "training" or self.episode_done:
                    self.observation, _ = self.env.reset()
                    self.episode_done = False
                self.env_role = "training"
                self.controller = "td3"
                self.mode = "training"
                initial_steps = self.training["steps"]
                target = initial_steps + steps
                self.training["total_steps"] = target
                self.training["exploration_noise"] = exploration_noise
                self.message = "Collecting transitions and training the actor and two critics."

            def run(stop: threading.Event) -> None:
                started = time.monotonic()
                while not stop.is_set():
                    with self.lock:
                        if self.training["steps"] >= target:
                            self.mode = "paused"
                            self.message = "Training budget complete. Run TD3 or evaluate it without noise."
                            self.training["phase"] = "Training budget complete"
                            self.training["steps_per_second"] = 0.0
                            return
                        obs = self.observation
                        if self.training["steps"] < self.warmup_steps:
                            self.action = self.rng.uniform(-1, 1, size=2).astype(np.float32)
                            self.training["phase"] = "Random warm-up"
                        else:
                            noise = self.rng.normal(0, exploration_noise, size=2)
                            self.action = np.clip(self.agent.select_action(obs) + noise, -1, 1).astype(np.float32)
                            self.training["phase"] = "Learning from replay"
                        nxt, reward, terminated, truncated, info = self.env.step(self.action)
                        # Truncation ends collection, but does not erase a continuing task's bootstrap.
                        self.replay.add(obs, self.action, reward, nxt, terminated)
                        self.observation = nxt
                        self.training["steps"] += 1
                        self.training["buffer_size"] = len(self.replay)
                        if (self.training["steps"] > self.warmup_steps
                                and len(self.replay) >= self.td3_config.batch_size):
                            metrics = self.agent.update(self.replay)
                            for key, value in metrics.items():
                                if value is not None:
                                    self.training[key] = value
                        if terminated or truncated:
                            snap = self.env.snapshot()
                            self.training["episodes"] += 1
                            self.history.append({"episode": self.training["episodes"],
                                                 "steps": self.training["steps"],
                                                 "return": snap["episode_return"],
                                                 "length": snap["episode_step"],
                                                 "outcome": info["outcome"]})
                            self.observation, _ = self.env.reset()
                        elapsed = max(1e-6, time.monotonic() - started)
                        self.training["steps_per_second"] = (self.training["steps"] - initial_steps) / elapsed
                    # Release the GIL/lock so status and pause remain responsive during fast CPU runs.
                    if self.training["steps"] % 20 == 0:
                        time.sleep(0)

            self._start_worker(run)
            return self.state()

    def start_evaluation(self, episodes: int = 10, controller: str = "td3") -> dict:
        if not isinstance(episodes, int) or not 1 <= episodes <= 100:
            raise ValueError("Evaluate between 1 and 100 episodes.")
        if controller not in {"td3", "baseline", "random"}:
            raise ValueError("Unknown controller.")
        with self.command_lock:
            self._halt()
            with self.lock:
                self.mode = "evaluating"
                self.controller = controller
                self.message = "Evaluating on separate fixed seeds with learning disabled."
                self.evaluation = None
                evaluation_seed = self.config["seed"] + 10_000
                settings = dict(self.config)

            def run(stop: threading.Event) -> None:
                env = AUVEnv(EnvConfig(scenario=settings["scenario"],
                                      current_strength=settings["current_strength"]))
                eval_rng = np.random.default_rng(evaluation_seed)
                records = []
                for i in range(episodes):
                    if stop.is_set():
                        return
                    with self.lock:
                        obs, _ = env.reset(seed=evaluation_seed + i)
                        self._evaluation_env = env
                        self._evaluation_action = np.zeros(2, dtype=np.float32)
                    done = False
                    while not done and not stop.is_set():
                        with self.lock:
                            if controller == "random":
                                action = eval_rng.uniform(-1, 1, 2).astype(np.float32)
                            else:
                                action = self._choose(controller, env, obs)
                            self._evaluation_action = action.copy()
                            obs, _, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                        if env.snapshot()["episode_step"] % 20 == 0:
                            time.sleep(0)
                    if stop.is_set():
                        return
                    snap = env.snapshot()
                    records.append({"return": snap["episode_return"],
                                    "steps": snap["episode_step"], "outcome": info["outcome"]})
                    with self.lock:
                        self.message = f"Evaluating {controller}: {i + 1}/{episodes} episodes complete."
                with self.lock:
                    self.evaluation = {
                        "controller": controller, "episodes": episodes,
                        "success_rate": sum(r["outcome"] == "success" for r in records) / episodes,
                        "collision_rate": sum(r["outcome"] == "collision" for r in records) / episodes,
                        "timeout_rate": sum(r["outcome"] == "timeout" for r in records) / episodes,
                        "mean_return": float(np.mean([r["return"] for r in records])),
                        "mean_steps": float(np.mean([r["steps"] for r in records])),
                        "seed": evaluation_seed, "training_steps": self.training["steps"],
                        "scenario": settings["scenario"], "records": records,
                    }
                    self.mode = "paused"
                    self._evaluation_env = None
                    self.message = "Evaluation complete. The policy and replay buffer were not changed."

            self._start_worker(run)
            return self.state()

    def wait(self, timeout: float | None = None) -> dict:
        worker = self.worker
        if worker:
            worker.join(timeout=timeout)
        return self.state()

    def _checkpoint_list(self) -> list[dict]:
        checkpoints = []
        for path in sorted(self.data_dir.glob("td3-*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(path.with_suffix(".json").read_text())
                checkpoints.append({"name": path.name, "steps": meta["training_steps"],
                                    "created": meta["created"]})
            except (OSError, ValueError, KeyError):
                continue
        return checkpoints

    def save(self) -> dict:
        with self.command_lock:
            self._halt()
            with self.lock:
                now = datetime.now(timezone.utc)
                name = f"td3-{self.training['steps']:08d}-{now.strftime('%Y%m%dT%H%M%S%f')}.pt"
                path = self.data_dir / name
                meta = {"created": now.isoformat(), "training_steps": self.training["steps"],
                        "config": self.config, "replay_saved": False}
                payload = {"format_version": 1, "agent": self.agent.state_dict(),
                           "metadata": meta, "training": self.training,
                           "history": list(self.history), "evaluation": self.evaluation}
                temporary = path.with_suffix(".tmp")
                try:
                    torch.save(payload, temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
                self.message = f"Saved {name}. Networks, targets and optimizers are saved; replay is not."
                return self.state()

    def load(self, name: str) -> dict:
        if not re.fullmatch(r"td3-[A-Za-z0-9_-]{1,80}\.pt", name):
            raise ValueError("Choose a checkpoint from the saved checkpoint list.")
        with self.command_lock:
            path = self.data_dir / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Checkpoint not found.")
            self._halt()
            # weights_only excludes arbitrary Python object loading.
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint format.")
            config = payload["metadata"]["config"]
            td3_config = TD3Config(**payload["agent"]["config"])
            # Build candidates first so an invalid checkpoint cannot partially replace the live policy.
            env = AUVEnv(EnvConfig(scenario=config["scenario"], current_strength=config["current_strength"]))
            obs, _ = env.reset(seed=config["seed"])
            agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], td3_config,
                         seed=config["seed"])
            agent.load_state_dict(payload["agent"])
            training = dict(payload["training"])
            if not set(self.training).issubset(training):
                raise ValueError("Checkpoint is missing training metadata.")
            if training["updates"] != agent.updates or training["actor_updates"] != agent.actor_updates:
                raise ValueError("Checkpoint learner counters do not match its training metadata.")
            if not isinstance(payload["history"], list):
                raise ValueError("Checkpoint history must be a list.")
            history = deque(payload["history"], maxlen=500)
            evaluation = payload["evaluation"]
            if evaluation is not None and not isinstance(evaluation, dict):
                raise ValueError("Checkpoint evaluation must be an object or null.")
            json.dumps({"training": training, "history": list(history), "evaluation": evaluation}, allow_nan=False)
            replay = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0],
                                  capacity=td3_config.buffer_capacity, seed=config["seed"] + 2)
            rng = np.random.default_rng(config["seed"] + 1)
            training.update(buffer_size=0, steps_per_second=0, phase="Loaded; replay will refill")
            with self.lock:
                self.config = config
                self.td3_config = td3_config
                self.env = env
                self.observation = obs
                self.agent = agent
                self.replay = replay
                self.rng = rng
                self.training = training
                self.history = history
                self.evaluation = evaluation
                self.action = np.zeros(2, dtype=np.float32)
                self.env_role = "preview"
                self.episode_done = False
                self.mode = "paused"
                self.controller = "td3"
                self.message = "Loaded TD3. Replay and environment history restart; this is not exact run resumption."
                return self.state()

    def state(self) -> dict:
        with self.lock:
            env = self._evaluation_env if self._evaluation_env is not None else self.env
            action = self._evaluation_action if self._evaluation_env is not None else self.action
            return {
                "mode": self.mode, "controller": self.controller, "message": self.message,
                "env": env.snapshot(), "training": dict(self.training),
                "history": list(self.history), "evaluation": self.evaluation,
                "checkpoints": self._checkpoint_list(), "config": dict(self.config),
                "action": action.tolist(), "policy_ready": self.training["actor_updates"] > 0,
            }

    def export(self, path: Path | str) -> None:
        with self.lock:
            report = {"config": self.config, "td3_config": asdict(self.td3_config),
                      "training": self.training, "history": list(self.history),
                      "evaluation": self.evaluation,
                      "limitations": "2D kinematics; no validated hydrodynamics or hardware model."}
            Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
