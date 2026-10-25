# AUV Practice

**Watch a simulated underwater vehicle move, train a real TD3 policy, and inspect what it learns.**

A local Python/PyTorch learning lab with a browser interface. Start with a map-aware steering demonstration, then train an initially untrained actor with two critics. The world view shows the vehicle, sonar rays, obstacles, current, goal, and trajectory. The inspector shows actual rewards, replay size, update counts, losses, and episode returns.

This first version is a **2D educational simulator** with simplified motion. It is not a hydrodynamic simulator or an exact reproduction of the three Ocean Engineering AUV studies. A real 50,000-step TD3 checkpoint is included for immediate playback; the default experiment starts with fresh weights so you can also observe learning from scratch.

## Start on Windows

Install **Python 3.11 or 3.12** (3.12 was used for validation), then clone or download this repository.

**Double-click `start-windows.cmd`.** It creates a project-local virtual environment, installs CPU-only PyTorch and the app, and starts the server. The first launch needs internet access for those packages. Open **http://127.0.0.1:8000** in your browser and leave the terminal open.

Or use PowerShell from the repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m auv_sim
```

You do not need to activate the virtual environment or change PowerShell execution policy. Opening `index.html` directly does not start the Python simulator.

## Start on macOS or Linux

```bash
python3 -m venv .venv
# Linux CPU installation; macOS users should run: .venv/bin/python -m pip install torch
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
.venv/bin/python -m auv_sim
```

Open **http://127.0.0.1:8000**. The app needs no API keys, external AI services, Node installation, or GPU. It runs locally after package installation. If port 8000 is occupied, use `python -m auv_sim serve --port 8001` with your environment's Python executable.

## Your first experiment

**To watch trained TD3 immediately:** open **Checkpoints**, choose `td3-demo-reef-50000.pt`, and click **Load**. The controller switches to TD3. Click **Play** to watch it, or **Evaluate** to run the fixed test suite. This checkpoint achieved 10/10 goal arrivals in the recorded reef evaluation; that small test is not a guarantee on other conditions. See [validation results](docs/VALIDATION.md) for the exact training procedure and limits.

**To learn from scratch**, reset the experiment and follow these steps:

1. Keep **Map-aware baseline** selected and run it. Watch the sonar, current arrow, and path. This controller uses an A* route and steering rules; it knows the full obstacle map. It is a demonstration, not a trained TD3 policy or an equal-information benchmark.
2. Pause and choose **TD3**. Run or single-step it to see the initial actor's behavior. No training occurs in this preview mode.
3. Request a training budget, for example **10,000 additional steps**. The first 1,000 training transitions use random commands; subsequent commands use the actor plus exploration noise. Critic learning starts after warm-up when a batch is available.
4. Observe actual episode returns, outcomes, critic losses, and actor-update counts. A short run demonstrates learning mechanics; it does not guarantee successful navigation.
5. Pause and **evaluate TD3**. Evaluation uses separate fixed seeds and no exploration noise. It reports success, collision, mean return, and episode length without changing the model or replay buffer.
6. Save a checkpoint. Train for an additional budget, evaluate again, and compare the measured results.

Use **Open water** to learn the basic control task, **Reef** for scattered obstacles, and **Channel** for a more constrained course. Applying a new experiment resets both the environment and the learned policy; save first if you want to keep the model. Preview, pause, and additional training budgets preserve existing learning.

Training is intentionally CPU-friendly: two hidden layers of 128 units, one PyTorch thread, a 100,000-transition float32 buffer, and a batch size of 128. The network width and warm-up budget differ from the reference repository discussed in the source-code guide; the TD3 learning mechanisms are retained.

## What is actually implemented

| Component | Behavior |
| --- | --- |
| Actor | Observation → 128 ReLU → 128 ReLU → two tanh action components |
| Twin critics | Separate state/action value networks; the smaller target estimate builds the bootstrap |
| Target networks | Copies of both online networks, updated slowly with `tau=0.005` |
| Delayed policy updates | Critics update every learning call; actor and targets every second call |
| Target smoothing | Gaussian target-action noise, noise clipping, and action clipping |
| Replay | Uniform sampling with replacement; only true termination disables bootstrapping |
| Evaluation | Fixed held-out seed sequence, frozen weights, no exploration |
| Checkpoints | Actor, critics, targets, optimizers, learner counters/noise RNG, settings, and recent metrics |

Checkpoints **do not save replay, rollout RNG, or the full simulator state**. Loading restores a useful policy and learner state but is not exact training resumption. Replay refills when training continues. A model already past warm-up immediately collects with actor plus exploration noise and resumes optimizer updates after a fresh batch is available; it does not repeat the initial 1,000 random steps. Only load checkpoints you created or trust; the app uses PyTorch's `weights_only=True` loading mode.

Prioritized replay, N-step returns, C-APF guidance, adaptive research-specific noise schedules, moving obstacles, and 3D dynamics are **future extensions**. They are not represented as implemented paper results.

## Headless training and evaluation

After installation, using the virtual environment's Python:

```bash
python -m auv_sim train --scenario reef --seed 7 --steps 20000 --eval-episodes 10
python -m auv_sim evaluate --controller baseline --scenario reef --episodes 20
python -m auv_sim --help
```

Training prints the resulting checkpoint filename. To evaluate a saved model, supply that filename to `python -m auv_sim evaluate --checkpoint NAME.pt`. Checkpoint names must be files in the selected `--data-dir` (default `runs`). Loading uses the checkpoint's scenario, seed, and current configuration. `train --load NAME.pt --steps 20000` continues from a saved learner with fresh replay.

The CLI saves `runs/latest-results.json` with configuration, measured metrics, recent episode history, and evaluation results. The browser also exposes the current metrics at `/api/results`. `runs/` is ignored by Git.

## Source map

| File | Responsibility |
| --- | --- |
| `auv_sim/environment.py` | Motion, seeded courses, ray observations, swept collision tests, rewards, baseline |
| `auv_sim/td3.py` | Neural networks, replay, TD3 update, learner checkpoint state |
| `auv_sim/experiment.py` | Training/preview/evaluation lifecycles and synchronization |
| `auv_sim/server.py` | Validated local API and static interface hosting |
| `auv_sim/static/` | Canvas renderer, controls, real-data charts, and telemetry |
| `auv_sim/__main__.py` | Browser-server and headless commands |
| `tests/` | Algorithm, environment, checkpoint, lifecycle, and API tests |

See [the simulation specification](docs/SIMULATION.md) for observation/action definitions, reward terms, and limitations, and [the development notes](docs/DEVELOPMENT.md) for testing and extension points.

## Run the tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests exercise actual PyTorch updates, target masking, delayed actor updates, save/load behavior, collision geometry, Gymnasium compatibility, and the local API. They do not establish navigation reliability in real water.

## Research and implementation references

- Fujimoto, van Hoof, and Meger, [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), 2018.
- Scott Fujimoto's [TD3 reference implementation](https://github.com/sfujim/TD3/tree/34770ccdcb51df6f6c7d85c1ba7e40d71b940d16).
- OpenAI Spinning Up, [TD3 equations and pseudocode](https://spinningup.openai.com/en/latest/algorithms/td3.html).
- Gymnasium, [custom environments](https://gymnasium.farama.org/introduction/create_custom_env/) and [time-limit handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- Zhang et al. (2022), [AUV path tracking with real-time obstacle avoidance via reinforcement learning under adaptive constraints](https://www.sciencedirect.com/science/article/pii/S0029801822008320).
- Li and Yu (2025), [Obstacle avoidance path planning for AUVs in a three-dimensional unknown environment based on the C-APF-TD3 algorithm](https://www.sciencedirect.com/science/article/pii/S0029801824032244).
- Su et al. (2026), [AUV path planning in complex 3D underwater environments based on an improved TD3 algorithm](https://www.sciencedirect.com/science/article/pii/S0029801825033700).

The simulator code was written for this repository using those algorithmic references. The paper-specific author implementations were not available for verification. See [NOTICE](NOTICE) for attribution.
