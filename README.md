# AUV Practice

**Watch a simulated underwater vehicle move, train a real TD3 policy, and inspect what it learns.**

## Start here: open the single HTML file

**Download [auv-practice.html](auv-practice.html), save it to your computer, and open it in a browser.** That one file contains the interface, simulator, JavaScript TD3 learner, and a real trained example. Opening and using it requires **no packages, Python, Node, server, API keys, or network connection** after the file has been downloaded.

On GitHub, open the file and use **Download raw file**. Open the saved `.html` in a browser tab; GitHub's source viewer does not run the app.

The included reef policy is loaded and paused at startup. Wait for **Browser engine ready**, then press **Play**. Use **Pause** or single-step to inspect the vehicle, sonar rays, obstacles, current, goal, trajectory, and actual reward terms. The inspector shows replay size, update counts, losses, and episode returns.

Use **Code walkthrough** to place the tank beside a wide source panel and advance through real execution stages. Each complete code block expands to its full height, with key values below it. Scroll the page for longer code or opened implementation details; the controls remain available, and the tank stays alongside on desktop. Expand **All variables** for the full record or **Options** for pace and exploration settings. Follow observations into the actor, see the action move the AUV, and inspect replay, TD targets, and delayed updates during training. See [the walkthrough guide](docs/BROWSER.md#code-walkthrough) for how live motion and learning stages fit together.

The example was trained with real TD3 updates for 50,000 cumulative interactions in the Python implementation. Its weights and recorded history are embedded in the HTML; those initial charts describe prior training. **Reset experiment** starts a fresh, untrained JavaScript learner. All simulation and browser training run on your device, and speed depends on its hardware.

This is a **2D educational simulator** with simplified motion. It is not a hydrodynamic simulator or an exact reproduction of the three Ocean Engineering AUV studies.

### Two supported workflows

| Workflow | Start | Use |
| --- | --- | --- |
| **Single-file browser app** | Open `auv-practice.html` directly | Immediate offline playback, experiments, and genuine JavaScript TD3 training |
| **Python/PyTorch app** | Install the project and run `python -m auv_sim` | Python development, headless training, and the local FastAPI interface |

The implementations have different defaults, checkpoint formats, and random sequences. See the [browser guide](docs/BROWSER.md) for the offline workflow and the [Python setup](#python-implementation-optional) below for the server-based implementation.

### Your first browser experiment

1. **Try the included policy.** Press **Play** with **TD3 policy** selected. Preview uses the actor without exploration noise or training updates.
2. **Evaluate it.** Pause, choose an episode count, and click **Evaluate**. The app measures success, collisions, mean return, and episode length on its fixed JavaScript seed sequence. Evaluation does not update the model or replay.
3. **Start fresh.** Choose Open water, Reef field, or Narrow channel; set the seed and current; then click **Reset experiment**. Reset clears the active learner, replay, training history, and evaluation. Saved entries remain in the current tab's checkpoint list.
4. **Train.** Request the default **2,000 additional environment steps** as a short practice run. The first 1,000 cumulative training transitions use random commands. After warm-up, the actor collects with exploration noise and the learner updates from replay.
5. **Inspect and compare.** Watch actual returns, outcomes, losses, and update counts. Pause to preview or evaluate again. A short run demonstrates learning mechanics; it does not guarantee successful navigation.
6. **Keep your work.** Use **Save & download policy** while paused for model JSON, or **Export results** for a separate JSON record of settings, metrics, retained history, and evaluation.

**Checkpoint entries and unsaved learning belong to the current tab.** Keep the downloaded checkpoint JSON before closing or reloading. On a later opening, **Import checkpoint** reads that JSON and loads the model immediately. It accepts browser checkpoint JSON up to 8 MiB, not Python `.pt` files. The original HTML remains the complete application; model and results downloads are optional outputs.

To return to the included example after a reset, select `td3-demo-reef-50000.json` and click **Load**. Reopening the HTML also restores the embedded example.

The **Map-aware baseline** is a separate, hand-written A* route planner with steering rules and access to the full obstacle map. TD3 receives local observations. The baseline is a demonstration/reference, not an equal-information benchmark.

### Browser learning and checkpoint limits

The learner performs forward passes, backpropagation, Adam updates, replay sampling, and the TD3 update in JavaScript inside an embedded **Blob Web Worker**. Training is real; the page is not a pre-recorded animation or a remote inference client.

| Setting | Fresh browser model | Included model in browser | Python default |
| --- | --- | --- | --- |
| Hidden layers in actor and each critic | 64 → 64 | 128 → 128 | 128 → 128 |
| Training batch size | 32 | 32 for continued learning | 128 |
| Replay capacity | 50,000 | 50,000 | 100,000 |
| Initial state | Created by Reset | Loaded and paused at startup | Fresh model at startup |
| Checkpoint format | Browser JSON | Browser JSON | PyTorch `.pt` |

Both paths use twin critics, clipped target-action noise, delayed actor/target updates, slowly updated target networks, uniform replay, and a bootstrap mask that distinguishes true termination from an external time limit.

Loading restores a useful model and learner state, but **is not exact training resumption**. Replay, simulator state, and rollout random streams restart. Already-trained models collect with their actor plus exploration noise and resume updates once fresh replay contains a batch; completed warm-up is not repeated. JavaScript seeded sequences differ from NumPy's, and the imported model uses a smaller browser batch than its original Python training. Matching a seed does not reproduce the original Python trajectory.

The recorded offline browser check opened the file with networking disabled, observed **zero network requests and page errors**, and verified real updates, checkpoint download/import, result export, and mobile layout. The included model reached 10/10 goals in one fixed JavaScript reef evaluation. See [browser validation and its limits](docs/BROWSER.md#recorded-browser-validation) and [the recorded JSON report](docs/browser-validation.json); this is a small diagnostic, not a general navigation guarantee.

## Python implementation (optional)

This path retains the original Python/PyTorch simulator, headless commands, and local server. It is separate from opening the standalone HTML.

### Start on Windows

Install **Python 3.11 or 3.12** (3.12 was used for validation), then clone or download this repository.

**Double-click `start-windows.cmd`.** It creates a project-local virtual environment, installs CPU-only PyTorch and the app, and starts the server. The first launch needs internet access for those packages. Open **http://127.0.0.1:8000** in your browser and leave the terminal open.

Or use PowerShell from the repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m auv_sim
```

You do not need to activate the virtual environment or change PowerShell execution policy. Opening `auv_sim/static/index.html` directly does not start the Python simulator; use the root-level `auv-practice.html` for the standalone application.

### Start on macOS or Linux

```bash
python3 -m venv .venv
# Linux CPU installation; macOS users should run: .venv/bin/python -m pip install torch
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
.venv/bin/python -m auv_sim
```

Open **http://127.0.0.1:8000**. The app needs no API keys, external AI services, Node installation, or GPU. It runs locally after package installation. If port 8000 is occupied, use `python -m auv_sim serve --port 8001` with your environment's Python executable.

### Your first Python experiment

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

### Python learner and checkpoints

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

### Headless Python training and evaluation

After installation, using the virtual environment's Python:

```bash
python -m auv_sim train --scenario reef --seed 7 --steps 20000 --eval-episodes 10
python -m auv_sim evaluate --controller baseline --scenario reef --episodes 20
python -m auv_sim --help
```

Training prints the resulting checkpoint filename. To evaluate a saved model, supply that filename to `python -m auv_sim evaluate --checkpoint NAME.pt`. Checkpoint names must be files in the selected `--data-dir` (default `runs`). Loading uses the checkpoint's scenario, seed, and current configuration. `train --load NAME.pt --steps 20000` continues from a saved learner with fresh replay.

The CLI saves `runs/latest-results.json` with configuration, measured metrics, recent episode history, and evaluation results. The Python server exposes current metrics at `/api/results`. In the standalone app, use **Export results**; its `/api/...` names are internal worker messages, not HTTP requests. `runs/` is ignored by Git.

## Source map

| File | Responsibility |
| --- | --- |
| `auv-practice.html` | Complete offline application, including the trained example |
| `web_src/environment.js` | JavaScript motion, courses, sonar, geometry, rewards, and baseline |
| `web_src/td3.js` | JavaScript networks, gradients, Adam, replay, TD3 updates, and serialization |
| `web_src/worker.js` | Browser preview, training, staged walkthrough execution, evaluation, and in-tab checkpoints |
| `web_src/bridge.js` | Embedded Blob Worker, message handling, and JSON downloads |
| `web_src/ui.html`, `web_src/ui.js` | Offline template, controls, canvas, charts, and telemetry |
| `web_src/trace_ui.js`, `web_src/trace.css` | Walkthrough controls, live values, numbered source, and responsive layout |
| `scripts/build_html.py` | Model export, walkthrough source extraction, and assembly of the single HTML file |
| `auv_sim/environment.py` | Motion, seeded courses, ray observations, swept collision tests, rewards, baseline |
| `auv_sim/td3.py` | Neural networks, replay, TD3 update, learner checkpoint state |
| `auv_sim/experiment.py` | Training/preview/evaluation lifecycles and synchronization |
| `auv_sim/server.py` | Validated local API and static interface hosting |
| `auv_sim/static/` | Python frontend; its stylesheet is also embedded in the offline HTML |
| `auv_sim/__main__.py` | Browser-server and headless commands |
| `tests/` | JavaScript and Python algorithm, environment, checkpoint, and lifecycle tests |

See [the browser guide](docs/BROWSER.md), [simulation specification](docs/SIMULATION.md), and [Python development notes](docs/DEVELOPMENT.md) for behavior, assumptions, and extension points.

## Rebuild the single file

From the repository root, using the project's Python/PyTorch environment:

```bash
python scripts/build_html.py
```

The builder uses Python, NumPy, and PyTorch to export the bundled model and inline the source, CSS, and model into `auv-practice.html`. There is no Node/npm bundler or CDN step. These dependencies are **not needed to open the generated HTML**. See [browser source and rebuilding](docs/BROWSER.md#source-and-rebuilding) for details.

## Run the tests

JavaScript tests use Node's built-in runner:

```bash
node tests/browser_environment.cjs
node tests/browser_td3.cjs
node tests/browser_parity.cjs
node tests/browser_lifecycle.cjs
node tests/browser_trace.cjs
```

Node is needed only for these development tests, not to open or train in the HTML. The optional browser automation scripts `scripts/check_browser.cjs` and `scripts/check_trace_browser.cjs` additionally use Playwright. See the [browser guide](docs/BROWSER.md#tests) for their scope.

For the Python implementation:

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
