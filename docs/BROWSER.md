# Single-file browser guide

## Open and use the app

Download [auv-practice.html](../auv-practice.html) and open the saved file in a browser. That one file contains the interface, styling, simulator, JavaScript TD3 learner, and trained example. Once downloaded, it needs **no package installation, Python, Node, web server, API key, or network connection**.

Use a browser that supports JavaScript, canvas, Web Workers created from Blob URLs, and local downloads. Open the actual HTML in a browser tab; a source-code viewer or embedded document preview may not execute it.

When **Browser engine ready** appears, the included TD3 reef example is loaded and paused. Press **Play** to run it. You can train further, evaluate, download model JSON, or reset to learn from scratch. All simulation and learning happen on your device.

The default model was trained with real TD3 updates in the Python implementation: 20,000 interactions, followed by loading that checkpoint and training another 30,000 interactions with fresh replay. Its learned parameters and retained history are embedded in the HTML. The initial 50,000-step count and charts describe that prior training; the browser has not just performed those interactions. See [Python validation](VALIDATION.md) for the original experiment and [browser validation](#recorded-browser-validation) below for the separate JavaScript measurements.

## Controls and experiment lifecycle

| Control | Behavior |
| --- | --- |
| **Play** | Runs the selected preview controller, without training or replay writes. |
| **Pause** | Stops preview, training, or evaluation; current weights and replay remain in the tab. |
| **Single step** | Advances one preview transition using the selected controller. |
| **Preview controller** | Selects TD3, the map-aware baseline, or random actions for preview and evaluation. |
| **Speed** | Changes preview playback speed, not the training budget or integration interval. |
| **Sonar / Path / Current** | Toggles visual overlays without changing policy observations or physics. |
| **Environment / seed / current** | Stages settings that take effect when you click Reset experiment. |
| **Reset experiment** | Creates a fresh policy, replay buffer, environment, and metrics using those settings. Saved entries remain in the tab's checkpoint list. |
| **Train policy / Continue training** | Adds the requested number of environment interactions to the current learner. |
| **Exploration noise** | Sets noise added to collected training actions after warm-up. Preview and TD3 evaluation omit it. |
| **Evaluate** | Tests the selected controller on fixed JavaScript seeds in a separate environment, without model updates or replay writes. |
| **Save & download policy** | Adds a checkpoint to this tab and downloads its JSON. |
| **Load** | Restores a listed model/settings and starts fresh replay and environment state. |
| **Import checkpoint** | Reads browser checkpoint JSON, adds it to the list, and immediately loads it. |
| **Export results** | Downloads `auv-results.json` with settings, learner configuration, metrics, retained history, and evaluation. It is not a model checkpoint. |

Save, load, and import controls require a paused run. To continue learning after a pause, use **Continue training** with an additional budget. **Play** starts preview.

The map-aware baseline uses a hand-written A* route and steering rules with the complete obstacle map. TD3 observes local range rays and its supplied goal, speed, and current features. The baseline has an information advantage and is intended as a demonstration/reference, not an equal-information policy benchmark.

### A first learning run

1. Choose Open water, Reef field, or Narrow channel, set the seed and current, and click **Reset experiment**. This replaces the loaded model with a fresh learner.
2. Select TD3 and preview or single-step to inspect its initial behavior. Preview does not improve the weights.
3. Request the default **2,000 additional steps** as a short practice run. The first 1,000 cumulative training steps collect random actions. After warm-up, the actor supplies commands with exploration noise and learning samples the replay buffer.
4. Inspect actual returns, outcomes, and update counters. Critic loss is an optimization diagnostic, not a navigation success measure.
5. Pause, evaluate TD3, and export results. Add another training budget and repeat the same evaluation procedure if desired.
6. Download a checkpoint to keep the model beyond this tab's lifetime.

A short run is not expected to reproduce the included model's history. Speed depends on processor performance, browser scheduling, network width, and batch size. Background tabs may run more slowly. Reset creates the smaller browser model described below and discards the active learned weights unless you saved them.

## What learns in the browser

The learner in [web_src/td3.js](../web_src/td3.js) uses typed arrays, explicit neural-network derivatives, Adam optimizer state, and uniform replay. Simulation, inference, replay, and gradient updates run in a **Blob Web Worker** created from code embedded in the HTML. The main page displays snapshots and sends commands.

Names such as `/api/train` are internal message identifiers passed through `window.auvRequest`; they make no HTTP requests. There is no external model service or server process.

### Model settings

| Setting | Fresh model after Reset | Included 50,000-step model |
| --- | --- | --- |
| Actor hidden layers | 64 → 64, ReLU | 128 → 128, ReLU |
| Hidden layers in each critic | 64 → 64, ReLU | 128 → 128, ReLU |
| Output actions | Two tanh components in `[-1, 1]` | Same |
| Browser training batch | 32 | 32 for continued browser training |
| Replay capacity | 50,000 transitions | 50,000 transitions in the browser |
| Replay after reset/load | Empty | Empty |
| Initial random warm-up | First 1,000 cumulative environment steps | Already completed in the imported history |

The included model was originally trained in Python with 128-unit hidden layers, batch size 128, and replay capacity 100,000. Export preserves its width and learned parameters while setting browser batch size to 32 and replay capacity to 50,000. Continued browser training is a new continuation with different sampling conditions.

Both browser models use discount `0.99`, Adam learning rate `3e-4`, target interpolation `tau=0.005`, and actor/target updates every second critic update. Target-action noise has standard deviation `0.2` and is clipped to `±0.5` before action clipping. This target smoothing is separate from the collection exploration noise in the interface.

The environment supplies 21 float32 observations: 16 range rays, two body-frame goal offsets, speed, and two body-frame current components. Action 1 requests acceleration; negative values brake without reversing. Action 2 requests yaw rate; positive values turn counterclockwise. See the [simulation specification](SIMULATION.md) for units, motion, rewards, and termination rules.

Success and collision disable bootstrapping. The external time limit ends the rollout but retains the bootstrap from the real final observation. TD3 targets use the smaller target-critic estimate; the actor's objective uses Q1. Reward penalties do not supply a runtime collision veto.

### Reading the inspector

- **Episode return, distance, speed, clearance, and reward terms** describe the displayed environment. Clearance is current hull clearance, not the episode's historical minimum.
- **Training steps** are cumulative training interactions. The denominator is the cumulative target after adding the requested budget.
- **Critic/actor updates** count optimizer updates. They differ from interaction counts; actor updates are deliberately less frequent.
- **Batch Q1/Q2, TD error, and losses** describe the latest sampled training batch and may remain visible during preview.
- **History charts** show completed TD3 training episodes, including history loaded with a checkpoint. Preview/evaluation episodes are separate. Up to 500 records are retained; charts plot up to the latest 240.
- **Evaluation** starts at experiment seed + 10,000 and increases the seed once per episode. Its result describes that configured test set, not general safety or reliability.

## Save, import, and session lifetime

Pause and click **Save & download policy**, then keep the resulting `td3-....json` file. The same checkpoint appears in the current tab's list. On a later opening, click **Import checkpoint** and choose that JSON; import validates and immediately loads it.

The list is held in memory. **Closing or reloading loses unsaved learning and tab-only checkpoint entries.** The app does not persist them through localStorage, IndexedDB, or another browser store. A download preserves a separate JSON file; it does not rewrite the original HTML.

The included `td3-demo-reef-50000.json` is embedded, so it returns whenever you reopen the original HTML. Reset within the same tab also leaves it available in the list. Load it to return to the example policy.

Import accepts this app's version-1 browser checkpoint format, with a maximum size of **8 MiB (8,388,608 bytes)**. **Python `.pt` files cannot be imported through this control.** The build script converted the bundled example; the browser does not contain PyTorch or a general `.pt` reader.

### What is restored

Browser JSON includes the actor, both critics, target networks, optimizer moments/counters, learner noise RNG state, learner configuration, experiment settings, cumulative counters, and retained metrics/history. It can contain an evaluation record for inspection, but loading starts with no active evaluation result. Run Evaluate again for a new measurement.

It omits replay, the live simulator/episode state, and rollout random streams. Loading starts a new environment and empty replay. A model beyond warm-up collects with its actor plus exploration noise and resumes optimizer updates once a fresh batch is available. Its cumulative counters remain, but the original running experiment is not reconstructed.

**Matching seeds does not make browser and Python runs identical.** JavaScript generators differ from NumPy's, the converted example begins with a JavaScript-specific learner RNG state, and browser continuation uses batch size 32. The same numeric seed can produce different randomized courses and commands across implementations. Browser checkpoints retain their own learner noise RNG, but restarted replay sampling and rollout state still prevent exact resumption.

### Export a measurement

**Export results** downloads `auv-results.json` containing the active settings, learner configuration, training metrics/history, and any completed evaluation, with the engine identified as JavaScript. It contains no model tensors and cannot be imported as a checkpoint. Keep a policy JSON and results JSON together when you need both a model and its measurement record.

## Recorded browser validation

The [recorded browser report](browser-validation.json) comes from opening the generated HTML through **`file://` with networking disabled**, using Chromium `134.0.6998.35`. It recorded **zero network requests and zero page errors**. Controls, checkpoint download/import, results export, and invalid-import preservation were exercised. A 390-pixel mobile viewport had a 390-pixel document width, with no horizontal overflow.

A fresh run completed **1,200 training interactions**, producing **200 critic updates and 100 actor updates** after the 1,000-step warm-up. Pause responded in **98 ms** in that recorded run. Those timings describe the tested environment and are not a performance promise for other devices.

The included 50,000-step model was evaluated in the JavaScript reef environment with current strength **0.08 m/s**, no exploration noise, and reset seeds **10007–10016**:

| Episodes | Reached goal | Collisions | Timeouts | Mean return | Mean episode steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 10 / 10 | 0 / 10 | 0 / 10 | 213.5720 | 144.8 |

This small fixed suite supports the stated result for that model and configuration. It does not establish performance across new map families, multiple training seeds, changed sensors/currents, 3D dynamics, or physical vehicles. The Python evaluation is a separate result with a different random sequence.

The JavaScript verification suite passed **77 tests**: 32 environment, 12 TD3, 26 cross-language parity, and 7 lifecycle checks. Across 12 fixed comparison states and 36 transitions, the maximum actor-output discrepancy was approximately **3.58 × 10⁻⁷** and the maximum Q discrepancy approximately **5.70 × 10⁻⁶**. These comparisons check numerical implementation on supplied states; they do not make JavaScript and NumPy seeded rollouts identical. The comparison fixtures are in [tests/fixtures/browser-parity.json](../tests/fixtures/browser-parity.json).

## Source and rebuilding

Opening the distributed HTML needs no development tools. Edit the source and rebuild when changing the program:

| Source | Purpose |
| --- | --- |
| [web_src/ui.html](../web_src/ui.html) | Page template and inline insertion markers |
| [web_src/ui.js](../web_src/ui.js) | Controls, charts, canvas, and command dispatch |
| [auv_sim/static/styles.css](../auv_sim/static/styles.css) | Shared responsive styling embedded by the builder |
| [web_src/environment.js](../web_src/environment.js) | Seeded environment, observations, geometry, and baseline |
| [web_src/td3.js](../web_src/td3.js) | Networks, gradients, Adam, replay, TD3, and checkpoint encoding |
| [web_src/worker.js](../web_src/worker.js) | Preview/training/evaluation lifecycle and in-tab checkpoints |
| [web_src/bridge.js](../web_src/bridge.js) | Blob Worker startup, message promises, and downloads |
| [scripts/build_html.py](../scripts/build_html.py) | Model conversion and complete HTML assembly |

From the repository root, using the project's Python/PyTorch environment:

```bash
python scripts/build_html.py
```

The normal builder imports **NumPy and PyTorch**, reads `auv_sim/assets/td3-demo-reef-50000.pt`, exports numerical tensors and optimizer state, and inlines everything into `auv-practice.html`. It needs no Node/npm build chain or CDN dependency. These build dependencies are not needed to open the result. Editing source alone does not update an existing HTML; rebuild and reopen it.

To regenerate the Python/JavaScript comparison fixtures as well:

```bash
python scripts/build_html.py --fixtures
```

Fixture generation additionally imports the Python environment and learner, so it uses the installed project's Python dependencies. The normal single-file build does not start a server.

### Tests

JavaScript tests use Node's built-in runner without npm packages:

```bash
node tests/browser_environment.cjs
node tests/browser_td3.cjs
node tests/browser_parity.cjs
node tests/browser_lifecycle.cjs
```

The suite checks gradients, Adam, real updates, actor/target delay, replay masks, serialization, seeded geometry, sonar, collision events, numerical parity, and lifecycle behavior. CI also checks the JavaScript tests and HTML rebuilding.

The optional [scripts/check_browser.cjs](../scripts/check_browser.cjs) exercises the actual HTML with Playwright; Playwright is required only for that development check. It tests offline loading and user interactions and records browser results. **Neither Node nor Playwright is needed to open or train in the HTML app.** Python workflows remain documented in the [README](../README.md#python-implementation-optional) and [development notes](DEVELOPMENT.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Source text or inactive controls | Save the actual `.html` and open it in a browser tab that permits scripts and Blob Workers. |
| Engine cannot start | Read the visible error and check whether the browser/viewer permits Blob Web Workers. Use a normal browser tab for the saved file. |
| Training is slow | Use a smaller budget, keep the tab active, or reset for the fresh 64-unit model. Save the active learned model first if needed. |
| Edited scene/seed/current has no effect | Click Reset experiment to apply staged settings; this also replaces the learner. |
| Reload shows the original example | Reopening restores the embedded model. Import your downloaded checkpoint to restore a separately saved policy. |
| Import rejected | Use browser checkpoint JSON, not results JSON or Python `.pt`, within 8 MiB. The error identifies invalid format/tensor/configuration data. |
| Download is missing | Check the browser's download location and whether it blocked the download. A tab-only entry is not a saved file. |
| Evaluation differs from Python | Random sequences and continued-training batches differ. Compare measurements with their engine and configuration stated. |

The environment remains an idealized 2D teaching model. It does not simulate realistic sonar propagation, noisy state estimation, moving obstacles, 3D motion, or validated vehicle hydrodynamics. Inspectable learning experiments do not establish readiness for deployment on an AUV.
