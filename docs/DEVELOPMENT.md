# Development and extension notes

## Architecture

The server owns one `Experiment`. A background worker runs either preview, training, or evaluation. A command lock serializes mutations, and a state lock protects the policy/environment during updates and snapshots. Pause signals and joins the existing worker before starting a different operation. Status polling only reads snapshots.

Preview never writes to replay. Training stores `(obs, action, reward, next_obs, terminated)` and uses the current actor plus collection noise after warm-up. Evaluation has an independent environment and random stream; it preserves the trained model and buffer.

The frontend is plain HTML, CSS, and JavaScript served by FastAPI. It has no package build step or third-party CDN. The server binds to `127.0.0.1` by default and is intended for one local user. This MVP has no multi-user isolation or authentication; it is not an internet hosting configuration.

## API

| Method/path | Purpose |
| --- | --- |
| `GET /health` | Process health |
| `GET /api/state` | Simulation snapshot, metrics, history, saved checkpoint list |
| `POST /api/reset` | New policy and experiment: `scenario`, `seed`, `current_strength` |
| `POST /api/control` | `command` (`play`, `pause`, `step`), `controller`, `speed` |
| `POST /api/train` | Additional `steps` and `exploration_noise` |
| `POST /api/evaluate` | `episodes` and `controller` |
| `POST /api/save` | Save the current learner and metadata; pauses work |
| `POST /api/load` | Load a listed checkpoint by `name` |
| `GET /api/results` | Exportable current metrics/history/evaluation JSON |

Generated request documentation is available at `/docs`. Inputs reject unsupported fields, nonfinite numbers, and out-of-range budgets. Checkpoint loading accepts listed-style local filenames, not arbitrary paths.

`training.steps` is cumulative TD3 environment interactions. `training.total_steps` is the cumulative target of the requested training budget. `training.updates` counts critic optimizer steps; `actor_updates` counts actor optimizer steps. Batch Q metrics are pre-update sampled means. `td_error` is the mean of the two critics' absolute TD errors; a large value is prediction disagreement, not a collision probability.

## Validation

Install development dependencies with `python -m pip install -e ".[dev]"` and run `python -m pytest`.

Tests target meaningful failure modes: shapes and termination masks, real weight changes, actor/target delay, target-action bounds, save/load predictions and optimizer continuity, geometric tunneling, seed reproducibility, Gymnasium environment conformance, lifecycle operations, and API validation.

Training quality is a separate experiment. Record the configuration and evaluate deterministic policies on multiple seeds. Do not treat a passing unit test or a decreasing loss as evidence that navigation is solved.

## Suggested extension sequence

1. **Experiment tracking:** retain a comparison table across several seeds and fixed evaluation suites; add CSV downloads and trajectory replays.
2. **Observation realism:** introduce explicit sensor-noise/latency settings, matching estimator assumptions, and additional diagnostics.
3. **Prioritized replay:** return indices and weights, apply per-sample weighted critic loss, and define how two TD errors produce priorities. Retain uniform replay as a control.
4. **N-step learning:** collect episode-bounded contiguous segments, actual segment lengths, and discounted reward sums. Keep final observations at truncations and address off-policy mismatch explicitly.
5. **Paper-specific methods:** reproduce the actual C-APF/adaptive constraint/noise definitions from the studies; distinguish reported methods from new design choices.
6. **3D and dynamics:** add depth, pitch and appropriate motion/actuator models only after the simpler task has stable measurements.

Change one mechanism at a time and keep identical evaluation conditions for comparisons.
