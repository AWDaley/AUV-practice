# Recorded validation

Prepared 25 October 2026. These results come from executing this project's Python implementation, not from the research papers.

## Automated checks

**59 automated tests passed.** The suite covers the Gymnasium environment contract, seeded observations, swept hull collisions and wall/goal event ordering, true termination versus time-limit truncation, bounded actions, real optimizer updates, delayed actor/target behavior, exact numerical targets, checkpoint restoration, experiment lifecycle, and API validation.

Browser checks exercised Play, pause, single-step, form-value persistence during polling, 1,200 real training interactions, checkpoint save/load, evaluation without learning, and experiment reset. The loaded example reached the goal in all five episodes of that browser-triggered evaluation. No JavaScript runtime errors occurred. Desktop and 390-pixel mobile screenshots were reviewed; no horizontal overflow or clipped controls were found.

The validation environment used Python 3.12.14, CPU PyTorch 2.14.1, NumPy 2.5.4, Gymnasium 1.4.0, and FastAPI 0.162.0. Tests use temporary run directories and small networks where appropriate; navigation performance below uses the default 128-unit networks.

## Measured TD3 experiment

Configuration: reef scenario, seed 7, current strength 0.08 m/s, default course variation, 128-unit hidden layers, batch size 128, 100,000-record replay capacity, 1,000 initial random steps, exploration standard deviation 0.15, discount 0.99, tau 0.005, and delayed policy updates every two critic updates.

The first run trained for 20,000 interactions, then evaluated and saved. A new process loaded that checkpoint and trained for another 30,000 interactions. As documented, loading starts with an empty replay buffer, so this is a **20k + resumed 30k experiment**, not an uninterrupted 50k run. The cumulative update count at the end is 48,873 critic updates and 24,436 actor updates.

Both evaluations used 10 deterministic TD3 episodes without exploration, reset seeds 10007 through 10016, and no optimizer updates:

| Cumulative training steps | Goal arrivals | Collisions | Timeouts | Mean return | Mean episode steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 0 / 10 | 1 / 10 | 9 / 10 | 52.7930 | 484.6 |
| 50,000 (after resume) | 10 / 10 | 0 / 10 | 0 / 10 | 214.5729 | 144.8 |

The final checkpoint is included at `auv_sim/assets/td3-demo-reef-50000.pt`. On first use of a run directory it is copied into that directory alongside its metadata; it is offered in the checkpoint menu. This does not replace the freshly initialized actor until you choose Load. The model was trained with actual TD3 updates, without using the map-aware baseline to generate its actions or rewards.

Detailed final results, including all 10 evaluation episodes and the training history retained by the app, are in [validation-results.json](validation-results.json).

## What the result supports

The included model reaches the goal on this small fixed evaluation set, and the training/evaluation/checkpoint pipeline executes successfully. It does not establish performance over multiple training seeds, unseen map families, different current distributions, moving obstacles, noisy sensors, 3D dynamics, or real vehicles. A 10,000-step practice budget can be useful for watching learning but should not be expected to reproduce the final result.

The map-aware A* baseline has a different information advantage and is kept separate from these learned-policy measurements.
