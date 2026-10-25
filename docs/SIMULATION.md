# Simulation specification

## Purpose and scope

This environment makes the TD3 data flow observable: state → continuous action → transition/reward → replay → critic update → delayed actor update. It is a 2D navigation exercise with idealized range observations and known current. It is not validated against a physical vehicle.

## World and vehicle

- World: 20 × 20 m, with bounded walls and circular static obstacles.
- Vehicle hull: circle of radius 0.25 m.
- Goal region: radius 0.75 m.
- Control interval: 0.1 s; external episode limit: 500 steps.
- Through-water forward speed: 0 to 1.8 m/s; no reverse motion.
- Forward acceleration scale: 0.9 m/s²; linear drag coefficient: 0.3 s⁻¹.
- Yaw-rate scale: 1.2 rad/s, positive counterclockwise.
- Uniform current magnitude: configurable from 0 to 0.4 m/s.

The default course contains modest seeded variations in start, heading, goal, obstacle positions, and current direction. Reusing a seed reproduces a reset. Unseeded subsequent resets continue that environment's random stream. The optional `course_variety=False` setting disables those course variations.

For command `(u_accel, u_turn)`, both components are clipped to `[-1, 1]`. Speed integrates acceleration and drag; heading integrates yaw rate. Position then advances using through-water velocity plus current. The vehicle's negative acceleration command brakes but does not drive backward. There is no vertical axis, buoyancy model, thruster allocation, or coupled six-degree-of-freedom dynamics.

Collision uses the entire movement segment against circles expanded by the hull radius and against hull-safe wall boundaries. A long integration step cannot simply skip through an obstacle. The first contact or goal-entry event clips the position along that segment. Speed and heading use the full discrete integration interval. Physical success/collision takes precedence over the external time limit.

## Observation vector: 21 float32 values

| Indices | Meaning | Scaling |
| --- | --- | --- |
| `0:16` | 16 idealized range rays, beginning forward and rotating counterclockwise | Range / 6 m, clipped to `[0,1]` |
| `16` | Goal displacement forward in the body frame | Divide by world diagonal |
| `17` | Goal displacement left in the body frame | Divide by world diagonal |
| `18` | Forward through-water speed | Divide by 1.8 m/s |
| `19` | Current component forward in the body frame | Divide by 0.4 m/s |
| `20` | Current component left in the body frame | Divide by 0.4 m/s |

The policy receives those values, not obstacle coordinates or an A* route. Range rays do not simulate acoustic propagation, reflections, noise, occlusion processing, or sensor latency. The current estimate is supplied exactly. Hidden map structure can make local observations partially informative despite the simulator itself knowing the full world.

## Reward

Each step exposes a dictionary whose signed components sum to the scalar reward:

- `progress`: `6 × (previous goal distance − new goal distance)`.
- `goal_bonus`: `+100` on success.
- `collision_penalty`: `−100` on collision.
- `time_cost`: `−0.02 × elapsed_scale`.
- `control_cost`: `−0.01 × elapsed_scale × (u_accel² + 0.3 × u_turn²)`.
- `proximity_cost`: `−0.12 × elapsed_scale × max(0, (0.75 − clearance)/0.75)²`.

Here `elapsed_scale = dt / 0.1`: per-step costs use the configured integration interval, even when a terminal event clips the movement segment. This is a simple discrete-step convention, not a partial-event-time dynamics solver. `clearance` is the current distance from the vehicle hull to the nearest obstacle or wall. It is not the historical minimum over the whole episode.

Progress reward encourages movement toward the goal; collision penalties and clearance cost favor avoiding obstacles; time cost discourages indefinite waiting. The learned policy can still fail. Reward design favors behavior statistically through learning and does not impose an execution-time collision veto.

## Episodes and learning targets

Success and collision set `terminated=True`. The external 500-step cap sets `truncated=True` if no physical terminal event occurs. Both end the rollout episode and require reset. Replay stores the real final observation before that reset. Only `terminated` disables the bootstrap term.

The default TD3 update is:

```
next_action = clip(target_actor(next_obs) + clipped_noise, -1, 1)
target = reward + 0.99 * (1 - terminated) * min(target_Q1, target_Q2)
critic_loss = mean((Q1 - target)^2) + mean((Q2 - target)^2)
actor_loss = -mean(Q1(obs, actor(obs)))  # every second update
target_weights = 0.005 * online_weights + 0.995 * target_weights
```

Target noise has standard deviation 0.2 and is clipped to ±0.5. Exploration noise defaults to 0.15 in the collection loop. These are separate mechanisms.

## Evaluation and baseline

Evaluation seeds begin at experiment seed + 10,000 and increase once per episode. TD3 evaluation uses no exploration, no replay writes, and no optimizer updates. Results contain success, collision and timeout rates, mean return, and mean episode length.

The map-aware baseline computes a coarse A* path using the complete obstacle map and follows it with current-compensating steering. It makes the simulator immediately explorable. Its map access exceeds the TD3 observation, so its score is not an equal-information comparison.

An evaluation on ten seeds is a small diagnostic, not a broad generalization claim. New current distributions, sensors, maps, moving obstacles, and vehicle dynamics require their own measurements.
