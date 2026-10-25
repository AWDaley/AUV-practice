"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {AUVEnv, SeededRNG} = require("../web_src/environment.js");

function makeEnv(config = {}) {
  return new AUVEnv({scenario: "open", current_strength: 0, course_variety: false, ...config});
}
function close(actual, expected, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`);
}
function checkObservation(observation) {
  assert.ok(observation instanceof Float32Array);
  assert.equal(observation.length, 21);
  for (let i = 0; i < 21; i++) {
    assert.ok(Number.isFinite(observation[i]), `non-finite observation at ${i}`);
    assert.ok(observation[i] >= (i < 16 || i === 18 ? 0 : -1) && observation[i] <= 1,
      `out-of-bounds observation at ${i}: ${observation[i]}`);
  }
}
function checkFiniteJSON(value) {
  if (typeof value === "number") assert.ok(Number.isFinite(value));
  else if (Array.isArray(value)) value.forEach(checkFiniteJSON);
  else if (value && typeof value === "object") Object.values(value).forEach(checkFiniteJSON);
}
function horizontalMotion(config = {}) {
  const env = makeEnv({dt: 2, ...config});
  env.position = [2, 5]; env.goal = [18, 5];
  env.heading = 0; env.speed = 1.8;
  return env;
}

test("module works as a browser global and a CommonJS export", () => {
  assert.equal(globalThis.AUVEnvironment.AUVEnv, AUVEnv);
  assert.equal(AUVEnv.OBSERVATION_DIM, 21);
  assert.equal(makeEnv().MAX_YAW_RATE, 1.2);
});

test("RNG is deterministic and round-trips mixed uniform/normal/integer state", () => {
  const a = new SeededRNG(2718), b = new SeededRNG(2718);
  for (let i = 0; i < 20; i++) {
    assert.equal(a.uniform(-2, 5), b.uniform(-2, 5));
    assert.equal(a.normal(), b.normal());
    assert.equal(a.integer(100), b.integer(100));
  }
  a.normal(); // Checkpoint a cached Box-Muller value as well as the integer state.
  const state = JSON.parse(JSON.stringify(a.getState()));
  const restored = new SeededRNG(0).setState(state);
  for (let i = 0; i < 100; i++) {
    assert.equal(a.normal(), restored.normal());
    assert.equal(a.uniform(), restored.uniform());
    assert.equal(a.integer(7), restored.integer(7));
  }
  assert.throws(() => restored.setState({...state, state: -1}));
  assert.throws(() => restored.setState({...state, normal_spare: NaN}));
  assert.throws(() => a.integer(0));
  assert.throws(() => a.uniform(2, 1));
});

test("seeded resets and complete transition results are reproducible", () => {
  const a = new AUVEnv({current_strength: 0.2});
  const b = new AUVEnv({current_strength: 0.2});
  const first = a.reset(123);
  assert.deepEqual(first, b.reset(123));
  assert.deepEqual(a.snapshot(), b.snapshot());
  for (const action of [[0.8, -0.1], [0.9, 0.2], [-0.4, 0.8]]) {
    assert.deepEqual(a.step(action), b.step(action));
  }
  assert.deepEqual(a.reset(), b.reset());
  assert.notDeepEqual(a._observation(), first.observation);
  assert.deepEqual(a.reset(123), first);
  assert.notDeepEqual(a.reset(124), first);
});

test("course variation stays bounded and reset option can disable it", () => {
  const env = new AUVEnv({current_strength: 0.4});
  for (let seed = 0; seed < 10; seed++) {
    env.reset(seed);
    assert.ok(Math.abs(env.position[0] - 2) <= 0.25);
    assert.ok(Math.abs(env.position[1] - 3) <= 0.25);
    assert.ok(Math.abs(env.goal[0] - 17) <= 0.35);
    close(Math.hypot(...env.current), 0.4);
  }
  env.reset(999, {course_variety: false});
  assert.deepEqual(env.position, [2, 3]);
  assert.deepEqual(env.goal, [17, 17]);
  assert.deepEqual(env.current, [0.4, 0]);
  assert.deepEqual(env.obstacles[0], [7, 7.7, 1]);
});

test("sweep catches a small obstacle when the proposed endpoint is beyond it", () => {
  const env = horizontalMotion();
  env.obstacles = [[3.5, 5, 0.35]];
  const result = env.step(new Float32Array([1, 0]));
  assert.equal(result.terminated, true);
  assert.equal(result.truncated, false);
  assert.equal(result.info.outcome, "collision");
  close(env.position[0], 2.9);
  close(result.info.min_clearance, 0);
  checkObservation(result.observation);
});

test("sweep treats tangent contact as collision", () => {
  const env = horizontalMotion();
  env.obstacles = [[3.5, 5.6, 0.35]];
  const result = env.step([1, 0]);
  assert.equal(result.info.outcome, "collision");
  close(env.position[0], 3.5, 1e-6);
});

for (const [position, heading, axis, contact] of [
  [[19, 10], 0, 0, 19.75], [[1, 10], Math.PI, 0, 0.25],
  [[10, 19], Math.PI / 2, 1, 19.75], [[10, 1], -Math.PI / 2, 1, 0.25],
]) {
  test(`wall contact preserves the full hull: ${position}, heading ${heading}`, () => {
    const env = makeEnv({dt: 1});
    env.position = position.slice(); env.heading = heading; env.speed = 1.8;
    const result = env.step([1, 0]);
    assert.equal(result.info.outcome, "collision");
    assert.equal(result.terminated, true);
    assert.equal(result.truncated, false);
    close(env.position[axis], contact);
    assert.ok(env.position.every(v => v >= 0.25 && v <= 19.75));
  });
}

test("first physical event wins, with collision winning a simultaneous goal entry", () => {
  let env = horizontalMotion({dt: 4});
  env.goal = [4, 5]; env.obstacles = [[7, 5, 0.4]];
  assert.equal(env.step([1, 0]).info.outcome, "success");
  close(env.position[0], 3.25);

  env = horizontalMotion({dt: 4});
  env.goal = [7, 5]; env.obstacles = [[4, 5, 0.4]];
  assert.equal(env.step([1, 0]).info.outcome, "collision");
  close(env.position[0], 3.35);

  env = horizontalMotion();
  env.goal = [3.65, 5]; env.obstacles = [[3.5, 5, 0.35]];
  assert.equal(env.step([1, 0]).info.outcome, "collision");
  close(env.position[0], 2.9);
});

test("time limit truncates, while physical events at the horizon terminate", () => {
  const env = makeEnv({max_steps: 1});
  let result = env.step([0, 0]);
  assert.equal(result.truncated, true); assert.equal(result.terminated, false);
  assert.equal(result.info.outcome, "timeout");
  assert.throws(() => env.step([0, 0]), /call reset/);
  env.reset();
  env.goal = [env.position[0] + 0.5, env.position[1]];
  result = env.step([0, 0]);
  assert.equal(result.terminated, true); assert.equal(result.truncated, false);
  assert.equal(result.info.outcome, "success");
  assert.equal(result.info.reward_terms.goal_bonus, 100);

  const collisionEnv = horizontalMotion({max_steps: 1});
  collisionEnv.obstacles = [[3.5, 5, 0.35]];
  result = collisionEnv.step([1, 0]);
  assert.equal(result.terminated, true); assert.equal(result.truncated, false);
});

test("terminal contact clips position while dynamics and costs retain full dt", () => {
  const env = makeEnv({dt: 1});
  env.position = [19, 10]; env.heading = 0; env.speed = 1;
  const result = env.step([1, 0.2]);
  assert.equal(result.info.outcome, "collision");
  close(env.position[0], 19.75);
  close(env.speed, 1.6);
  close(env.heading, 0.24);
  close(result.info.reward_terms.time_cost, -0.2);
  close(result.info.reward_terms.control_cost, -0.1 * (1 + 0.3 * 0.2 ** 2));
  close(result.reward, Object.values(result.info.reward_terms).reduce((sum, v) => sum + v, 0));
  close(env.episode_return, result.reward);
});

test("body-frame observation geometry and sensor center origin", () => {
  const env = makeEnv();
  env.position = [10, 10]; env.heading = Math.PI / 2;
  env.goal = [10, 14]; env.current = [0.1, 0.2]; env.speed = 0.9;
  env.obstacles = [[10, 13, 0.5]];
  const observation = env._observation();
  close(observation[0], 2.5 / 6, 1e-7);
  close(observation[16], 4 / Math.hypot(20, 20), 1e-7);
  close(observation[17], 0, 1e-7);
  close(observation[18], 0.5);
  close(observation[19], 0.5);
  close(observation[20], -0.25);
  const ray = env.snapshot().sonar[0];
  close(ray.angle, Math.PI / 2);
  close(ray.end_x, 10); close(ray.end_y, 12.5);
  env.obstacles = []; env.position = [2, 3]; env.heading = Math.PI;
  close(env.snapshot().sonar[0].distance, 2);
  close(env._min_clearance(), 1.75);
});

test("sonar chooses the nearest forward hit and ignores circles behind a ray", () => {
  const env = makeEnv();
  env.position = [10, 10]; env.heading = 0;
  env.obstacles = [[15, 10, 1], [13, 10, 0.5], [8, 10, 0.5]];
  const sonar = env.snapshot().sonar;
  close(sonar[0].distance, 2.5);
  close(sonar[8].distance, 1.5);
  close(sonar[4].distance, 6);
});

for (const scenario of ["open", "reef", "channel"]) {
  test(`observations, snapshots, and rewards stay finite for ${scenario}`, () => {
    const env = new AUVEnv({scenario, current_strength: 0.4, max_steps: 150, seed: 31});
    const rng = new SeededRNG(99);
    checkObservation(env._observation());
    for (let i = 0; i < 150; i++) {
      const result = env.step([rng.uniform(-1, 1), rng.uniform(-1, 1)]);
      checkObservation(result.observation);
      checkFiniteJSON(env.snapshot());
      checkFiniteJSON(result.info);
      assert.ok(Number.isFinite(result.reward));
      close(result.reward, Object.values(result.info.reward_terms).reduce((sum, v) => sum + v, 0));
      if (result.terminated || result.truncated) break;
    }
    assert.equal(env.snapshot().sonar.length, 16);
    assert.ok(env.heading >= -Math.PI && env.heading <= Math.PI);
  });
}

test("snapshot has exact serializable fields and no alias to mutable state", () => {
  const env = new AUVEnv(), snapshot = env.snapshot();
  assert.deepEqual(Object.keys(snapshot).sort(), ["world", "vehicle", "goal", "obstacles", "sonar", "current",
    "trail", "episode_step", "episode_return", "last_reward", "distance_to_goal", "min_clearance",
    "outcome", "reward_terms"].sort());
  assert.deepEqual(JSON.parse(JSON.stringify(snapshot)), snapshot);
  const before = JSON.stringify(env.snapshot());
  snapshot.trail[0][0] = 999;
  snapshot.obstacles[0].x = 999;
  snapshot.reward_terms.goal_bonus = 999;
  assert.equal(JSON.stringify(env.snapshot()), before);
});

test("invalid actions fail before changing state; valid actions are clipped", () => {
  const env = makeEnv();
  const before = env.snapshot();
  for (const action of [[NaN, 0], [0, Infinity], [0], [[0, 0]], ["1", 0], null]) {
    assert.throws(() => env.step(action));
    assert.deepEqual(env.snapshot(), before);
  }
  const other = makeEnv();
  assert.deepEqual(env.step([100, -100]), other.step([1, -1]));
});

test("invalid configuration and reset options are rejected", () => {
  for (const config of [{scenario: "missing"}, {scenario: "toString"}, {current_strength: NaN},
    {current_strength: 0.41}, {current_strength: -0.1}, {dt: 0}, {dt: Infinity},
    {max_steps: 0}, {max_steps: 1.2}, {seed: -1}, {seed: true}, {course_variety: 1}]) {
    assert.throws(() => new AUVEnv(config));
  }
  const env = makeEnv(), before = env.snapshot();
  assert.throws(() => env.reset(-1));
  assert.throws(() => env.reset(7, {course_variety: 1}));
  assert.throws(() => env.reset(7, []));
  assert.deepEqual(env.snapshot(), before);
});

test("negative acceleration cannot reverse and trail memory is bounded", () => {
  const env = makeEnv({max_steps: 700});
  for (let i = 0; i < 650; i++) {
    const result = env.step([-1, 1]);
    assert.equal(env.speed, 0);
    assert.equal(result.terminated, false); assert.equal(result.truncated, false);
    checkObservation(result.observation);
  }
  assert.equal(env.snapshot().trail.length, 600);
});

test("core and baseline never depend on Math.random", () => {
  const original = Math.random;
  try {
    Math.random = () => { throw new Error("unexpected global random access"); };
    const env = new AUVEnv({seed: 42});
    env.step(env.heuristicAction());
    env.reset();
    env.step([0.5, -0.5]);
    const rng = new SeededRNG(42);
    rng.uniform(); rng.normal(); rng.integer(100);
  } finally { Math.random = original; }
});

for (const scenario of ["open", "reef", "channel"]) {
  for (const seed of [0, 7, 42]) {
    test(`map-aware baseline succeeds: ${scenario}, seed ${seed}, current 0.4`, () => {
      const env = new AUVEnv({scenario, seed, current_strength: 0.4});
      for (let i = 0; i < env.config.max_steps; i++) {
        const before = env.snapshot();
        const action = env.heuristicAction();
        assert.ok(action instanceof Float32Array && action.length === 2);
        assert.ok(Array.from(action).every(v => Number.isFinite(v) && Math.abs(v) <= 1));
        // Planning may fill a cache, but never advances the simulator or reward.
        assert.deepEqual(env.snapshot(), before);
        const result = env.step(action);
        if (result.terminated || result.truncated) break;
      }
      assert.equal(env.outcome, "success");
      assert.deepEqual(env.heuristicAction(), new Float32Array(2));
    });
  }
}
