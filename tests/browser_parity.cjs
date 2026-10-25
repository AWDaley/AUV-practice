"use strict";

// Python/JavaScript checks use identical injected states: their seeded course
// generators intentionally differ. Run the independent demo evaluation with:
//   node tests/browser_parity.cjs --evaluate
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {AUVEnv} = require("../web_src/environment.js");
const {TD3} = require("../web_src/td3.js");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "auv-practice.html"), "utf8");
const embedded = html.match(/<script\b[^>]*\bid=["']auv-demo-model["'][^>]*>([\s\S]*?)<\/script\s*>/i);
assert.ok(embedded, "The offline HTML must contain its own demo checkpoint");
const checkpoint = JSON.parse(embedded[1]);
assert.equal(checkpoint.format, "auv-practice-browser");
assert.equal(checkpoint.version, 1);
const agent = TD3.fromJSON(checkpoint.learner);

function evaluateDemo() {
  const records = [];
  for (let seed = 10007; seed <= 10016; seed++) {
    const env = new AUVEnv({scenario: "reef", seed, current_strength: 0.08});
    let observation = env._observation();
    for (let i = 0; i < env.config.max_steps; i++) {
      const result = env.step(agent.selectAction(observation));
      observation = result.observation;
      if (result.terminated || result.truncated) break;
    }
    records.push({seed, outcome: env.outcome, steps: env.episode_step,
      return: env.episode_return, distance_to_goal: env.snapshot().distance_to_goal});
  }
  const count = outcome => records.filter(record => record.outcome === outcome).length;
  const mean = key => records.reduce((sum, record) => sum + record[key], 0) / records.length;
  console.log(JSON.stringify({
    source: "Embedded auv-demo-model; deterministic actor, no exploration or updates",
    scenario: "reef", current_strength: 0.08, seed_from: 10007, seed_to: 10016,
    training_steps: checkpoint.training.steps, episodes: records.length,
    successes: count("success"), collisions: count("collision"), timeouts: count("timeout"),
    success_rate: count("success") / records.length,
    collision_rate: count("collision") / records.length,
    timeout_rate: count("timeout") / records.length,
    mean_return: mean("return"), mean_steps: mean("steps"), records,
  }, null, 2));
}

if (process.argv.includes("--evaluate")) {
  evaluateDemo();
} else {
  const test = require("node:test");
  const fixtures = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "browser-parity.json"), "utf8"));
  const maximumErrors = {observation: 0, position: 0, heading: 0, speed: 0, reward: 0, action: 0, q: 0};

  function close(actual, expected, tolerance, kind, label) {
    assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${label}: values must be finite`);
    const error = Math.abs(actual - expected);
    maximumErrors[kind] = Math.max(maximumErrors[kind], error);
    assert.ok(error <= tolerance,
      `${label}: JavaScript ${actual}, Python ${expected}, absolute error ${error}, tolerance ${tolerance}`);
  }
  function closeArray(actual, expected, tolerance, kind, label) {
    assert.equal(actual.length, expected.length, `${label}: shape mismatch`);
    for (let i = 0; i < expected.length; i++) close(actual[i], expected[i], tolerance, kind, `${label}[${i}]`);
  }
  function fixtureEnvironment(row) {
    const env = new AUVEnv({scenario: row.scenario, seed: row.seed, current_strength: 0.08});
    env.position = row.start.position.slice();
    env.goal = row.start.goal.slice();
    env.obstacles = row.start.obstacles.map(circle => circle.slice());
    env.current = row.start.current.slice();
    env.heading = row.start.heading;
    env.speed = row.start.speed;
    env.trail = [env.position.slice()];
    return env;
  }

  test("fixture and embedded checkpoint schemas match the intended parity cases", () => {
    assert.equal(fixtures.version, 1);
    assert.equal(fixtures.rows.length, 12);
    assert.deepEqual([...new Set(fixtures.rows.map(row => row.scenario))].sort(), ["channel", "open", "reef"]);
    assert.equal(agent.obsDim, 21);
    assert.equal(agent.actionDim, 2);
    assert.equal(agent.updates, checkpoint.training.updates);
    assert.equal(agent.actor_updates, checkpoint.training.actor_updates);
  });

  for (const row of fixtures.rows) {
    const label = `${row.scenario}, Python seed ${row.seed}`;
    test(`environment parity: ${label}`, () => {
      const env = fixtureEnvironment(row);
      closeArray(env._observation(), row.observation, 1e-6, "observation", "initial observation");
      assert.equal(row.transitions.length, 3);
      for (let i = 0; i < row.transitions.length; i++) {
        const expected = row.transitions[i], transition = `${label}, step ${i + 1}`;
        // Browser actors/replay use float32 actions; Python fixture commands
        // were doubles. The 1e-8 motion/reward tolerance includes this rounding.
        const result = env.step(new Float32Array(expected.action));
        closeArray(result.observation, expected.observation, 1e-6, "observation", `${transition} observation`);
        closeArray(env.position, expected.position, 1e-8, "position", `${transition} position`);
        close(env.heading, expected.heading, 1e-8, "heading", `${transition} heading`);
        close(env.speed, expected.speed, 1e-8, "speed", `${transition} speed`);
        close(result.reward, expected.reward, 1e-8, "reward", `${transition} reward`);
        assert.equal(result.info.outcome, expected.outcome, `${transition} outcome`);
        assert.equal(result.terminated, ["success", "collision"].includes(expected.outcome));
        assert.equal(result.truncated, expected.outcome === "timeout");
      }
    });

    test(`imported actor and twin-critic parity: ${label}`, () => {
      const observation = fixtureEnvironment(row)._observation();
      const action = agent.selectAction(observation);
      closeArray(action, row.policy_action, 1e-6, "action", `${label} actor action`);
      // First isolate critic import from small actor arithmetic differences.
      const qAtPythonAction = agent.actionValues(observation, new Float32Array(row.policy_action));
      closeArray(qAtPythonAction, row.q, 1e-4, "q", `${label} critics at Python action`);
      // Also verify the complete JS observation -> actor -> critics path.
      closeArray(agent.actionValues(observation, action), row.q, 1e-4, "q", `${label} critics at JS action`);
    });
  }

  test("report maximum absolute differences across the 12 parity fixtures", context => {
    context.diagnostic(JSON.stringify(maximumErrors));
  });
}
