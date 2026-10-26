"use strict";

// These integration checks compare paced execution with ordinary execution.
// They deliberately inspect real weights, replay contents and random streams,
// rather than accepting the displayed values as proof that learning happened.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
require("../web_src/environment.js");
const { TD3, ReplayBuffer } = require("../web_src/td3.js");
const { BrowserLab } = require("../web_src/worker.js");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "auv-practice.html"), "utf8");
const demo = JSON.parse(html.match(/<script id="auv-demo-model" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function trace(lab, command, extra = {}) {
  return lab.request("/api/trace", { command, ...extra });
}

function transition(lab, start = false, extra = {}) {
  const stages = [];
  let state = trace(lab, start ? "start" : "step", extra);
  for (let count = 0; count < 32; count++) {
    assert.equal(state.trace.enabled, true);
    assert.notEqual(state.mode, "error", state.message);
    stages.push(JSON.parse(JSON.stringify(state.trace.stage)));
    if (state.trace.at_boundary) return { state, stages };
    state = trace(lab, "step");
  }
  throw new Error("A traced transition did not reach a boundary within 32 stages");
}

function stepUntil(lab, wanted, maximum = 32) {
  for (let count = 0; count < maximum; count++) {
    const state = lab.state();
    if (state.trace.stage?.key === wanted) return state;
    trace(lab, "step");
  }
  throw new Error(`Trace did not reach ${wanted}`);
}

async function finish(lab) {
  const deadline = performance.now() + 20000;
  while (["training", "evaluating"].includes(lab.mode)) {
    assert.ok(performance.now() < deadline, "Timed out waiting for the ordinary worker loop");
    await pause(5);
  }
  assert.notEqual(lab.mode, "error", lab.message);
}

function replayState(replay) {
  return {
    size: replay.length, position: replay.position, rng: replay.rng.toJSON(),
    observations: Array.from(replay.observations.slice(0, replay.length * replay.obsDim)),
    actions: Array.from(replay.actions.slice(0, replay.length * replay.actionDim)),
    rewards: Array.from(replay.rewards.slice(0, replay.length)),
    next_observations: Array.from(replay.next_observations.slice(0, replay.length * replay.obsDim)),
    not_done: Array.from(replay.not_done.slice(0, replay.length)),
  };
}

function coreState(lab) {
  const metrics = {};
  for (const key of ["steps", "updates", "actor_updates", "episodes", "buffer_size", "critic_loss", "actor_loss", "q1", "q2", "td_error"]) {
    metrics[key] = lab.training[key];
  }
  return {
    environment: lab.env.snapshot(), observation: Array.from(lab.obs),
    action: Array.from(lab.action), rollout_rng: lab.rng.getState(),
    environment_rng: lab.env.rng.getState(), learner: lab.agent.toJSON(),
    replay: replayState(lab.replay), history: lab.history, metrics,
  };
}

for (const controller of ["td3", "random", "baseline"]) {
  test(`traced ${controller} preview matches normal steps without learning or extra RNG draws`, () => {
    const paced = new BrowserLab(demo), ordinary = new BrowserLab(demo);
    try {
      const initialLearner = paced.agent.toJSON();
      for (let index = 0; index < 6; index++) {
        const result = transition(paced, index === 0, { role: "preview", controller });
        ordinary.control({ command: "step", controller });
        assert.deepEqual(coreState(paced), coreState(ordinary));
        assert.ok(result.stages.some(stage => stage.key === "environment"));
        assert.ok(!result.stages.some(stage => stage.key === "replay_add"));
      }
      assert.deepEqual(paced.agent.toJSON(), initialLearner);
      assert.equal(paced.replay.length, 0);
    } finally { paced.stop(); ordinary.stop(); }
  });
}

test("staged training matches fast training across warm-up, replay filling and delayed updates", async () => {
  const paced = new BrowserLab(), ordinary = new BrowserLab();
  try {
    // A short warm-up exercises both branches without 1,000 UI commands.
    paced.warmupSteps = ordinary.warmupSteps = 2;
    ordinary.startTraining({ steps: 35, exploration_noise: 0.19 });
    await finish(ordinary);
    const seen = new Set();
    for (let index = 0; index < 35; index++) {
      const result = transition(paced, index === 0, { role: "training", exploration_noise: 0.19 });
      result.stages.forEach(stage => seen.add(stage.key));
    }
    assert.deepEqual(coreState(paced), coreState(ordinary));
    for (const key of ["random", "actor", "exploration", "replay_add", "learning_wait", "replay_sample", "target_actions", "bellman_target", "critic", "actor_update", "actor_skip", "soft_targets"]) {
      assert.ok(seen.has(key), `Missing execution branch: ${key}`);
    }
    assert.equal(paced.agent.updates, 4);
    assert.equal(paced.agent.actor_updates, 2);
  } finally { paced.stop(); ordinary.stop(); }
});

test("an environment stage moves the AUV once and keeps it still during replay learning", () => {
  const lab = new BrowserLab(demo);
  try {
    trace(lab, "start", { role: "training", exploration_noise: 0.1 });
    const before = lab.env.snapshot();
    const initialReplay = lab.replay.length;
    stepUntil(lab, "actor");
    assert.deepEqual(lab.env.snapshot(), before);
    stepUntil(lab, "exploration");
    assert.deepEqual(lab.env.snapshot(), before);
    stepUntil(lab, "environment");
    const after = lab.env.snapshot();
    assert.equal(after.episode_step, before.episode_step + 1);
    assert.notDeepEqual(after.vehicle, before.vehicle);
    assert.equal(lab.replay.length, initialReplay);
    stepUntil(lab, "replay_add");
    assert.equal(lab.replay.length, initialReplay + 1);
    assert.deepEqual(lab.env.snapshot(), after);
    stepUntil(lab, "transition_end");
    assert.deepEqual(lab.env.snapshot(), after);
  } finally { lab.stop(); }
});

for (const terminalKind of ["success", "timeout"]) {
  test(`training holds the ${terminalKind} frame and saves the real successor with the correct bootstrap mask`, () => {
    const lab = new BrowserLab(demo);
    try {
      // Enter training before positioning, so its role change cannot reset the fixture.
      trace(lab, "start", { role: "training", exploration_noise: 0 });
      trace(lab, "exit");
      lab.env.obstacles = [];
      lab.env.current = [0, 0];
      if (terminalKind === "success") {
        lab.env.position = [16.2, 17];
        lab.env.goal = [17, 17];
        lab.env.heading = 0;
        lab.env.speed = 1.8;
      } else {
        lab.env.position = [4, 4];
        lab.env.goal = [17, 17];
        lab.env.episode_step = lab.env.config.max_steps - 1;
      }
      lab.obs = lab.env._observation();
      trace(lab, "start", { role: "training", exploration_noise: 0 });
      const contact = stepUntil(lab, "environment");
      assert.equal(contact.env.outcome, terminalKind);
      const successor = Array.from(lab.obs);
      const contactPosition = lab.env.position.slice();
      const slot = lab.replay.position;
      stepUntil(lab, "replay_add");
      assert.equal(lab.replay.not_done[slot], terminalKind === "success" ? 0 : 1);
      assert.deepEqual(Array.from(lab.replay.next_observations.slice(slot * 21, (slot + 1) * 21)), successor);
      const final = stepUntil(lab, "transition_end");
      assert.equal(final.env.outcome, terminalKind);
      assert.deepEqual(lab.env.position, contactPosition);
      assert.equal(lab.history.at(-1).outcome, terminalKind);
      const next = trace(lab, "step");
      assert.equal(next.trace.stage.key, "reset");
      assert.equal(next.env.episode_step, 0);
      assert.equal(next.env.outcome, "running");
      assert.notDeepEqual(lab.env.position, contactPosition);
    } finally { lab.stop(); }
  });
}

test("exiting after movement finishes that training transition exactly once", async () => {
  const paced = new BrowserLab(demo), ordinary = new BrowserLab(demo);
  try {
    trace(paced, "start", { role: "training", exploration_noise: 0.13 });
    stepUntil(paced, "environment");
    const ended = trace(paced, "exit");
    assert.equal(ended.trace.enabled, false);
    assert.equal(ended.mode, "paused");
    ordinary.startTraining({ steps: 1, exploration_noise: 0.13 });
    await finish(ordinary);
    assert.deepEqual(coreState(paced), coreState(ordinary));
    const complete = coreState(paced);
    trace(paced, "exit");
    assert.deepEqual(coreState(paced), complete);
  } finally { paced.stop(); ordinary.stop(); }
});

test("tracing does not advance in wall-clock time without a step command", async () => {
  const lab = new BrowserLab(demo);
  try {
    const first = trace(lab, "start", { role: "preview" });
    const before = coreState(lab);
    await pause(35);
    assert.deepEqual(coreState(lab), before);
    assert.equal(lab.state().trace.sequence, first.trace.sequence);
    trace(lab, "step");
    assert.equal(lab.state().trace.sequence, first.trace.sequence + 1);
  } finally { lab.stop(); }
});


test("the learning trace exposes the sampled row, real Bellman target, critic loss and delayed parameter changes", () => {
  const agent = new TD3(3, 2, { hidden_dim: 8, batch_size: 8, buffer_capacity: 64 }, 17);
  const replay = new ReplayBuffer(3, 2, 64, 23);
  for (let index = 0; index < 48; index++) {
    replay.add([index / 50, Math.sin(index), -0.2], [0.1, -0.3], index / 10 - 2,
      [(index + 1) / 50, Math.sin(index + 1), 0.4], index % 9 === 0);
  }
  let actualBatch;
  const sample = replay.sample.bind(replay);
  replay.sample = (...args) => { actualBatch = sample(...args); return actualBatch; };
  const clamp = (number, minimum, maximum) => Math.max(minimum, Math.min(maximum, number));
  const networkState = () => agent.toJSON().networks;
  for (let update = 1; update <= 2; update++) {
    const before = agent.toJSON();
    const oracle = TD3.fromJSON(before);
    const iterator = agent.updateSteps(replay);
    const sampleStage = iterator.next().value;
    assert.equal(sampleStage.key, "replay_sample");
    assert.deepEqual(agent.toJSON(), before, "sampling alone must not change the learner");
    const row = sampleStage.values.sample_index;
    assert.deepEqual(sampleStage.values.observation,
      Array.from(replay.observations.slice(row * 3, row * 3 + 3)));
    assert.deepEqual(sampleStage.values.next_observation,
      Array.from(replay.next_observations.slice(row * 3, row * 3 + 3)));
    assert.deepEqual(sampleStage.values.action, Array.from(replay.actions.slice(row * 2, row * 2 + 2)));
    assert.equal(sampleStage.values.reward, replay.rewards[row]);
    assert.equal(sampleStage.values.not_done, replay.not_done[row]);

    const smooth = iterator.next().value;
    assert.equal(smooth.key, "target_actions");
    const proposed = oracle.actorTarget.forward(actualBatch.next_observations, 8).slice();
    const targetActions = new Float32Array(16);
    const raw = [], clipped = [];
    for (let index = 0; index < 16; index++) {
      const noise = oracle.rng.normal() * oracle.config.policy_noise;
      const clippedNoise = clamp(noise, -oracle.config.noise_clip, oracle.config.noise_clip);
      targetActions[index] = clamp(proposed[index] + clippedNoise, -1, 1);
      if (index < 2) { raw.push(noise); clipped.push(clippedNoise); }
    }
    assert.deepEqual(smooth.values.proposed_action, Array.from(proposed.slice(0, 2)));
    assert.deepEqual(smooth.values.raw_noise, raw);
    assert.deepEqual(smooth.values.clipped_noise, clipped);
    assert.deepEqual(smooth.values.smoothed_action, Array.from(targetActions.slice(0, 2)));
    assert.deepEqual(agent.rng.toJSON(), oracle.rng.toJSON(), "tracing must consume exactly the learning-noise draws");

    const bellman = iterator.next().value;
    assert.equal(bellman.key, "bellman_target");
    const next = new Float32Array(8 * 5), current = new Float32Array(8 * 5);
    for (let index = 0; index < 8; index++) {
      next.set(actualBatch.next_observations.slice(index * 3, index * 3 + 3), index * 5);
      next.set(targetActions.slice(index * 2, index * 2 + 2), index * 5 + 3);
      current.set(actualBatch.observations.slice(index * 3, index * 3 + 3), index * 5);
      current.set(actualBatch.actions.slice(index * 2, index * 2 + 2), index * 5 + 3);
    }
    const nextQ1 = oracle.q1Target.forward(next, 8), nextQ2 = oracle.q2Target.forward(next, 8);
    const currentQ1 = oracle.q1.forward(current, 8), currentQ2 = oracle.q2.forward(current, 8);
    let loss = 0;
    for (let index = 0; index < 8; index++) {
      const target = actualBatch.rewards[index] + oracle.config.discount * actualBatch.not_done[index]
        * Math.min(nextQ1[index], nextQ2[index]);
      if (index === 0) {
        assert.equal(bellman.values.target_q1, nextQ1[index]);
        assert.equal(bellman.values.target_q2, nextQ2[index]);
        assert.equal(bellman.values.min_target_q, Math.min(nextQ1[index], nextQ2[index]));
        assert.equal(bellman.values.target, target);
        assert.equal(bellman.values.not_done, actualBatch.not_done[index]);
      }
      const first = currentQ1[index] - target, second = currentQ2[index] - target;
      loss += (first * first + second * second) / 8;
    }
    const critic = iterator.next().value;
    assert.equal(critic.key, "critic");
    assert.equal(critic.values.critic_loss, loss);
    assert.equal(critic.values.first_q1, currentQ1[0]);
    assert.equal(critic.values.first_q2, currentQ2[0]);
    const afterCritic = networkState();
    assert.notEqual(afterCritic.q1, before.networks.q1);
    assert.notEqual(afterCritic.q2, before.networks.q2);
    for (const name of ["actor", "actor_target", "q1_target", "q2_target"]) {
      assert.equal(afterCritic[name], before.networks[name]);
    }
    const actor = iterator.next().value;
    if (update === 1) {
      assert.equal(actor.key, "actor_skip");
      assert.deepEqual(networkState(), afterCritic);
    } else {
      assert.equal(actor.key, "actor_update");
      const afterActor = networkState();
      assert.notEqual(afterActor.actor, before.networks.actor);
      for (const name of ["q1", "q2", "actor_target", "q1_target", "q2_target"]) {
        assert.equal(afterActor[name], afterCritic[name]);
      }
      const target = iterator.next().value;
      assert.equal(target.key, "soft_targets");
      const numbers = target.values;
      assert.deepEqual(numbers.target_weights_after, numbers.target_weights_before.map((value, index) =>
        Math.fround((1 - numbers.tau) * value + numbers.tau * numbers.online_weights[index])));
      for (const name of ["actor_target", "q1_target", "q2_target"]) {
        assert.notEqual(networkState()[name], before.networks[name]);
      }
    }
    const completed = iterator.next();
    assert.equal(completed.done, true);
    assert.equal(completed.value.updates, update);
    assert.equal(completed.value.actor_updates, Math.floor(update / 2));
  }
});


test("pause preserves an in-flight learning block; saving then completes the remaining target updates", async () => {
  const paced = new BrowserLab(demo), ordinary = new BrowserLab(demo);
  try {
    paced.startTraining({ steps: 33, exploration_noise: 0.17 });
    ordinary.startTraining({ steps: 34, exploration_noise: 0.17 });
    await Promise.all([finish(paced), finish(ordinary)]);
    trace(paced, "start", { role: "training", exploration_noise: 0.17 });
    stepUntil(paced, "critic");
    const beforePause = coreState(paced);
    const sequence = paced.trace.sequence;
    const held = paced.control({ command: "pause" });
    assert.equal(held.mode, "walkthrough");
    assert.equal(held.trace.enabled, true);
    assert.equal(held.trace.sequence, sequence);
    assert.deepEqual(coreState(paced), beforePause);
    stepUntil(paced, "actor_update");
    const saved = paced.save();
    assert.equal(saved.state.trace.enabled, false);
    assert.equal(saved.state.mode, "paused");
    assert.deepEqual(coreState(paced), coreState(ordinary));
    const restored = TD3.fromJSON(saved.download.payload.learner);
    assert.deepEqual(restored.toJSON(), ordinary.agent.toJSON());
    assert.equal(paced.traceIterator, null);
  } finally { paced.stop(); ordinary.stop(); }
});

test("invalid input preserves a suspended trace; reset and import discard its continuation safely", () => {
  const lab = new BrowserLab(demo);
  try {
    trace(lab, "start", { role: "training", exploration_noise: 0.1 });
    stepUntil(lab, "environment");
    const before = coreState(lab), sequence = lab.trace.sequence;
    const invalid = JSON.parse(JSON.stringify(demo));
    invalid.learner.networks.actor = "invalid";
    assert.throws(() => lab.importCheckpoint(invalid));
    assert.throws(() => lab.newExperiment({ scenario: "invalid", seed: 7, current_strength: 0 }));
    assert.throws(() => trace(lab, "start", { role: "invalid" }));
    assert.deepEqual(coreState(lab), before);
    assert.equal(lab.trace.enabled, true);
    assert.equal(lab.trace.sequence, sequence);
    lab.newExperiment({ scenario: "open", seed: 42, current_strength: 0 });
    assert.equal(lab.trace.enabled, false);
    assert.equal(lab.traceIterator, null);
    assert.equal(lab.training.steps, 0);
    assert.equal(lab.replay.length, 0);
    assert.throws(() => trace(lab, "step"));
    trace(lab, "start", { role: "preview", controller: "random" });
    stepUntil(lab, "random");
    lab.importCheckpoint(demo);
    assert.equal(lab.trace.enabled, false);
    assert.equal(lab.traceIterator, null);
    assert.equal(lab.env.episode_step, 0);
    assert.equal(lab.replay.length, 0);
    assert.deepEqual(lab.agent.toJSON(), TD3.fromJSON(demo.learner).toJSON());
  } finally { lab.stop(); }
});

test("Next AUV step finishes a partial transition or performs exactly one next complete transition", () => {
  const lab = new BrowserLab(demo);
  try {
    trace(lab, "start", { role: "training" });
    const initial = lab.training.steps;
    stepUntil(lab, "environment");
    const moved = lab.env.episode_step;
    const first = trace(lab, "tick");
    assert.equal(first.trace.at_boundary, true);
    assert.equal(first.trace.stage.key, "transition_end");
    assert.equal(first.env.episode_step, moved);
    assert.equal(first.training.steps, initial + 1);
    const second = trace(lab, "tick");
    assert.equal(second.trace.at_boundary, true);
    assert.equal(second.env.episode_step, moved + 1);
    assert.equal(second.training.steps, initial + 2);
    assert.equal(second.training.buffer_size, 2);
  } finally { lab.stop(); }
});

test("evaluation after a suspended training step first completes that step and leaves no stale iterator", async () => {
  const lab = new BrowserLab(demo), reference = new BrowserLab(demo);
  try {
    trace(lab, "start", { role: "training", exploration_noise: 0.08 });
    stepUntil(lab, "exploration");
    lab.startEvaluation({ episodes: 1, controller: "td3" });
    reference.startTraining({ steps: 1, exploration_noise: 0.08 });
    await Promise.all([finish(lab), finish(reference)]);
    assert.equal(lab.trace.enabled, false);
    assert.equal(lab.traceIterator, null);
    assert.deepEqual(coreState(lab), coreState(reference));
    assert.equal(lab.evaluation.episodes, 1);
  } finally { lab.stop(); reference.stop(); }
});
