"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
require("../web_src/environment.js");
require("../web_src/td3.js");
const { BrowserLab } = require("../web_src/worker.js");

const html = fs.readFileSync(path.join(__dirname, "../auv-practice.html"), "utf8");
const demo = JSON.parse(html.match(/<script id="auv-demo-model" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function finish(lab) {
  const deadline = performance.now() + 20000;
  while (["training", "evaluating"].includes(lab.mode)) {
    assert.ok(performance.now() < deadline, "Browser job exceeded its test deadline");
    await delay(5);
  }
  assert.notEqual(lab.mode, "error", lab.message);
}

test("the bundled Python policy loads with real weights and learning counters", () => {
  const lab = new BrowserLab(demo);
  try {
    const state = lab.state();
    assert.equal(state.training.steps, 50000);
    assert.equal(state.training.updates, 48873);
    assert.equal(state.training.actor_updates, 24436);
    assert.equal(state.policy_ready, true);
    assert.equal(lab.agent.config.hidden_dim, 128);
    assert.equal(state.training.buffer_size, 0);
    assert.equal(state.evaluation, null); // Python results are not presented as JS results.
    assert.equal(state.controller, "td3");
    assert.equal(state.mode, "paused");
  } finally { lab.stop(); }
});

test("preview stepping moves the vehicle without modifying the learner or replay", () => {
  const lab = new BrowserLab(demo);
  try {
    const before = lab.agent.toJSON();
    const position = lab.env.position.slice();
    for (let i = 0; i < 8; i++) lab.control({ command: "step", controller: "td3" });
    assert.notDeepEqual(lab.env.position, position);
    assert.equal(lab.env.episode_step, 8);
    assert.equal(lab.replay.length, 0);
    assert.equal(lab.training.steps, 50000);
    assert.deepEqual(lab.agent.toJSON(), before);
  } finally { lab.stop(); }
});

test("a fresh experiment performs the warm-up and actual delayed neural updates", async () => {
  const lab = new BrowserLab();
  try {
    const actor = lab.agent.actor.params.slice();
    const critic = lab.agent.q1.params.slice();
    lab.startTraining({ steps: 1064, exploration_noise: 0.1 });
    await finish(lab);
    assert.equal(lab.training.steps, 1064);
    assert.equal(lab.training.updates, 64);
    assert.equal(lab.training.actor_updates, 32);
    assert.equal(lab.replay.length, 1064);
    assert.notDeepEqual(lab.agent.actor.params, actor);
    assert.notDeepEqual(lab.agent.q1.params, critic);
    assert.ok(Number.isFinite(lab.training.critic_loss));
    assert.ok(Number.isFinite(lab.training.actor_loss));
    assert.ok(lab.history.length > 0);
    const before = lab.agent.toJSON();
    const bufferLength = lab.replay.length;
    lab.startEvaluation({ episodes: 2, controller: "td3" });
    await finish(lab);
    assert.deepEqual(lab.agent.toJSON(), before);
    assert.equal(lab.replay.length, bufferLength);
    assert.equal(lab.evaluation.records.length, 2);
    assert.equal(lab.evaluation.engine, "JavaScript");
    assert.equal(lab.evaluation.seed, 10007);
  } finally { lab.stop(); }
});

test("pause cancels a scheduled training budget and preview timers", async () => {
  const lab = new BrowserLab();
  try {
    lab.startTraining({ steps: 100000 });
    await delay(25);
    lab.control({ command: "pause" });
    const steps = lab.training.steps;
    await delay(30);
    assert.equal(lab.training.steps, steps);
    assert.ok(steps < 100000);
    lab.control({ command: "play", controller: "baseline", speed: 12 });
    await delay(40);
    lab.control({ command: "pause" });
    const previewStep = lab.env.episode_step;
    await delay(30);
    assert.equal(lab.env.episode_step, previewStep);
    assert.equal(lab.mode, "paused");
  } finally { lab.stop(); }
});

test("save/import preserves the full learner and clears replay before resuming", async () => {
  const lab = new BrowserLab(demo);
  try {
    const initial = lab.agent.toJSON();
    const saved = lab.save();
    assert.match(saved.download.filename, /^td3-.*\.json$/);
    lab.newExperiment({ seed: 8, scenario: "open", current_strength: 0 });
    assert.equal(lab.training.steps, 0);
    assert.equal(lab.agent.config.hidden_dim, 64);
    assert.equal(lab.agent.config.batch_size, 32);
    lab.importCheckpoint(JSON.parse(JSON.stringify(saved.download.payload)));
    assert.deepEqual(lab.agent.toJSON(), initial);
    assert.equal(lab.replay.length, 0);
    assert.equal(lab.config.scenario, "reef");
    lab.startTraining({ steps: 34, exploration_noise: 0.1 });
    await finish(lab);
    // Existing learner skips initial random warm-up; updates begin with the first full new batch.
    assert.equal(lab.training.steps, 50034);
    assert.equal(lab.training.updates, 48876);
    assert.equal(lab.training.actor_updates, 24438);
    assert.equal(lab.replay.length, 34);
    assert.notDeepEqual(lab.agent.toJSON().networks.actor, initial.networks.actor);
  } finally { lab.stop(); }
});

test("invalid checkpoint import is atomic and preserves the active experiment", () => {
  const lab = new BrowserLab(demo);
  try {
    const before = lab.agent.toJSON();
    const bad = JSON.parse(JSON.stringify(demo));
    bad.learner.networks.actor = "AAAA";
    assert.throws(() => lab.importCheckpoint(bad));
    assert.deepEqual(lab.agent.toJSON(), before);
    assert.equal(lab.training.steps, 50000);
    const missing = JSON.parse(JSON.stringify(demo));
    delete missing.training.critic_loss;
    assert.throws(() => lab.importCheckpoint(missing));
    assert.deepEqual(lab.agent.toJSON(), before);
    assert.throws(() => lab.request("/api/reset", { seed: 7, scenario: "bogus", current_strength: 0 }));
    assert.deepEqual(lab.agent.toJSON(), before);
  } finally { lab.stop(); }
});

test("fixed-seed evaluation is repeatable and does not consume the training environment RNG", async () => {
  const lab = new BrowserLab(demo);
  try {
    const before = lab.env.snapshot();
    lab.startEvaluation({ episodes: 3, controller: "td3" });
    await finish(lab);
    const first = lab.evaluation;
    lab.startEvaluation({ episodes: 3, controller: "td3" });
    await finish(lab);
    assert.deepEqual(lab.evaluation, first);
    assert.deepEqual(lab.env.snapshot(), before);
    assert.equal(lab.training.steps, 50000);
    assert.equal(lab.replay.length, 0);
    const download = lab.request("/api/export-results").download;
    assert.equal(download.filename, "auv-results.json");
    assert.deepEqual(download.payload.evaluation, first);
  } finally { lab.stop(); }
});
