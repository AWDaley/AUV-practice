"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { TD3, TD3Config, ReplayBuffer, MLP, Adam, RNG, encodeFloat32, decodeFloat32 } = require("../web_src/td3.js");

function near(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by more than ${tolerance}`);
}

function fillBuffer(obsDim = 3, actionDim = 2, sampleSeed = 19, count = 64) {
  const buffer = new ReplayBuffer(obsDim, actionDim, 128, sampleSeed), rng = new RNG(31);
  for (let step = 0; step < count; step++) {
    const obs = Float32Array.from({ length: obsDim }, () => rng.normal());
    const action = Float32Array.from({ length: actionDim }, () => 2 * rng.uniform() - 1);
    const next = Float32Array.from(obs, value => value + 0.1 * rng.normal());
    const reward = -obs.reduce((sum, value) => sum + value * value, 0);
    buffer.add(obs, action, reward, next, step % 11 === 0);
  }
  return buffer;
}

function setConstant(network, value) {
  network.params.fill(0);
  network.params[network.layers[2].b_offset] = value;
}

for (const activation of ["linear", "tanh"]) {
  test(`analytic ${activation} MLP gradients agree with finite differences`, () => {
    const network = new MLP(3, 4, 2, activation, new RNG(19));
    for (let i = 0; i < network.params.length; i++) network.params[i] = 0.04 * Math.sin(i + 1);
    for (const layer of network.layers.slice(0, 2)) {
      for (let i = 0; i < layer.output_dim; i++) network.params[layer.b_offset + i] = 0.3;
    }
    const input = new Float64Array([0.2, -0.1, 0.4, -0.3, 0.2, 0.5]);
    const weights = new Float64Array([0.3, -0.7, 0.2, 0.5]);
    const objective = () => network.forward(input, 2).reduce((sum, value, i) => sum + weights[i] * value, 0);
    network.forward(input, 2, true);
    network.zeroGrad();
    const inputGradient = network.backward(weights).slice();
    const parameterGradient = network.gradients.slice();
    for (let i = 0; i < network.params.length; i++) {
      const original = network.params[i], epsilon = 1e-3;
      network.params[i] = original + epsilon;
      const plus = network.params[i], high = objective();
      network.params[i] = original - epsilon;
      const minus = network.params[i], low = objective();
      network.params[i] = original;
      near(parameterGradient[i], (high - low) / (plus - minus), 2e-5);
    }
    for (let i = 0; i < input.length; i++) {
      const original = input[i], epsilon = 1e-5;
      input[i] = original + epsilon;
      const high = objective();
      input[i] = original - epsilon;
      const low = objective();
      input[i] = original;
      near(inputGradient[i], (high - low) / (2 * epsilon), 1e-7);
    }
  });
}

test("Adam uses descent, bias corrections, epsilon placement, and restorable moments", () => {
  const params = new Float32Array(3), optimizer = new Adam(params, 0.001);
  optimizer.step(new Float64Array([1, -2, 1e-9]));
  near(params[0], -0.001, 1e-8);
  near(params[1], 0.001, 1e-8);
  near(params[2], -0.001 / 11, 1e-10);
  near(optimizer.m[0], 0.1, 1e-8);
  near(optimizer.v[1], 0.004, 1e-9);
  const restoredParams = params.slice(), restored = new Adam(restoredParams, 0.001);
  restored.loadJSON(JSON.parse(JSON.stringify(optimizer.toJSON())));
  const next = new Float64Array([-0.2, 0.5, -1e-9]);
  optimizer.step(next); restored.step(next);
  assert.deepEqual(restoredParams, params);
  assert.equal(restored.t, 2);
});

test("replay wraps, samples real records, and masks only true termination", () => {
  const buffer = new ReplayBuffer(3, 2, 2, 3);
  for (const [index, terminal] of [[1, false], [2, true], [3, false]]) {
    buffer.add([index, index, index], [0.25, -0.25], index, [index + 1, index + 1, index + 1], terminal);
  }
  assert.equal(buffer.length, 2);
  const batch = buffer.sample(64), seen = new Set();
  assert.equal(batch.observations.length, 192);
  assert.equal(batch.actions.length, 128);
  assert.equal(batch.rewards.length, 64);
  for (let i = 0; i < 64; i++) {
    const index = batch.observations[3 * i];
    seen.add(index);
    assert.equal(batch.rewards[i], index);
    assert.equal(batch.next_observations[3 * i], index + 1);
    assert.equal(batch.not_done[i], index === 2 ? 0 : 1);
  }
  assert.deepEqual(seen, new Set([2, 3]));
});

test("real weight updates obey the actor delay and target interpolation", () => {
  const agent = new TD3(3, 2, { hidden_dim: 16, batch_size: 8, tau: 0.2 });
  const buffer = fillBuffer();
  const actor = agent.actor.params.slice(), q1 = agent.q1.params.slice();
  const actorTarget = agent.actorTarget.params.slice(), q1Target = agent.q1Target.params.slice(), q2Target = agent.q2Target.params.slice();
  const first = agent.update(buffer);
  assert.equal(first.updates, 1); assert.equal(first.actor_updates, 0); assert.equal(first.actor_loss, null);
  assert.notDeepEqual(agent.q1.params, q1);
  assert.deepEqual(agent.actor.params, actor);
  assert.deepEqual(agent.actorTarget.params, actorTarget);
  assert.deepEqual(agent.q1Target.params, q1Target);
  assert.deepEqual(agent.q2Target.params, q2Target);
  const second = agent.update(buffer);
  assert.equal(second.updates, 2); assert.equal(second.actor_updates, 1);
  assert.notEqual(second.actor_loss, null);
  assert.notDeepEqual(agent.actor.params, actor);
  for (const [previous, online, target] of [[actorTarget, agent.actor, agent.actorTarget], [q1Target, agent.q1, agent.q1Target], [q2Target, agent.q2, agent.q2Target]]) {
    for (let i = 0; i < previous.length; i++) near(target.params[i], Math.fround(0.8 * previous[i] + 0.2 * online.params[i]), 0);
  }
  assert.ok(Object.values(second).every(Number.isFinite));
});

for (const terminated of [false, true]) {
  test(`real critic target has the correct ${terminated ? "terminal" : "nonterminal"} loss`, () => {
    const agent = new TD3(3, 2, { hidden_dim: 8, batch_size: 1 });
    setConstant(agent.q1, 2); setConstant(agent.q2, 4);
    setConstant(agent.q1Target, 8); setConstant(agent.q2Target, 5);
    const buffer = new ReplayBuffer(3, 2, 1);
    buffer.add([0, 0, 0], [0, 0], -2, [1, 1, 1], terminated);
    const metrics = agent.update(buffer);
    near(metrics.critic_loss, terminated ? 52 : 2.005, 1e-10);
    near(metrics.td_error, terminated ? 5 : 1, 1e-10);
    assert.equal(metrics.q1, 2); assert.equal(metrics.q2, 4);
    const first = agent.q1.params[agent.q1.layers[2].b_offset];
    assert.ok(terminated ? first < 2 : first > 2);
    assert.ok(agent.q2.params[agent.q2.layers[2].b_offset] < 4);
  });
}

test("target noise clipping precedes action clipping and uses the saved local RNG", () => {
  const agent = new TD3(3, 2, { hidden_dim: 8, policy_noise: 10, noise_clip: 0.25 });
  agent.actorTarget.params.fill(0);
  const offset = agent.actorTarget.layers[2].b_offset;
  agent.actorTarget.params[offset] = Math.atanh(0.9);
  agent.actorTarget.params[offset + 1] = Math.atanh(-0.9);
  const input = new Float32Array(64 * 3);
  const proposed = agent.actorTarget.forward(input, 64).slice();
  const rng = RNG.fromJSON(agent.rng.toJSON());
  const expected = Float32Array.from(proposed, value => {
    const noise = Math.max(-0.25, Math.min(0.25, 10 * rng.normal()));
    return Math.max(-1, Math.min(1, value + noise));
  });
  const actual = agent._targetActions(input, 64);
  assert.deepEqual(actual, expected);
  assert.equal(Math.max(...actual), 1); assert.equal(Math.min(...actual), -1);
  for (const obs of [[0, 0, 0], [1000, -1000, 1000]]) {
    assert.ok([...agent.selectAction(obs)].every(value => Number.isFinite(value) && Math.abs(value) <= 1));
  }
});

test("actor ascends Q1 through its action input while critic parameters stay fixed", () => {
  const agent = new TD3(3, 2, { hidden_dim: 8, batch_size: 4, policy_delay: 1 });
  agent.actor.params.fill(0);
  for (const [network, slope] of [[agent.q1, 1], [agent.q2, -1]]) {
    network.params.fill(0);
    const [first, second, last] = network.layers;
    network.params[first.w_offset + 3] = slope;
    network.params[first.b_offset] = 1;
    network.params[second.w_offset] = 1;
    network.params[last.w_offset] = 1;
    network.params[last.b_offset] = -1;
  }
  const firstBefore = agent.q1.params.slice(), secondBefore = agent.q2.params.slice();
  const buffer = new ReplayBuffer(3, 2, 1);
  buffer.add([0, 0, 0], [0, 0], 0, [0, 0, 0], true);
  const metrics = agent.update(buffer);
  assert.equal(metrics.critic_loss, 0);
  assert.equal(metrics.actor_loss, 0);
  assert.ok(agent.selectAction([0, 0, 0])[0] > 0);
  assert.equal(agent.selectAction([0, 0, 0])[1], 0);
  assert.deepEqual(agent.q1.params, firstBefore);
  assert.deepEqual(agent.q2.params, secondBefore);
});

test("checkpoint restores networks, optimizers, counters and the exact next update", () => {
  const original = new TD3(3, 1, { hidden_dim: 16, batch_size: 7 }, 5);
  const buffer = fillBuffer(3, 1);
  for (let i = 0; i < 3; i++) original.update(buffer);
  const restored = TD3.fromJSON(JSON.parse(JSON.stringify(original.toJSON())));
  assert.equal(restored.updates, 3); assert.equal(restored.actor_updates, 1);
  assert.deepEqual(restored.selectAction([0.1, -0.5, 0.7]), original.selectAction([0.1, -0.5, 0.7]));
  assert.deepEqual(restored.actionValues([0.1, -0.5, 0.7], [0.2]), original.actionValues([0.1, -0.5, 0.7], [0.2]));
  assert.deepEqual(original.update(fillBuffer(3, 1, 101)), restored.update(fillBuffer(3, 1, 101)));
  assert.deepEqual(original.toJSON(), restored.toJSON());
});

test("base64 uses the specified little-endian float32 layout", () => {
  const values = new Float32Array([1, -2.5, 0.25]);
  const bytes = Buffer.alloc(12);
  values.forEach((value, i) => bytes.writeFloatLE(value, i * 4));
  assert.equal(encodeFloat32(values), bytes.toString("base64"));
  assert.deepEqual(decodeFloat32(encodeFloat32(values), 3), values);
});

test("invalid input, checkpoint tensors, moments and counters are rejected", () => {
  assert.throws(() => new TD3Config({ discount: NaN }), /finite/);
  assert.throws(() => new TD3Config({ tau: 2 }), /\[0, 1\]/);
  assert.throws(() => new TD3Config({ batch_size: 0 }), /integer/);
  assert.throws(() => new TD3Config({ toString: 1 }), /Unknown/);
  const buffer = new ReplayBuffer(3, 2);
  assert.throws(() => buffer.sample(1), /empty/);
  assert.throws(() => buffer.add([0, 0, 0], [0, 0], NaN, [0, 0, 0], false), /finite/);
  assert.throws(() => buffer.add([0, 0, 0], [2, 0], 0, [0, 0, 0], false), /\[-1, 1\]/);
  assert.throws(() => buffer.add([0, 0, 0], [0, 0], 0, [0, 0, 0], "false"), /terminated/);
  assert.throws(() => encodeFloat32(new Float32Array([Infinity])), /finite/);
  const valid = new TD3(3, 2, { hidden_dim: 8 }).toJSON();
  const broken = () => JSON.parse(JSON.stringify(valid));
  let payload = broken(); payload.networks.actor = "AA==";
  assert.throws(() => TD3.fromJSON(payload), /length/);
  payload = broken(); delete payload.obs_dim;
  assert.throws(() => TD3.fromJSON(payload), /dimension/);
  payload = broken(); delete payload.config.discount;
  assert.throws(() => TD3.fromJSON(payload), /incomplete/);
  payload = broken(); payload.actor_updates = 2;
  assert.throws(() => TD3.fromJSON(payload), /exceed/);
  payload = broken(); payload.optimizers.q1.step = 1;
  assert.throws(() => TD3.fromJSON(payload), /counter/);
  payload = broken(); payload.rng.state = 0;
  assert.throws(() => TD3.fromJSON(payload), /RNG state/);
  payload = broken();
  const negative = decodeFloat32(payload.optimizers.q1.v, new TD3(3, 2, { hidden_dim: 8 }).q1.params.length);
  negative[0] = -1; payload.optimizers.q1.v = encodeFloat32(negative);
  assert.throws(() => TD3.fromJSON(payload), /nonnegative/);
});
