/* Independently implemented, one-step TD3. No dependencies or network access.
 * Actions are normalized to [-1, 1]. This file works in a Worker and in Node.
 */
(function (scope) {
  "use strict";

  function integer(value, name, minimum = 1, maximum = Number.MAX_SAFE_INTEGER) {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      throw new Error(`${name} must be an integer in [${minimum}, ${maximum}]`);
    }
    return value;
  }

  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${name} must be finite`);
    }
    return value;
  }

  function arrayLength(values, length, name) {
    if (!values || values.length !== length) throw new Error(`${name} must have length ${length}`);
  }

  function finiteArray(values, length, name) {
    arrayLength(values, length, name);
    for (let i = 0; i < length; i++) {
      if (typeof values[i] !== "number" || !Number.isFinite(values[i])) {
        throw new Error(`${name}[${i}] must be finite`);
      }
    }
  }

  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }

  const DEFAULTS = Object.freeze({
    hidden_dim: 64, learning_rate: 3e-4, discount: 0.99, tau: 0.005,
    policy_noise: 0.2, noise_clip: 0.5, policy_delay: 2,
    batch_size: 32, buffer_capacity: 50000,
  });

  class TD3Config {
    constructor(options = {}) {
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw new Error("TD3 configuration must be an object");
      }
      for (const key of Object.keys(options)) {
        if (!Object.hasOwn(DEFAULTS, key)) throw new Error(`Unknown TD3 setting: ${key}`);
      }
      const values = { ...DEFAULTS, ...options };
      integer(values.hidden_dim, "hidden_dim", 1, 1024);
      integer(values.batch_size, "batch_size", 1, 4096);
      integer(values.buffer_capacity, "buffer_capacity", 1, 1000000);
      integer(values.policy_delay, "policy_delay");
      for (const key of ["learning_rate", "discount", "tau", "policy_noise", "noise_clip"]) {
        finite(values[key], key);
      }
      if (values.learning_rate <= 0) throw new Error("learning_rate must be positive");
      if (values.discount < 0 || values.discount > 1 || values.tau < 0 || values.tau > 1) {
        throw new Error("discount and tau must lie in [0, 1]");
      }
      if (values.policy_noise < 0 || values.noise_clip < 0) throw new Error("Noise settings must be nonnegative");
      Object.assign(this, values);
      Object.freeze(this);
    }
  }

  class RNG {
    constructor(seed = 7) {
      integer(seed, "seed", 0, 0xffffffff);
      this.state = (seed >>> 0) || 0x6d2b79f5;
      this.spare = null;
    }
    uniform() {
      let value = this.state;
      value ^= value << 13;
      value ^= value >>> 17;
      value ^= value << 5;
      this.state = value >>> 0;
      return (this.state + 0.5) / 4294967296;
    }
    normal() {
      if (this.spare !== null) {
        const result = this.spare;
        this.spare = null;
        return result;
      }
      const radius = Math.sqrt(-2 * Math.log(this.uniform()));
      const angle = 2 * Math.PI * this.uniform();
      this.spare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    }
    toJSON() { return { algorithm: "xorshift32-boxmuller-v1", state: this.state, spare: this.spare }; }
    static fromJSON(value) {
      if (!value || value.algorithm !== "xorshift32-boxmuller-v1") throw new Error("Unsupported RNG state");
      integer(value.state, "RNG state", 1, 0xffffffff);
      if (value.spare !== null) finite(value.spare, "RNG spare");
      const rng = new RNG(value.state);
      rng.spare = value.spare;
      return rng;
    }
  }

  // Explicit little-endian float32 makes checkpoints portable and matches the
  // Python exporter. Base64 contains only numeric tensors, never executable code.
  function encodeFloat32(values) {
    const bytes = new Uint8Array(values.length * 4);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < values.length; i++) {
      const value = Math.fround(finite(values[i], "Checkpoint parameter"));
      finite(value, "Float32 checkpoint parameter");
      view.setFloat32(4 * i, value, true);
    }
    const parts = [];
    for (let i = 0; i < bytes.length; i += 8192) {
      parts.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
    }
    return btoa(parts.join(""));
  }

  function decodeFloat32(encoded, length, name = "Checkpoint tensor") {
    integer(length, "Tensor length", 0);
    const byteLength = length * 4;
    if (typeof encoded !== "string" || encoded.length !== 4 * Math.ceil(byteLength / 3)) {
      throw new Error(`${name} has the wrong encoded length`);
    }
    let binary;
    try { binary = atob(encoded); } catch (_) { throw new Error(`${name} is not valid base64`); }
    if (binary.length !== byteLength) throw new Error(`${name} has the wrong byte length`);
    const bytes = new Uint8Array(byteLength);
    for (let i = 0; i < byteLength; i++) bytes[i] = binary.charCodeAt(i);
    const view = new DataView(bytes.buffer);
    const values = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      values[i] = view.getFloat32(4 * i, true);
      finite(values[i], name);
    }
    return values;
  }

  class MLP {
    // Flat parameter order: row-major W1, b1, W2, b2, W3, b3.
    constructor(inputDim, hiddenDim, outputDim, activation = "linear", rng = new RNG(7)) {
      this.inputDim = integer(inputDim, "Network input dimension", 1, 8192);
      this.hiddenDim = integer(hiddenDim, "Network hidden dimension", 1, 1024);
      this.outputDim = integer(outputDim, "Network output dimension", 1, 1024);
      if (activation !== "linear" && activation !== "tanh") throw new Error("Unknown output activation");
      this.activation = activation;
      const widths = [inputDim, hiddenDim, hiddenDim, outputDim];
      this.layers = [];
      let size = 0;
      for (let i = 0; i < 3; i++) {
        const input = widths[i], output = widths[i + 1];
        const layer = { input_dim: input, output_dim: output, w_offset: size,
          b_offset: size + input * output, activation: i === 2 ? activation : "relu" };
        this.layers.push(layer);
        size += input * output + output;
      }
      this.params = new Float32Array(size);
      this.gradients = new Float64Array(size);
      this._workspaces = new Map();
      this._cache = null;
      if (rng !== null) {
        for (const layer of this.layers) {
          const bound = 1 / Math.sqrt(layer.input_dim);
          const end = layer.b_offset + layer.output_dim;
          for (let i = layer.w_offset; i < end; i++) this.params[i] = (2 * rng.uniform() - 1) * bound;
        }
      }
    }

    clone() {
      const result = new MLP(this.inputDim, this.hiddenDim, this.outputDim, this.activation, null);
      result.params.set(this.params);
      return result;
    }

    _workspace(batchSize) {
      let workspace = this._workspaces.get(batchSize);
      if (!workspace) {
        const widths = [this.inputDim, this.hiddenDim, this.hiddenDim, this.outputDim];
        workspace = {
          batchSize,
          values: widths.map(width => new Float64Array(batchSize * width)),
          deltas: widths.map(width => new Float64Array(batchSize * width)),
        };
        // Usually there are only inference (1) and training (batch_size) buffers.
        // Bound retained workspaces when a caller probes many different sizes.
        if (this._workspaces.size >= 4) this._workspaces.delete(this._workspaces.keys().next().value);
        this._workspaces.set(batchSize, workspace);
      }
      return workspace;
    }

    // @trace-start network_forward
    forward(inputs, batchSize = 1, retain = false) {
      integer(batchSize, "Network batch size", 1, 4096);
      finiteArray(inputs, batchSize * this.inputDim, "Network input");
      const workspace = this._workspace(batchSize);
      workspace.values[0].set(inputs);
      for (let index = 0; index < this.layers.length; index++) {
        const layer = this.layers[index], input = workspace.values[index], output = workspace.values[index + 1];
        const ni = layer.input_dim, no = layer.output_dim;
        for (let batch = 0; batch < batchSize; batch++) {
          const inputStart = batch * ni, outputStart = batch * no;
          for (let row = 0; row < no; row++) {
            let sum = this.params[layer.b_offset + row];
            const start = layer.w_offset + row * ni;
            for (let column = 0; column < ni; column++) sum += this.params[start + column] * input[inputStart + column];
            if (!Number.isFinite(sum)) throw new Error("Non-finite network activation; learning stopped");
            output[outputStart + row] = layer.activation === "relu" ? Math.max(0, sum)
              : layer.activation === "tanh" ? Math.tanh(sum) : sum;
          }
        }
      }
      this._cache = retain ? workspace : null;
      // This view is reused by the next forward call of the same batch size.
      return workspace.values[3];
    }
    // @trace-end network_forward

    zeroGrad() { this.gradients.fill(0); }

    backward(outputGradient, options = {}) {
      const accumulate = options.accumulateParameters !== false;
      const returnInput = options.inputGradient !== false;
      const workspace = this._cache;
      if (!workspace) throw new Error("Backward requires a retained forward pass");
      finiteArray(outputGradient, workspace.batchSize * this.outputDim, "Output gradient");
      workspace.deltas[3].set(outputGradient);
      for (let index = 2; index >= 0; index--) {
        const layer = this.layers[index], input = workspace.values[index], output = workspace.values[index + 1];
        const delta = workspace.deltas[index + 1], previous = workspace.deltas[index];
        const ni = layer.input_dim, no = layer.output_dim;
        const computePrevious = index > 0 || returnInput;
        if (computePrevious) previous.fill(0);
        for (let batch = 0; batch < workspace.batchSize; batch++) {
          const inputStart = batch * ni, outputStart = batch * no;
          for (let row = 0; row < no; row++) {
            const value = output[outputStart + row];
            let derivative = delta[outputStart + row];
            if (layer.activation === "relu" && value <= 0) derivative = 0;
            else if (layer.activation === "tanh") derivative *= 1 - value * value;
            if (!Number.isFinite(derivative)) throw new Error("Non-finite network gradient; learning stopped");
            if (accumulate) this.gradients[layer.b_offset + row] += derivative;
            const start = layer.w_offset + row * ni;
            for (let column = 0; column < ni; column++) {
              if (accumulate) this.gradients[start + column] += derivative * input[inputStart + column];
              if (computePrevious) previous[inputStart + column] += derivative * this.params[start + column];
            }
          }
        }
      }
      this._cache = null;
      return returnInput ? workspace.deltas[0] : null;
    }

    // @trace-start soft_update
    softUpdate(online, tau) {
      arrayLength(online.params, this.params.length, "Online parameters");
      for (let i = 0; i < this.params.length; i++) {
        this.params[i] = (1 - tau) * this.params[i] + tau * online.params[i];
        if (!Number.isFinite(this.params[i])) throw new Error("Non-finite target parameter; learning stopped");
      }
    }
    // @trace-end soft_update
  }

  class Adam {
    constructor(params, learningRate = 3e-4) {
      this.params = params;
      this.learningRate = finite(learningRate, "Adam learning rate");
      if (learningRate <= 0) throw new Error("Adam learning rate must be positive");
      this.m = new Float32Array(params.length);
      this.v = new Float32Array(params.length);
      this.t = 0;
    }

    step(gradients) {
      finiteArray(gradients, this.params.length, "Parameter gradient");
      this.t = integer(this.t + 1, "Adam step");
      const correction1 = 1 - Math.pow(0.9, this.t);
      const correction2 = Math.sqrt(1 - Math.pow(0.999, this.t));
      const stepSize = this.learningRate / correction1;
      for (let i = 0; i < this.params.length; i++) {
        const gradient = gradients[i];
        this.m[i] = 0.9 * this.m[i] + 0.1 * gradient;
        this.v[i] = 0.999 * this.v[i] + 0.001 * gradient * gradient;
        if (!Number.isFinite(this.m[i]) || !Number.isFinite(this.v[i])) {
          throw new Error("Non-finite Adam moment; learning stopped");
        }
        const denominator = Math.sqrt(this.v[i]) / correction2 + 1e-8;
        this.params[i] -= stepSize * this.m[i] / denominator;
        if (!Number.isFinite(this.params[i])) throw new Error("Non-finite learned parameter; learning stopped");
      }
    }

    toJSON() { return { step: this.t, m: encodeFloat32(this.m), v: encodeFloat32(this.v) }; }
    loadJSON(value) {
      if (!value || typeof value !== "object") throw new Error("Missing Adam state");
      const step = integer(value.step, "Adam step", 0);
      const m = decodeFloat32(value.m, this.params.length, "Adam first moment");
      const v = decodeFloat32(value.v, this.params.length, "Adam second moment");
      for (let i = 0; i < v.length; i++) if (v[i] < 0) throw new Error("Adam second moments must be nonnegative");
      this.m.set(m);
      this.v.set(v);
      this.t = step;
    }
  }

  class ReplayBuffer {
    constructor(obsDim, actionDim, capacity = 50000, seed = 7) {
      this.obsDim = integer(obsDim, "Observation dimension", 1, 4096);
      this.actionDim = integer(actionDim, "Action dimension", 1, 64);
      this.capacity = integer(capacity, "Replay capacity", 1, 1000000);
      this.observations = new Float32Array(capacity * obsDim);
      this.actions = new Float32Array(capacity * actionDim);
      this.rewards = new Float32Array(capacity);
      this.next_observations = new Float32Array(capacity * obsDim);
      this.not_done = new Float32Array(capacity);
      this.rng = new RNG(seed);
      this.position = 0;
      this._size = 0;
    }
    get length() { return this._size; }

    add(obs, action, reward, nextObs, terminated) {
      finiteArray(obs, this.obsDim, "Observation");
      finiteArray(nextObs, this.obsDim, "Next observation");
      finiteArray(action, this.actionDim, "Action");
      finite(Math.fround(finite(reward, "Reward")), "Float32 reward");
      if (![true, false, 0, 1].includes(terminated)) throw new Error("terminated must be a boolean or 0 or 1");
      for (let i = 0; i < action.length; i++) if (Math.abs(action[i]) > 1) throw new Error("Actions must lie in [-1, 1]");
      for (let i = 0; i < this.obsDim; i++) {
        finite(Math.fround(obs[i]), "Float32 observation");
        finite(Math.fround(nextObs[i]), "Float32 next observation");
      }
      this.observations.set(obs, this.position * this.obsDim);
      this.next_observations.set(nextObs, this.position * this.obsDim);
      this.actions.set(action, this.position * this.actionDim);
      this.rewards[this.position] = reward;
      // A timeout resets the rollout, but only true termination removes bootstrap.
      this.not_done[this.position] = 1 - Number(terminated);
      this.position = (this.position + 1) % this.capacity;
      this._size = Math.min(this._size + 1, this.capacity);
    }

    sample(batchSize, trace = false) {
      integer(batchSize, "Replay batch size", 1, 4096);
      if (!this._size) throw new Error("Cannot sample empty replay");
      const batch = {
        observations: new Float32Array(batchSize * this.obsDim),
        actions: new Float32Array(batchSize * this.actionDim),
        rewards: new Float32Array(batchSize),
        next_observations: new Float32Array(batchSize * this.obsDim),
        not_done: new Float32Array(batchSize), batch_size: batchSize,
      };
      if (trace) batch.indices = new Uint32Array(batchSize);
      for (let row = 0; row < batchSize; row++) {
        const index = Math.floor(this.rng.uniform() * this._size);
        if (trace) batch.indices[row] = index;
        for (let i = 0; i < this.obsDim; i++) {
          batch.observations[row * this.obsDim + i] = this.observations[index * this.obsDim + i];
          batch.next_observations[row * this.obsDim + i] = this.next_observations[index * this.obsDim + i];
        }
        for (let i = 0; i < this.actionDim; i++) batch.actions[row * this.actionDim + i] = this.actions[index * this.actionDim + i];
        batch.rewards[row] = this.rewards[index];
        batch.not_done[row] = this.not_done[index];
      }
      return batch;
    }
  }

  class TD3 {
    constructor(obsDim = 21, actionDim = 2, config = {}, seed = 7) {
      this.obsDim = integer(obsDim, "Observation dimension", 1, 4096);
      this.actionDim = integer(actionDim, "Action dimension", 1, 64);
      this.config = new TD3Config(config);
      const initialization = new RNG(seed), width = this.config.hidden_dim;
      this.actor = new MLP(obsDim, width, actionDim, "tanh", initialization);
      this.q1 = new MLP(obsDim + actionDim, width, 1, "linear", initialization);
      this.q2 = new MLP(obsDim + actionDim, width, 1, "linear", initialization);
      this.actorTarget = this.actor.clone();
      this.q1Target = this.q1.clone();
      this.q2Target = this.q2.clone();
      this.optimizers = {
        actor: new Adam(this.actor.params, this.config.learning_rate),
        q1: new Adam(this.q1.params, this.config.learning_rate),
        q2: new Adam(this.q2.params, this.config.learning_rate),
      };
      this.rng = new RNG((seed ^ 0x9e3779b9) >>> 0);
      this.updates = 0;
      this.actor_updates = 0;
      this._scratch = null;
    }

    selectAction(obs) {
      // @trace-start actor
      finiteArray(obs, this.obsDim, "Observation");
      return Float32Array.from(this.actor.forward(obs));
      // @trace-end actor
    }

    actionValues(obs, action) {
      finiteArray(obs, this.obsDim, "Observation");
      finiteArray(action, this.actionDim, "Action");
      const input = new Float32Array(this.obsDim + this.actionDim);
      input.set(obs);
      for (let i = 0; i < action.length; i++) {
        if (Math.abs(action[i]) > 1) throw new Error("Actions must lie in [-1, 1]");
        input[this.obsDim + i] = action[i];
      }
      return [this.q1.forward(input)[0], this.q2.forward(input)[0]];
    }

    _targetActions(nextObs, batchSize, output = new Float32Array(batchSize * this.actionDim), trace = null) {
      // @trace-start target_actions
      const proposed = this.actorTarget.forward(nextObs, batchSize);
      for (let i = 0; i < output.length; i++) {
        const rawNoise = this.rng.normal() * this.config.policy_noise;
        const noise = clamp(rawNoise, -this.config.noise_clip, this.config.noise_clip);
        output[i] = clamp(proposed[i] + noise, -1, 1);
        if (!Number.isFinite(output[i])) throw new Error("Non-finite target action");
        if (trace && i < this.actionDim) {
          trace.proposed_action.push(proposed[i]);
          trace.raw_noise.push(rawNoise);
          trace.clipped_noise.push(noise);
          trace.smoothed_action.push(output[i]);
        }
      }
      return output;
      // @trace-end target_actions
    }

    _join(observations, actions, output, batchSize) {
      const width = this.obsDim + this.actionDim;
      for (let row = 0; row < batchSize; row++) {
        for (let i = 0; i < this.obsDim; i++) output[row * width + i] = observations[row * this.obsDim + i];
        for (let i = 0; i < this.actionDim; i++) output[row * width + this.obsDim + i] = actions[row * this.actionDim + i];
      }
    }

    _workspace() {
      if (!this._scratch) {
        const size = this.config.batch_size, joined = size * (this.obsDim + this.actionDim);
        this._scratch = {
          current: new Float32Array(joined), next: new Float32Array(joined), actorInput: new Float32Array(joined),
          targetActions: new Float32Array(size * this.actionDim), target: new Float64Array(size),
          gradient1: new Float64Array(size), gradient2: new Float64Array(size),
          actorCriticGradient: new Float64Array(size), actorGradient: new Float64Array(size * this.actionDim),
        };
      }
      return this._scratch;
    }

    // Fast training and the walkthrough execute this same generator. Pausing at
    // a yield does not sample again, run a second forward pass, or alter RNGs.
    update(buffer) {
      const run = this.updateSteps(buffer, false);
      let next = run.next();
      while (!next.done) next = run.next();
      return next.value;
    }

    *updateSteps(buffer, trace = true) {
      if (!buffer || buffer.obsDim !== this.obsDim || buffer.actionDim !== this.actionDim) {
        throw new Error("Replay dimensions do not match the learner");
      }
      const event = (key, title, explanation, values) => ({ key, source: key, title, explanation, values });
      // @trace-start replay_sample
      const size = this.config.batch_size;
      const batch = buffer.sample(size, trace);
      const work = this._workspace();
      // @trace-end replay_sample
      if (trace) yield event("replay_sample", "Sample past experience", "This is the first row of the actual random minibatch, not necessarily the AUV's current position. Sampling is uniform with replacement.", {
        batch_size: size, replay_size: buffer.length, sample_index: batch.indices[0],
        observation: Array.from(batch.observations.subarray(0, this.obsDim)),
        action: Array.from(batch.actions.subarray(0, this.actionDim)),
        reward: batch.rewards[0], next_observation: Array.from(batch.next_observations.subarray(0, this.obsDim)),
        not_done: batch.not_done[0],
      });

      const targetTrace = trace ? { proposed_action: [], raw_noise: [], clipped_noise: [], smoothed_action: [] } : null;
      this._targetActions(batch.next_observations, size, work.targetActions, targetTrace);
      if (trace) yield event("target_actions", "Smooth the target action", "The target actor proposes the next action for each replay row. Clipped target noise is used only for learning; it does not steer the visible AUV.", {
        sample_index: batch.indices[0], policy_noise: this.config.policy_noise, noise_clip: this.config.noise_clip, ...targetTrace,
      });
      // @trace-start bellman_target
      this._join(batch.next_observations, work.targetActions, work.next, size);
      const next1 = this.q1Target.forward(work.next, size);
      const next2 = this.q2Target.forward(work.next, size);
      for (let i = 0; i < size; i++) {
        if (batch.not_done[i] !== 0 && batch.not_done[i] !== 1) throw new Error("Invalid replay continuation mask");
        work.target[i] = batch.rewards[i] + this.config.discount * batch.not_done[i] * Math.min(next1[i], next2[i]);
        finite(work.target[i], "TD target");
      }
      // @trace-end bellman_target
      if (trace) yield event("bellman_target", "Build the TD target", "For the same sampled row: reward + discount × not_done × the smaller target-critic estimate. True termination removes the future term; an external timeout retains it.", {
        sample_index: batch.indices[0], reward: batch.rewards[0], discount: this.config.discount,
        not_done: batch.not_done[0], target_q1: next1[0], target_q2: next2[0],
        min_target_q: Math.min(next1[0], next2[0]), target: work.target[0],
      });
      // @trace-start critic
      this._join(batch.observations, batch.actions, work.current, size);
      const current1 = this.q1.forward(work.current, size, true);
      const current2 = this.q2.forward(work.current, size, true);
      let loss = 0, q1Mean = 0, q2Mean = 0, error = 0;
      for (let i = 0; i < size; i++) {
        const first = current1[i] - work.target[i], second = current2[i] - work.target[i];
        loss += (first * first + second * second) / size;
        error += (Math.abs(first) + Math.abs(second)) / (2 * size);
        q1Mean += current1[i] / size;
        q2Mean += current2[i] / size;
        work.gradient1[i] = 2 * first / size;
        work.gradient2[i] = 2 * second / size;
      }
      finite(loss, "Critic loss");
      this.q1.zeroGrad(); this.q2.zeroGrad();
      this.q1.backward(work.gradient1, { inputGradient: false });
      this.q2.backward(work.gradient2, { inputGradient: false });
      finiteArray(this.q1.gradients, this.q1.params.length, "Q1 gradient");
      finiteArray(this.q2.gradients, this.q2.params.length, "Q2 gradient");
      this.optimizers.q1.step(this.q1.gradients);
      this.optimizers.q2.step(this.q2.gradients);
      this.updates++;
      // @trace-end critic
      if (trace) yield { ...event("critic", "Update both critics", "The two critics learn from this minibatch's target. The displayed Q values and TD error were measured before these optimizer steps.", {
        sample_index: batch.indices[0], first_q1: current1[0], first_q2: current2[0], first_target: work.target[0],
        critic_loss: loss, q1_mean: q1Mean, q2_mean: q2Mean, mean_absolute_td_error: error,
        critic_updates: this.updates,
      }), metrics: { critic_loss: loss, q1: q1Mean, q2: q2Mean, td_error: error, updates: this.updates } };
      let actorLoss = null;
      // @trace-start actor_skip
      const updateActor = this.updates % this.config.policy_delay === 0;
      // @trace-end actor_skip
      if (updateActor) {
        // @trace-start actor_update
        const proposed = this.actor.forward(batch.observations, size, true);
        this._join(batch.observations, proposed, work.actorInput, size);
        const policyValues = this.q1.forward(work.actorInput, size, true);
        actorLoss = 0;
        for (let i = 0; i < size; i++) actorLoss -= policyValues[i] / size;
        finite(actorLoss, "Actor loss");
        work.actorCriticGradient.fill(-1 / size);
        // Derivative through Q1's action input, without critic parameter updates.
        const inputGradient = this.q1.backward(work.actorCriticGradient, { accumulateParameters: false });
        const width = this.obsDim + this.actionDim;
        for (let row = 0; row < size; row++) {
          for (let i = 0; i < this.actionDim; i++) work.actorGradient[row * this.actionDim + i] = inputGradient[row * width + this.obsDim + i];
        }
        this.actor.zeroGrad();
        this.actor.backward(work.actorGradient, { inputGradient: false });
        this.optimizers.actor.step(this.actor.gradients);
        this.actor_updates++;
        // @trace-end actor_update
        if (trace) yield { ...event("actor_update", "Improve the actor through Q1", "This delayed update changes the actor to increase Q1's estimate. Gradients pass through Q1's action input, but this step does not update Q1's parameters.", {
          actor_loss: actorLoss, critic_updates: this.updates, actor_updates: this.actor_updates,
          policy_delay: this.config.policy_delay,
        }), metrics: { actor_loss: actorLoss, actor_updates: this.actor_updates } };
        const before = trace ? [this.actorTarget.params[0], this.q1Target.params[0], this.q2Target.params[0]] : null;
        // @trace-start soft_targets
        this.actorTarget.softUpdate(this.actor, this.config.tau);
        this.q1Target.softUpdate(this.q1, this.config.tau);
        this.q2Target.softUpdate(this.q2, this.config.tau);
        // @trace-end soft_targets
        if (trace) yield event("soft_targets", "Move the target networks slowly", "All three targets move a small fraction tau toward their online networks. These are actual first-weight values, in actor / Q1 / Q2 order.", {
          tau: this.config.tau, target_weights_before: before,
          online_weights: [this.actor.params[0], this.q1.params[0], this.q2.params[0]],
          target_weights_after: [this.actorTarget.params[0], this.q1Target.params[0], this.q2Target.params[0]],
        });
      } else if (trace) {
        yield event("actor_skip", "Keep the actor and targets unchanged", "This critic update is between delayed actor updates. The actor and all target networks keep their current parameters.", {
          critic_updates: this.updates, actor_updates: this.actor_updates,
          policy_delay: this.config.policy_delay, update_actor: false,
        });
      }
      // Q values and absolute TD error refer to the pre-update sampled batch.
      return { critic_loss: loss, actor_loss: actorLoss, q1: q1Mean, q2: q2Mean,
        td_error: error, updates: this.updates, actor_updates: this.actor_updates };
    }

    toJSON() {
      return {
        version: 1, obs_dim: this.obsDim, action_dim: this.actionDim, config: { ...this.config },
        updates: this.updates, actor_updates: this.actor_updates,
        networks: {
          actor: encodeFloat32(this.actor.params), actor_target: encodeFloat32(this.actorTarget.params),
          q1: encodeFloat32(this.q1.params), q1_target: encodeFloat32(this.q1Target.params),
          q2: encodeFloat32(this.q2.params), q2_target: encodeFloat32(this.q2Target.params),
        },
        optimizers: { actor: this.optimizers.actor.toJSON(), q1: this.optimizers.q1.toJSON(), q2: this.optimizers.q2.toJSON() },
        rng: this.rng.toJSON(),
      };
    }

    static fromJSON(payload) {
      if (!payload || payload.version !== 1) throw new Error("Unsupported TD3 checkpoint version");
      integer(payload.obs_dim, "Checkpoint observation dimension", 1, 4096);
      integer(payload.action_dim, "Checkpoint action dimension", 1, 64);
      if (!payload.config || typeof payload.config !== "object"
          || Object.keys(DEFAULTS).some(key => !Object.hasOwn(payload.config, key))) {
        throw new Error("Checkpoint configuration is incomplete");
      }
      integer(payload.updates, "Update count", 0);
      integer(payload.actor_updates, "Actor update count", 0);
      if (payload.actor_updates > payload.updates) throw new Error("Actor updates cannot exceed critic updates");
      const agent = new TD3(payload.obs_dim, payload.action_dim, payload.config);
      if (!payload.networks || !payload.optimizers) throw new Error("Checkpoint lacks networks or optimizers");
      const names = { actor: "actor", actor_target: "actorTarget", q1: "q1", q1_target: "q1Target", q2: "q2", q2_target: "q2Target" };
      for (const [key, attribute] of Object.entries(names)) {
        const network = agent[attribute];
        network.params.set(decodeFloat32(payload.networks[key], network.params.length, key));
      }
      for (const key of ["actor", "q1", "q2"]) {
        agent.optimizers[key].loadJSON(payload.optimizers[key]);
        const expected = key === "actor" ? payload.actor_updates : payload.updates;
        if (agent.optimizers[key].t !== expected) throw new Error(`${key} optimizer step does not match the learner counter`);
      }
      agent.rng = RNG.fromJSON(payload.rng);
      agent.updates = payload.updates;
      agent.actor_updates = payload.actor_updates;
      return agent;
    }
  }

  const api = { TD3, TD3Config, ReplayBuffer, MLP, Adam, RNG, encodeFloat32, decodeFloat32 };
  scope.AUVTD3 = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
