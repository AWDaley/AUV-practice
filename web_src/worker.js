/* Single-user simulation lifecycle. All computation runs inside a Blob Worker. */
(() => {
  "use strict";
  const { AUVEnv, SeededRNG } = globalThis.AUVEnvironment;
  const { TD3, ReplayBuffer } = globalThis.AUVTD3;
  const now = () => performance.now();
  const clone = value => JSON.parse(JSON.stringify(value));
  const bounded = (value, low, high, name, integer = false) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high || (integer && !Number.isInteger(value))) {
      throw new Error(`${name} must be ${integer ? "an integer " : ""}between ${low} and ${high}.`);
    }
    return value;
  };
  const trainingDefaults = () => ({
    steps: 0, total_steps: 0, updates: 0, actor_updates: 0, episodes: 0,
    buffer_size: 0, exploration_noise: 0.15, actor_loss: null, critic_loss: null,
    q1: null, q2: null, td_error: null, steps_per_second: 0, phase: "Fresh browser policy · 64 units",
  });

  class BrowserLab {
    constructor(demo = null) {
      this.saved = new Map();
      this.timer = null;
      this.generation = 0;
      this.saveCounter = 0;
      this.mode = "idle";
      this.warmupSteps = 1000;
      this.newExperiment({ seed: 7, scenario: "reef", current_strength: 0.08 });
      if (demo) {
        this.importCheckpoint(demo, true);
        this.message = "Trained TD3 example ready. Press Play, or reset to train a fresh browser policy.";
      }
    }

    validateConfig(config = {}) {
      if (!["open", "reef", "channel"].includes(config.scenario)) throw new Error("Choose open water, reef, or channel.");
      return {
        scenario: config.scenario,
        seed: bounded(config.seed, 0, 2147483647, "Seed", true),
        current_strength: bounded(config.current_strength, 0, 0.4, "Current"),
      };
    }

    stop(message = "Paused. Weights and replay remain in this tab.") {
      this.generation++;
      if (this.timer !== null) clearTimeout(this.timer);
      this.timer = null;
      this.evaluationEnv = null;
      this.mode = "paused";
      if (this.training) this.training.steps_per_second = 0;
      this.message = message;
    }

    schedule(callback, delay = 0, generation = this.generation) {
      this.timer = setTimeout(() => {
        if (generation !== this.generation) return;
        this.timer = null;
        try { callback(); } catch (error) {
          this.stop(String(error.message || error));
          this.mode = "error";
        }
      }, delay);
    }

    newExperiment(config) {
      const checked = this.validateConfig(config);
      const env = new AUVEnv(checked);
      const agent = new TD3(21, 2, { hidden_dim: 64, batch_size: 32, buffer_capacity: 50000 }, checked.seed);
      const replay = new ReplayBuffer(21, 2, 50000, checked.seed + 2);
      this.stop();
      this.config = checked;
      this.env = env;
      this.obs = env.reset(checked.seed).observation;
      this.agent = agent;
      this.replay = replay;
      this.rng = new SeededRNG(checked.seed + 1);
      this.training = trainingDefaults();
      this.history = [];
      this.evaluation = null;
      this.action = new Float32Array(2);
      this.evaluationAction = new Float32Array(2);
      this.envRole = "preview";
      this.episodeDone = false;
      this.controller = "td3";
      this.mode = "idle";
      this.message = "Fresh TD3 policy. Train it here, or load the included trained example.";
      return this.state();
    }

    choose(controller, env, observation, rng = this.rng) {
      if (controller === "td3") return this.agent.selectAction(observation);
      if (controller === "baseline") return env.heuristicAction();
      if (controller === "random") return new Float32Array([rng.uniform(-1, 1), rng.uniform(-1, 1)]);
      throw new Error("Unknown controller.");
    }

    previewTick() {
      if (this.episodeDone) {
        this.obs = this.env.reset().observation;
        this.episodeDone = false;
      }
      this.action = this.choose(this.controller, this.env, this.obs);
      const result = this.env.step(this.action);
      this.obs = result.observation;
      this.episodeDone = result.terminated || result.truncated;
    }

    control({ command, controller = "td3", speed = 4 }) {
      if (!["play", "pause", "step"].includes(command)) throw new Error("Unknown playback command.");
      if (!["td3", "baseline", "random"].includes(controller)) throw new Error("Unknown controller.");
      bounded(speed, 0.1, 32, "Playback speed");
      this.stop();
      if (command === "pause") return this.state();
      this.controller = controller;
      if (this.envRole !== "preview") {
        this.obs = this.env.reset().observation;
        this.episodeDone = false;
      }
      this.envRole = "preview";
      if (command === "step") {
        this.previewTick();
        this.message = "Advanced one simulation step. Preview does not train the model.";
        return this.state();
      }
      this.mode = "preview";
      this.message = controller === "baseline"
        ? "Map-aware baseline: hand-written A* guidance with access to all obstacles."
        : controller === "td3" ? "Running the learned actor without exploration or weight updates."
          : "Random commands; no learning updates.";
      const generation = this.generation;
      const tick = () => {
        const start = now();
        this.previewTick();
        this.schedule(tick, this.episodeDone ? 650 : Math.max(0, 100 / speed - (now() - start)), generation);
      };
      this.schedule(tick, 0, generation);
      return this.state();
    }

    startTraining({ steps = 2000, exploration_noise = 0.15 }) {
      bounded(steps, 1, 1000000, "Additional steps", true);
      bounded(exploration_noise, 0, 1, "Exploration noise");
      this.stop();
      if (this.envRole !== "training" || this.episodeDone) {
        this.obs = this.env.reset().observation;
        this.episodeDone = false;
      }
      this.envRole = "training";
      this.controller = "td3";
      this.mode = "training";
      const startSteps = this.training.steps;
      const target = startSteps + steps;
      const startTime = now();
      const generation = this.generation;
      this.training.total_steps = target;
      this.training.exploration_noise = exploration_noise;
      this.message = `Training locally in a worker (${this.agent.config.hidden_dim} hidden units, batch ${this.agent.config.batch_size}).`;
      const chunk = () => {
        const deadline = now() + 12;
        do {
          if (this.training.steps >= target) {
            this.mode = "paused";
            this.training.steps_per_second = 0;
            this.training.phase = "Training budget complete";
            this.message = "Budget complete. Preview or evaluate TD3 without exploration noise.";
            return;
          }
          if (this.training.steps < this.warmupSteps) {
            this.action = this.choose("random", this.env, this.obs);
            this.training.phase = "Random warm-up";
          } else {
            this.action = this.agent.selectAction(this.obs);
            for (let i = 0; i < 2; i++) this.action[i] = Math.max(-1, Math.min(1, this.action[i] + exploration_noise * this.rng.normal()));
            this.training.phase = "Learning from replay";
          }
          const result = this.env.step(this.action);
          this.replay.add(this.obs, this.action, result.reward, result.observation, result.terminated);
          this.obs = result.observation;
          this.training.steps++;
          this.training.buffer_size = this.replay.length;
          if (this.training.steps > this.warmupSteps && this.replay.length >= this.agent.config.batch_size) {
            const metrics = this.agent.update(this.replay);
            for (const [key, value] of Object.entries(metrics)) if (value !== null) this.training[key] = value;
          }
          if (result.terminated || result.truncated) {
            const s = this.env.snapshot();
            this.training.episodes++;
            this.history.push({ episode: this.training.episodes, steps: this.training.steps,
              return: s.episode_return, length: s.episode_step, outcome: s.outcome });
            if (this.history.length > 500) this.history.shift();
            this.obs = this.env.reset().observation;
          }
          this.training.steps_per_second = 1000 * (this.training.steps - startSteps) / Math.max(1, now() - startTime);
        } while (now() < deadline);
        this.schedule(chunk, 0, generation);
      };
      this.schedule(chunk, 0, generation);
      return this.state();
    }

    startEvaluation({ episodes = 10, controller = "td3" }) {
      bounded(episodes, 1, 100, "Evaluation episodes", true);
      if (!["td3", "baseline", "random"].includes(controller)) throw new Error("Unknown controller.");
      this.stop();
      this.mode = "evaluating";
      this.controller = controller;
      this.evaluation = null;
      this.message = "Evaluating on fixed JavaScript seeds. The policy and replay remain unchanged.";
      const generation = this.generation;
      const seed = this.config.seed + 10000;
      const env = new AUVEnv(this.config);
      const rng = new SeededRNG(seed);
      let observation = env.reset(seed).observation;
      const records = [];
      this.evaluationEnv = env;
      this.evaluationAction = new Float32Array(2);
      const chunk = () => {
        const deadline = now() + 12;
        do {
          this.evaluationAction = this.choose(controller, env, observation, rng);
          const result = env.step(this.evaluationAction);
          observation = result.observation;
          if (result.terminated || result.truncated) {
            const s = env.snapshot();
            records.push({ return: s.episode_return, steps: s.episode_step, outcome: s.outcome });
            this.message = `Evaluating ${controller}: ${records.length}/${episodes} complete.`;
            if (records.length === episodes) {
              const mean = key => records.reduce((sum, r) => sum + r[key], 0) / episodes;
              const rate = outcome => records.filter(r => r.outcome === outcome).length / episodes;
              this.evaluation = { controller, episodes, success_rate: rate("success"),
                collision_rate: rate("collision"), timeout_rate: rate("timeout"),
                mean_return: mean("return"), mean_steps: mean("steps"), seed,
                training_steps: this.training.steps, scenario: this.config.scenario, engine: "JavaScript", records };
              this.evaluationEnv = null;
              this.mode = "paused";
              this.message = "Evaluation complete. No learning updates or replay writes were performed.";
              return;
            }
            observation = env.reset(seed + records.length).observation;
            this.evaluationAction = new Float32Array(2);
          }
        } while (now() < deadline);
        this.schedule(chunk, 0, generation);
      };
      this.schedule(chunk, 0, generation);
      return this.state();
    }

    prepareCheckpoint(payload) {
      if (!payload || payload.format !== "auv-practice-browser" || payload.version !== 1) {
        throw new Error("Choose an AUV browser checkpoint JSON file. Python .pt files cannot be imported here.");
      }
      if (typeof payload.name !== "string" || !/^td3-[A-Za-z0-9_-]{1,90}\.json$/.test(payload.name)) throw new Error("Invalid checkpoint name.");
      const config = this.validateConfig(payload.config);
      const source = payload.learner;
      if (!source || source.obs_dim !== 21 || source.action_dim !== 2) throw new Error("Checkpoint needs 21 observations and 2 actions.");
      bounded(source.config?.hidden_dim, 1, 256, "Hidden width", true);
      bounded(source.config?.buffer_capacity, 1, 100000, "Replay capacity", true);
      bounded(source.config?.batch_size, 1, 256, "Batch size", true);
      const agent = TD3.fromJSON(source);
      const training = trainingDefaults();
      const input = payload.training;
      if (!input || typeof input !== "object") throw new Error("Checkpoint is missing training metadata.");
      for (const key of ["steps", "total_steps", "updates", "actor_updates", "episodes"]) training[key] = bounded(input[key], 0, Number.MAX_SAFE_INTEGER, key, true);
      if (training.updates !== agent.updates || training.actor_updates !== agent.actor_updates) throw new Error("Checkpoint counters do not match the learner.");
      training.exploration_noise = bounded(input.exploration_noise, 0, 1, "Exploration noise");
      for (const key of ["critic_loss", "actor_loss", "q1", "q2", "td_error"]) {
        if (input[key] !== null) {
          if (typeof input[key] !== "number" || !Number.isFinite(input[key])) throw new Error("Invalid checkpoint metric.");
          training[key] = input[key];
        }
      }
      if (!Array.isArray(payload.history) || payload.history.length > 500) throw new Error("Invalid episode history.");
      const history = payload.history.map(row => {
        const copy = {};
        for (const key of ["episode", "steps", "length"]) copy[key] = bounded(row[key], 0, Number.MAX_SAFE_INTEGER, key, true);
        if (!Number.isFinite(row.return) || !["success", "collision", "timeout"].includes(row.outcome)) throw new Error("Invalid episode record.");
        copy.return = row.return; copy.outcome = row.outcome;
        return copy;
      });
      const env = new AUVEnv(config);
      const observation = env.reset(config.seed).observation;
      const replay = new ReplayBuffer(21, 2, agent.config.buffer_capacity, config.seed + 2);
      training.phase = `Loaded policy · ${agent.config.hidden_dim} units`;
      return { config, agent, training, history, env, observation, replay };
    }

    importCheckpoint(payload, included = false) {
      const prepared = this.prepareCheckpoint(payload);
      this.stop();
      this.config = prepared.config;
      this.agent = prepared.agent;
      this.training = prepared.training;
      this.history = prepared.history;
      this.env = prepared.env;
      this.obs = prepared.observation;
      this.replay = prepared.replay;
      this.rng = new SeededRNG(this.config.seed + 1);
      this.action = new Float32Array(2);
      this.evaluationAction = new Float32Array(2);
      this.evaluation = null;
      this.envRole = "preview";
      this.episodeDone = false;
      this.controller = "td3";
      this.saved.set(payload.name, clone(payload));
      this.message = included ? "Included trained example loaded. Press Play."
        : "Loaded browser policy. Replay restarts; this does not restore an exact running experiment.";
      return this.state();
    }

    checkpoint() {
      const stamp = new Date().toISOString();
      const name = `td3-${this.training.steps}-${stamp.replace(/[^0-9]/g, "")}-${++this.saveCounter}.json`;
      return { format: "auv-practice-browser", version: 1, name, created: stamp,
        config: clone(this.config), learner: this.agent.toJSON(), training: clone(this.training),
        history: clone(this.history), evaluation: clone(this.evaluation), source: "AUV Practice browser",
        resume_note: "Replay, simulator state and rollout RNG restart on import." };
    }

    save() {
      this.stop();
      const checkpoint = this.checkpoint();
      this.saved.set(checkpoint.name, checkpoint);
      this.message = "Checkpoint saved in this tab and exported as JSON. Keep the download to use it after closing.";
      return { state: this.state(), download: { filename: checkpoint.name, payload: checkpoint } };
    }

    load(name) {
      if (!this.saved.has(name)) throw new Error("Choose a checkpoint from this tab, or import a JSON file.");
      return this.importCheckpoint(this.saved.get(name));
    }

    results() {
      return { config: clone(this.config), learner_config: { ...this.agent.config },
        training: clone(this.training), history: clone(this.history), evaluation: clone(this.evaluation),
        engine: "JavaScript", limitations: "2D idealized navigation; JavaScript and NumPy random sequences differ." };
    }

    state() {
      const env = this.evaluationEnv || this.env;
      return { mode: this.mode, controller: this.controller, message: this.message, env: env.snapshot(),
        training: { ...this.training }, history: this.history.map(row => ({ ...row })),
        evaluation: clone(this.evaluation), config: { ...this.config },
        checkpoints: [...this.saved.values()].map(p => ({ name: p.name, steps: p.training.steps, created: p.created })).reverse(),
        action: Array.from(this.evaluationEnv ? this.evaluationAction : this.action), policy_ready: this.training.actor_updates > 0 };
    }

    request(path, payload = {}) {
      switch (path) {
        case "/api/state": return this.state();
        case "/api/reset": return this.newExperiment(payload);
        case "/api/control": return this.control(payload);
        case "/api/train": return this.startTraining(payload);
        case "/api/evaluate": return this.startEvaluation(payload);
        case "/api/save": return this.save();
        case "/api/load": return this.load(payload.name);
        case "/api/import": return this.importCheckpoint(payload.checkpoint);
        case "/api/results": return this.results();
        case "/api/export-results": return { state: this.state(), download: { filename: "auv-results.json", payload: this.results() } };
        default: throw new Error("Unknown browser operation.");
      }
    }
  }

  globalThis.AUVLab = { BrowserLab };
  if (typeof module !== "undefined" && module.exports) module.exports = { BrowserLab };
  if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
    let lab;
    self.onmessage = event => {
      const { id, path, payload } = event.data;
      try {
        let data;
        if (path === "/init") { lab = new BrowserLab(payload.demo); data = lab.state(); }
        else {
          if (!lab) throw new Error("The local engine has not initialized yet.");
          data = lab.request(path, payload);
        }
        self.postMessage({ id, ok: true, data });
      } catch (error) { self.postMessage({ id, ok: false, error: String(error.message || error) }); }
    };
  }
})();
