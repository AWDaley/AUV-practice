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
      this.trace = { enabled: false, role: "preview", sequence: 0, transition: 0, stage: null, at_boundary: true };
      this.traceIterator = null;
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
      this.finishTrace();
      this.generation++;
      if (this.timer !== null) clearTimeout(this.timer);
      this.timer = null;
      this.evaluationEnv = null;
      this.mode = "paused";
      if (this.training) this.training.steps_per_second = 0;
      this.message = message;
    }

    finishTrace() {
      // Never leave a consumed action, replay write, or optimizer update half
      // committed when another operation takes ownership of the experiment.
      if (this.trace?.enabled && this.traceIterator && !this.trace.at_boundary) {
        while (!this.trace.at_boundary) this.advanceTrace();
      }
      if (this.traceIterator) this.traceIterator.return();
      this.traceIterator = null;
      if (this.trace) this.trace.enabled = false;
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
      this.trace = { enabled: false, role: "preview", sequence: 0, transition: 0, stage: null, at_boundary: true };
      this.message = "Fresh TD3 policy. Train it here, or load the included trained example.";
      return this.state();
    }

    choose(controller, env, observation, rng = this.rng) {
      if (controller === "td3") return this.agent.selectAction(observation);
      if (controller === "baseline") return env.heuristicAction();
      if (controller === "random") return new Float32Array([rng.uniform(-1, 1), rng.uniform(-1, 1)]);
      throw new Error("Unknown controller.");
    }

    // Both ordinary playback/training and the code walkthrough use this exact
    // transition. The trace flag only copies values and exposes yield points.
    *transitionSteps(role, trace = false) {
      const training = role === "training";
      const event = (key, title, explanation, values) => ({ key, source: key, title, explanation, values });
      if (this.episodeDone) {
        // @trace-start reset
        this.obs = this.env.reset().observation;
        this.episodeDone = false;
        // @trace-end reset
        if (trace) yield event("reset", "Start the next episode", "The previous episode is finished. Reset creates a new course; replay and learned weights remain available.", {
          episode_step: this.env.episode_step, position: this.env.position.slice(), goal: this.env.goal.slice(),
        });
      }
      // @trace-start observation
      const observation = this.obs;
      // @trace-end observation
      if (trace) yield event("observation", "Read the current observation", "The actor receives 21 features: 16 normalized sonar ranges, 2 body-frame goal offsets, normalized speed, and 2 body-frame current components. This observation came from reset or the preceding step.", {
        observation: Array.from(observation), sonar: Array.from(observation.subarray(0, 16)),
        goal_body: Array.from(observation.subarray(16, 18)), speed_normalized: observation[18],
        current_body: Array.from(observation.subarray(19, 21)), episode_step: this.env.episode_step,
      });
      const randomWarmup = training && this.training.steps < this.warmupSteps;
      const controller = randomWarmup ? "random" : training ? "td3" : this.controller;
      if (controller === "random") {
        // @trace-start random
        this.action = this.choose("random", this.env, observation);
        // @trace-end random
      } else if (controller === "baseline") {
        // @trace-start baseline
        this.action = this.choose("baseline", this.env, observation);
        // @trace-end baseline
      } else {
        this.action = this.agent.selectAction(observation);
      }
      if (training) this.training.phase = randomWarmup ? "Random warm-up" : "Learning from replay";
      if (trace) yield event(controller === "td3" ? "actor" : controller,
        randomWarmup ? "Collect a random warm-up action" : controller === "td3" ? "Run the actor" : controller === "baseline" ? "Run the map-aware baseline" : "Sample a random action",
        randomWarmup ? "The first 1,000 cumulative training interactions collect random commands. The actor does not choose this action."
          : controller === "td3" ? "One forward pass produces acceleration and yaw commands in [-1, 1]. The critics do not approve or veto this command."
            : controller === "baseline" ? "The hand-written A* controller has the full obstacle map; it is separate from the learned TD3 actor."
              : "This controller samples commands without using a learned policy.",
        { action: Array.from(this.action), controller, random_warmup: randomWarmup,
          training_steps: this.training.steps, actor_shape: controller === "td3" ? [21, this.agent.config.hidden_dim, this.agent.config.hidden_dim, 2] : null });
      if (training && !randomWarmup) {
        const before = trace ? Array.from(this.action) : null, noise = trace ? [] : null;
        // @trace-start exploration
        for (let i = 0; i < 2; i++) {
          const perturbation = this.training.exploration_noise * this.rng.normal();
          this.action[i] = Math.max(-1, Math.min(1, this.action[i] + perturbation));
          if (trace) noise.push(perturbation);
        }
        // @trace-end exploration
        if (trace) yield event("exploration", "Add exploration noise", "This noise changes the command sent to the AUV during training. Preview uses the actor's action directly. Target smoothing is a different operation later in the learner.", {
          actor_action: before, exploration_std: this.training.exploration_noise, noise, executed_action: Array.from(this.action),
        });
      }
      const before = trace ? { position: this.env.position.slice(), speed: this.env.speed, heading: this.env.heading } : null;
      // @trace-start environment
      const result = this.env.step(this.action);
      this.obs = result.observation;
      this.episodeDone = result.terminated || result.truncated;
      // @trace-end environment
      if (trace) yield { ...event("environment", "Move the AUV and measure reward", "The environment applies acceleration, yaw, drag, and current, checks the swept path for a goal or collision, then returns the next observation and reward. Only this stage moves the AUV.", {
        action: Array.from(this.action), dt: this.env.config.dt,
        position_before: before.position, position_after: this.env.position.slice(),
        speed_before: before.speed, speed_after: this.env.speed,
        heading_before: before.heading, heading_after: this.env.heading,
        reward: result.reward, reward_terms: { ...this.env.reward_terms },
        terminated: result.terminated, truncated: result.truncated, outcome: this.env.outcome,
        goal_distance: Math.hypot(this.env.goal[0] - this.env.position[0], this.env.goal[1] - this.env.position[1]),
      }), details_source: "physics" };
      if (training) {
        const slot = this.replay.position;
        // @trace-start replay_add
        this.replay.add(observation, this.action, result.reward, result.observation, result.terminated);
        this.training.steps++;
        this.training.buffer_size = this.replay.length;
        // @trace-end replay_add
        if (trace) {
          this.training.total_steps = Math.max(this.training.total_steps, this.training.steps);
          yield event("replay_add", "Store this transition", "Replay stores the real final observation before any reset. Only true termination sets not_done to zero; a time limit retains the bootstrap.", {
            slot, replay_size: this.replay.length, training_steps: this.training.steps,
            reward: result.reward, terminated: result.terminated, truncated: result.truncated,
            not_done: 1 - Number(result.terminated),
          });
        }
        // @trace-start learning_wait
        const canLearn = this.training.steps > this.warmupSteps
          && this.replay.length >= this.agent.config.batch_size;
        // @trace-end learning_wait
        if (canLearn) {
          const update = this.agent.updateSteps(this.replay, trace);
          let next = update.next();
          while (!next.done) {
            if (next.value.metrics) Object.assign(this.training, next.value.metrics);
            yield next.value;
            next = update.next();
          }
          for (const [key, value] of Object.entries(next.value)) if (value !== null) this.training[key] = value;
        } else if (trace) {
          yield event("learning_wait", "Collect more experience before learning", "No optimizer update ran on this transition. A fresh learner needs to finish warm-up; a loaded learner still needs one full new replay batch.", {
            can_learn: false, training_steps: this.training.steps, warmup_steps: this.warmupSteps,
            replay_size: this.replay.length, batch_size: this.agent.config.batch_size,
          });
        }
        if (this.episodeDone) {
          // @trace-start episode
          const s = this.env.snapshot();
          this.training.episodes++;
          this.history.push({ episode: this.training.episodes, steps: this.training.steps,
            return: s.episode_return, length: s.episode_step, outcome: s.outcome });
          if (this.history.length > 500) this.history.shift();
          // @trace-end episode
          if (trace) yield event("episode", "Record the completed episode", "The outcome and return are added to training history. The final position stays visible until you advance into the next episode.", {
            outcome: s.outcome, episode_return: s.episode_return, episode_steps: s.episode_step,
            completed_episodes: this.training.episodes,
          });
        }
      }
      // @trace-start transition_end
      const summary = { episode_step: this.env.episode_step, outcome: this.env.outcome,
        training_steps: this.training.steps, critic_updates: this.agent.updates,
        actor_updates: this.agent.actor_updates, replay_size: this.replay.length };
      // @trace-end transition_end
      if (trace) yield event("transition_end", "One AUV transition complete", training
        ? "The environment step and all eligible learning updates are complete. Advance to inspect the next transition."
        : "Preview is complete for this step. It did not write replay or change any network weights.", summary);
      return result;
    }

    previewTick() {
      const run = this.transitionSteps("preview");
      while (!run.next().done) { /* Shared execution path, with tracing off. */ }
    }

    trainingTick() {
      const run = this.transitionSteps("training");
      while (!run.next().done) { /* Shared execution path, with tracing off. */ }
      // Fast training keeps its immediate reset. Walkthroughs hold the final
      // state until the next transition's explicit reset stage.
      if (this.episodeDone) {
        this.obs = this.env.reset().observation;
        this.episodeDone = false;
      }
    }

    advanceTrace() {
      if (!this.trace.enabled) throw new Error("Open Code walkthrough before stepping code.");
      if (this.trace.at_boundary && this.traceIterator) {
        this.traceIterator.return();
        this.traceIterator = null;
      }
      if (!this.traceIterator) {
        this.trace.transition++;
        this.trace.at_boundary = false;
        this.traceIterator = this.transitionSteps(this.trace.role, true);
      }
      let next;
      try { next = this.traceIterator.next(); }
      catch (error) {
        this.traceIterator = null;
        this.trace.enabled = false;
        this.trace.at_boundary = true;
        this.mode = "error";
        throw error;
      }
      if (next.done) throw new Error("The walkthrough ended without a transition boundary.");
      this.trace.sequence++;
      this.trace.stage = { ...next.value, sequence: this.trace.sequence,
        transition: this.trace.transition, episode_step: this.env.episode_step,
        training_steps: this.training.steps };
      this.trace.at_boundary = next.value.key === "transition_end";
      this.message = `${next.value.title}. ${this.trace.role === "training" ? "Training" : "Preview"} code walkthrough.`;
      return this.trace.stage;
    }

    traceControl({ command, role = "preview", controller = "td3", exploration_noise = this.training.exploration_noise } = {}) {
      if (!["start", "step", "tick", "exit"].includes(command)) throw new Error("Unknown code walkthrough command.");
      if (command === "exit") {
        this.stop("Code walkthrough paused. The current transition is complete; weights and replay remain in this tab.");
        return this.state();
      }
      if (command === "start") {
        if (!["preview", "training"].includes(role)) throw new Error("Choose preview or training for the walkthrough.");
        if (!["td3", "baseline", "random"].includes(controller)) throw new Error("Unknown controller.");
        bounded(exploration_noise, 0, 1, "Exploration noise");
        this.stop();
        this.controller = role === "training" ? "td3" : controller;
        if (this.envRole !== role) {
          this.obs = this.env.reset().observation;
          this.episodeDone = false;
        }
        this.envRole = role;
        this.mode = "walkthrough";
        this.training.steps_per_second = 0;
        if (role === "training") this.training.exploration_noise = exploration_noise;
        this.trace = { enabled: true, role, sequence: 0, transition: 0, stage: null, at_boundary: true };
      }
      this.advanceTrace();
      if (command === "tick") while (!this.trace.at_boundary) this.advanceTrace();
      return this.state();
    }

    control({ command, controller = "td3", speed = 4 }) {
      if (!["play", "pause", "step"].includes(command)) throw new Error("Unknown playback command.");
      if (!["td3", "baseline", "random"].includes(controller)) throw new Error("Unknown controller.");
      bounded(speed, 0.1, 32, "Playback speed");
      if (command === "pause" && this.trace.enabled) {
        this.message = "Code walkthrough paused at the current stage.";
        return this.state();
      }
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
          this.trainingTick();
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
      this.trace = { enabled: false, role: "preview", sequence: 0, transition: 0, stage: null, at_boundary: true };
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
        evaluation: clone(this.evaluation), config: { ...this.config }, trace: clone(this.trace),
        checkpoints: [...this.saved.values()].map(p => ({ name: p.name, steps: p.training.steps, created: p.created })).reverse(),
        action: Array.from(this.evaluationEnv ? this.evaluationAction : this.action), policy_ready: this.training.actor_updates > 0 };
    }

    request(path, payload = {}) {
      switch (path) {
        case "/api/state": return this.state();
        case "/api/reset": return this.newExperiment(payload);
        case "/api/control": return this.control(payload);
        case "/api/trace": return this.traceControl(payload);
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
