/* Live browser stages connected to the original Python reference or executing JavaScript. */
(() => {
  "use strict";

  const groupFor = {
    reset: "observe", observation: "observe", actor: "act", random: "act", baseline: "act",
    exploration: "act", environment: "move", replay_add: "store", replay_sample: "learn",
    target_actions: "learn", bellman_target: "learn", critic: "learn", actor_update: "learn",
    actor_skip: "learn", soft_targets: "learn", learning_wait: "learn", episode: "finish",
    transition_end: "finish",
  };
  const groupNames = { observe: "Observe", act: "Choose an action", move: "Step the environment",
    store: "Write replay", learn: "Learn from replay", finish: "Complete the transition" };
  const detailFor = { observation: "observation_features", actor: "network_forward",
    environment: "physics", soft_targets: "soft_update" };
  const detailTitles = { observation_features: "See how the 21 features are calculated",
    network_forward: "See the actor's matrix operations and activations",
    physics: "See motion, collision checks, and the reward calculation",
    soft_update: "See the target-weight interpolation",
    replay_sampling: "See how the Python replay buffer samples a batch",
    replay_storage: "See how the Python replay buffer stores a transition",
    critic_forward: "See the two Python critic networks" };
  const pythonDetailFor = { ...detailFor, replay_sample: "replay_sampling", replay_add: "replay_storage",
    critic: "critic_forward" };
  const focusFor = { bellman_target: "work.target[i] =", critic: "this.optimizers.q1.step",
    actor_update: "this.optimizers.actor.step", target_actions: "output[i] = clamp" };
  const learningSamples = new Set(["replay_sample", "target_actions", "bellman_target", "critic", "actor_update"]);
  const featureNames = Array.from({ length: 21 }, (_, i) => i < 16 ? `sonar ${i}`
    : ["goal forward", "goal left", "speed", "current forward", "current left"][i - 16]);
  const shortExplanation = {
    reset: "Start the next episode. The policy and replay stay available.",
    observation: "Read 21 normalized features describing sonar, the goal, speed, and current.",
    actor: "One neural-network pass chooses acceleration and turn.",
    random: "Choose a random acceleration and turn command.",
    baseline: "The map-aware controller follows its planned route.",
    exploration: "Add sampled exploration noise, then clip the command to [−1, 1].",
    environment: "Apply the command, move the sub, and check the reward and outcome.",
    replay_add: "Save this transition and its real successor before any episode reset.",
    replay_sample: "Draw a batch of past experience. This row may differ from the live tank.",
    target_actions: "Smooth the target actor's action for learning with clipped noise.",
    bellman_target: "Add the reward to the discounted, smaller target-Q estimate.",
    critic: "Fit both critics to the targets. These batch values were measured before the update.",
    actor_update: "Improve the actor through Q1's action gradient.",
    actor_skip: "The delay rule skips actor and target updates on this critic step.",
    soft_targets: "Move target weights a fraction τ toward the online networks.",
    learning_wait: "Collect more experience before running an optimizer update.",
    episode: "Record the outcome. The final position stays visible until the next reset.",
    transition_end: "This control step and its eligible learning work are complete.",
  };

  function keyValues(stage) {
    const v = stage.values || {};
    const pick = (key, label, unit = "") => ({ key, label, value: v[key], unit });
    const action = key => [
      { key: `${key}[0]`, label: "Acceleration", value: v[key]?.[0] },
      { key: `${key}[1]`, label: "Turn", value: v[key]?.[1] },
    ];
    let items;
    switch (stage.key) {
      case "observation": items = [pick("goal_body", "Goal offset"), pick("speed_normalized", "Normalized speed"),
        pick("current_body", "Body-frame current"), { key: "observation.length", label: "Input features", value: v.observation?.length }]; break;
      case "actor": items = [...action("action"), { key: "actor_shape", label: "Network shape", value: v.actor_shape?.join(" → ") },
        pick("controller", "Controller")]; break;
      case "random": case "baseline": items = [...action("action"), pick("controller", "Controller"), pick("random_warmup", "Random warm-up")]; break;
      case "exploration": items = [pick("actor_action", "Actor command"), pick("noise", "Sampled noise"),
        pick("executed_action", "Applied command"), pick("exploration_std", "Noise scale")]; break;
      case "environment": items = [pick("position_after", "Position", "m"), pick("goal_distance", "Goal distance", "m"),
        pick("reward", "Reward"), pick("outcome", "Outcome")]; break;
      case "replay_add": items = [pick("reward", "Stored reward"), pick("not_done", "Bootstrap mask"),
        pick("replay_size", "Stored transitions"), pick("slot", "Replay slot")]; break;
      case "replay_sample": items = [pick("action", "Stored command"), pick("reward", "Stored reward"),
        pick("not_done", "Bootstrap mask"), pick("batch_size", "Batch size")]; break;
      case "target_actions": items = [pick("proposed_action", "Target proposal"), pick("raw_noise", "Sampled noise"),
        pick("clipped_noise", "Clipped noise"), pick("smoothed_action", "Smoothed action")]; break;
      case "bellman_target": items = [pick("reward", "Reward"), pick("min_target_q", "Smaller target Q"),
        pick("not_done", "Bootstrap mask"), pick("target", "TD target")]; break;
      case "critic": items = [pick("critic_loss", "Critic loss"), pick("mean_absolute_td_error", "Mean |TD error|"),
        pick("q1_mean", "Batch Q1"), pick("q2_mean", "Batch Q2")]; break;
      case "actor_update": items = [pick("actor_loss", "Actor loss"), pick("actor_updates", "Actor updates"),
        pick("critic_updates", "Critic updates"), pick("policy_delay", "Update interval")]; break;
      case "actor_skip": items = [pick("update_actor", "Update actor"), pick("policy_delay", "Update interval"),
        pick("critic_updates", "Critic updates"), pick("actor_updates", "Actor updates")]; break;
      case "soft_targets": items = [pick("tau", "Target mix · τ"), ...["Actor", "Q1", "Q2"].map((name, i) => ({
        key: `target_weights_before[${i}] → target_weights_after[${i}]`, label: `${name} target · first weight`,
        value: `${v.target_weights_before[i].toFixed(6)} → ${v.target_weights_after[i].toFixed(6)}`,
      }))]; break;
      case "learning_wait": items = [pick("training_steps", "Collected steps"), pick("warmup_steps", "Warm-up requirement"),
        pick("replay_size", "Stored transitions"), pick("batch_size", "Required batch size")]; break;
      case "episode": items = [pick("outcome", "Outcome"), pick("episode_return", "Episode return"),
        pick("episode_steps", "Episode steps"), pick("completed_episodes", "Completed episodes")]; break;
      case "reset": items = [pick("position", "Start position", "m"), pick("goal", "Goal", "m"), pick("episode_step", "Episode step")]; break;
      case "transition_end": items = [pick("episode_step", "Episode step"), pick("outcome", "Outcome"),
        pick("critic_updates", "Critic updates"), pick("actor_updates", "Actor updates")]; break;
      default: items = Object.keys(v).slice(0, 4).map(key => pick(key, key));
    }
    return items.filter(item => item.value !== undefined && item.value !== null);
  }

  function valueText(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return String(value);
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Number.isInteger(value)) return String(value);
      if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(3);
      return value.toFixed(4);
    }
    if (Array.isArray(value)) return `[${value.map(valueText).join(", ")}]`;
    if (typeof value === "object") return Object.entries(value).map(([key, entry]) => `${key}: ${valueText(entry)}`).join(" · ");
    return String(value);
  }

  function highlightedLine(text, language) {
    const fragment = document.createDocumentFragment();
    // Token styling only. Source text is never interpreted as HTML or executed.
    const token = language === "python"
      ? /(#.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|class|return|if|elif|else|for|while|in|with|as|import|from|not|and|or|is|None|True|False|self|raise|try|except|finally|pass|yield|break|continue)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g
      : /(\/\/.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b(?:const|let|return|if|else|for|while|this|new|throw|true|false|null|typeof|yield)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g;
    let from = 0, match;
    while ((match = token.exec(text)) !== null) {
      fragment.append(document.createTextNode(text.slice(from, match.index)));
      const span = document.createElement("span");
      span.className = "trace-token-" + (match[0].startsWith(language === "python" ? "#" : "//") ? "comment"
        : /^["'`]/.test(match[0]) ? "string" : /^\d/.test(match[0]) ? "number" : "keyword");
      span.textContent = match[0];
      fragment.append(span);
      from = token.lastIndex;
    }
    fragment.append(document.createTextNode(text.slice(from)));
    return fragment;
  }

  function showSource(container, source, focusText) {
    container.replaceChildren();
    container.dataset.language = source?.language || "javascript";
    if (!source) {
      container.textContent = "Source excerpt unavailable. Rebuild the HTML from its current sources.";
      return;
    }
    const nonempty = source.lines.filter(line => line.trim());
    const indent = nonempty.length ? Math.min(...nonempty.map(line => line.match(/^\s*/)[0].length)) : 0;
    const fragment = document.createDocumentFragment();
    source.lines.forEach((text, index) => {
      const line = document.createElement("span");
      line.className = "trace-code-line";
      if (source.focus?.includes(source.start_line + index) || focusText && text.includes(focusText)) line.classList.add("is-focus");
      const number = document.createElement("span");
      number.className = "trace-line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = String(source.start_line + index);
      const code = document.createElement("span");
      code.className = "trace-line-text";
      code.append(highlightedLine(text.slice(indent), source.language || "javascript"));
      line.append(number, code);
      fragment.append(line);
    });
    container.append(fragment);
  }

  window.createAUVTraceView = ({ elements, getState, sendCommand, setText, redraw, refreshControls, isLocked }) => {
    const sources = JSON.parse(document.getElementById("auv-trace-source").textContent);
    const pythonReference = JSON.parse(document.getElementById("auv-python-trace-source").textContent);
    let sourceView = "python";
    let playing = false;
    let timer = null;
    let generation = 0;
    let signature = "";
    let visible = false;
    let sourceKey = "";
    let lastRole = "preview";

    function stopPlaying() {
      playing = false;
      generation++;
      window.clearTimeout(timer);
      timer = null;
      refreshControls();
    }

    function updateControls({ locked = isLocked() } = {}) {
      const enabled = Boolean(getState()?.trace?.enabled);
      elements["code-view"].disabled = locked || enabled;
      elements["simulation-view"].disabled = locked || !enabled;
      elements["trace-play"].disabled = !enabled || (locked && !playing);
      elements["trace-next"].disabled = locked || !enabled || playing;
      elements["trace-tick"].disabled = locked || !enabled || playing;
      elements["trace-role"].disabled = locked || playing;
      elements["trace-noise"].disabled = locked || playing;
      elements["trace-source-view"].disabled = locked || !enabled;
      setText("trace-play-label", playing ? "Pause blocks" : "Play blocks");
      elements["trace-play"].setAttribute("aria-pressed", String(playing));
      elements["trace-play"].querySelector("path").setAttribute("d", playing ? "M5 4h3v12H5zm7 0h3v12h-3z" : "m6 3 11 7-11 7z");
    }

    async function startTrace() {
      stopPlaying();
      const noise = elements["trace-noise"];
      if (!noise.checkValidity()) { noise.reportValidity(); return; }
      const role = elements["trace-role"].value;
      const data = await sendCommand("/api/trace", { command: "start", role,
        controller: elements.controller.value, exploration_noise: Number(noise.value) });
      if (!data) elements["trace-role"].value = lastRole;
      else {
        elements["trace-options"].open = false;
        elements["trace-next"].focus({ preventScroll: true });
        setText("trace-announcement", `${role === "training" ? "Training" : "Preview"} walkthrough. ${data.trace.stage.title}. Use Next block to continue.`);
      }
    }

    async function autoplay(run) {
      if (!playing || generation !== run || !getState()?.trace?.enabled) return;
      if (isLocked()) { timer = window.setTimeout(() => autoplay(run), 50); return; }
      const data = await sendCommand("/api/trace", { command: "step" });
      if (!data || !data.trace?.enabled) { stopPlaying(); return; }
      if (playing && generation === run) {
        timer = window.setTimeout(() => autoplay(run), Number(elements["trace-pace"].value));
      }
    }

    function renderValues(stage) {
      const values = stage.values || {};
      function renderList(container, items, raw = false) {
        const fragment = document.createDocumentFragment();
        for (const { key, label: name, value, unit = "" } of items) {
          const row = document.createElement("div");
          row.className = "trace-watch";
          row.dataset.key = key;
          if (raw && (value && typeof value === "object" && !Array.isArray(value) || Array.isArray(value) && value.length > 4)) row.classList.add("wide");
          const label = document.createElement("dt"), result = document.createElement("dd");
          label.textContent = name;
          label.title = key;
          result.textContent = valueText(value) + (unit ? ` ${unit}` : "");
          result.title = typeof value === "object" ? JSON.stringify(value) : String(value);
          if (typeof value === "boolean") result.className = value ? "is-true" : "is-false";
          row.append(label, result);
          fragment.append(row);
        }
        container.replaceChildren(fragment);
      }
      renderList(elements["trace-values"], keyValues(stage));
      renderList(elements["trace-raw-values"], Object.entries(values).map(([key, value]) => ({ key, label: key, value })), true);
      setText("trace-full-explanation", stage.explanation);
      const observation = values.observation;
      const hasObservation = Array.isArray(observation) && observation.length === 21;
      elements["trace-observation-detail"].hidden = !hasObservation;
      if (hasObservation) {
        const nextObservation = values.next_observation;
        elements["trace-observation-detail"].querySelector("summary").textContent = nextObservation
          ? "Inspect the sampled observations (state → next state)" : "Inspect the 21 observation features";
        elements["trace-observation-detail"].querySelector("p").textContent = nextObservation
          ? "These are the actual stored features of the sampled replay row. They may differ from the live tank. All values are normalized."
          : "16 sonar ranges, 2 body-frame goal offsets, speed, and 2 body-frame current components. All values are normalized.";
        const features = document.createDocumentFragment();
        observation.forEach((value, index) => {
          const feature = document.createElement("div");
          feature.className = "trace-feature";
          const name = document.createElement("span"), amount = document.createElement("strong");
          name.textContent = `[${index}] ${featureNames[index]}`;
          amount.textContent = nextObservation ? `${valueText(value)} → ${valueText(nextObservation[index])}` : valueText(value);
          feature.append(name, amount);
          features.append(feature);
        });
        elements["trace-observation-grid"].replaceChildren(features);
      }
      const equation = elements["trace-equation"];
      equation.hidden = !["bellman_target", "actor_skip", "soft_targets"].includes(stage.key);
      equation.replaceChildren();
      if (stage.key === "bellman_target") {
        const result = document.createElement("strong"), note = document.createElement("small");
        result.textContent = valueText(values.target);
        equation.append(document.createTextNode(`y = ${valueText(values.reward)} + ${valueText(values.discount)} × ${values.not_done} × min(${valueText(values.target_q1)}, ${valueText(values.target_q2)}) = `), result);
        note.textContent = `Replay row ${values.sample_index} · rounded for display`;
        equation.append(note);
      } else if (stage.key === "actor_skip") {
        equation.textContent = `${values.critic_updates} % ${values.policy_delay} ≠ 0 → skip actor and target updates`;
      } else if (stage.key === "soft_targets") {
        equation.textContent = `target ← (1 − ${valueText(values.tau)}) × target + ${valueText(values.tau)} × online`;
      }
    }

    function renderState(data) {
      const trace = data.trace || {};
      const enabled = Boolean(trace.enabled);
      if (enabled !== visible) {
        visible = enabled;
        document.body.classList.toggle("walkthrough-on", enabled);
        elements.workspace.classList.toggle("code-mode", enabled);
        for (const id of ["trace-toolbar", "trace-panel", "trace-tank-status"]) elements[id].hidden = !enabled;
        elements["code-view"].classList.toggle("is-selected", enabled);
        elements["simulation-view"].classList.toggle("is-selected", !enabled);
        elements["code-view"].setAttribute("aria-pressed", String(enabled));
        elements["simulation-view"].setAttribute("aria-pressed", String(!enabled));
        setText("view-description", enabled ? "Connect Python TD3 source to the live simulation, one stage at a time."
          : "Watch the vehicle, then follow the code behind each move.");
        elements["run-message"].setAttribute("aria-live", enabled ? "off" : "polite");
        redraw();
      }
      if (!enabled) {
        if (playing) stopPlaying();
        signature = "";
        return;
      }
      lastRole = trace.role;
      elements["trace-role"].value = trace.role;
      const training = trace.role === "training";
      elements["trace-noise-field"].hidden = !training;
      setText("trace-role-note", training ? "Each control step writes replay. Eligible steps change the learned weights."
        : data.controller === "td3" ? "The actor controls the sub. Weights and replay stay unchanged."
          : `${data.controller === "baseline" ? "The map-aware baseline" : "Random actions"} controls the sub. No learning occurs.`);
      setText("active-controller", `${data.controller === "td3" ? "TD3 policy" : data.controller === "baseline" ? "Map-aware baseline" : "Random actions"} · ${training ? "training" : "preview"} walkthrough`);
      const stage = trace.stage;
      if (!stage) return;
      const nextSignature = `${trace.sequence}:${trace.transition}:${stage.key}:${trace.role}:${sourceView}`;
      if (nextSignature === signature) return;
      signature = nextSignature;
      elements["trace-panel"].dataset.stage = stage.key;
      elements["trace-panel"].dataset.sequence = String(trace.sequence);
      elements["trace-panel"].dataset.sourceView = sourceView;
      elements.simulation.dataset.traceStage = stage.key;
      const group = groupFor[stage.key] || "finish";
      const groupIndex = Object.keys(groupNames).indexOf(group);
      elements["trace-flow"].querySelectorAll("li").forEach((item, index) => {
        const active = item.dataset.group === group;
        item.classList.toggle("is-current", active);
        item.classList.toggle("is-past", index < groupIndex);
        item.classList.toggle("is-inactive", !training && ["store", "learn"].includes(item.dataset.group));
        if (active) item.setAttribute("aria-current", "step"); else item.removeAttribute("aria-current");
      });
      setText("trace-stage-kind", groupNames[group].toUpperCase());
      setText("trace-sequence", `Step ${trace.transition}`);
      setText("trace-stage-title", stage.title);
      setText("trace-explanation", stage.key === "transition_end" && !training
        ? "This move is complete. Preview leaves weights and replay unchanged."
        : shortExplanation[stage.key] || stage.explanation);
      setText("trace-boundary", trace.at_boundary ? "Control step complete" : "Browser stage completed");
      setText("trace-tick", trace.at_boundary ? "Next AUV step" : "Finish this step");
      setText("trace-clock", training ? `Critic ${data.training?.updates ?? 0} · Actor ${data.training?.actor_updates ?? 0}` : "Preview · weights unchanged");
      setText("trace-values-context", learningSamples.has(stage.key)
        ? ["actor_update", "critic"].includes(stage.key) ? "Replay minibatch" : `Replay row ${stage.values.sample_index}`
        : ["soft_targets", "actor_skip", "learning_wait"].includes(stage.key) ? "Learner state" : "Live AUV");
      const outcome = data.env?.outcome;
      const finished = outcome && outcome !== "running";
      setText("trace-motion-note", finished
        ? `${outcome === "success" ? "Goal reached" : outcome === "collision" ? "Collision" : "Time limit"} · final position held until reset`
        : stage.key === "environment" ? `Moved once: environment step ${data.env.episode_step}, Δt = ${valueText(stage.values.dt)} s.`
          : stage.key === "reset" ? "New episode · waiting for the next move"
            : learningSamples.has(stage.key) || ["soft_targets", "actor_skip", "learning_wait"].includes(stage.key)
              ? "Holding position · learning from replay"
              : "Holding position · waiting for the move block");
      const python = sourceView === "python";
      const sourceSet = python ? pythonReference.sources : sources;
      const mappedKey = python && stage.key === "random" && stage.values.random_warmup ? "random_warmup" : stage.source;
      const source = sourceSet[mappedKey];
      const upstream = python && source?.origin === "td3-reference";
      const contextOnly = python && source?.kind === "context";
      setText("trace-source-file", source?.file || "Source excerpt unavailable");
      setText("trace-source-range", source ? `L${source.start_line}–${source.end_line}` : "");
      setText("trace-source-origin", python
        ? upstream ? `${pythonReference.repository} · Python reference` : "AUV-practice · Python reference"
        : "Browser engine · JavaScript");
      setText("trace-source-note", python
        ? contextOnly ? "Browser pause boundary; the Python shown is loop context. Live values come from the JavaScript simulation."
          : "Python reference for this stage. Live values come from the JavaScript browser simulation; Python is not executing here."
        : "This JavaScript block has executed in the browser. The live values below are captured from that execution.");
      setText("trace-values-title", python ? "Live browser values" : "Key values");
      setText("trace-execution-note", python
        ? contextOnly ? "Python loop context · browser transition complete." : "Python reference · corresponding browser stage completed."
        : "Highlighted JavaScript has just executed.");
      elements["trace-code-scroll"].setAttribute("aria-label", python ? "Python reference code for this stage" : "Executed JavaScript source code");
      elements["trace-inspector"].setAttribute("aria-label", python ? "Live browser values for the corresponding algorithm stage" : "Values from the executed JavaScript block");
      setText("trace-context-title", python ? "How this maps to the simulation" : "About this source");
      setText("trace-source-mapping", python ? source?.note || "This Python excerpt illustrates the corresponding browser stage."
        : "The highlighted source is extracted directly from the JavaScript files bundled in this HTML. Each stage is displayed after it runs.");
      setText("trace-source-precision", python
        ? "The original TD3 repository uses different model sizes, defaults, and random streams. These readings are browser measurements, not Python local variables or a reproduced Python run. All variables keeps the browser's diagnostic names."
        : "The Python source view connects these stages to the original TD3 reference. It does not change which code the browser runs.");
      elements["trace-license"].hidden = !python;
      setText("trace-source-license", pythonReference.license);
      const baseUrl = source?.url || (source && !python
        ? `https://github.com/AWDaley/AUV-practice/blob/75b7b262641b097c8056b2005235910ffafa4af4/${source.file}` : "");
      const sourceLink = elements["trace-source-link"];
      sourceLink.hidden = !baseUrl;
      if (baseUrl) sourceLink.href = `${baseUrl.split("#")[0]}#L${source.start_line}-L${source.end_line}`;
      else sourceLink.removeAttribute("href");
      elements["trace-code"].dataset.source = mappedKey;
      const nextSourceKey = `${sourceView}:${mappedKey}`;
      if (sourceKey !== nextSourceKey) {
        sourceKey = nextSourceKey;
        showSource(elements["trace-code"], source, python ? null : focusFor[stage.key]);
      }
      const detailKey = python ? pythonDetailFor[stage.key] : stage.details_source || detailFor[stage.key];
      const detail = sourceSet[detailKey];
      elements["trace-source-detail"].hidden = !detail;
      if (detail) {
        setText("trace-source-detail-label", detailTitles[detailKey] || "Open the called implementation");
        setText("trace-detail-location", `${detail.file} · L${detail.start_line}–${detail.end_line}`);
        elements["trace-detail-note"].hidden = !python || !detail.note;
        setText("trace-detail-note", python ? detail.note || "" : "");
        showSource(elements["trace-detail-code"], detail);
      }
      renderValues(stage);
    }

    elements["code-view"].addEventListener("click", () => {
      elements["trace-role"].value = "preview";
      elements["trace-noise"].value = elements["exploration-noise"].value;
      startTrace();
    });
    elements["simulation-view"].addEventListener("click", async () => {
      stopPlaying();
      elements["trace-options"].open = false;
      const data = await sendCommand("/api/trace", { command: "exit" });
      if (data) {
        elements["code-view"].focus({ preventScroll: true });
        setText("trace-announcement", "Returned to Simulation. The current control step is complete and the run is paused.");
      }
    });
    elements["trace-role"].addEventListener("change", startTrace);
    elements["trace-source-view"].addEventListener("change", () => {
      stopPlaying();
      sourceView = elements["trace-source-view"].value;
      renderState(getState());
      setText("trace-announcement", sourceView === "python"
        ? "Showing corresponding Python reference code. Live values still come from the browser simulation."
        : "Showing the JavaScript executed by the browser.");
    });
    elements["trace-noise"].addEventListener("change", () => {
      if (!elements["trace-noise"].checkValidity()) { elements["trace-noise"].reportValidity(); return; }
      elements["exploration-noise"].value = elements["trace-noise"].value;
      setText("noise-output", Number(elements["trace-noise"].value).toFixed(2));
      startTrace();
    });
    elements["trace-play"].addEventListener("click", () => {
      if (playing) { stopPlaying(); return; }
      if (isLocked() || !getState()?.trace?.enabled) return;
      playing = true;
      const run = ++generation;
      refreshControls();
      timer = window.setTimeout(() => autoplay(run), Number(elements["trace-pace"].value));
    });
    async function manualStep(command, id) {
      const data = await sendCommand("/api/trace", { command });
      if (data) {
        elements[id].focus({ preventScroll: true });
        setText("trace-announcement", data.trace.stage.title);
      }
    }
    elements["trace-next"].addEventListener("click", () => manualStep("step", "trace-next"));
    elements["trace-tick"].addEventListener("click", () => manualStep("tick", "trace-tick"));
    for (const id of ["trace-options", "trace-source-detail", "trace-source-context", "trace-license", "trace-all-values", "trace-observation-detail"]) {
      elements[id].addEventListener("toggle", () => { if (elements[id].open) stopPlaying(); });
    }
    document.addEventListener("pointerdown", event => {
      if (!elements["trace-options"].contains(event.target)) elements["trace-options"].open = false;
    });
    elements["trace-options"].addEventListener("keydown", event => {
      if (event.key === "Escape") {
        elements["trace-options"].open = false;
        elements["trace-options"].querySelector("summary").focus();
        event.preventDefault();
      }
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlaying(); });
    window.addEventListener("pagehide", stopPlaying);

    return { renderState, updateControls, stopPlaying, get playing() { return playing; } };
  };
})();
