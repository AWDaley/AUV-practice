/* A serial, block-by-block view of the worker's actual execution. */
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
    soft_update: "See the target-weight interpolation" };
  const learningSamples = new Set(["replay_sample", "target_actions", "bellman_target", "critic", "actor_update"]);
  const featureNames = Array.from({ length: 21 }, (_, i) => i < 16 ? `sonar ${i}`
    : ["goal forward", "goal left", "speed", "current forward", "current left"][i - 16]);

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

  function highlightedLine(text) {
    const fragment = document.createDocumentFragment();
    // Token styling only. Source text is never interpreted as HTML or executed.
    const token = /(\/\/.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b(?:const|let|return|if|else|for|while|this|new|throw|true|false|null|typeof|yield)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g;
    let from = 0, match;
    while ((match = token.exec(text)) !== null) {
      fragment.append(document.createTextNode(text.slice(from, match.index)));
      const span = document.createElement("span");
      span.className = "trace-token-" + (match[0].startsWith("//") ? "comment"
        : /^["'`]/.test(match[0]) ? "string" : /^\d/.test(match[0]) ? "number" : "keyword");
      span.textContent = match[0];
      fragment.append(span);
      from = token.lastIndex;
    }
    fragment.append(document.createTextNode(text.slice(from)));
    return fragment;
  }

  function showSource(container, source) {
    container.replaceChildren();
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
      const number = document.createElement("span");
      number.className = "trace-line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = String(source.start_line + index);
      const code = document.createElement("span");
      code.className = "trace-line-text";
      code.append(highlightedLine(text.slice(indent)));
      line.append(number, code);
      fragment.append(line);
    });
    container.append(fragment);
  }

  window.createAUVTraceView = ({ elements, getState, sendCommand, setText, redraw, refreshControls, isLocked }) => {
    const sources = JSON.parse(document.getElementById("auv-trace-source").textContent);
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
      const fragment = document.createDocumentFragment();
      for (const [key, value] of Object.entries(values)) {
        if (["observation", "next_observation", "sonar"].includes(key) || value === null) continue;
        const row = document.createElement("div");
        row.className = "trace-watch";
        if (value && typeof value === "object" && !Array.isArray(value) || Array.isArray(value) && value.length > 4) row.classList.add("wide");
        const label = document.createElement("dt"), result = document.createElement("dd");
        label.textContent = key;
        result.textContent = valueText(value);
        if (typeof value === "boolean") result.className = value ? "is-true" : "is-false";
        row.append(label, result);
        fragment.append(row);
      }
      elements["trace-values"].replaceChildren(fragment);
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
        note.textContent = `First sampled replay row, index ${values.sample_index}. Values are rounded for display; learning uses their full precision.`;
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
        setText("view-description", enabled ? "One block at a time. The same code that runs the simulation."
          : "Watch the vehicle, then follow the code behind each move.");
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
      const nextSignature = `${trace.sequence}:${trace.transition}:${stage.key}:${trace.role}`;
      if (nextSignature === signature) return;
      signature = nextSignature;
      elements["trace-panel"].dataset.stage = stage.key;
      elements["trace-panel"].dataset.sequence = String(trace.sequence);
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
      setText("trace-sequence", `Step ${trace.transition} · Block ${trace.sequence}`);
      setText("trace-stage-title", stage.title);
      setText("trace-explanation", stage.explanation);
      setText("trace-boundary", trace.at_boundary ? "Control step complete" : "Block executed");
      setText("trace-tick", trace.at_boundary ? "Next AUV step" : "Finish this step");
      setText("trace-clock", `Environment step ${data.env?.episode_step ?? 0} · Critic update ${data.training?.updates ?? 0} · Actor update ${data.training?.actor_updates ?? 0}`);
      setText("trace-values-context", learningSamples.has(stage.key)
        ? stage.key === "actor_update" ? "Replay minibatch" : "Sampled replay row"
        : ["soft_targets", "actor_skip", "learning_wait"].includes(stage.key) ? "Learner state" : "Live AUV");
      const outcome = data.env?.outcome;
      const finished = outcome && outcome !== "running";
      setText("trace-motion-note", finished
        ? `${outcome === "success" ? "Goal reached" : outcome === "collision" ? "Collision" : "Time limit"}. The final position stays visible until the next reset block.`
        : stage.key === "environment" ? `Moved once: environment step ${data.env.episode_step}, Δt = ${valueText(stage.values.dt)} s.`
          : stage.key === "reset" ? "New episode. The next movement happens in the environment block."
            : learningSamples.has(stage.key) || ["soft_targets", "actor_skip", "learning_wait"].includes(stage.key)
              ? "The sub holds position while the learner processes replay."
              : "The sub holds position between environment blocks.");
      const source = sources[stage.source];
      setText("trace-source-file", source?.file || "Source excerpt unavailable");
      setText("trace-source-range", source ? `L${source.start_line}–${source.end_line}` : "");
      elements["trace-code"].dataset.source = stage.source;
      if (sourceKey !== stage.source) {
        sourceKey = stage.source;
        showSource(elements["trace-code"], source);
        elements["trace-code-scroll"].scrollTop = 0;
        elements["trace-code-scroll"].scrollLeft = 0;
      }
      const detailKey = stage.details_source || detailFor[stage.key];
      const detail = sources[detailKey];
      elements["trace-source-detail"].hidden = !detail;
      if (detail) {
        setText("trace-source-detail-label", detailTitles[detailKey] || "Open the called implementation");
        setText("trace-detail-location", `${detail.file} · L${detail.start_line}–${detail.end_line}`);
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
      await sendCommand("/api/trace", { command: "exit" });
    });
    elements["trace-role"].addEventListener("change", startTrace);
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
    elements["trace-next"].addEventListener("click", () => sendCommand("/api/trace", { command: "step" }));
    elements["trace-tick"].addEventListener("click", () => sendCommand("/api/trace", { command: "tick" }));
    document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlaying(); });
    window.addEventListener("pagehide", stopPlaying);

    return { renderState, updateControls, stopPlaying, get playing() { return playing; } };
  };
})();
