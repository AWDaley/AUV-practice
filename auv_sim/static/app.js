(() => {
  "use strict";

  const elements = {};
  document.querySelectorAll("[id]").forEach((element) => { elements[element.id] = element; });
  const names = { baseline: "Map-aware baseline", td3: "TD3 policy", random: "Random actions" };
  const scenes = { open: "Open water", reef: "Reef field", channel: "Narrow channel" };
  const integerFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const compactFormat = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
  const palette = { teal: "#61dfc4", coral: "#ef9b87", muted: "#7e9eb1", grid: "#213747", label: "#7893a5" };
  let state = null;
  let firstState = true;
  let commandBusy = false;
  let pollPromise = null;
  let pollTimer = null;
  let stopped = false;
  let activeRequest = null;
  let errorKind = null;
  let errorMessage = "";
  let dismissedError = "";
  let checkpointSignature = "";
  let historySignature = "";
  let chartHistory = [];
  let drawQueued = false;
  const edited = new Set();

  function number(value) {
    if (value === null || value === undefined || value === "") return null;
    const converted = Number(value);
    return Number.isFinite(converted) ? converted : null;
  }

  function fixed(value, decimals = 2) {
    const n = number(value);
    if (n === null) return "—";
    if (Math.abs(n) >= 10000) return compactFormat.format(n);
    return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  }

  function count(value) { return integerFormat.format(number(value) ?? 0); }
  function percent(value) { const n = number(value); return n === null ? "—" : `${(n * 100).toFixed(1)}%`; }
  function signed(value, decimals = 3) { const n = number(value); return n === null ? "—" : `${n > 0 ? "+" : ""}${fixed(n, decimals)}`; }
  function setText(id, text) { if (elements[id] && elements[id].textContent !== String(text)) elements[id].textContent = text; }
  function titleCase(value) { return String(value ?? "").replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }

  function outcomeKind(value) {
    const raw = String(value ?? "").toLowerCase();
    if (["success", "goal", "reached_goal", "goal_reached"].includes(raw)) return "success";
    if (["collision", "crash", "obstacle_collision", "boundary_collision"].includes(raw)) return "collision";
    if (["timeout", "time_limit", "truncated", "truncation"].includes(raw)) return "timeout";
    return raw;
  }

  function outcomeName(value) {
    const kind = outcomeKind(value);
    if (kind === "success") return "Reached goal";
    if (kind === "collision") return "Collision";
    if (kind === "timeout") return "Time limit";
    if (!kind || kind === "running" || kind === "none") return "—";
    return titleCase(kind);
  }

  function showError(message, kind = "action") {
    errorKind = kind;
    errorMessage = String(message);
    if (dismissedError === errorMessage) return;
    setText("error-text", errorMessage);
    elements["error-banner"].hidden = false;
  }

  function clearError(kind) {
    if (kind && errorKind !== kind) return;
    elements["error-banner"].hidden = true;
    errorMessage = "";
    errorKind = null;
    dismissedError = "";
  }

  function setConnection(value) {
    elements["connection-status"].classList.toggle("connected", value);
    elements["connection-status"].classList.toggle("disconnected", !value);
    setText("connection-text", value ? "Simulator connected" : "Connection interrupted");
  }

  function errorDetail(data) {
    const detail = data?.detail ?? data?.message ?? data?.error;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
      return `${location ? `${location}: ` : ""}${item.msg || "Invalid value"}`;
    }).join("; ");
    return "";
  }

  async function fetchJSON(path, payload) {
    const abort = new AbortController();
    activeRequest = abort;
    const timeout = window.setTimeout(() => abort.abort(), payload === undefined ? 7000 : 20000);
    try {
      const options = { method: payload === undefined ? "GET" : "POST", signal: abort.signal, cache: "no-store" };
      if (payload !== undefined) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(payload);
      }
      const response = await fetch(path, options);
      let data;
      try { data = await response.json(); } catch (_) { throw new Error(`The simulator returned an unreadable response (HTTP ${response.status}).`); }
      if (!response.ok) throw new Error(errorDetail(data) || `The request failed (HTTP ${response.status}).`);
      if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("The simulator returned an unexpected response.");
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The simulator did not respond in time. Please try again.");
      if (error instanceof TypeError) throw new Error("Cannot reach the local simulator. Check that the server is running.");
      throw error;
    } finally {
      window.clearTimeout(timeout);
      if (activeRequest === abort) activeRequest = null;
    }
  }

  function schedulePoll() {
    window.clearTimeout(pollTimer);
    if (!stopped) pollTimer = window.setTimeout(pollState, 250);
  }

  async function pollState() {
    if (stopped) return;
    if (commandBusy || pollPromise) { schedulePoll(); return; }
    pollPromise = (async () => {
      try {
        const data = await fetchJSON("/api/state");
        setConnection(true);
        clearError("connection");
        applyState(data);
      } catch (error) {
        if (stopped) return;
        setConnection(false);
        showError(`${error.message} Retrying automatically.`, "connection");
        if (!state) setText("run-message", "The page is ready. Waiting for the local simulator to reconnect.");
      }
    })();
    try { await pollPromise; } finally { pollPromise = null; schedulePoll(); }
  }

  async function sendCommand(path, payload, onSuccess) {
    if (commandBusy) return;
    commandBusy = true;
    window.clearTimeout(pollTimer);
    updateControls();
    try {
      // Finish the current read before sending a mutation, so an older poll cannot overwrite its result.
      if (pollPromise) await pollPromise;
      const data = await fetchJSON(path, payload);
      setConnection(true);
      clearError();
      applyState(data);
      if (onSuccess) onSuccess(data);
    } catch (error) {
      dismissedError = "";
      showError(error.message, "action");
    } finally {
      commandBusy = false;
      updateControls();
      schedulePoll();
    }
  }

  function initializeInputs(data) {
    const config = data.config || {};
    const initial = { scenario: config.scenario, seed: config.seed, "current-strength": config.current_strength, controller: data.controller, "exploration-noise": data.training?.exploration_noise };
    Object.entries(initial).forEach(([id, value]) => {
      if (value !== undefined && value !== null && !edited.has(id)) {
        if (elements[id].tagName === "SELECT" && !Array.from(elements[id].options).some((option) => option.value === String(value))) return;
        elements[id].value = String(value);
      }
    });
    updateNoise();
    updateControllerNote();
  }

  function applyState(data) {
    state = data;
    if (firstState) { initializeInputs(data); firstState = false; }
    const env = data.env || {};
    const training = data.training || {};
    const mode = data.mode || "idle";
    const cumulativeSteps = number(training.steps) ?? 0;
    const targetSteps = number(training.total_steps) ?? 0;
    const progress = targetSteps > 0 ? Math.min(1, Math.max(0, cumulativeSteps / targetSteps)) : 0;

    setText("mode-pill", titleCase(mode).toUpperCase());
    elements["mode-pill"].dataset.mode = mode;
    setText("run-message", data.message || (mode === "idle" ? "Choose a controller and press Play, or start a training run." : titleCase(mode)));
    const ready = Boolean(data.policy_ready);
    setText("policy-status", ready ? `TD3 · ${count(training.updates)} updates` : "TD3 · untrained");
    elements["policy-status"].classList.toggle("ready", ready);
    setText("scenario-label", scenes[data.config?.scenario] || titleCase(data.config?.scenario) || "—");
    setText("episode-step", count(env.episode_step));
    setText("episode-return", fixed(env.episode_return, 1));
    setText("goal-distance", fixed(env.distance_to_goal, 2));
    setText("vehicle-speed", fixed(env.vehicle?.speed, 2));
    setText("min-clearance", fixed(env.min_clearance, 2));
    const currentX = number(env.current?.x) ?? 0;
    const currentY = number(env.current?.y) ?? 0;
    setText("current-vector", `Current ${fixed(Math.hypot(currentX, currentY), 2)} m/s`);
    const controller = mode === "training" ? "td3" : data.controller;
    setText("active-controller", `${names[controller] || "AUV simulator"}${mode === "training" ? " · exploration on" : mode === "evaluating" ? " · evaluating" : ""}`);
    elements["tank-loading"].hidden = Boolean(env.vehicle && env.world);

    setText("critic-updates", count(training.updates));
    setText("actor-updates", count(training.actor_updates));
    setText("buffer-size", count(training.buffer_size));
    setText("completed-episodes", count(training.episodes));
    setText("steps-per-second", fixed(training.steps_per_second, 0));
    setText("q1", fixed(training.q1, 3));
    setText("q2", fixed(training.q2, 3));
    setText("td-error", fixed(training.td_error, 3));
    setText("critic-loss", fixed(training.critic_loss, 4));
    setText("actor-loss", fixed(training.actor_loss, 4));
    setText("training-phase", titleCase(training.phase) || "Not started");
    setText("training-progress-text", `${count(cumulativeSteps)} / ${count(targetSteps)}`);
    elements["training-progress"].max = Math.max(1, targetSteps);
    elements["training-progress"].value = Math.min(cumulativeSteps, targetSteps);
    setText("training-percent", `${Math.round(progress * 100)}%`);
    setText("training-caption", mode === "training" ? (titleCase(training.phase) || "Training") : cumulativeSteps > 0 ? `${count(training.episodes)} training episodes completed` : "Ready for the first run");
    setText("last-reward", signed(env.last_reward, 3));
    setText("action-1", signed(data.action?.[0], 2));
    setText("action-2", signed(data.action?.[1], 2));
    renderRewardTerms(env.reward_terms);
    renderEvaluation(data.evaluation);
    renderCheckpoints(data.checkpoints || []);
    renderHistory(data.history || []);
    updateConfigDirty();
    updateControllerNote();
    updateControls();
    queueDraw();
  }

  function updateControls() {
    const mode = state?.mode || "idle";
    const activeJob = mode === "training" || mode === "evaluating";
    const noState = !state;
    const locked = commandBusy || noState;
    elements["play-button"].disabled = locked || activeJob;
    elements["step-button"].disabled = locked || activeJob;
    elements["pause-button"].disabled = locked || !["training", "preview", "evaluating"].includes(mode);
    elements["train-button"].disabled = locked || activeJob;
    elements["evaluate-button"].disabled = locked || activeJob;
    elements["reset-button"].disabled = locked;
    const canCheckpoint = !locked && !["training", "preview", "evaluating"].includes(mode);
    elements["save-button"].disabled = !canCheckpoint || !state?.policy_ready;
    elements["save-button"].title = !state?.policy_ready ? "Train the TD3 policy before saving." : !canCheckpoint ? "Pause the run before saving." : "Save the current TD3 policy.";
    const hasCheckpoint = Boolean(elements["checkpoint-select"].value);
    elements["checkpoint-select"].disabled = !canCheckpoint || !hasCheckpoint;
    elements["load-button"].disabled = !canCheckpoint || !hasCheckpoint;
    elements["load-button"].title = !canCheckpoint ? "Pause the run before loading." : "Load the selected saved policy.";
    setText("train-label", mode === "training" ? "Training…" : state?.policy_ready ? "Continue training" : "Train policy");
    setText("evaluate-button", mode === "evaluating" ? "Evaluating…" : "Evaluate");
    elements["train-button"].setAttribute("aria-busy", String(mode === "training"));
    elements["evaluate-button"].setAttribute("aria-busy", String(mode === "evaluating"));
  }

  function updateControllerNote() {
    const value = elements.controller.value;
    let note;
    if (value === "baseline") note = "Hand-written baseline with access to the obstacle map. TD3 uses local observations; this is a demonstration, not a fair learned-policy benchmark.";
    else if (value === "td3") note = state?.policy_ready ? "The current TD3 actor controls the AUV. Preview uses no exploration noise and does not train the policy." : "TD3 is untrained. Start a training run to update its weights; preview does not learn.";
    else note = "Samples random actions. This controller does not learn, and preview adds no training experience.";
    if (state?.mode === "preview" && state.controller !== value) note += " Press Play to switch the running preview.";
    setText("controller-note", note);
    setText("evaluation-controller", names[value] || titleCase(value));
  }

  function updateNoise() { setText("noise-output", Number(elements["exploration-noise"].value).toFixed(2)); }

  function updateConfigDirty() {
    if (!state?.config) { elements["config-dirty"].hidden = true; return; }
    const config = state.config;
    const changed = elements.scenario.value !== String(config.scenario) || Number(elements.seed.value) !== Number(config.seed) || Math.abs(Number(elements["current-strength"].value) - Number(config.current_strength)) > 1e-7;
    elements["config-dirty"].hidden = !changed;
  }

  function renderRewardTerms(terms) {
    const container = elements["reward-terms"];
    const entries = Object.entries(terms && typeof terms === "object" ? terms : {}).filter(([, value]) => number(value) !== null);
    container.replaceChildren();
    if (!entries.length) {
      const empty = document.createElement("p");
      empty.className = "empty-copy";
      empty.textContent = "Take a step to inspect the reward terms.";
      container.append(empty);
      return;
    }
    entries.forEach(([name, value]) => {
      const row = document.createElement("div");
      row.className = "reward-term";
      const label = document.createElement("span");
      label.textContent = titleCase(name);
      const amount = document.createElement("strong");
      amount.textContent = signed(value, 3);
      amount.className = number(value) > 0 ? "positive" : number(value) < 0 ? "negative" : "";
      row.append(label, amount);
      container.append(row);
    });
  }

  function renderEvaluation(result) {
    elements["evaluation-empty"].hidden = Boolean(result);
    elements["evaluation-results"].hidden = !result;
    if (!result) return;
    setText("evaluated-controller", names[result.controller] || titleCase(result.controller));
    setText("evaluated-episodes", `${count(result.episodes)} episodes`);
    setText("eval-success", percent(result.success_rate));
    setText("eval-collision", percent(result.collision_rate));
    setText("eval-return", fixed(result.mean_return, 1));
    setText("eval-steps", fixed(result.mean_steps, 1));
    setText("evaluation-seed", result.seed === undefined || result.seed === null ? "Latest completed evaluation" : `Latest completed evaluation · seed ${result.seed}`);
  }

  function renderCheckpoints(checkpoints) {
    const list = Array.isArray(checkpoints) ? checkpoints.filter((checkpoint) => checkpoint && typeof checkpoint.name === "string") : [];
    const signature = list.map((item) => `${item.name}|${item.steps}|${item.created}`).join(";");
    if (signature === checkpointSignature && elements["checkpoint-select"].options.length) return;
    checkpointSignature = signature;
    const select = elements["checkpoint-select"];
    const previous = select.value;
    select.replaceChildren();
    if (!list.length) {
      select.add(new Option("No checkpoints yet", ""));
    } else {
      list.forEach((item) => {
        const option = new Option(`${count(item.steps)} steps · ${item.name}`, item.name);
        option.title = item.name;
        select.add(option);
      });
      if (list.some((item) => item.name === previous)) select.value = previous;
    }
    select.title = select.value || "No saved policies";
  }

  function renderHistory(history) {
    const valid = Array.isArray(history) ? history.filter((entry) => entry && number(entry.return) !== null) : [];
    const signature = JSON.stringify(valid.map((entry) => [entry.episode, entry.return, entry.outcome, entry.steps]));
    if (signature === historySignature) return;
    historySignature = signature;
    chartHistory = valid;
    const successful = valid.filter((entry) => outcomeKind(entry.outcome) === "success").length;
    setText("history-count", `${count(valid.length)} recorded`);
    setText("history-window-note", valid.length > 240 ? `Latest 240 of ${count(valid.length)} recorded TD3 training episodes. Preview and evaluation runs are separate.` : "Completed TD3 training episodes. Preview and evaluation runs are separate.");
    setText("history-success", valid.length ? percent(successful / valid.length) : "—");
    setText("last-outcome", valid.length ? outcomeName(valid[valid.length - 1].outcome) : "—");
    elements["return-empty"].hidden = valid.length > 0;
    elements["outcome-empty"].hidden = valid.length > 0;
    elements["return-chart"].setAttribute("aria-label", valid.length ? `Returns from ${valid.length} completed TD3 training episodes. Latest return ${fixed(valid[valid.length - 1].return, 2)}.` : "Episode return chart. No completed training episodes yet.");
    elements["outcome-chart"].setAttribute("aria-label", valid.length ? `Rolling success and collision rates from TD3 training. ${successful} of ${valid.length} recorded episodes reached the goal.` : "Outcome chart. No completed training episodes yet.");
    drawCharts();
  }

  function validNumber(id, integer = false) {
    const input = elements[id];
    if (!input.checkValidity()) { input.reportValidity(); return null; }
    const value = Number(input.value);
    if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
      showError(`Enter a valid ${integer ? "whole number" : "number"} for ${input.labels?.[0]?.textContent?.trim() || id}.`);
      return null;
    }
    return value;
  }

  function sendControl(command) {
    return sendCommand("/api/control", { command, controller: elements.controller.value, speed: Number(elements["preview-speed"].value) });
  }

  elements["play-button"].addEventListener("click", () => sendControl("play"));
  elements["pause-button"].addEventListener("click", () => sendControl("pause"));
  elements["step-button"].addEventListener("click", () => sendControl("step"));
  elements["reset-button"].addEventListener("click", () => {
    const seed = validNumber("seed", true);
    const strength = validNumber("current-strength");
    if (seed === null || strength === null) return;
    sendCommand("/api/reset", { scenario: elements.scenario.value, seed, current_strength: strength });
  });
  elements["train-button"].addEventListener("click", () => {
    const steps = validNumber("training-steps", true);
    const noise = validNumber("exploration-noise");
    if (steps === null || noise === null) return;
    sendCommand("/api/train", { steps, exploration_noise: noise }, () => {
      elements.controller.value = "td3";
      edited.add("controller");
      updateControllerNote();
    });
  });
  elements["evaluate-button"].addEventListener("click", () => sendCommand("/api/evaluate", { episodes: Number(elements["evaluation-episodes"].value), controller: elements.controller.value }));
  elements["save-button"].addEventListener("click", () => {
    const previousNames = new Set((state?.checkpoints || []).map((checkpoint) => checkpoint.name));
    sendCommand("/api/save", {}, (data) => {
      const saved = (data.checkpoints || []).find((checkpoint) => !previousNames.has(checkpoint.name));
      if (saved) {
        elements["checkpoint-select"].value = saved.name;
        elements["checkpoint-select"].title = saved.name;
      }
    });
  });
  elements["load-button"].addEventListener("click", () => {
    const name = elements["checkpoint-select"].value;
    if (!name) { showError("Choose a saved policy first."); return; }
    sendCommand("/api/load", { name }, (data) => {
      elements.controller.value = "td3";
      edited.add("controller");
      // Loading explicitly changes the experiment; reflect that configuration once, not on every poll.
      if (data.config) {
        elements.scenario.value = String(data.config.scenario);
        elements.seed.value = String(data.config.seed);
        elements["current-strength"].value = String(data.config.current_strength);
        updateConfigDirty();
      }
      updateControllerNote();
    });
  });
  elements["dismiss-error"].addEventListener("click", () => {
    dismissedError = errorMessage;
    elements["error-banner"].hidden = true;
  });
  ["scenario", "seed", "current-strength", "controller", "exploration-noise"].forEach((id) => elements[id].addEventListener("input", () => {
    edited.add(id);
    if (id === "controller") updateControllerNote();
    if (id === "exploration-noise") updateNoise();
    updateConfigDirty();
  }));
  elements["preview-speed"].addEventListener("change", () => { if (state?.mode === "preview") sendControl("play"); });
  elements["checkpoint-select"].addEventListener("change", () => { elements["checkpoint-select"].title = elements["checkpoint-select"].value; updateControls(); });
  ["show-sonar", "show-trail", "show-current"].forEach((id) => elements[id].addEventListener("change", queueDraw));

  // All canvas coordinates below use CSS pixels; the backing store follows the display pixel ratio.
  function canvasContext(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
  }

  function queueDraw() {
    if (drawQueued) return;
    drawQueued = true;
    window.requestAnimationFrame(() => { drawQueued = false; drawTank(); });
  }

  function roundedRect(context, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.arcTo(x + width, y, x + width, y + height, r);
    context.arcTo(x + width, y + height, x, y + height, r);
    context.arcTo(x, y + height, x, y, r);
    context.arcTo(x, y, x + width, y, r);
    context.closePath();
  }

  function drawArrow(context, x, y, dx, dy, color, size = 3) {
    const angle = Math.atan2(dy, dx);
    const endX = x + dx;
    const endY = y + dy;
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(endX, endY);
    context.moveTo(endX - Math.cos(angle - 0.55) * size, endY - Math.sin(angle - 0.55) * size);
    context.lineTo(endX, endY);
    context.lineTo(endX - Math.cos(angle + 0.55) * size, endY - Math.sin(angle + 0.55) * size);
    context.stroke();
  }

  function niceSpacing(length, targetCount = 10) {
    const raw = length / targetCount;
    const power = 10 ** Math.floor(Math.log10(raw));
    const ratio = raw / power;
    return (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
  }

  function drawTank() {
    const setup = canvasContext(elements["tank-canvas"]);
    if (!setup) return;
    const { context: ctx, width, height } = setup;
    ctx.fillStyle = "#091923";
    ctx.fillRect(0, 0, width, height);
    const env = state?.env;
    const worldWidth = number(env?.world?.width);
    const worldHeight = number(env?.world?.height);
    if (!env?.vehicle || !worldWidth || !worldHeight || worldWidth <= 0 || worldHeight <= 0) return;
    const insetX = width < 420 ? 30 : 41;
    const scale = Math.min((width - insetX * 2) / worldWidth, (height - 91) / worldHeight);
    if (scale <= 0) return;
    const tankWidth = worldWidth * scale;
    const tankHeight = worldHeight * scale;
    const originX = (width - tankWidth) / 2;
    const originY = (height - tankHeight) / 2 + 1;
    const point = (x, y) => [originX + x * scale, originY + (worldHeight - y) * scale];
    const goal = env.goal || {};
    const goalPoint = point(number(goal.x) ?? 0, number(goal.y) ?? 0);
    const glow = ctx.createRadialGradient(goalPoint[0], goalPoint[1], 0, goalPoint[0], goalPoint[1], Math.max(tankWidth, tankHeight) * 0.8);
    glow.addColorStop(0, "#153c3b55");
    glow.addColorStop(1, "#0a1d2900");
    ctx.fillStyle = glow;
    ctx.fillRect(originX, originY, tankWidth, tankHeight);
    const spacing = niceSpacing(Math.max(worldWidth, worldHeight), 12);
    ctx.lineWidth = 0.7;
    ctx.strokeStyle = "#23424c65";
    ctx.beginPath();
    for (let x = spacing; x < worldWidth; x += spacing) { const p = point(x, 0); ctx.moveTo(p[0], originY); ctx.lineTo(p[0], originY + tankHeight); }
    for (let y = spacing; y < worldHeight; y += spacing) { const p = point(0, y); ctx.moveTo(originX, p[1]); ctx.lineTo(originX + tankWidth, p[1]); }
    ctx.stroke();
    ctx.strokeStyle = "#3a626c85";
    ctx.lineWidth = 1;
    ctx.strokeRect(originX, originY, tankWidth, tankHeight);
    ctx.font = "8px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#527786";
    ctx.textAlign = "center";
    const labelSpacing = niceSpacing(worldWidth, width < 420 ? 4 : 6);
    for (let x = 0; x <= worldWidth + 1e-8; x += labelSpacing) { const p = point(x, 0); ctx.fillText(`${Number(x.toFixed(2))}`, p[0], originY + tankHeight + 13); }
    ctx.textAlign = "right";
    const ySpacing = niceSpacing(worldHeight, 4);
    for (let y = 0; y <= worldHeight + 1e-8; y += ySpacing) { const p = point(0, y); ctx.fillText(`${Number(y.toFixed(2))}`, originX - 8, p[1] + 3); }
    ctx.textAlign = "right";
    ctx.fillText("m", originX + tankWidth + 13, originY + tankHeight + 13);

    ctx.save();
    ctx.beginPath();
    ctx.rect(originX + 0.5, originY + 0.5, tankWidth - 1, tankHeight - 1);
    ctx.clip();
    const currentX = number(env.current?.x) ?? 0;
    const currentY = number(env.current?.y) ?? 0;
    const currentLength = Math.hypot(currentX, currentY);
    if (elements["show-current"].checked && currentLength > 0.0001) {
      const length = 8 + Math.min(0.4, currentLength) * 37;
      const dx = currentX / currentLength * length;
      const dy = -currentY / currentLength * length;
      ctx.lineWidth = 0.9;
      for (let x = worldWidth / 12; x < worldWidth; x += worldWidth / 6) {
        for (let y = worldHeight / 10; y < worldHeight; y += worldHeight / 5) {
          const p = point(x, y);
          drawArrow(ctx, p[0] - dx / 2, p[1] - dy / 2, dx, dy, "#416e8150", 3.4);
        }
      }
    }

    if (elements["show-trail"].checked && Array.isArray(env.trail) && env.trail.length > 1) {
      ctx.beginPath();
      let started = false;
      env.trail.forEach((position) => {
        if (!Array.isArray(position) || number(position[0]) === null || number(position[1]) === null) return;
        const p = point(Number(position[0]), Number(position[1]));
        if (!started) { ctx.moveTo(...p); started = true; } else ctx.lineTo(...p);
      });
      ctx.strokeStyle = "#61dfc474";
      ctx.lineWidth = 1.8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    (Array.isArray(env.obstacles) ? env.obstacles : []).forEach((obstacle) => {
      const x = number(obstacle.x);
      const y = number(obstacle.y);
      const radius = number(obstacle.radius);
      if (x === null || y === null || radius === null || radius < 0) return;
      const p = point(x, y);
      const r = radius * scale;
      const gradient = ctx.createRadialGradient(p[0] - r * 0.3, p[1] - r * 0.4, 0, p[0], p[1], Math.max(1, r));
      gradient.addColorStop(0, "#304b5c");
      gradient.addColorStop(1, "#18313f");
      ctx.beginPath(); ctx.arc(...p, r, 0, Math.PI * 2);
      ctx.fillStyle = gradient; ctx.fill();
      ctx.strokeStyle = "#7595a688"; ctx.lineWidth = 1; ctx.stroke();
      if (r > 12) { ctx.beginPath(); ctx.arc(p[0], p[1], r * 0.72, 3.7, 5.55); ctx.strokeStyle = "#5a7d8d35"; ctx.stroke(); }
    });

    const vehicle = env.vehicle;
    const vehicleX = number(vehicle.x) ?? 0;
    const vehicleY = number(vehicle.y) ?? 0;
    const vehiclePoint = point(vehicleX, vehicleY);
    if (elements["show-sonar"].checked && Array.isArray(env.sonar)) {
      ctx.lineWidth = 0.8;
      env.sonar.forEach((ray) => {
        let endX = number(ray.end_x);
        let endY = number(ray.end_y);
        if (endX === null || endY === null) {
          const angle = number(ray.angle);
          const distance = number(ray.distance);
          if (angle === null || distance === null) return;
          endX = vehicleX + Math.cos(angle) * distance;
          endY = vehicleY + Math.sin(angle) * distance;
        }
        const end = point(endX, endY);
        const gradient = ctx.createLinearGradient(...vehiclePoint, ...end);
        gradient.addColorStop(0, "#61dfc425"); gradient.addColorStop(1, "#61dfc46b");
        ctx.beginPath(); ctx.moveTo(...vehiclePoint); ctx.lineTo(...end); ctx.strokeStyle = gradient; ctx.stroke();
        ctx.beginPath(); ctx.arc(...end, 1.7, 0, Math.PI * 2); ctx.fillStyle = "#8cebd08a"; ctx.fill();
      });
    }

    const goalRadius = Math.max(4, (number(goal.radius) ?? 0.5) * scale);
    ctx.beginPath(); ctx.arc(...goalPoint, goalRadius, 0, Math.PI * 2);
    ctx.fillStyle = "#61dfc419"; ctx.fill(); ctx.strokeStyle = "#61dfc4b0"; ctx.lineWidth = 1.3; ctx.stroke();
    ctx.beginPath(); ctx.arc(...goalPoint, Math.max(2.5, goalRadius * 0.25), 0, Math.PI * 2); ctx.fillStyle = palette.teal; ctx.fill();
    ctx.save(); ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.arc(...goalPoint, goalRadius + 5, 0, Math.PI * 2); ctx.strokeStyle = "#61dfc44f"; ctx.lineWidth = 0.7; ctx.stroke(); ctx.restore();
    ctx.font = "600 8px ui-sans-serif, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.fillStyle = "#97e6d5";
    const goalLabelY = goalPoint[1] - goalRadius - 12 > originY + 9 ? goalPoint[1] - goalRadius - 12 : goalPoint[1] + goalRadius + 17;
    ctx.fillText("GOAL", goalPoint[0], goalLabelY);

    const radius = Math.max(5, (number(vehicle.radius) ?? 0.25) * scale);
    ctx.save();
    ctx.translate(...vehiclePoint);
    ctx.rotate(-(number(vehicle.heading) ?? 0));
    ctx.beginPath(); ctx.arc(0, 0, radius + 5, 0, Math.PI * 2); ctx.fillStyle = "#71ead010"; ctx.fill();
    ctx.beginPath();
    ctx.moveTo(radius * 1.45, 0); ctx.lineTo(-radius * 0.95, -radius * 0.8); ctx.lineTo(-radius * 0.5, 0); ctx.lineTo(-radius * 0.95, radius * 0.8); ctx.closePath();
    ctx.fillStyle = "#e4f4f5"; ctx.shadowColor = "#61dfc4"; ctx.shadowBlur = 8; ctx.fill();
    ctx.shadowBlur = 0; ctx.lineWidth = 0.9; ctx.strokeStyle = "#91cec9"; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(radius * 0.6, 0); ctx.lineTo(-radius * 0.2, 0); ctx.strokeStyle = "#24585b"; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.restore();
    ctx.restore();

    const outcome = outcomeKind(env.outcome);
    if (outcome && !["running", "none"].includes(outcome)) {
      const text = outcomeName(env.outcome);
      ctx.font = "500 10px ui-sans-serif, system-ui, sans-serif";
      const boxWidth = ctx.measureText(text).width + 22;
      const boxX = width - boxWidth - 18;
      roundedRect(ctx, boxX, 11, boxWidth, 24, 5);
      ctx.fillStyle = outcome === "success" ? "#16443b" : outcome === "collision" ? "#472d31" : "#253746"; ctx.fill();
      ctx.fillStyle = outcome === "success" ? palette.teal : outcome === "collision" ? palette.coral : "#bdced8";
      ctx.textAlign = "center"; ctx.fillText(text, boxX + boxWidth / 2, 27);
    }
    elements["tank-canvas"].setAttribute("aria-label", `${scenes[state?.config?.scenario] || "Navigation"} tank, ${worldWidth} by ${worldHeight} metres. AUV at x ${fixed(vehicleX, 1)}, y ${fixed(vehicleY, 1)}. Goal distance ${fixed(env.distance_to_goal, 1)} metres. ${outcomeName(env.outcome) !== "—" ? outcomeName(env.outcome) : "Episode in progress"}.`);
  }

  function drawChart(canvas, series, xValues, range, percentage = false) {
    const setup = canvasContext(canvas);
    if (!setup) return;
    const { context: ctx, width, height } = setup;
    const padding = { left: 37, right: 9, top: 10, bottom: 24 };
    const plotWidth = Math.max(1, width - padding.left - padding.right);
    const plotHeight = Math.max(1, height - padding.top - padding.bottom);
    ctx.lineWidth = 0.7;
    ctx.font = "8px ui-sans-serif, system-ui, sans-serif";
    if (!xValues.length) {
      ctx.strokeStyle = "#21374788";
      for (let i = 0; i < 4; i += 1) { const y = padding.top + i * plotHeight / 3; ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(width - padding.right, y); ctx.stroke(); }
      return;
    }
    const [minimum, maximum] = range;
    const x = (index) => padding.left + (xValues.length === 1 ? plotWidth / 2 : index / (xValues.length - 1) * plotWidth);
    const y = (value) => padding.top + (maximum - value) / (maximum - minimum) * plotHeight;
    for (let i = 0; i <= 3; i += 1) {
      const value = maximum - (maximum - minimum) * i / 3;
      const top = padding.top + plotHeight * i / 3;
      ctx.strokeStyle = palette.grid;
      ctx.beginPath(); ctx.moveTo(padding.left, top); ctx.lineTo(width - padding.right, top); ctx.stroke();
      ctx.fillStyle = palette.label; ctx.textAlign = "right";
      const label = percentage ? `${Math.round(value * 100)}%` : Math.abs(value) >= 1000 ? compactFormat.format(value) : Math.abs(maximum - minimum) < 3 ? value.toFixed(1) : Math.round(value).toString();
      ctx.fillText(label, padding.left - 7, top + 3);
    }
    const labels = [...new Set([0, Math.floor((xValues.length - 1) / 2), xValues.length - 1])];
    ctx.fillStyle = palette.label; ctx.textAlign = "center";
    labels.forEach((index) => ctx.fillText(String(xValues[index]), x(index), height - 6));
    ctx.save();
    ctx.beginPath(); ctx.rect(padding.left - 2, padding.top - 3, plotWidth + 4, plotHeight + 6); ctx.clip();
    series.forEach((line) => {
      ctx.beginPath();
      line.values.forEach((value, index) => { if (index === 0) ctx.moveTo(x(index), y(value)); else ctx.lineTo(x(index), y(value)); });
      ctx.strokeStyle = line.color; ctx.lineWidth = line.width || 1.7; ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.setLineDash(line.dashed ? [3, 3] : []); ctx.stroke(); ctx.setLineDash([]);
      if (line.values.length === 1) { ctx.beginPath(); ctx.arc(x(0), y(line.values[0]), 2.7, 0, Math.PI * 2); ctx.fillStyle = line.color; ctx.fill(); }
    });
    ctx.restore();
  }

  function drawCharts() {
    const start = Math.max(0, chartHistory.length - 240);
    const returns = [];
    const averages = [];
    const successes = [];
    const collisions = [];
    const episodes = [];
    for (let i = start; i < chartHistory.length; i += 1) {
      const entry = chartHistory[i];
      episodes.push(number(entry.episode) ?? i + 1);
      returns.push(Number(entry.return));
      const averageWindow = chartHistory.slice(Math.max(0, i - 9), i + 1);
      averages.push(averageWindow.reduce((sum, item) => sum + Number(item.return), 0) / averageWindow.length);
      const outcomeWindow = chartHistory.slice(Math.max(0, i - 19), i + 1);
      successes.push(outcomeWindow.filter((item) => outcomeKind(item.outcome) === "success").length / outcomeWindow.length);
      collisions.push(outcomeWindow.filter((item) => outcomeKind(item.outcome) === "collision").length / outcomeWindow.length);
    }
    const values = [...returns, ...averages];
    const smallest = values.length ? Math.min(...values) : 0;
    const largest = values.length ? Math.max(...values) : 1;
    const margin = Math.max((largest - smallest) * 0.15, Math.abs(largest) * 0.03, 1);
    drawChart(elements["return-chart"], [{ values: returns, color: palette.teal, width: 1.5 }, { values: averages, color: palette.muted, width: 1.4, dashed: true }], episodes, [smallest - margin, largest + margin]);
    drawChart(elements["outcome-chart"], [{ values: successes, color: palette.teal }, { values: collisions, color: palette.coral }], episodes, [0, 1], true);
  }

  const resize = () => { queueDraw(); drawCharts(); };
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(resize);
    observer.observe(elements["tank-wrap"]);
    observer.observe(elements["return-chart"].parentElement);
    observer.observe(elements["outcome-chart"].parentElement);
  }
  window.addEventListener("resize", resize);
  window.addEventListener("pagehide", () => { stopped = true; window.clearTimeout(pollTimer); if (activeRequest) activeRequest.abort(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) { stopped = false; schedulePoll(); } });

  setText("action-1-label", "Acceleration command");
  setText("action-2-label", "Turn command");
  elements["action-1"].title = "Normalized acceleration command from −1 to +1. Negative values brake; there is no reverse thrust.";
  elements["action-2"].title = "Normalized turn command from −1 to +1. Positive turns are counterclockwise in world coordinates.";
  elements.controller.options[0].textContent = names.baseline;
  updateControllerNote();
  updateNoise();
  updateControls();
  queueDraw();
  drawCharts();
  pollState();
})();
