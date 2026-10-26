/* Optional browser integration check for Code walkthrough; no dependencies at runtime. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.AUV_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname, "..");
const artifacts = path.join(root, "playwright-artifacts", "walkthrough");
fs.mkdirSync(artifacts, { recursive: true });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  try {
    const context = await browser.newContext({ offline: true, viewport: { width: 1440, height: 1100 } });
    const page = await context.newPage();
    const errors = [], requests = [], stages = new Set();
    page.on("pageerror", error => errors.push(error.message));
    context.on("request", request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
    const state = () => page.evaluate(() => window.auvRequest("/api/state"));
    async function until(predicate, timeout = 20000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        const value = await state();
        assert.notEqual(value.mode, "error", value.message);
        if (predicate(value)) return value;
        await pause(30);
      }
      throw new Error("Browser did not reach the expected walkthrough state");
    }
    async function rendered() {
      const value = await state();
      await page.waitForFunction(sequence => Number(document.getElementById("trace-panel").dataset.sequence) === sequence,
        value.trace.sequence);
      stages.add(value.trace.stage.key);
      assert.equal(await page.locator("#trace-panel").getAttribute("data-stage"), value.trace.stage.key);
      assert.ok((await page.locator("#trace-code").innerText()).trim().length > 0, "Executed block is empty");
      assert.ok(!(await page.locator("#trace-code").innerText()).includes("Source unavailable"));
      assert.match(await page.locator("#trace-source-file").innerText(), /^web_src\/(worker|td3|environment)\.js/);
      return value;
    }
    async function advance(button = "#trace-next") {
      const before = await state();
      await page.locator(button).click();
      await until(value => value.trace.sequence > before.trace.sequence);
      return rendered();
    }
    async function stepTo(key) {
      for (let index = 0; index < 32; index++) {
        const value = await rendered();
        if (value.trace.stage.key === key) return value;
        await advance();
      }
      throw new Error(`Did not reach the ${key} block`);
    }

    await page.goto(pathToFileURL(path.join(root, "auv-practice.html")).href);
    await page.waitForFunction(() => document.getElementById("connection-text").textContent === "Browser engine ready");
    const initial = await state();
    await page.locator("#code-view").click();
    await page.locator("#trace-panel").waitFor({ state: "visible" });
    let value = await rendered();
    assert.equal(value.trace.role, "preview");
    assert.equal(value.trace.stage.key, "observation");
    assert.equal(await page.locator("#train-button").isEnabled(), false);
    const stationary = value.env.vehicle;
    await stepTo("actor");
    assert.deepEqual((await state()).env.vehicle, stationary);
    value = await stepTo("environment");
    assert.notDeepEqual(value.env.vehicle, stationary);
    assert.equal(value.env.episode_step, initial.env.episode_step + 1);
    const movementStep = value.env.episode_step;
    value = await advance("#trace-tick");
    assert.equal(value.trace.at_boundary, true);
    assert.equal(value.env.episode_step, movementStep);
    value = await advance("#trace-tick");
    assert.equal(value.env.episode_step, movementStep + 1);
    assert.equal(value.training.updates, initial.training.updates);
    assert.equal(value.training.buffer_size, 0);

    await page.locator("#trace-pace").selectOption("300");
    const sequence = value.trace.sequence;
    await page.locator("#trace-play").click();
    await until(current => current.trace.sequence >= sequence + 3);
    await page.locator("#trace-play").click();
    // A request already in flight may finish before the pause click is handled.
    await pause(100);
    const held = await state();
    await pause(420);
    assert.equal((await state()).trace.sequence, held.trace.sequence);
    assert.deepEqual((await state()).env.vehicle, held.env.vehicle);

    await page.locator("#simulation-view").click();
    await until(current => !current.trace.enabled && current.mode === "paused");
    assert.equal(await page.locator("#trace-panel").isVisible(), false);
    await page.locator("#training-steps").fill("34");
    await page.locator("#train-button").click();
    await until(current => current.mode === "paused" && current.training.steps === 50034);
    await page.locator("#code-view").click();
    await page.locator("#trace-role").selectOption("training");
    await until(current => current.trace.enabled && current.trace.role === "training");
    value = await stepTo("replay_sample");
    const learningVehicle = value.env.vehicle;
    const sampledIndex = value.trace.stage.values.sample_index;
    await stepTo("target_actions");
    value = await stepTo("bellman_target");
    assert.equal(value.trace.stage.values.sample_index, sampledIndex);
    assert.deepEqual(value.env.vehicle, learningVehicle);
    assert.equal(await page.locator("#trace-equation").isVisible(), true);
    assert.match(await page.locator("#trace-values-context").innerText(), /replay|sample|minibatch/i);
    await page.screenshot({ path: path.join(artifacts, "desktop-target.png"), fullPage: true });
    value = await stepTo("critic");
    assert.deepEqual(value.env.vehicle, learningVehicle);
    value = await stepTo("actor_skip");
    assert.equal(value.training.updates, 48877);
    value = await advance("#trace-tick");
    assert.equal(value.trace.at_boundary, true);
    value = await stepTo("actor_update");
    assert.equal(value.training.actor_updates, 24439);
    value = await stepTo("soft_targets");
    assert.equal(value.training.updates, 48878);
    const softTarget = value.trace.stage.values;
    assert.equal(softTarget.tau, 0.005);

    await page.setViewportSize({ width: 390, height: 844 });
    await pause(100);
    const mobile = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth }));
    assert.ok(mobile.document <= mobile.width, `Mobile document overflows ${mobile.document} > ${mobile.width}`);
    await page.screenshot({ path: path.join(artifacts, "mobile-target.png"), fullPage: true });
    await page.locator("#trace-next").click();
    await until(current => current.trace.at_boundary);
    await page.locator("#simulation-view").click();
    const final = await until(current => !current.trace.enabled && current.mode === "paused");
    assert.equal(final.training.steps, 50036);
    assert.equal(await page.locator("#train-button").isEnabled(), true);
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    const report = {
      browser: await browser.version(), protocol: "file://", offline: true,
      network_requests: requests.length, page_errors: errors.length,
      preview_does_not_train: true, environment_moves_only_at_environment_block: true,
      next_auv_step_advances_exactly_one_transition: true, autoplay_pause_preserves_stage: true,
      real_replay_target_visible: true, delayed_actor_and_soft_targets_visible: true,
      training_steps: final.training.steps, critic_updates: final.training.updates,
      actor_updates: final.training.actor_updates, observed_stages: [...stages], mobile,
    };
    fs.writeFileSync(path.join(artifacts, "validation.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
