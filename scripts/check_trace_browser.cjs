/* Optional browser integration check for Code walkthrough; no dependencies at runtime. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.AUV_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname, "..");
const artifacts = path.join(root, "playwright-artifacts", "walkthrough-cleanup");
fs.mkdirSync(artifacts, { recursive: true });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  try {
    const context = await browser.newContext({ offline: true, viewport: { width: 1366, height: 768 } });
    const page = await context.newPage();
    const errors = [], requests = [], stages = new Set();
    const layout = {};
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

    async function focusIs(id) {
      await page.waitForFunction(expected => document.activeElement?.id === expected, id);
    }
    async function setDisclosure(id, open) {
      const details = page.locator(`#${id}`);
      if (await details.evaluate(element => element.open) !== open) await details.locator(":scope > summary").click();
      assert.equal(await details.evaluate(element => element.open), open);
    }
    async function inspectWithoutStepping(inspect) {
      const before = await state();
      await inspect();
      const after = await state();
      assert.equal(after.trace.sequence, before.trace.sequence, "Inspecting information must not execute code");
      assert.equal(after.trace.transition, before.trace.transition);
      assert.deepEqual(after.env, before.env);
      assert.deepEqual(after.training, before.training);
    }
    async function dashboardVisible(expected) {
      for (const selector of [".learning-panel", ".diagnostics-grid", ".control-column"]) {
        assert.equal(await page.locator(selector).isVisible(), expected, `${selector} visibility`);
      }
    }
    async function sourceReadable(minimumFont) {
      const metrics = await page.locator("#trace-code-scroll").evaluate(element => {
        const text = element.querySelector(".trace-line-text"), style = getComputedStyle(text);
        return { font_size: parseFloat(style.fontSize), white_space: style.whiteSpace,
          width: element.clientWidth, scroll_width: element.scrollWidth };
      });
      assert.ok(metrics.font_size >= minimumFont - 0.1, `Code is too small: ${metrics.font_size}px`);
      assert.ok(["pre-wrap", "break-spaces", "normal"].includes(metrics.white_space), `Code does not wrap: ${metrics.white_space}`);
      assert.ok(metrics.scroll_width <= metrics.width + 2, "Wrapped source still needs horizontal scrolling");
      return metrics;
    }
    async function focusedStatementVisible(expectedText) {
      // The UI positions the focus line on the next animation frame. Poll with
      // rAF so this checks the settled source pane, not the preceding paint.
      await page.waitForFunction(expected => {
        const pane = document.getElementById("trace-code-scroll");
        const statement = document.querySelector("#trace-code .trace-code-line.is-focus");
        if (!statement || !statement.textContent.includes(expected)) return false;
        const bounds = pane.getBoundingClientRect(), line = statement.getBoundingClientRect();
        return line.top >= bounds.top - 1 && line.bottom <= bounds.bottom + 1
          && line.left >= bounds.left - 1 && line.right <= bounds.right + 1;
      }, expectedText, { polling: "raf" });
      return page.evaluate(() => {
        const pane = document.getElementById("trace-code-scroll");
        const statement = document.querySelector("#trace-code .trace-code-line.is-focus");
        const bounds = pane.getBoundingClientRect(), line = statement.getBoundingClientRect();
        return { text: statement.querySelector(".trace-line-text").textContent,
          pane_top: bounds.top, pane_bottom: bounds.bottom, line_top: line.top,
          line_bottom: line.bottom, scroll_top: pane.scrollTop };
      });
    }
    async function compactDesktopLayout() {
      await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
      const boxes = await page.evaluate(() => {
        const bounds = selector => {
          const rect = document.querySelector(selector).getBoundingClientRect();
          return { x: rect.x, y: rect.y, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height };
        };
        return { viewport: { width: innerWidth, height: innerHeight }, document_height: document.documentElement.scrollHeight,
          toolbar: bounds("#trace-toolbar"), tank: bounds("#simulation"), panel: bounds("#trace-panel"),
          source: bounds("#trace-code-scroll"), values: bounds("#trace-values") };
      });
      assert.ok(boxes.tank.right <= boxes.panel.x + 1, "Desktop tank and code must sit beside each other without overlap");
      assert.ok(boxes.source.y >= boxes.toolbar.bottom - 1, "Toolbar covers the source");
      assert.ok(boxes.source.y < boxes.viewport.height && boxes.source.bottom > boxes.source.y,
        "Source is not visible in the desktop workspace");
      assert.ok(boxes.values.y >= boxes.source.bottom - 1, "Source overlaps the key values");
      assert.ok(boxes.values.bottom <= boxes.viewport.height + 2, "Key values are below the desktop viewport");
      const watches = await page.locator("#trace-values > div").count();
      assert.ok(watches > 0 && watches <= 4, `Expected up to four key values, received ${watches}`);
      return { ...boxes, key_values: watches, code: await sourceReadable(13) };
    }
    async function mobileContentReachable(selector) {
      await page.locator(selector).evaluate(element => element.scrollIntoView({ block: "center", inline: "nearest", behavior: "instant" }));
      const boxes = await page.evaluate(selector => {
        const rect = element => {
          const bounds = element.getBoundingClientRect();
          return { top: bounds.top, bottom: bounds.bottom, left: bounds.left, right: bounds.right,
            height: bounds.height, width: bounds.width };
        };
        return { height: innerHeight, toolbar: rect(document.getElementById("trace-toolbar")),
          content: rect(document.querySelector(selector)),
          buttons: ["trace-play", "trace-next", "trace-tick"].map(id => ({ id, ...rect(document.getElementById(id)) })) };
      }, selector);
      assert.ok(boxes.toolbar.top >= -1 && boxes.toolbar.bottom < boxes.height,
        "The mobile stepping toolbar must remain on screen while inspecting code");
      assert.ok(boxes.content.top >= boxes.toolbar.bottom - 1, `Sticky controls cover ${selector}`);
      assert.ok(boxes.content.bottom <= boxes.height + 2, `${selector} cannot be read in the remaining mobile viewport: ${JSON.stringify(boxes)}`);
      for (const button of boxes.buttons) {
        assert.ok(button.top >= boxes.toolbar.top - 1 && button.bottom <= boxes.toolbar.bottom + 1);
        assert.ok(button.width >= 40 && button.height >= 40, `${button.id} needs a usable touch target`);
      }
      return boxes;
    }

    await page.goto(pathToFileURL(path.join(root, "auv-practice.html")).href);
    await page.waitForFunction(() => document.getElementById("connection-text").textContent === "Browser engine ready");
    const initial = await state();
    await page.locator("#code-view").click();
    await page.locator("#trace-panel").waitFor({ state: "visible" });
    let value = await rendered();
    assert.equal(value.trace.role, "preview");
    assert.equal(value.trace.stage.key, "observation");
    await focusIs("trace-next");
    await dashboardVisible(false);
    assert.equal(await page.locator("#train-button").isEnabled(), false);
    assert.equal(await page.getByRole("button", { name: "Next block", exact: true }).isVisible(), true);
    assert.equal(await page.locator("#code-view").getAttribute("aria-pressed"), "true");
    assert.equal(await page.locator("#code-view").evaluate(element => getComputedStyle(element).opacity), "1");
    await inspectWithoutStepping(async () => {
      await setDisclosure("trace-source-detail", true);
      assert.equal(await page.locator("#trace-detail-code").isVisible(), true);
      assert.ok((await page.locator("#trace-detail-code").innerText()).includes("observation"));
      await setDisclosure("trace-source-detail", false);
      await setDisclosure("trace-observation-detail", true);
      assert.equal(await page.locator("#trace-observation-grid > div").count(), 21);
      assert.equal(await page.locator("#trace-observation-grid").isVisible(), true);
      await setDisclosure("trace-observation-detail", false);
    });
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

    await inspectWithoutStepping(async () => {
      await setDisclosure("trace-options", true);
      await page.locator("#trace-pace").selectOption("300");
      await setDisclosure("trace-options", false);
    });
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
    await focusIs("code-view");
    await dashboardVisible(true);
    await page.locator("#training-steps").fill("34");
    await page.locator("#train-button").click();
    await until(current => current.mode === "paused" && current.training.steps === 50034);
    await page.locator("#code-view").click();
    await page.locator("#trace-role").selectOption("training");
    await until(current => current.trace.enabled && current.trace.role === "training");
    value = await stepTo("replay_sample");
    const learningVehicle = value.env.vehicle;
    const sampledIndex = value.trace.stage.values.sample_index;
    await inspectWithoutStepping(async () => {
      await setDisclosure("trace-observation-detail", true);
      assert.equal(await page.locator("#trace-observation-grid > div").count(), 21);
      assert.match(await page.locator("#trace-observation-grid").innerText(), /→/);
      await setDisclosure("trace-observation-detail", false);
    });
    await stepTo("target_actions");
    value = await stepTo("bellman_target");
    assert.equal(value.trace.stage.values.sample_index, sampledIndex);
    assert.deepEqual(value.env.vehicle, learningVehicle);
    assert.equal(await page.locator("#trace-equation").isVisible(), true);
    assert.match(await page.locator("#trace-values-context").innerText(), /replay|sample|minibatch/i);
    layout.desktop = await compactDesktopLayout();
    await inspectWithoutStepping(async () => {
      layout.desktop.focused_statement = await focusedStatementVisible("work.target[i] =");
    });
    await page.screenshot({ path: path.join(artifacts, "desktop-target.png"), fullPage: true });
    await inspectWithoutStepping(async () => {
      await setDisclosure("trace-all-values", true);
      assert.equal(await page.locator("#trace-raw-values").isVisible(), true);
      const rawLabels = await page.locator("#trace-raw-values dt").allTextContents();
      for (const key of Object.keys(value.trace.stage.values)) assert.ok(rawLabels.includes(key), `Missing raw value: ${key}`);
      await setDisclosure("trace-all-values", false);
    });
    value = await stepTo("critic");
    assert.equal(await page.locator("#trace-values-context").innerText(), "Replay minibatch",
      "Critic losses and means describe the full minibatch, not only the illustrated replay row");
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
    const mobile = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth,
      height: document.documentElement.scrollHeight }));
    assert.ok(mobile.document <= mobile.width, `Mobile document overflows ${mobile.document} > ${mobile.width}`);
    await dashboardVisible(false);
    layout.mobile = { ...mobile, code: await sourceReadable(12) };
    await inspectWithoutStepping(async () => {
      layout.mobile.source = await mobileContentReachable("#trace-code .trace-code-line:first-child");
      await page.screenshot({ path: path.join(artifacts, "mobile-source-viewport.png") });
      layout.mobile.values = await mobileContentReachable("#trace-values");
      await page.screenshot({ path: path.join(artifacts, "mobile-values-viewport.png") });
      await setDisclosure("trace-source-detail", true);
      assert.equal(await page.locator("#trace-detail-code").isVisible(), true);
      await page.screenshot({ path: path.join(artifacts, "mobile-expanded-source.png"), fullPage: true });
      await setDisclosure("trace-source-detail", false);
      await setDisclosure("trace-all-values", true);
      assert.equal(await page.locator("#trace-raw-values").isVisible(), true);
      await setDisclosure("trace-all-values", false);
    });
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
    await page.screenshot({ path: path.join(artifacts, "mobile-target.png"), fullPage: true });
    await page.locator("#trace-next").click();
    await until(current => current.trace.at_boundary);
    await page.locator("#simulation-view").click();
    const final = await until(current => !current.trace.enabled && current.mode === "paused");
    assert.equal(final.training.steps, 50036);
    assert.equal(await page.locator("#train-button").isEnabled(), true);
    await focusIs("code-view");
    await dashboardVisible(true);
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
      disclosures_preserve_execution: true, focus_restored_on_mode_change: true,
      dashboard_hidden_only_during_walkthrough: true, source_wraps_without_horizontal_scroll: true,
      raw_variables_accessible: true, bellman_assignment_visible_without_manual_scroll: true, layout,
    };
    fs.writeFileSync(path.join(artifacts, "validation.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
