/* Optional integration check: install Playwright for development, not to use the HTML. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.AUV_PLAYWRIGHT_MODULE || "playwright");

const root = path.resolve(__dirname, "..");
const artifacts = path.join(root, "playwright-artifacts", "standalone");
fs.mkdirSync(artifacts, { recursive: true });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  try {
    const context = await browser.newContext({ offline: true, acceptDownloads: true, viewport: { width: 1440, height: 1100 } });
    const page = await context.newPage();
    const errors = [], network = [];
    page.on("pageerror", error => errors.push(error.message));
    context.on("request", request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
    const state = () => page.evaluate(() => window.auvRequest("/api/state"));
    async function until(predicate, timeout = 20000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        const value = await state();
        assert.notEqual(value.mode, "error", value.message);
        if (predicate(value)) return value;
        await pause(50);
      }
      throw new Error("Browser state did not reach the expected condition");
    }

    await page.goto(pathToFileURL(path.join(root, "auv-practice.html")).href);
    await page.waitForFunction(() => document.getElementById("connection-text").textContent === "Browser engine ready");
    const initial = await state();
    assert.equal(initial.training.steps, 50000);
    assert.equal(initial.policy_ready, true);
    assert.equal(initial.controller, "td3");
    assert.equal(await page.locator("#tank-loading").isVisible(), false);
    await page.locator("#step-button").click();
    await until(s => s.env.episode_step === 1);
    await page.locator("#play-button").click();
    await until(s => s.mode === "preview" && s.env.episode_step >= 4);
    await page.locator("#pause-button").click();
    const preview = await until(s => s.mode === "paused");
    assert.equal(preview.training.steps, 50000);
    assert.equal(preview.training.buffer_size, 0);

    await page.locator("#evaluate-button").click();
    const evaluated = await until(s => s.mode === "paused" && s.evaluation?.episodes === 10);
    assert.equal(evaluated.training.updates, initial.training.updates);
    assert.equal(evaluated.training.buffer_size, 0);
    assert.equal(evaluated.evaluation.engine, "JavaScript");
    await page.locator("#evaluation-results").waitFor({ state: "visible" });

    const modelDownload = page.waitForEvent("download");
    await page.locator("#save-button").click();
    const download = await modelDownload;
    const checkpointPath = path.join(artifacts, download.suggestedFilename());
    await download.saveAs(checkpointPath);
    const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, "utf8"));
    assert.equal(checkpoint.learner.updates, 48873);
    assert.equal(checkpoint.training.steps, 50000);

    await page.locator("#reset-button").click();
    await until(s => s.training.steps === 0);
    await page.locator("#import-file").setInputFiles(checkpointPath);
    await until(s => s.training.steps === 50000);
    assert.equal(await page.locator("#error-banner").isVisible(), false);

    await page.locator("#reset-button").click();
    await until(s => s.training.steps === 0);
    await page.locator("#training-steps").fill("1200");
    await page.locator("#train-button").click();
    const trained = await until(s => s.mode === "paused" && s.training.steps === 1200, 60000);
    assert.equal(trained.training.updates, 200);
    assert.equal(trained.training.actor_updates, 100);
    assert.ok(Number.isFinite(trained.training.critic_loss));
    assert.ok(Number.isFinite(trained.training.actor_loss));

    await page.locator("#training-steps").fill("100000");
    await page.locator("#train-button").click();
    await until(s => s.mode === "training" && s.training.steps >= 1210);
    const pauseStarted = Date.now();
    await page.locator("#pause-button").click();
    const paused = await until(s => s.mode === "paused");
    const pauseResponseMs = Date.now() - pauseStarted;
    await pause(150);
    assert.equal((await state()).training.steps, paused.training.steps);
    assert.ok(paused.training.steps < 101200);

    const resultsDownload = page.waitForEvent("download");
    await page.locator("#export-results-button").click();
    const resultsFile = await resultsDownload;
    const resultsPath = path.join(artifacts, "auv-results.json");
    await resultsFile.saveAs(resultsPath);
    const exported = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
    assert.equal(exported.training.steps, paused.training.steps);
    assert.equal(exported.learner_config.hidden_dim, 64);

    await page.locator("#import-file").setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from('{"format":"invalid"}') });
    await page.locator("#error-banner").waitFor({ state: "visible" });
    assert.equal((await state()).training.steps, paused.training.steps);
    await page.locator("#dismiss-error").click();

    // Restore the bundled example, run a visible trajectory, and capture both layouts.
    await page.locator("#checkpoint-select").selectOption("td3-demo-reef-50000.json");
    await page.locator("#load-button").click();
    await until(s => s.training.steps === 50000);
    await page.locator("#preview-speed").selectOption("12");
    await page.locator("#play-button").click();
    await until(s => s.env.outcome === "success" || s.env.outcome === "collision" || s.env.outcome === "timeout");
    await page.locator("#pause-button").click();
    const visibleEpisode = await until(s => s.mode === "paused");
    await page.locator("#evaluate-button").click();
    await until(s => s.mode === "paused" && s.evaluation?.episodes === 10);
    await page.locator("#evaluation-results").waitFor({ state: "visible" });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(artifacts, "desktop.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await pause(150);
    const size = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth }));
    assert.ok(size.document <= size.width, `Mobile page overflows: ${size.document} > ${size.width}`);
    await page.screenshot({ path: path.join(artifacts, "mobile.png"), fullPage: true });

    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    const report = {
      browser: await browser.version(), protocol: "file://", offline: true,
      network_requests: network.length, page_errors: errors.length,
      evaluation: evaluated.evaluation,
      fresh_training: { steps: 1200, critic_updates: trained.training.updates, actor_updates: trained.training.actor_updates,
        critic_loss: trained.training.critic_loss, actor_loss: trained.training.actor_loss },
      checkpoint_download_and_import: true, result_export: true, invalid_import_preserved_model: true,
      pause_response_ms: pauseResponseMs, default_preview_outcome: visibleEpisode.env.outcome,
      mobile: size,
    };
    fs.writeFileSync(path.join(artifacts, "validation.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
