/* Browser ↔ local Worker bridge. There are no network requests. */
(() => {
  "use strict";
  const pending = new Map();
  let sequence = 0;
  let worker;
  let failed = null;
  let objectURL;

  function fail(error) {
    failed = error instanceof Error ? error : new Error(String(error));
    for (const request of pending.values()) { clearTimeout(request.timeout); request.reject(failed); }
    pending.clear();
  }

  function call(path, payload) {
    if (failed) return Promise.reject(failed);
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error("The browser engine is taking too long. Pause or reduce the training budget.")); }, 30000);
      pending.set(id, { resolve, reject, timeout });
      worker.postMessage({ id, path, payload });
    });
  }

  function download({ filename, payload }) {
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  let ready;
  try {
    const source = JSON.parse(document.getElementById("auv-worker-source").textContent);
    const demo = JSON.parse(document.getElementById("auv-demo-model").textContent);
    objectURL = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
    worker = new Worker(objectURL);
    worker.onerror = event => fail(new Error(event.message || "This browser could not start the local simulation worker."));
    worker.onmessageerror = () => fail(new Error("The browser could not read a message from the simulation."));
    worker.onmessage = event => {
      const response = event.data;
      const request = pending.get(response.id);
      if (!request) return;
      clearTimeout(request.timeout);
      pending.delete(response.id);
      if (response.ok) request.resolve(response.data);
      else request.reject(new Error(response.error || "Local simulation error."));
    };
    ready = call("/init", { demo });
    ready.then(() => URL.revokeObjectURL(objectURL), () => URL.revokeObjectURL(objectURL));
  } catch (error) {
    fail(error);
    ready = Promise.reject(failed);
  }
  // Let the UI show startup failures through its ordinary error panel.
  ready.catch(() => {});
  window.auvRequest = async (path, payload) => {
    await ready;
    const response = await call(path, payload);
    if (response?.download) { download(response.download); return response.state; }
    return response;
  };
})();
