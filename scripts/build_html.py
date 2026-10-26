"""Build the offline HTML, exporting the real PyTorch checkpoint into JS arrays.

Run with the project's Python environment. Opening the resulting HTML needs
neither Python nor a server; these dependencies are only for rebuilding it.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import torch

try:
    from .python_trace_sources import python_trace_bundle
except ImportError:  # Direct invocation: python scripts/build_html.py
    from python_trace_sources import python_trace_bundle


ROOT = Path(__file__).resolve().parents[1]


def trace_sources() -> dict:
    """Extract the displayed code from the exact executable source at build time."""
    blocks = {}
    marker = re.compile(r"^\s*// @trace-(start|end) ([a-z_]+)\s*$")
    for name in ("environment", "td3", "worker"):
        filename = f"web_src/{name}.js"
        lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
        opened = {}
        for number, line in enumerate(lines, start=1):
            match = marker.match(line)
            if not match:
                continue
            kind, key = match.groups()
            if kind == "start":
                if key in opened or key in blocks:
                    raise ValueError(f"Duplicate trace source marker: {key}")
                opened[key] = number + 1
            else:
                if key not in opened:
                    raise ValueError(f"Unmatched trace source end: {key}")
                start = opened.pop(key)
                if number <= start:
                    raise ValueError(f"Empty trace source block: {key}")
                blocks[key] = {"file": filename, "start_line": start,
                               "end_line": number - 1, "lines": lines[start - 1:number - 1]}
        if opened:
            raise ValueError(f"Unclosed trace source markers in {filename}: {list(opened)}")
    if not blocks:
        raise ValueError("The walkthrough needs source markers in the executable code")
    return blocks


def python_trace_sources() -> dict:
    """Reference Python follows the same stage keys, with explicit provenance."""
    return python_trace_bundle(ROOT, expected_keys=trace_sources())


def encode(tensors) -> str:
    flat = np.concatenate([t.detach().cpu().numpy().reshape(-1) for t in tensors]).astype("<f4")
    if not np.isfinite(flat).all():
        raise ValueError("Checkpoint contains nonfinite values")
    return base64.b64encode(flat.tobytes()).decode("ascii")


def network(state: dict, prefix: str) -> str:
    keys = [f"{prefix}{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")]
    return encode([state[key] for key in keys])


def adam(state: dict, indices: list[int]) -> dict:
    parameters = state["param_groups"][0]["params"]
    entries = [state["state"][parameters[index]] for index in indices]
    steps = [int(entry["step"].item()) for entry in entries]
    if len(set(steps)) != 1:
        raise ValueError("Cannot combine Adam states with different step counts")
    return {"step": steps[0], "m": encode([entry["exp_avg"] for entry in entries]),
            "v": encode([entry["exp_avg_sq"] for entry in entries])}


def export_demo() -> dict:
    payload = torch.load(ROOT / "auv_sim/assets/td3-demo-reef-50000.pt", map_location="cpu", weights_only=True)
    a = payload["agent"]
    config = dict(a["config"])
    config.update(batch_size=32, buffer_capacity=50_000)
    learner = {
        "version": 1, "obs_dim": a["obs_dim"], "action_dim": a["action_dim"], "config": config,
        "updates": a["updates"], "actor_updates": a["actor_updates"],
        "networks": {
            "actor": network(a["actor"], "net."),
            "actor_target": network(a["actor_target"], "net."),
            "q1": network(a["critic"], "q1_net."),
            "q2": network(a["critic"], "q2_net."),
            "q1_target": network(a["critic_target"], "q1_net."),
            "q2_target": network(a["critic_target"], "q2_net."),
        },
        "optimizers": {
            "actor": adam(a["actor_optimizer"], list(range(6))),
            "q1": adam(a["critic_optimizer"], list(range(6))),
            "q2": adam(a["critic_optimizer"], list(range(6, 12))),
        },
        "rng": {"algorithm": "xorshift32-boxmuller-v1", "state": 7, "spare": None},
    }
    training = dict(payload["training"])
    training.update(buffer_size=0, steps_per_second=0, phase="Imported Python policy · 128 units")
    return {
        "format": "auv-practice-browser", "version": 1,
        "name": "td3-demo-reef-50000.json", "created": payload["metadata"]["created"],
        "source": "Real Python TD3 policy: 20,000 steps, then 30,000 resumed steps.",
        "config": dict(payload["metadata"]["config"]), "learner": learner,
        "training": training, "history": payload["history"], "evaluation": None,
        "resume_note": "Replay and rollout RNG restart. Browser minibatch is 32. Target-noise RNG is JavaScript-specific.",
    }


def export_fixtures(demo: dict, path: Path) -> None:
    from auv_sim.environment import AUVEnv, EnvConfig
    from auv_sim.td3 import TD3, TD3Config
    source = torch.load(ROOT / "auv_sim/assets/td3-demo-reef-50000.pt", map_location="cpu", weights_only=True)
    agent = TD3(21, 2, TD3Config(**source["agent"]["config"]), seed=7)
    agent.load_state_dict(source["agent"])
    rows = []
    for scenario in ("open", "reef", "channel"):
        for seed in (7, 10007, 10008, 10009):
            env = AUVEnv(EnvConfig(scenario=scenario, current_strength=0.08))
            obs, _ = env.reset(seed=seed)
            start = {"position": env.position.tolist(), "goal": env.goal.tolist(),
                     "obstacles": env.obstacles.tolist(), "current": env.current.tolist(),
                     "heading": env.heading, "speed": env.speed}
            actions = [[0.7, -0.15], [0.25, 0.4], [-0.3, -0.1]]
            transitions = []
            for action in actions:
                nxt, reward, terminated, truncated, info = env.step(action)
                transitions.append({"action": action, "observation": nxt.tolist(), "reward": reward,
                                    "position": env.position.tolist(), "heading": env.heading,
                                    "speed": env.speed, "outcome": info["outcome"]})
            policy_action = agent.select_action(obs)
            rows.append({"scenario": scenario, "seed": seed, "start": start, "observation": obs.tolist(),
                         "policy_action": policy_action.tolist(), "q": agent.action_values(obs, policy_action),
                         "transitions": transitions})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "rows": rows}, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", action="store_true", help="Regenerate Python/JS comparison fixtures")
    args = p.parse_args()
    demo = export_demo()
    if args.fixtures:
        export_fixtures(demo, ROOT / "tests/fixtures/browser-parity.json")
    template = (ROOT / "web_src/ui.html").read_text(encoding="utf-8")
    worker = "\n".join((ROOT / f"web_src/{name}.js").read_text(encoding="utf-8") for name in ("environment", "td3", "worker"))
    bridge = (ROOT / "web_src/bridge.js").read_text(encoding="utf-8")
    ui = (ROOT / "web_src/ui.js").read_text(encoding="utf-8")
    trace_ui = (ROOT / "web_src/trace_ui.js").read_text(encoding="utf-8")
    styles = (ROOT / "auv_sim/static/styles.css").read_text(encoding="utf-8")
    styles += "\n" + (ROOT / "web_src/trace.css").read_text(encoding="utf-8")
    safe_json = lambda value: json.dumps(value, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    scripts = (f'<script id="auv-worker-source" type="application/json">{safe_json(worker)}</script>\n'
               f'<script id="auv-demo-model" type="application/json">{safe_json(demo)}</script>\n'
               f'<script id="auv-trace-source" type="application/json">{safe_json(trace_sources())}</script>\n'
               f'<script id="auv-python-trace-source" type="application/json">{safe_json(python_trace_sources())}</script>\n'
               f'<script>\n{bridge}\n</script>\n<script>\n{trace_ui}\n</script>\n<script>\n{ui}\n</script>')
    if template.count("<!-- INLINE_STYLES -->") != 1 or template.count("<!-- INLINE_SCRIPTS -->") != 1:
        raise ValueError("Template must contain exactly one styles and scripts placeholder")
    html = template.replace("<!-- INLINE_STYLES -->", f"<style>\n{styles}\n</style>")
    html = html.replace("<!-- INLINE_SCRIPTS -->", scripts)
    destination = ROOT / "auv-practice.html"
    destination.write_text(html, encoding="utf-8", newline="\n")
    print(json.dumps({"file": str(destination), "bytes": destination.stat().st_size,
                      "model_steps": demo["training"]["steps"], "external_dependencies": 0}))


if __name__ == "__main__":
    main()
