"""Bundle exact, attributed Python references for the browser walkthrough.

These files are displayed as references; the offline browser executes JavaScript.
No upstream module is imported or executed by this build helper. SHA256 checks
pin both the original TD3 source and the separate AUV Python implementation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable


UPSTREAM_REPOSITORY = "sfujim/TD3"
UPSTREAM_REVISION = "34770ccdcb51df6f6c7d85c1ba7e40d71b940d16"
UPSTREAM_FILES = {
    "TD3.py": "70bec1b3b9e7c623e732c1f81d95fb30618a550919bfcbaeea31c82d54754df0",
    "main.py": "507a93012bdd1d83f61947d7ab40074085311aecd918f610bf00fe9cc39c2f24",
    "utils.py": "202d694690980c259e91a4e49259c131e95bed215b2bcf1ba108900d4edc1db0",
    "LICENSE": "5a596c473413553b1fb3e8db602e32524b20166adbbd0f58cbcda60b16e19195",
}
AUV_REPOSITORY = "AWDaley/AUV-practice"
AUV_REVISION = "75b7b262641b097c8056b2005235910ffafa4af4"
AUV_FILES = {
    "auv_sim/environment.py": "99eab386d26afd102e233e4bbb0b026569ea6e20d4086b2a908142a6349855d5",
    "auv_sim/experiment.py": "9626ff73386c231f422c98d052c5c2cfdc8d752a6abb749b4b680227ea02f69c",
}


def python_trace_bundle(root: Path, expected_keys: Iterable[str] = ()) -> dict:
    """Return verified, unmodified source excerpts with absolute line numbers.

    Updating a source requires an intentional revision/hash/mapping update. A
    stale line number or newly added unmapped JavaScript stage fails the build.
    """
    root = Path(root)
    vendor = root / "third_party/sfujim_td3"
    manifest = json.loads((vendor / "provenance.json").read_text(encoding="utf-8"))
    if (manifest.get("repository"), manifest.get("revision"), manifest.get("license")) != (
        UPSTREAM_REPOSITORY, UPSTREAM_REVISION, "MIT"
    ):
        raise ValueError("The TD3 reference provenance does not match the pinned upstream revision")
    if set(manifest.get("files", {})) != set(UPSTREAM_FILES):
        raise ValueError("The TD3 reference provenance must describe exactly the four upstream files")

    files = {}
    for name, expected in {**UPSTREAM_FILES, **AUV_FILES}.items():
        upstream = name in UPSTREAM_FILES
        path = vendor / name if upstream else root / name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"Pinned Python reference changed: {name}")
        repository = UPSTREAM_REPOSITORY if upstream else AUV_REPOSITORY
        revision = UPSTREAM_REVISION if upstream else AUV_REVISION
        url = f"https://github.com/{repository}/blob/{revision}/{name}"
        if upstream:
            entry = manifest["files"][name]
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if (entry.get("sha256"), entry.get("git_blob_sha"), entry.get("bytes"), entry.get("url")) != (
                digest, blob, len(data), url
            ):
                raise ValueError(f"Upstream provenance mismatch: {name}")
        text = data.decode("utf-8")
        files[name] = {
            "file": name, "local_file": path.relative_to(root).as_posix(),
            "language": "python", "origin": "td3-reference" if upstream else "auv-python",
            "url": url, "revision": revision, "sha256": digest,
            "text": text, "lines": text.splitlines(),
            "tree": ast.parse(text, filename=name) if name.endswith(".py") else None,
        }

    def section(name, start, end, *, first=None, last=None, focus=(), note="", kind="reference"):
        source = files[name]
        lines = source["lines"]
        if not (1 <= start <= end <= len(lines)):
            raise ValueError(f"Invalid Python source range: {name}:{start}-{end}")
        if first is not None and lines[start - 1].strip() != first:
            raise ValueError(f"Python source start anchor changed: {name}:{start}")
        if last is not None and lines[end - 1].strip() != last:
            raise ValueError(f"Python source end anchor changed: {name}:{end}")
        focused = list(focus or range(start, end + 1))
        if len(set(focused)) != len(focused) or any(not start <= line <= end for line in focused):
            raise ValueError(f"Python focus lies outside excerpt: {name}:{start}-{end}")
        return {
            key: value for key, value in source.items() if key not in {"text", "lines", "tree"}
        } | {
            "start_line": start, "end_line": end, "lines": lines[start - 1:end],
            "url": f'{source["url"]}#L{start}-L{end}',
            "focus": focused, "note": note, "kind": kind,
        }

    def definition(name, qualified_name, **options):
        node = files[name]["tree"]
        for part in qualified_name.split("."):
            matches = [child for child in node.body
                       if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == part]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous Python definition: {name}:{qualified_name}")
            node = matches[0]
        return section(name, node.lineno, node.end_lineno, **options)

    upstream_note = "Original sfujim/TD3 Python reference; the live values come from the browser's matching JavaScript stage."
    auv_note = "AUV-practice Python environment code, separate from the original TD3 repository; displayed as a reference to this browser stage."
    sources = {
        "actor": definition("TD3.py", "TD3.select_action", focus=[100, 101],
            note=f"{upstream_note} Here state corresponds to the 21-feature browser observation; max_action is 1 for this AUV."),
        "network_forward": definition("TD3.py", "Actor.forward", focus=[26, 27, 28],
            note="Original Actor.forward. The upstream constructor uses 256-unit hidden layers; this browser uses 128 for the included model or 64 after Reset. The same two-ReLU/tanh structure is used, with max_action = 1."),
        "actor_architecture": definition("TD3.py", "Actor", focus=[18, 19, 20, 26, 27, 28],
            note="This is the unmodified upstream 256 → 256 actor architecture, not the browser's current network dimensions."),
        "critic_forward": definition("TD3.py", "Critic.forward", focus=[47, 49, 50, 51, 53, 54, 55, 56],
            note="Original twin-critic forward pass. State and action are concatenated; two independent branches each return one Q estimate. The upstream hidden width is 256; the browser uses the loaded model's width."),
        "target_actions": section("TD3.py", 110, 118, first="with torch.no_grad():",
            last=").clamp(-self.max_action, self.max_action)", focus=[112, 113, 114, 116, 117, 118],
            note=f"{upstream_note} Upstream noise already contains the clipped noise; the browser inspector shows raw and clipped noise separately. max_action = 1 here; JavaScript and PyTorch draw different random sequences."),
        "bellman_target": section("TD3.py", 120, 123, first="# Compute the target Q value",
            last="target_Q = reward + not_done * self.discount * target_Q", focus=[121, 122, 123],
            note="Inside torch.no_grad(), target_Q first holds the smaller target-critic estimate on line 122, then is overwritten by the final TD target on line 123. The inspector's Smaller target Q and TD target distinguish these values for one actual browser replay row, not necessarily the visible AUV's current state."),
        "critic": section("TD3.py", 125, 134, first="# Get current Q estimates",
            last="self.critic_optimizer.step()", focus=[126, 129, 132, 133, 134],
            note="Original critic loss and optimizer step. Displayed Q1/Q2 are batch means corresponding to the current_Q1/current_Q2 tensors. Live loss is the sum of both minibatch MSEs. Upstream uses one optimizer for both independent critics; the browser uses separate Adam states."),
        "actor_skip": section("TD3.py", 136, 137, first="# Delayed policy updates",
            last="if self.total_it % self.policy_freq == 0:", focus=[137],
            note="This condition is false at the current browser stage, so the actor and target updates are skipped. Original total_it increments at train() entry, line 105; the browser counts the critic update after its optimizer steps."),
        "actor_update": section("TD3.py", 139, 145, first="# Compute actor losse",
            last="self.actor_optimizer.step()", focus=[140, 143, 144, 145],
            note="Original actor update, inside the delayed-policy condition on line 137. Both implementations optimize −mean(Q1(s, actor(s))). The browser avoids accumulating unused critic parameter gradients during this step."),
        "soft_targets": section("TD3.py", 147, 152, first="# Update the frozen target models",
            last="target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)",
            focus=[148, 149, 151, 152],
            note="Original target interpolation, performed only with a delayed actor update. Live first-weight examples are ordered actor / Q1 / Q2; upstream groups Q1 and Q2 inside one critic module."),
        "replay_sample": section("TD3.py", 107, 108, first="# Sample replay buffer",
            last="state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)", focus=[108],
            note="Original replay-sampling call. Its tuple order is state, action, next_state, reward, not_done; the live inspector labels the corresponding browser arrays. Upstream defaults to batch 256, browser to 32. total_it increments earlier on line 105."),
        "replay_sampling": definition("utils.py", "ReplayBuffer.sample", focus=[32, 35, 36, 37, 38, 39],
            note="Exact original uniform sampling with replacement. The browser's seed generator differs from NumPy's, so the shown row is from this browser run, not a rerun of this Python code."),
        "replay_add": definition("utils.py", "ReplayBuffer.add", focus=[21, 22, 23, 24, 25, 27, 28],
            note="Exact original replay write. Upstream main.py converts old Gym done to done_bool before this call; this AUV directly uses terminated and keeps time-limit truncations bootstrappable. Argument order differs, so follow the named values."),
        "exploration": section("main.py", 120, 123, first="action = (",
            last=").clip(-max_action, max_action)", focus=[121, 122, 123],
            note="Original action-selection expression includes the actor call. The browser has already shown that forward pass at its actor stage and now adds the displayed exploration noise. max_action = 1 for this AUV."),
        "random_warmup": section("main.py", 117, 118, first="if t < args.start_timesteps:",
            last="action = env.action_space.sample()", focus=[117, 118],
            note="Original training warm-up branch. Upstream defaults to 25,000 random interactions; this browser uses 1,000 cumulative training interactions. This branch is not the random preview controller."),
        "random": section("auv_sim/experiment.py", 130, 131, first='if controller == "random":',
            last="return self.rng.uniform(-1, 1, size=2).astype(np.float32)", focus=[131],
            note="AUV-practice's random preview controller, not a TD3 policy or the upstream training warm-up branch. It does not learn."),
        "observation": section("auv_sim/experiment.py", 217, 217, first="obs = self.observation",
            note=f"{auv_note} Reads an observation already computed by reset or the preceding environment step; the sensor construction is shown in Observation features."),
        "observation_features": definition("auv_sim/environment.py", "AUVEnv._observation",
            focus=[302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312],
            note="The 21-feature sonar/goal/speed/current encoding belongs to this AUV environment. It is not part of sfujim/TD3. These features are computed during reset or step, then read by the actor."),
        "environment": section("auv_sim/experiment.py", 225, 225,
            first="nxt, reward, terminated, truncated, info = self.env.step(self.action)",
            note="AUV-practice Python training-loop call. The environment computes motion, reward, and termination. This Gymnasium API separates terminated from truncated; the original TD3 main.py uses the older Gym done API."),
        "physics": definition("auv_sim/environment.py", "AUVEnv.step",
            focus=[212, 214, 215, 216, 217, 218, 219, 222, 223, 233, 240, 241, 242, 243, 244, 249, 250, 251, 252, 253, 254, 255],
            note="Exact AUV-practice Python dynamics and reward code. These are the separate 2D teaching environment, not part of the original TD3 algorithm. Browser calculations follow the corresponding JavaScript implementation."),
        "reset": section("auv_sim/experiment.py", 137, 139, first="if self.episode_done:",
            last="self.episode_done = False", focus=[138, 139],
            note="AUV-practice Python preview reset. The browser walkthrough also delays a training reset until the next visible transition, so a final goal/collision frame remains inspectable; ordinary Python training resets immediately."),
        "baseline": section("auv_sim/experiment.py", 128, 129, first='if controller == "baseline":',
            last="return env.heuristic_action()", focus=[129],
            note="AUV-practice's hand-written map-aware controller. It has no counterpart in the original TD3 algorithm and does not update a policy."),
        "learning_wait": section("auv_sim/experiment.py", 231, 233,
            first='if (self.training["steps"] > self.warmup_steps', last="metrics = self.agent.update(self.replay)",
            focus=[231, 232],
            note="AUV-practice's update guard is false here: warm-up or a full replay minibatch is still required. The original main.py checks start_timesteps; this guard additionally supports loading a trained model with empty replay."),
        "episode": section("auv_sim/experiment.py", 237, 244, first="if terminated or truncated:",
            last='"outcome": info["outcome"]})', focus=[237, 238, 239, 240, 241, 242, 243, 244],
            note="AUV-practice records a completed training episode. Python resets on the following line; the browser walkthrough holds this final position until you advance."),
        "transition_end": definition("auv_sim/experiment.py", "Experiment._preview_tick", focus=[140, 141, 142],
            note="Browser pause boundary: there is no separate Python statement corresponding to transition_end. This exact Python preview loop is context for a completed transition; training additionally stores replay and may update the networks.", kind="context"),
    }
    sources["soft_update"] = {**sources["soft_targets"]}
    sources["replay_storage"] = {**sources["replay_add"]}
    missing = set(expected_keys) - sources.keys()
    if missing:
        raise ValueError(f"Python reference mappings missing for walkthrough stages: {sorted(missing)}")
    return {
        "repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION,
        "url": f"https://github.com/{UPSTREAM_REPOSITORY}/tree/{UPSTREAM_REVISION}",
        "license": files["LICENSE"]["text"], "license_name": "MIT",
        "license_url": files["LICENSE"]["url"],
        "provenance": manifest,
        "auv_reference": {"repository": AUV_REPOSITORY, "revision": AUV_REVISION, "sha256": AUV_FILES},
        "mapping_note": "Python source is a synchronized reference. The standalone browser executes JavaScript; the displayed values are captured from that browser run.",
        "sources": sources,
    }
