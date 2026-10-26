"""Source provenance and attribution gates for the Python reference viewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from scripts.python_trace_sources import python_trace_bundle


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PIN = "34770ccdcb51df6f6c7d85c1ba7e40d71b940d16"
AUV_PIN = "75b7b262641b097c8056b2005235910ffafa4af4"
GITHUB_BLOBS = {
    "TD3.py": "0b953d34412d86b398329afff5d4bb47f18ec5fb",
    "main.py": "a616475c7263aa5ad61093891680b6117b749550",
    "utils.py": "bf0f3198ad2b6d4e6dd82e7203f142aa86af700c",
    "LICENSE": "eec7f732facf8f3ff0753d024250cb8b2f95c467",
}


def test_vendored_files_match_the_pinned_github_blobs_byte_for_byte():
    vendor = ROOT / "third_party/sfujim_td3"
    provenance = json.loads((vendor / "provenance.json").read_text())
    assert provenance["repository"] == "sfujim/TD3"
    assert provenance["revision"] == UPSTREAM_PIN
    assert provenance["license"] == "MIT"
    for name, expected_blob in GITHUB_BLOBS.items():
        data = (vendor / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == expected_blob
        assert provenance["files"][name]["git_blob_sha"] == expected_blob
        assert provenance["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert provenance["files"][name]["bytes"] == len(data)
    # Upstream has mixed file endings. Preserve the original source bytes.
    assert b"\r\n" in (vendor / "utils.py").read_bytes()
    assert b"Copyright (c) 2020 Scott Fujimoto" in (vendor / "LICENSE").read_bytes()


def test_every_embedded_excerpt_is_an_exact_slice_with_pinned_line_links():
    bundle = python_trace_bundle(ROOT)
    assert bundle["revision"] == UPSTREAM_PIN
    assert bundle["auv_reference"]["revision"] == AUV_PIN
    assert bundle["license"].encode() == (ROOT / "third_party/sfujim_td3/LICENSE").read_bytes()
    for key, source in bundle["sources"].items():
        raw = (ROOT / source["local_file"]).read_bytes()
        original_lines = raw.decode("utf-8").splitlines()
        start, end = source["start_line"], source["end_line"]
        assert source["lines"] == original_lines[start - 1:end], key
        assert source["language"] == "python"
        assert source["sha256"] == hashlib.sha256(raw).hexdigest()
        assert source["focus"]
        assert all(start <= line <= end for line in source["focus"])
        if source["origin"] == "td3-reference":
            assert source["file"] in {"TD3.py", "main.py", "utils.py"}
            assert source["url"] == f"https://github.com/sfujim/TD3/blob/{UPSTREAM_PIN}/{source['file']}#L{start}-L{end}"
        else:
            assert source["origin"] == "auv-python"
            assert source["file"] in {"auv_sim/environment.py", "auv_sim/experiment.py"}
            assert source["url"] == f"https://github.com/AWDaley/AUV-practice/blob/{AUV_PIN}/{source['file']}#L{start}-L{end}"


def test_original_td3_blocks_and_auv_specific_blocks_have_correct_provenance():
    sources = python_trace_bundle(ROOT)["sources"]
    for key in ("actor", "target_actions", "bellman_target", "critic", "actor_update",
                "actor_skip", "soft_targets", "replay_add", "replay_sample", "exploration"):
        assert sources[key]["origin"] == "td3-reference"
    for key in ("observation", "observation_features", "environment", "physics", "reset",
                "episode", "learning_wait", "transition_end", "baseline", "random"):
        assert sources[key]["origin"] == "auv-python"
    assert sources["actor"]["start_line"] == 99
    assert sources["bellman_target"]["end_line"] == 123
    assert "torch.min(target_Q1, target_Q2)" in "\n".join(sources["bellman_target"]["lines"])
    assert "self.critic.Q1(state, self.actor(state)).mean()" in "\n".join(sources["actor_update"]["lines"])
    assert sources["replay_sample"]["start_line"] == 107  # Counter increment at 105 is not this stage.
    assert sources["transition_end"]["kind"] == "context"
    assert "no separate Python statement" in sources["transition_end"]["note"]
    assert sources["random_warmup"]["origin"] == "td3-reference"
    assert sources["random_warmup"]["start_line"] == 117
    assert "256" in sources["network_forward"]["note"]
    assert "128" in sources["network_forward"]["note"]
    assert "64" in sources["network_forward"]["note"]


def test_all_executable_walkthrough_source_keys_have_python_references():
    keys = set()
    for name in ("environment", "td3", "worker"):
        content = (ROOT / f"web_src/{name}.js").read_text()
        keys.update(re.findall(r"^\s*// @trace-start ([a-z_]+)\s*$", content, flags=re.MULTILINE))
    assert len(keys) >= 20
    assert keys <= python_trace_bundle(ROOT, expected_keys=keys)["sources"].keys()
    with pytest.raises(ValueError, match="mappings missing"):
        python_trace_bundle(ROOT, expected_keys={"unmapped_future_stage"})


def _copy_reference_inputs(destination):
    shutil.copytree(ROOT / "third_party", destination / "third_party")
    (destination / "auv_sim").mkdir()
    for name in ("environment.py", "experiment.py"):
        shutil.copyfile(ROOT / "auv_sim" / name, destination / "auv_sim" / name)


@pytest.mark.parametrize("filename", ["third_party/sfujim_td3/TD3.py", "auv_sim/environment.py"])
def test_source_changes_fail_closed_before_emitting_a_misleading_reference(tmp_path, filename):
    _copy_reference_inputs(tmp_path)
    source = tmp_path / filename
    source.write_bytes(source.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValueError, match="Pinned Python reference changed"):
        python_trace_bundle(tmp_path)


def test_wrong_provenance_revision_cannot_silently_relabel_the_source(tmp_path):
    _copy_reference_inputs(tmp_path)
    path = tmp_path / "third_party/sfujim_td3/provenance.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "0" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned upstream revision"):
        python_trace_bundle(tmp_path)
