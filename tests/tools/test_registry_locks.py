"""Shape checks for evidence/registry admission records and asset locks (R-013)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REG = Path(__file__).resolve().parents[2] / "evidence" / "registry"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load(name: str) -> dict:
    return tomllib.loads((REG / name).read_text(encoding="utf-8"))


def test_qwen35_lock_files_and_digests():
    lock = load("qwen3.5-0.8b.lock.toml")
    assert SHA1.match(lock["source"]["revision"])
    files = lock["files"]
    assert len(files) == 13
    paths = [f["path"] for f in files]
    assert len(set(paths)) == len(paths)
    assert "model.safetensors-00001-of-00001.safetensors" in paths
    assert "chat_template.jinja" in paths
    for f in files:
        assert f["size"] > 0
        assert SHA256.match(f["sha256"]), f["path"]


def test_qwen35_admission_agrees_with_lock():
    adm = load("qwen3.5-0.8b.admission.toml")
    lock = load("qwen3.5-0.8b.lock.toml")
    assert adm["source"]["revision"] == lock["source"]["revision"]
    assert adm["source"]["repo"] == lock["source"]["repo"]
    assert (REG.parents[1] / adm["lock"]).is_file()
    by_path = {f["path"]: f for f in lock["files"]}
    assert adm["source"]["weights"] in by_path
    ct = adm["comparison"]["chat_template"]
    assert ct["hub_sha256"] == by_path["chat_template.jinja"]["sha256"]
    assert ct["hub_sha256"] == ct["starling_published_sha256"]
    tj = adm["comparison"]["tokenizer_json"]
    assert tj["hub_sha256"] == by_path["tokenizer.json"]["sha256"]
    assert tj["hub_size"] == by_path["tokenizer.json"]["size"]
    arch = adm["architecture"]
    assert arch["full_attention_layers"] + arch["linear_attention_layers"] == arch["decoder_layers"]
    assert len(arch["full_attention_layer_indices"]) == arch["full_attention_layers"]
