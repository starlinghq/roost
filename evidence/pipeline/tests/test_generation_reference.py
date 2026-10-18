"""Reference file loading and the reference runner's torch-free parts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from roost_evidence import generation_reference, reference_runner
from synthetic import reference_cases, write_reference


def test_a_valid_reference_loads(tmp_path: Path) -> None:
    path = write_reference(tmp_path, reference_cases())
    assert list(generation_reference.load_reference(path)) == ["case-0", "case-1", "case-2"]


def write_lines(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "reference.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def good_row() -> dict:
    return {k: v for k, v in reference_cases()["case-0"].items() if not k.startswith("_")}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"finish_reason": "stop"}, "finish_reason"),
        ({"step_margin": [1.0]}, "one value per output id"),
        ({"prompt_ids": []}, "prompt_ids"),
        ({"id": "a/b"}, "id must be"),
        ({"step0_top": [[1, -3.0], [2, -1.0]]}, "best first"),
        ({"rendered": None}, "rendered"),
    ],
)
def test_malformed_rows_are_rejected(tmp_path: Path, change: dict, message: str) -> None:
    path = write_lines(tmp_path, {**good_row(), **change})
    with pytest.raises(generation_reference.ReferenceFileError, match=message):
        generation_reference.load_reference(path)


def test_a_missing_field_is_named(tmp_path: Path) -> None:
    row = good_row()
    del row["step0_vocab"]
    with pytest.raises(generation_reference.ReferenceFileError, match="step0_vocab"):
        generation_reference.load_reference(write_lines(tmp_path, row))


def test_a_repeated_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(generation_reference.ReferenceFileError, match="more than once"):
        generation_reference.load_reference(write_lines(tmp_path, good_row(), good_row()))


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(generation_reference.ReferenceFileError, match="no cases"):
        generation_reference.load_reference(write_lines(tmp_path))


def test_prompts_need_exactly_one_input_form(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "x", "messages": [], "prompt_ids": [1]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        reference_runner.load_prompts(path)


def test_a_hub_model_without_a_sha_is_refused_before_anything_is_imported(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt_ids": [1]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="40-character"):
        reference_runner.run_reference("org/model", path, tmp_path, revision="main")


def test_a_missing_extra_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt_ids": [1]}) + "\n", encoding="utf-8")
    with pytest.raises(reference_runner.ExtraMissingError, match="reference"):
        reference_runner.run_reference(str(tmp_path), path, tmp_path)
