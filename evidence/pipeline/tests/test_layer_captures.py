"""The capture file reader and writer, which need no torch."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from roost_evidence import layer_captures
from synthetic import reference_cases
from synthetic_layers import HIDDEN, LAYER_TYPES, make_capture


def written(tmp_path: Path) -> Path:
    case = reference_cases(1)["case-0"]
    metadata, tensors = make_capture(case, seed=1)
    path = tmp_path / "x.safetensors"
    layer_captures.write_capture(path, metadata, tensors)
    return path


def rewrite_header(path: Path, edit) -> Path:
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    edit(header)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    out = path.with_name("edited.safetensors")
    out.write_bytes(struct.pack("<Q", len(text)) + text + data[8 + size :])
    return out


def test_a_written_capture_parses(tmp_path: Path) -> None:
    case = reference_cases(1)["case-0"]
    capture = layer_captures.parse_capture(written(tmp_path))
    assert capture.layer_types == LAYER_TYPES
    assert capture.linear_layers == [0, 1, 2]
    assert capture.forwards == len(case["output_ids"])
    assert capture.hidden_size == HIDDEN
    assert capture.shapes["forward0/layer.1"] == (len(case["prompt_ids"]), HIDDEN)
    assert capture.shapes["forward1/layer.1"] == (1, HIDDEN)
    assert "forward0/layer.2/recurrent_state" in capture.shapes
    assert "forward0/layer.3/recurrent_state" not in capture.shapes
    assert "forward2/layer.0/recurrent_state" not in capture.shapes
    assert len(capture.values("forward1/layer.1")) == HIDDEN


def test_malformed_files_are_rejected(tmp_path: Path) -> None:
    good = written(tmp_path)
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01")
    with pytest.raises(layer_captures.CaptureError, match="shorter"):
        layer_captures.parse_capture(short)
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(good.read_bytes()[:-4])
    with pytest.raises(layer_captures.CaptureError, match="offsets|bytes"):
        layer_captures.parse_capture(truncated)
    with pytest.raises(layer_captures.CaptureError, match="cannot read"):
        layer_captures.parse_capture(tmp_path / "missing.safetensors")


def test_header_edits_are_rejected(tmp_path: Path) -> None:
    good = written(tmp_path)

    def dtype(header: dict) -> None:
        header["forward0/layer.0"]["dtype"] = "F16"

    def wrong_shape(header: dict) -> None:
        header["forward0/layer.0"]["shape"] = [1, HIDDEN]

    def stranger(header: dict) -> None:
        header["forward0/other"] = header.pop("forward0/layer.0")

    def no_metadata(header: dict) -> None:
        del header["__metadata__"]["layer_types"]

    def forwards(header: dict) -> None:
        header["__metadata__"]["forwards"] = "99"

    def format_name(header: dict) -> None:
        header["__metadata__"]["format"] = "other"

    def attention_state(header: dict) -> None:
        header["forward0/layer.3/conv_state"] = header.pop("forward0/layer.0/conv_state")

    def late_state(header: dict) -> None:
        header["forward5/layer.0/conv_state"] = header.pop("forward0/layer.0/conv_state")

    for edit, message in (
        (dtype, "F32"),
        (wrong_shape, "offsets|shape"),
        (stranger, "unknown tensor|lacks"),
        (no_metadata, "lacks"),
        (forwards, "forwards"),
        (format_name, "not a"),
        (attention_state, "outside"),
        (late_state, "outside"),
    ):
        with pytest.raises(layer_captures.CaptureError, match=message):
            layer_captures.parse_capture(rewrite_header(good, edit))


def test_overlapping_tensors_are_rejected(tmp_path: Path) -> None:
    good = written(tmp_path)

    def overlap(header: dict) -> None:
        header["forward0/layer.1"]["data_offsets"] = header["forward0/layer.0"]["data_offsets"]
        header["forward0/layer.1"]["shape"] = header["forward0/layer.0"]["shape"]

    with pytest.raises(layer_captures.CaptureError, match="gap|overlap"):
        layer_captures.parse_capture(rewrite_header(good, overlap))


def test_canonicalizing_a_header_makes_the_bytes_independent_of_key_order(tmp_path: Path) -> None:
    good = written(tmp_path)
    canonical = layer_captures.canonicalize_header(good)

    def reverse(header: dict) -> None:
        items = list(header.items())[::-1]
        header.clear()
        header.update(items)
        header["__metadata__"] = dict(list(header["__metadata__"].items())[::-1])

    shuffled = rewrite_header(good, reverse)
    assert shuffled.read_bytes() != good.read_bytes()
    assert layer_captures.canonicalize_header(shuffled) == canonical
    assert shuffled.read_bytes() == good.read_bytes()
    assert layer_captures.parse_capture(shuffled).sha256 == canonical
