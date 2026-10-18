"""Per-layer capture files (build plan R-011): one safetensors file per case, read here without torch.

A capture holds what the reference (transformers fp32 CPU hooks) or a candidate produced for one case, teacher-forced on
the same token sequence: the prompt, then every generated token except the last (a stop token or the token that ended
the budget is never fed back). A *forward* is one model call: forward 0 is the prefill over the prompt, forward n >= 1
feeds the n-th generated token. `forwards` is therefore the number of generated tokens.

Tensors, all little-endian float32:

    forward{n}/embeddings                  [positions, hidden]  output of the token embedding (optional)
    forward{n}/layer.{i}                   [positions, hidden]  output of decoder layer i, before the final norm
    forward{n}/final_norm                  [positions, hidden]  output of the final norm (optional)
    forward{n}/layer.{i}/recurrent_state   [heads, key_dim, value_dim]  Gated DeltaNet state after forward n
    forward{n}/layer.{i}/conv_state        [channels, kernel]           Gated DeltaNet convolution state after forward n

`positions` is the prompt length for forward 0 and 1 afterwards. Recurrent and convolution state exist only for layers
whose type is `linear_attention`, and only for the first `state_forwards` forwards (the states are large: a Qwen3.5-0.8B
layer holds 262144 floats, 18 layers per forward). A recurrent layer's state lives in the model's cache object, not in
what a forward hook returns, so a reference run reads it from the cache after each forward and copies it.

The file's `__metadata__` (string values, JSON where structured) carries `format` (`roost.layer-capture`),
`format_version`, `layer_types`, `hidden_size`, `stages` (the hidden-state tensor names in model order), `prompt_ids`,
`decode_ids`, `forwards` and `state_forwards`. `parse_capture` checks all of it; the evaluator compares two captures
only when their metadata agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "roost.layer-capture"
FORMAT_VERSION = 1
LAYERS_DIR = "layers"
LINEAR = "linear_attention"
EMBEDDINGS = "embeddings"
FINAL_NORM = "final_norm"
MAX_HEADER = 64 * 1024 * 1024
METADATA_KEYS = (
    "format",
    "format_version",
    "layer_types",
    "hidden_size",
    "stages",
    "prompt_ids",
    "decode_ids",
    "forwards",
    "state_forwards",
)


class CaptureError(ValueError):
    """A capture file is unreadable or malformed."""


def layer_stage(index: int) -> str:
    return f"layer.{index}"


def hidden_name(forward: int, stage: str) -> str:
    return f"forward{forward}/{stage}"


def state_name(forward: int, layer: int, kind: str) -> str:
    """`kind` is `recurrent_state` or `conv_state`."""
    return f"forward{forward}/layer.{layer}/{kind}"


def capture_path(directory: Path, case_id: str) -> Path:
    return directory / LAYERS_DIR / f"{case_id}.safetensors"


def metadata_for(
    *,
    layer_types: list[str],
    hidden_size: int,
    stages: list[str],
    prompt_ids: list[int],
    decode_ids: list[int],
    state_forwards: int,
) -> dict[str, str]:
    """The metadata block of a capture; `forwards` follows from the decode ids."""
    return {
        "format": FORMAT,
        "format_version": str(FORMAT_VERSION),
        "layer_types": json.dumps(layer_types),
        "hidden_size": str(hidden_size),
        "stages": json.dumps(stages),
        "prompt_ids": json.dumps(prompt_ids),
        "decode_ids": json.dumps(decode_ids),
        "forwards": str(len(decode_ids) + 1),
        "state_forwards": str(state_forwards),
    }


@dataclass
class Capture:
    """A parsed capture. Tensor values are decoded on demand from the file's bytes."""

    path: Path
    sha256: str
    metadata: dict[str, str]
    shapes: dict[str, tuple[int, ...]]
    layer_types: list[str]
    hidden_size: int
    stages: list[str]
    prompt_ids: list[int]
    decode_ids: list[int]
    forwards: int
    state_forwards: int
    _data: bytes = field(repr=False, default=b"")
    _ranges: dict[str, tuple[int, int]] = field(repr=False, default_factory=dict)

    @property
    def linear_layers(self) -> list[int]:
        return [i for i, kind in enumerate(self.layer_types) if kind == LINEAR]

    def values(self, name: str) -> array:
        start, end = self._ranges[name]
        values = array("f")
        values.frombytes(self._data[start:end])
        if sys.byteorder == "big":
            values.byteswap()
        return values


def _ints(text: str, what: str, where: Path) -> list[int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise CaptureError(f"{where}: metadata {what} is not JSON: {error}") from error
    if not (isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)):
        raise CaptureError(f"{where}: metadata {what} must be a list of token ids")
    return value


def _count(metadata: dict[str, str], key: str, where: Path, *, minimum: int) -> int:
    text = metadata[key]
    if not (text.isascii() and text.isdigit()) or int(text) < minimum:
        raise CaptureError(f"{where}: metadata {key} must be an integer of at least {minimum}")
    return int(text)


def parse_capture(path: Path) -> Capture:
    """Read and verify one capture file: container layout, metadata, and the tensors the metadata promises."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CaptureError(f"{path}: cannot read: {error}") from error
    if len(data) < 8:
        raise CaptureError(f"{path}: shorter than a safetensors header")
    (size,) = struct.unpack("<Q", data[:8])
    if size > MAX_HEADER or 8 + size > len(data):
        raise CaptureError(f"{path}: header length {size} does not fit the file")
    try:
        header = json.loads(data[8 : 8 + size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"{path}: header is not JSON: {error}") from error
    if not isinstance(header, dict):
        raise CaptureError(f"{path}: header is not an object")
    metadata = header.pop("__metadata__", None)
    if not (isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values())):
        raise CaptureError(f"{path}: __metadata__ must be an object of strings")
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise CaptureError(f"{path}: metadata lacks {missing}")
    if metadata["format"] != FORMAT or metadata["format_version"] != str(FORMAT_VERSION):
        raise CaptureError(f"{path}: not a {FORMAT} version {FORMAT_VERSION} file")

    base = 8 + size
    body = len(data) - base
    shapes: dict[str, tuple[int, ...]] = {}
    ranges: dict[str, tuple[int, int]] = {}
    spans: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if not (isinstance(entry, dict) and entry.get("dtype") == "F32"):
            raise CaptureError(f"{path}: tensor {name!r} is not an F32 tensor")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if not (
            isinstance(shape, list) and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
        ):
            raise CaptureError(f"{path}: tensor {name!r} has a malformed shape")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(o, int) for o in offsets)):
            raise CaptureError(f"{path}: tensor {name!r} has malformed data_offsets")
        begin, end = offsets
        if not 0 <= begin <= end <= body or end - begin != 4 * math.prod(shape):
            raise CaptureError(f"{path}: tensor {name!r} offsets {offsets} do not match shape {shape}")
        shapes[name] = tuple(shape)
        ranges[name] = (base + begin, base + end)
        spans.append((begin, end, name))
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            raise CaptureError(f"{path}: tensor {name!r} leaves a gap or overlaps before offset {begin}")
        cursor = end
    if cursor != body:
        raise CaptureError(f"{path}: {body - cursor} bytes of data belong to no tensor")

    try:
        layer_types = json.loads(metadata["layer_types"])
        stages = json.loads(metadata["stages"])
    except json.JSONDecodeError as error:
        raise CaptureError(f"{path}: metadata is not JSON: {error}") from error
    if not (isinstance(layer_types, list) and layer_types and all(isinstance(t, str) for t in layer_types)):
        raise CaptureError(f"{path}: metadata layer_types must be a non-empty list of strings")
    if not (isinstance(stages, list) and all(isinstance(s, str) for s in stages) and len(set(stages)) == len(stages)):
        raise CaptureError(f"{path}: metadata stages must be a list of distinct names")
    layers = [layer_stage(i) for i in range(len(layer_types))]
    if [s for s in stages if s not in (EMBEDDINGS, FINAL_NORM)] != layers or (
        EMBEDDINGS in stages and stages[0] != EMBEDDINGS
    ):
        raise CaptureError(
            f"{path}: stages must list embeddings (optional), every layer in order, final_norm (optional)"
        )
    if FINAL_NORM in stages and stages[-1] != FINAL_NORM:
        raise CaptureError(f"{path}: final_norm must be the last stage")
    prompt_ids = _ints(metadata["prompt_ids"], "prompt_ids", path)
    decode_ids = _ints(metadata["decode_ids"], "decode_ids", path)
    hidden = _count(metadata, "hidden_size", path, minimum=1)
    forwards = _count(metadata, "forwards", path, minimum=1)
    state_forwards = _count(metadata, "state_forwards", path, minimum=0)
    if not prompt_ids:
        raise CaptureError(f"{path}: prompt_ids is empty")
    if forwards != len(decode_ids) + 1:
        raise CaptureError(f"{path}: forwards is {forwards} but decode_ids has {len(decode_ids)} ids")
    if state_forwards > forwards:
        raise CaptureError(f"{path}: state_forwards {state_forwards} exceeds forwards {forwards}")

    expected: dict[str, tuple[int, ...]] = {}
    for forward in range(forwards):
        rows = len(prompt_ids) if forward == 0 else 1
        for stage in stages:
            expected[hidden_name(forward, stage)] = (rows, hidden)
    declared_states = {name for name in shapes if name.count("/") == 2}
    for name in declared_states:
        forward_text, layer_text, kind = name.split("/")
        ok = forward_text.startswith("forward") and forward_text[7:].isdigit() and layer_text.startswith("layer.")
        ok = ok and layer_text[6:].isdigit() and kind in ("recurrent_state", "conv_state")
        if not ok:
            raise CaptureError(f"{path}: unknown tensor {name!r}")
        forward, layer = int(forward_text[7:]), int(layer_text[6:])
        if forward >= state_forwards or layer >= len(layer_types) or layer_types[layer] != LINEAR:
            raise CaptureError(f"{path}: tensor {name!r} is outside the states the metadata declares")
    for name, shape in shapes.items():
        if name in declared_states:
            continue
        if name not in expected:
            raise CaptureError(f"{path}: unknown tensor {name!r}")
        if shape != expected[name]:
            raise CaptureError(f"{path}: tensor {name!r} has shape {list(shape)}, expected {list(expected[name])}")
    absent = [name for name in expected if name not in shapes]
    if absent:
        raise CaptureError(f"{path}: lacks hidden-state tensors {absent[:4]}")
    return Capture(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        metadata=metadata,
        shapes=shapes,
        layer_types=layer_types,
        hidden_size=hidden,
        stages=stages,
        prompt_ids=prompt_ids,
        decode_ids=decode_ids,
        forwards=forwards,
        state_forwards=state_forwards,
        _data=data,
        _ranges=ranges,
    )


def write_capture(path: Path, metadata: dict[str, str], tensors: dict[str, tuple[tuple[int, ...], list[float]]]) -> str:
    """Write a capture without torch (tests, and candidate producers in Python); return its sha256.

    `tensors` maps a name to (shape, values). The layout is safetensors': a little-endian u64 header length, a JSON
    header padded with spaces to a multiple of 8 bytes, then the tensor bytes in name order.
    """
    header: dict[str, Any] = {"__metadata__": metadata}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        shape, values = tensors[name]
        if len(values) != math.prod(shape):
            raise ValueError(f"tensor {name!r}: {len(values)} values for shape {list(shape)}")
        packed = array("f", values)
        if sys.byteorder == "big":
            packed.byteswap()
        blob = packed.tobytes()
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    data = struct.pack("<Q", len(text)) + text + b"".join(blobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def canonicalize_header(path: Path) -> str:
    """Rewrite the JSON header of a file another writer produced in a fixed key order; return the file's sha256.

    The safetensors library writes its metadata in hash-map order, so two runs of the same capture differ in bytes
    although every tensor is equal. Sorting the keys makes the digest a function of the content alone. The tensor
    bytes and their offsets (which count from the end of the header) are untouched.
    """
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size].decode("utf-8"))
    metadata = header.pop("__metadata__", {})
    ordered: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    ordered.update(sorted(header.items()))
    text = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    out = struct.pack("<Q", len(text)) + text + data[8 + size :]
    path.write_bytes(out)
    return hashlib.sha256(out).hexdigest()
