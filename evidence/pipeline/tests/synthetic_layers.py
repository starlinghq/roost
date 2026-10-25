"""Synthetic per-layer captures built without torch: three recurrent layers and one attention layer."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from roost_evidence import layer_captures
from synthetic import reference_cases, write_reference, write_tolerances

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
HIDDEN = 8
HEADS, KEY, VALUE, CONV_CHANNELS, KERNEL = 2, 4, 4, 12, 4
STAGES = [layer_captures.EMBEDDINGS, *[layer_captures.layer_stage(i) for i in range(4)], layer_captures.FINAL_NORM]
LAYER_TOLERANCES = {
    "float.layer_hidden_max_abs": 1e-4,
    "float.recurrent_state_max_abs": 1e-4,
    "float.conv_state_max_abs": 1e-4,
}

Tensors = dict[str, tuple[tuple[int, ...], list[float]]]


def make_capture(
    case: dict[str, Any], seed: int, state_forwards: int = 2, scales: dict[str, float] | None = None
) -> tuple[dict[str, str], Tensors]:
    """Captures of random values; `scales` multiplies the hidden values of the named stages (default 1)."""
    scales = scales or {}
    rng = random.Random(seed)
    prompt, decode = case["prompt_ids"], case["output_ids"][:-1]
    metadata = layer_captures.metadata_for(
        layer_types=LAYER_TYPES,
        hidden_size=HIDDEN,
        stages=STAGES,
        prompt_ids=prompt,
        decode_ids=decode,
        state_forwards=state_forwards,
    )
    tensors: Tensors = {}
    for forward in range(len(decode) + 1):
        rows = len(prompt) if forward == 0 else 1
        for stage in STAGES:
            tensors[layer_captures.hidden_name(forward, stage)] = (
                (rows, HIDDEN),
                [rng.uniform(-2, 2) * scales.get(stage, 1.0) for _ in range(rows * HIDDEN)],
            )
        if forward < state_forwards:
            for layer, kind in enumerate(LAYER_TYPES):
                if kind == layer_captures.LINEAR:
                    shapes = {"recurrent_state": (HEADS, KEY, VALUE), "conv_state": (CONV_CHANNELS, KERNEL)}
                    for name, shape in shapes.items():
                        size = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
                        tensors[layer_captures.state_name(forward, layer, name)] = (
                            shape,
                            [rng.uniform(-1, 1) for _ in range(size)],
                        )
    return metadata, tensors


def write_captures(
    directory: Path, cases: dict[str, dict[str, Any]], tensors: dict[str, Tensors], metadata: dict[str, dict[str, str]]
) -> dict[str, str]:
    return {
        name: layer_captures.write_capture(layer_captures.capture_path(directory, name), metadata[name], tensors[name])
        for name in cases
    }


class Pair:
    """A reference run directory and a candidate run directory with identical captures, and the means to perturb."""

    def __init__(self, tmp_path: Path, count: int = 2, scales: dict[str, float] | None = None) -> None:
        self.scales = scales or {}
        self.cases = reference_cases(count)
        self.reference_dir = tmp_path / "reference"
        self.candidate_dir = tmp_path / "candidate"
        self.metadata: dict[str, dict[str, str]] = {}
        self.tensors: dict[str, Tensors] = {}
        for index, (name, case) in enumerate(self.cases.items()):
            self.metadata[name], self.tensors[name] = make_capture(case, seed=100 + index, scales=self.scales)
        digests = write_captures(self.reference_dir, self.cases, self.tensors, self.metadata)
        for name, case in self.cases.items():
            case["layers_sha256"] = digests[name]
        self.reference = write_reference(self.reference_dir, self.cases)
        self.tolerances = write_tolerances(tmp_path / "limits.json", None)
        self.write_candidate()

    def tolerance_file(self, tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
        import json

        document = json.loads(self.tolerances.read_text(encoding="utf-8"))
        document["float"].update({k.removeprefix("float."): v for k, v in (extra or LAYER_TOLERANCES).items()})
        path = tmp_path / "layer-limits.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def write_candidate(self, tensors: dict[str, Tensors] | None = None, only: list[str] | None = None) -> None:
        tensors = tensors if tensors is not None else self.tensors
        for name in only if only is not None else list(self.cases):
            layer_captures.write_capture(
                layer_captures.capture_path(self.candidate_dir, name), self.metadata[name], tensors[name]
            )

    def perturbed(self, case: str, key: str, delta: float, positions: int | None = None) -> dict[str, Tensors]:
        """Tensors with `delta` added to the first value of every row of `key` (or to one row's first value)."""
        out = copy.deepcopy(self.tensors)
        shape, values = out[case][key]
        width = shape[-1]
        rows = len(values) // width
        for row in range(rows) if positions is None else [positions]:
            values[row * width] += delta
        return out
