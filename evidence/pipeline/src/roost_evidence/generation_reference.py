"""Reference side of the generation evaluator: the jsonl the fp32 CPU reference runner writes (build plan R-010).

Each line is one case of a reference run:

    id, rendered, prompt_ids, output_ids (with the stop token when finish_reason is eos), finish_reason (eos or
    length), step_margin (reference top-1 minus top-2 log-probability at every generated step), step0_top (the best
    [token, log-probability] pairs of the first generated step, best first), step0_vocab, and optionally step0_sha256,
    the digest of the full step-0 vector stored beside the file as `step0/<id>.f32`.

Roost defines this format itself; the evaluator reads it as the protected fixture once the maintainer admits one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roost_evidence import stage_results

REQUIRED = ("id", "rendered", "prompt_ids", "output_ids", "finish_reason", "step_margin", "step0_top", "step0_vocab")
FINISH_REASONS = ("eos", "length")
CASE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class ReferenceFileError(ValueError):
    """The reference file is unreadable or malformed."""


def _is_ids(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)


def _case_problems(case: Any) -> list[str]:
    if not isinstance(case, dict):
        return ["not an object"]
    problems = [f"lacks {key!r}" for key in REQUIRED if key not in case]
    if problems:
        return problems
    name = case["id"]
    if not isinstance(name, str) or not name or not set(name) <= CASE_ID_CHARS:
        problems.append("id must be a non-empty string of letters, digits, '.', '_' or '-'")
    if not isinstance(case["rendered"], str):
        problems.append("rendered must be a string")
    if not _is_ids(case["prompt_ids"]) or not case["prompt_ids"]:
        problems.append("prompt_ids must be a non-empty list of token ids")
    if not _is_ids(case["output_ids"]):
        problems.append("output_ids must be a list of token ids")
    if case["finish_reason"] not in FINISH_REASONS:
        problems.append(f"finish_reason must be one of {FINISH_REASONS}")
    margins = case["step_margin"]
    if not isinstance(margins, list) or not all(isinstance(m, int | float) and math.isfinite(m) for m in margins):
        problems.append("step_margin must be a list of finite numbers")
    elif _is_ids(case["output_ids"]) and len(margins) != len(case["output_ids"]):
        problems.append("step_margin needs one value per output id")
    top = case["step0_top"]
    if not (
        isinstance(top, list)
        and top
        and all(
            isinstance(p, list) and len(p) == 2 and isinstance(p[0], int) and isinstance(p[1], int | float) for p in top
        )
    ):
        problems.append("step0_top must be a non-empty list of [token, log-probability] pairs")
    elif any(not math.isfinite(p[1]) for p in top):
        problems.append("step0_top log-probabilities must be finite")
    elif any(a[1] < b[1] for a, b in zip(top, top[1:], strict=False)):
        problems.append("step0_top must be listed best first")
    vocab = case["step0_vocab"]
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab < 1:
        problems.append("step0_vocab must be a positive integer")
    digest = case.get("step0_sha256")
    if digest is not None and not (isinstance(digest, str) and len(digest) == 64):
        problems.append("step0_sha256 must be a sha256 hex digest")
    return problems


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    """The cases of a reference jsonl by id. Raises ReferenceFileError on any malformed line or repeated id."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReferenceFileError(f"{path}: cannot read: {error}") from error
    cases: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReferenceFileError(f"{path}:{number}: {error}") from error
        problems = _case_problems(case)
        if problems:
            raise ReferenceFileError(f"{path}:{number}: " + "; ".join(problems))
        if case["id"] in cases:
            raise ReferenceFileError(f"{path}:{number}: case {case['id']!r} is listed more than once")
        cases[case["id"]] = case
    if not cases:
        raise ReferenceFileError(f"{path}: no cases")
    return cases


@dataclass
class ReferenceVectors:
    """Full step-0 vectors of the reference by case id, each checked against `step0_sha256` and `step0_vocab`."""

    vectors: dict[str, list[float]]


def load_reference_vectors(directory: Path, cases: dict[str, dict[str, Any]]) -> ReferenceVectors:
    vectors: dict[str, list[float]] = {}
    for name, case in cases.items():
        path = directory / "step0" / f"{name}.f32"
        if not path.is_file():
            continue
        digest = case.get("step0_sha256")
        if digest is not None and stage_results.sha256_file(path) != digest:
            raise ReferenceFileError(f"{path}: sha256 does not match step0_sha256 of case {name!r}")
        values = stage_results.read_float32(path.read_bytes())
        if len(values) != case["step0_vocab"]:
            raise ReferenceFileError(f"{path}: {len(values)} values, step0_vocab is {case['step0_vocab']}")
        vectors[name] = values
    return ReferenceVectors(vectors)
