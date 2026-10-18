"""Stage results, format version 3, task generation (build plan R-010).

A run directory holds `template.results.json` and `dotnet.results.json`, plus the optional float32 sidecar the dotnet
file names under `tensors`. `validate_document` checks a document against evidence/schemas/stage-results.schema.json
and the rules the schema cannot express; `load_run` reads and verifies a whole directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from roost_evidence.manifest import Problem, load_text

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = REPO_ROOT / "evidence" / "schemas" / "stage-results.schema.json"
STAGES = ("template", "dotnet")
STEP0_TENSOR = "step0_logprobs"
FORBIDDEN_KEYS = frozenset({"tolerance", "tolerances", "threshold", "thresholds"})


class RunError(ValueError):
    """A run directory cannot be evaluated: unreadable, invalid or inconsistent input."""


def stage_file_name(stage: str) -> str:
    return f"{stage}.results.json"


@cache
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@cache
def _validator() -> Draft202012Validator:
    Draft202012Validator.check_schema(schema())
    return Draft202012Validator(schema())


def _where(parts: Any) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _forbidden_keys(node: Any, where: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_KEYS:
                found.append(f"{where}.{key}")
            found.extend(_forbidden_keys(value, f"{where}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_forbidden_keys(value, f"{where}[{index}]"))
    return found


def validate_document(document: object, path: str = "<memory>") -> list[Problem]:
    """Schema problems first; the semantic pass runs only on a document that passed the schema."""
    errors = sorted(_validator().iter_errors(document), key=lambda e: [str(p) for p in e.absolute_path])
    problems = [Problem(path, _where(e.absolute_path), e.message) for e in errors]
    if problems or not isinstance(document, dict):
        return problems
    for where in _forbidden_keys(document):
        problems.append(Problem(path, where, "tolerances belong in the protected tolerance file, not in results"))
    spec = document.get("tensors", {}).get(STEP0_TENSOR)
    if spec is not None:
        ids = [entry["id"] for entry in spec["cases"]]
        if len(set(ids)) != len(ids):
            problems.append(Problem(path, f"$.tensors.{STEP0_TENSOR}.cases", "a case is listed more than once"))
        for entry in spec["cases"]:
            if entry["id"] not in document["cases"]:
                problems.append(
                    Problem(path, f"$.tensors.{STEP0_TENSOR}.cases", f"tensor case {entry['id']!r} is not a case")
                )
    for name, case in document["cases"].items():
        if document["stage"] != "dotnet":
            break
        problems.extend(_dotnet_case_problems(path, name, case))
    return problems


def _dotnet_case_problems(path: str, name: str, case: dict[str, Any]) -> list[Problem]:
    problems: list[Problem] = []
    where = f"$.cases.{name}"
    top = case["step0_top"]
    values = [pair[1] for pair in top]
    # Non-finite values are not rejected here: the evaluator's nan_or_inf check reports them as a failed gate.
    if any(a < b for a, b in zip(values, values[1:], strict=False)):
        problems.append(Problem(path, f"{where}.step0_top", "pairs must be listed best first"))
    if len({pair[0] for pair in top}) != len(top):
        problems.append(Problem(path, f"{where}.step0_top", "a token is listed more than once"))
    return problems


# Sidecar -------------------------------------------------------------------------------------------------------


def read_float32(data: bytes) -> list[float]:
    """Little-endian float32 values, as every sidecar is written."""
    values = array("f")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return list(values)


def float32_bytes(values: list[float]) -> bytes:
    packed = array("f", values)
    if sys.byteorder == "big":
        packed.byteswap()
    return packed.tobytes()


def write_float32(path: Path, values: list[float]) -> None:
    path.write_bytes(float32_bytes(values))


def write_sidecar(directory: Path, file: str, vectors: dict[str, list[float]]) -> dict[str, Any]:
    """Write `vectors` (case id to step-0 log-probabilities) and return the `tensors.step0_logprobs` entry."""
    blob = bytearray()
    cases = []
    for case_id, vector in vectors.items():
        blob += float32_bytes(vector)
        cases.append({"id": case_id, "shape": [len(vector)]})
    (directory / file).write_bytes(bytes(blob))
    return {"file": file, "dtype": "float32", "sha256": hashlib.sha256(bytes(blob)).hexdigest(), "cases": cases}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sidecar(directory: Path, spec: dict[str, Any]) -> dict[str, list[float]]:
    """The vectors a `tensors.step0_logprobs` entry names, after checking digest and size."""
    path = directory / spec["file"]
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RunError(f"{path}: cannot read: {error}") from error
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise RunError(f"{path}: sha256 does not match the digest in the results file")
    expected = sum(entry["shape"][0] for entry in spec["cases"]) * 4
    if len(data) != expected:
        raise RunError(f"{path}: {len(data)} bytes, but the shapes name {expected}")
    values = read_float32(data)
    out: dict[str, list[float]] = {}
    offset = 0
    for entry in spec["cases"]:
        size = entry["shape"][0]
        out[entry["id"]] = values[offset : offset + size]
        offset += size
    return out


# Run directory -------------------------------------------------------------------------------------------------


@dataclass
class StageFile:
    path: Path
    document: dict[str, Any]
    sha256: str

    @property
    def cases(self) -> dict[str, dict[str, Any]]:
        return self.document["cases"]


@dataclass
class Run:
    """A verified run directory: both stage files and the step-0 vectors (empty without a sidecar)."""

    directory: Path
    profile: str
    template: StageFile
    dotnet: StageFile
    step0: dict[str, list[float]] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def load_stage_file(path: Path, stage: str) -> StageFile:
    try:
        text = path.read_text(encoding="utf-8")
        document = load_text(text, ".json")
    except (OSError, UnicodeDecodeError) as error:
        raise RunError(f"{path}: cannot read: {error}") from error
    except ValueError as error:
        raise RunError(f"{path}: cannot parse: {error}") from error
    problems = validate_document(document, str(path))
    if problems:
        raise RunError("\n".join(str(p) for p in problems))
    if document["stage"] != stage:
        raise RunError(f"{path}: stage is {document['stage']!r}, expected {stage!r}")
    return StageFile(path, document, sha256_file(path))


def load_run(directory: Path) -> Run:
    template = load_stage_file(directory / stage_file_name("template"), "template")
    dotnet = load_stage_file(directory / stage_file_name("dotnet"), "dotnet")
    profiles = {template.document["profile"], dotnet.document["profile"]}
    if len(profiles) != 1:
        raise RunError(f"{directory}: the stage files name different profiles {sorted(profiles)}")
    run = Run(directory, profiles.pop(), template, dotnet)
    run.files = {template.path.name: template.sha256, dotnet.path.name: dotnet.sha256}
    spec = dotnet.document.get("tensors", {}).get(STEP0_TENSOR)
    if spec is not None:
        run.step0 = read_sidecar(directory, spec)
        run.files[spec["file"]] = spec["sha256"]
    return run


def write_stage_file(directory: Path, document: dict[str, Any]) -> Path:
    path = directory / stage_file_name(document["stage"])
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
