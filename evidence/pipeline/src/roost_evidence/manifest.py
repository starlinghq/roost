"""Profile manifest loading and validation (build plan R-012).

Validation has two layers. The JSON Schema (evidence/schemas/profile-manifest.schema.json) checks shape, patterns
and the maturity rules. `semantic_problems` checks what a schema cannot express: unique names and indices, mesh
arithmetic and cross-field references.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = REPO_ROOT / "evidence" / "schemas" / "profile-manifest.schema.json"
MANIFEST_SUFFIXES = (".json", ".yaml", ".yml")


class DuplicateKeyError(ValueError):
    """A mapping repeats a key. Evidence records must not depend on which duplicate a parser keeps."""


class _Loader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys and keeps dates as strings."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise DuplicateKeyError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep)


_Loader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate key {key!r}")
        out[key] = value
    return out


def load_text(text: str, suffix: str) -> object:
    """Parse manifest text by file suffix. Raises ValueError (or a subclass) on malformed input."""
    if suffix == ".json":
        return json.loads(text, object_pairs_hook=_json_object)
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 - safe loader subclass
    except yaml.YAMLError as error:
        raise ValueError(str(error)) from error


@dataclass(frozen=True)
class Problem:
    path: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.location}: {self.message}"


@cache
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@cache
def _validator() -> Draft202012Validator:
    Draft202012Validator.check_schema(schema())
    return Draft202012Validator(schema())


def _location(parts: Any) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def schema_problems(document: object, path: str = "<memory>") -> list[Problem]:
    errors = sorted(_validator().iter_errors(document), key=lambda e: [str(p) for p in e.absolute_path])
    return [Problem(path, _location(e.absolute_path), e.message) for e in errors]


def _duplicates(values: list[Any]) -> list[Any]:
    return sorted(value for value, n in Counter(values).items() if n > 1)


def semantic_problems(document: dict[str, Any], path: str = "<memory>") -> list[Problem]:
    """Checks beyond the schema. Assumes the document passed `schema_problems`."""
    problems: list[Problem] = []

    def add(location: str, message: str) -> None:
        problems.append(Problem(path, location, message))

    profile = document["profile"]

    for name in _duplicates([g["group"] for g in profile["kv_dtype"]]):
        add("$.profile.kv_dtype", f"state group {name!r} is listed more than once")
    for arch in _duplicates([s["arch"] for s in profile["sm"]]):
        add("$.profile.sm", f"{arch} is listed more than once")

    declared = {s["arch"]: s["status"] for s in profile["sm"]}
    keys = [(v["op"], v["variant"], v["sm"]) for v in profile["selected_variants"]]
    for op, variant, arch in _duplicates(keys):
        add("$.profile.selected_variants", f"({op}, {variant}, {arch}) is selected more than once")
    for index, variant in enumerate(profile["selected_variants"]):
        status = declared.get(variant["sm"])
        where = f"$.profile.selected_variants[{index}].sm"
        if status is None:
            add(where, f"{variant['sm']} is not declared in profile.sm")
        elif status in ("untested", "unsupported"):
            add(where, f"{variant['sm']} is {status}; a variant cannot be selected for it")

    mesh = profile["mesh"]
    ranks = mesh["ranks"]
    product = mesh.get("tp", 1) * mesh.get("pp", 1) * mesh.get("dp", 1)
    if product != ranks:
        add("$.profile.mesh", f"tp*pp*dp is {product} but ranks is {ranks}")
    ep = mesh.get("ep", 1)
    if ranks % ep != 0:
        add("$.profile.mesh.ep", f"ep {ep} does not divide ranks {ranks}")

    layers = profile.get("layer_precision", [])
    for layer in _duplicates([entry["layer"] for entry in layers]):
        add("$.profile.layer_precision", f"layer {layer} is listed more than once")
    count = document["model"].get("num_layers")
    if layers and count is None:
        add("$.model.num_layers", "required when profile.layer_precision is present")
    for index, entry in enumerate(layers):
        if count is not None and entry["layer"] >= count:
            add(f"$.profile.layer_precision[{index}].layer", f"layer {entry['layer']} is outside num_layers {count}")

    return problems


def validate_document(document: object, path: str = "<memory>") -> list[Problem]:
    problems = schema_problems(document, path)
    if problems or not isinstance(document, dict):
        return problems
    return semantic_problems(document, path)


def validate_file(path: Path, *, registry: bool = False) -> list[Problem]:
    """Validate one manifest file. With `registry`, a synthetic manifest is also a problem."""
    name = str(path)
    try:
        document = load_text(path.read_text(encoding="utf-8"), path.suffix.lower())
    except (OSError, UnicodeDecodeError) as error:
        return [Problem(name, "$", f"cannot read: {error}")]
    except ValueError as error:
        return [Problem(name, "$", f"cannot parse: {error}")]
    problems = validate_document(document, name)
    if registry and not problems and document.get("synthetic"):  # type: ignore[union-attr]
        problems.append(Problem(name, "$.synthetic", "synthetic manifests are not allowed in the registry"))
    return problems


def find_manifests(paths: list[Path]) -> Iterator[Path]:
    """Files as given; directories expand to the manifest files below them, sorted."""
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in MANIFEST_SUFFIXES)
        else:
            yield path
