#!/usr/bin/env python3
"""Validate kernels/manifest.toml against the C5 schema (tools/kernel-manifest.schema.json) and the C1 header.

    python3 -I tools/validate-kernel-manifest.py [MANIFEST]     # default kernels/manifest.toml; exit 1 on errors

Standard library only, so it runs wherever the CI tools job runs. It implements the JSON Schema keywords the C5 schema
uses (type, const, enum, required, properties, additionalProperties, items, minItems, uniqueItems, pattern, minLength,
minimum, maximum, local $ref) and then checks what a schema cannot:

  - every op name is a C3 op (a ROOST_OP_* id in kernels/include/roost_kernels.h);
  - variant ids are unique within the manifest (conflicts between libraries resolve by variant id, C1);
  - every binary and constraint arch is in the SM list, and every variant source is a declared upstream, external
    library or "roost";
  - an upstream's ops are ops of the manifest, and a variant's triton_kernels are declared under triton_aot;
  - abi.major equals ROOST_ABI_MAJOR and abi.minor is at most ROOST_ABI_MINOR.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "tools" / "kernel-manifest.schema.json"
HEADER = ROOT / "kernels" / "include" / "roost_kernels.h"
DEFAULT_MANIFEST = ROOT / "kernels" / "manifest.toml"

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, TYPES[name])


def resolve(root: dict, ref: str) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate(value: Any, schema: dict, root: dict, path: str = "$") -> list[str]:
    """Errors for value against schema, each prefixed with a JSON path."""
    errors: list[str] = []
    if "$ref" in schema:
        errors += validate(value, resolve(root, schema["$ref"]), root, path)
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(is_type(value, n) for n in names):
            return errors + [f"{path}: expected {' or '.join(names)}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                errors += validate(item, properties[key], root, f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}: unknown key {key!r}")
            elif isinstance(extra, dict):
                errors += validate(item, extra, root, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in value]
            if len(seen) != len(set(seen)):
                errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                errors += validate(item, schema["items"], root, f"{path}[{index}]")
    return errors


def header_ops_and_abi(header: Path = HEADER) -> tuple[set[str], int, int]:
    """C3 op names in normalised form (upper case, no underscores) and the ABI version from the C1 header."""
    text = header.read_text(encoding="utf-8")
    ops = {m.group(1).replace("_", "") for m in re.finditer(r"^#define ROOST_OP_(\w+) \d+", text, re.MULTILINE)}
    major = int(re.search(r"^#define ROOST_ABI_MAJOR (\d+)", text, re.MULTILINE).group(1))
    minor = int(re.search(r"^#define ROOST_ABI_MINOR (\d+)", text, re.MULTILINE).group(1))
    return ops, major, minor


def semantic_errors(manifest: dict, header: Path = HEADER) -> list[str]:
    errors: list[str] = []
    known_ops, abi_major, abi_minor = header_ops_and_abi(header)
    abi = manifest.get("abi", {})
    if abi.get("major") != abi_major:
        errors.append(f"$.abi.major: {abi.get('major')} but roost_kernels.h has ROOST_ABI_MAJOR {abi_major}")
    if isinstance(abi.get("minor"), int) and abi["minor"] > abi_minor:
        errors.append(f"$.abi.minor: {abi['minor']} is newer than ROOST_ABI_MINOR {abi_minor}")

    archs = {entry.get("arch") for entry in manifest.get("sm", [])}
    sources = {"roost"}
    sources |= {u.get("name") for u in manifest.get("upstreams", [])}
    sources |= {lib.get("name") for lib in manifest.get("external_libraries", [])}
    triton = {k.get("name") for k in manifest.get("triton_aot", {}).get("kernels", [])}
    manifest_ops = set()
    variant_ids: dict[str, str] = {}

    for i, op in enumerate(manifest.get("ops", [])):
        name = op.get("op", "")
        manifest_ops.add(name)
        if name.upper() not in known_ops:
            errors.append(f"$.ops[{i}].op: {name!r} is not a C3 op (no ROOST_OP_ id in roost_kernels.h)")
        for j, variant in enumerate(op.get("variants", [])):
            where = f"$.ops[{i}].variants[{j}]"
            vid = variant.get("id")
            if vid in variant_ids:
                errors.append(f"{where}.id: {vid!r} already used by op {variant_ids[vid]!r}")
            variant_ids[vid] = name
            if variant.get("source") not in sources:
                errors.append(f"{where}.source: {variant.get('source')!r} is not an upstream, library or 'roost'")
            constraints = variant.get("constraints", {})
            for key in ("sm_min", "sm_max"):
                if key in constraints and constraints[key] not in archs:
                    errors.append(f"{where}.constraints.{key}: {constraints[key]} is not in the SM list")
            for k, binary in enumerate(variant.get("binaries", [])):
                if binary.get("arch") not in archs:
                    errors.append(f"{where}.binaries[{k}].arch: {binary.get('arch')} is not in the SM list")
            for kernel in variant.get("triton_kernels", []):
                if kernel not in triton:
                    errors.append(f"{where}.triton_kernels: {kernel!r} is not declared under triton_aot.kernels")

    for i, upstream in enumerate(manifest.get("upstreams", [])):
        for name in upstream.get("ops", []):
            if name not in manifest_ops:
                errors.append(f"$.upstreams[{i}].ops: {name!r} is not an op of this manifest")
    for i, kernel in enumerate(manifest.get("triton_aot", {}).get("kernels", [])):
        if kernel.get("upstream") not in sources:
            errors.append(f"$.triton_aot.kernels[{i}].upstream: {kernel.get('upstream')!r} is not an upstream")
    return errors


def check(manifest: dict, schema_path: Path = SCHEMA, header: Path = HEADER) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = validate(manifest, schema, schema)
    return errors if errors else semantic_errors(manifest, header)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_MANIFEST
    if not path.exists():
        print(f"{path}: not found (kernels/manifest.toml lands with R-016)")
        return 1
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)
    errors = check(manifest)
    for error in errors:
        print(f"{path}: {error}")
    if not errors:
        print(f"{path}: valid (C5 schema v{manifest['schema_version']})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
