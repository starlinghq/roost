#!/usr/bin/env python3
"""Digests of the built per-SM kernel libraries (R-016), and their record in kernels/manifest.toml.

    python3 -I kernels/tools/record_digests.py BUILD_DIR --json OUT.json   # digests of libroost_kernels_sm*.so
    python3 -I kernels/tools/record_digests.py BUILD_DIR --write           # rewrite the manifest's binaries blocks
    python3 -I kernels/tools/record_digests.py BUILD_DIR --check           # exit 1 if a build differs from the manifest

A library is one file per SM, so a digest is per (variant, SM). The binaries of a variant sit between the lines
`# BEGIN binaries <variant>` and `# END binaries <variant>` in the manifest; only SMs that were built are written, and
a block never keeps an entry for an SM that is no longer in the manifest's SM list. Binaries are release assets and are
never committed: only these digests are.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kernels" / "manifest.toml"
LIBRARY = re.compile(r"^(?:lib)?roost_kernels_sm(\d+)\.(?:so|dll)$")


def find_libraries(build_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(build_dir.rglob("*roost_kernels_sm*")):
        match = LIBRARY.match(path.name)
        if match and path.is_file():
            found[int(match.group(1))] = path
    return found


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def measured(build_dir: Path) -> dict[int, dict]:
    return {arch: digest(path) for arch, path in find_libraries(build_dir).items()}


def block(variant: str, digests: dict[int, dict]) -> str:
    lines = [f"# BEGIN binaries {variant}"]
    for arch in sorted(digests):
        lines += [
            "[[ops.variants.binaries]]",
            f"arch = {arch}",
            f'sha256 = "{digests[arch]["sha256"]}"',
            f"bytes = {digests[arch]['bytes']}",
            "",
        ]
    if digests:
        lines.pop()
    lines.append(f"# END binaries {variant}")
    return "\n".join(lines)


def variants_with_blocks(text: str) -> list[str]:
    return re.findall(r"^# BEGIN binaries (\S+)$", text, re.MULTILINE)


def rewrite(text: str, digests: dict[int, dict]) -> str:
    manifest = tomllib.loads(text)
    in_list = {entry["arch"] for entry in manifest["sm"]}
    previous: dict[str, dict[int, dict]] = {}
    for op in manifest["ops"]:
        for variant in op["variants"]:
            previous[variant["id"]] = {
                b["arch"]: {"sha256": b["sha256"], "bytes": b.get("bytes", 0)} for b in variant.get("binaries", [])
            }
    for variant in variants_with_blocks(text):
        merged = {a: d for a, d in previous.get(variant, {}).items() if a in in_list}
        merged.update({a: d for a, d in digests.items() if a in in_list})
        pattern = re.compile(
            rf"^# BEGIN binaries {re.escape(variant)}$.*?^# END binaries {re.escape(variant)}$",
            re.MULTILINE | re.DOTALL,
        )
        text = pattern.sub(lambda _m, v=variant, d=merged: block(v, d), text, count=1)
    return text


def check(text: str, digests: dict[int, dict]) -> list[str]:
    manifest = tomllib.loads(text)
    errors = []
    recorded = {}
    for op in manifest["ops"]:
        for variant in op["variants"]:
            for binary in variant.get("binaries", []):
                recorded[(variant["id"], binary["arch"])] = binary
    for arch, built in sorted(digests.items()):
        entries = [b for (_, a), b in recorded.items() if a == arch]
        if not entries:
            errors.append(f"sm_{arch}: built but no digest recorded in the manifest")
        for entry in entries:
            if entry["sha256"] != built["sha256"]:
                errors.append(f"sm_{arch}: built {built['sha256']} but the manifest records {entry['sha256']}")
    return errors


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    build_dir = Path(argv[0])
    digests = measured(build_dir)
    if not digests:
        print(f"{build_dir}: no libroost_kernels_sm<N> library found")
        return 1
    if "--json" in argv:
        out = Path(argv[argv.index("--json") + 1])
        out.write_text(json.dumps({str(a): d for a, d in sorted(digests.items())}, indent=2) + "\n", encoding="utf-8")
    for arch, d in sorted(digests.items()):
        print(f"sm_{arch}  {d['sha256']}  {d['bytes']} bytes")
    text = MANIFEST.read_text(encoding="utf-8")
    if "--write" in argv:
        MANIFEST.write_text(rewrite(text, digests), encoding="utf-8")
        print(f"{MANIFEST}: binaries updated")
    if "--check" in argv:
        errors = check(text, digests)
        for error in errors:
            print(error)
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
