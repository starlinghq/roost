#!/usr/bin/env python3
"""Fail a kernel upstream change that does not carry its provenance (R-016, AGENTS.md "Bounded authority").

    python3 -I kernels/tools/check_provenance.py                 # consistency of the tree
    python3 -I kernels/tools/check_provenance.py --base REF      # also: a pin change since REF needs both notice files
    (--root DIR checks another tree; the tests use it)

Consistency (always): every upstream in kernels/manifest.toml has a PROVENANCE.md row whose repository and full SHA
match, and a THIRD-PARTY-NOTICES.md section naming its repository and SHA; no PROVENANCE.md row for a manifest
repository carries another SHA; every patch the manifest lists exists with the recorded SHA-256, and every
kernels/upstream/<name>.patches/*.patch file is listed.

With --base (CI passes the pull request's base commit): an upstream added, removed, moved to another SHA or given
another patch set since the merge base of REF and HEAD requires PROVENANCE.md and THIRD-PARTY-NOTICES.md to be changed
in the same range. A bump that edits only the manifest therefore fails, even if the old text still looked consistent.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTICE_FILES = ("PROVENANCE.md", "THIRD-PARTY-NOTICES.md")


def pins(manifest: dict) -> dict[str, tuple]:
    """name -> (repo, sha, sorted patch (path, sha256) pairs)."""
    return {
        u["name"]: (u["repo"], u["sha"], tuple(sorted((p["path"], p["sha256"]) for p in u.get("patches", []))))
        for u in manifest.get("upstreams", [])
    }


def consistency_errors(manifest: dict, root: Path) -> list[str]:
    errors: list[str] = []
    provenance = (root / "PROVENANCE.md").read_text(encoding="utf-8")
    notices = (root / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    rows = [line for line in provenance.splitlines() if line.startswith("|")]
    for upstream in manifest.get("upstreams", []):
        name, repo, sha = upstream["name"], upstream["repo"], upstream["sha"]
        if not any(repo in row and sha in row for row in rows):
            errors.append(f"PROVENANCE.md: no row for {name} with repository {repo} and revision {sha}")
        for row in rows:
            if repo in row and sha not in row:
                errors.append(f"PROVENANCE.md: row for {repo} does not carry the manifest SHA {sha}: {row[:80]}")
        if repo not in notices or sha not in notices:
            errors.append(f"THIRD-PARTY-NOTICES.md: no section naming {repo} at {sha}")
        listed = set()
        for patch in upstream.get("patches", []):
            path = root / patch["path"]
            listed.add(path.resolve())
            if not path.is_file():
                errors.append(f"{patch['path']}: listed for {name} but not found")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
                errors.append(f"{patch['path']}: SHA-256 differs from the manifest")
        directory = root / "kernels" / "upstream" / f"{name}.patches"
        if directory.is_dir():
            for path in sorted(directory.glob("*.patch")):
                if path.resolve() not in listed:
                    errors.append(f"{path.relative_to(root).as_posix()}: patch not listed in the manifest")
    return errors


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def diff_errors(manifest: dict, root: Path, base: str) -> list[str]:
    merge_base = git(root, "merge-base", base, "HEAD")
    if merge_base.returncode != 0:
        return [f"git merge-base {base} HEAD failed: {merge_base.stderr.strip()}"]
    ref = merge_base.stdout.strip()
    old = git(root, "show", f"{ref}:kernels/manifest.toml")
    before = pins(tomllib.loads(old.stdout)) if old.returncode == 0 else {}
    after = pins(manifest)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    if not changed:
        return []
    names = git(root, "diff", "--name-only", ref, "HEAD").stdout.split()
    return [
        f"{name}: pin or patch set changed since {ref[:12]}, but {file} is not changed in the same range"
        for name in changed
        for file in NOTICE_FILES
        if file not in names
    ]


def main(argv: list[str]) -> int:
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else ROOT
    manifest_path = root / "kernels" / "manifest.toml"
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    errors = consistency_errors(manifest, root)
    if "--base" in argv:
        errors += diff_errors(manifest, root, argv[argv.index("--base") + 1])
    for error in errors:
        print(error)
    if not errors:
        print(f"provenance: {len(manifest.get('upstreams', []))} upstreams consistent")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
