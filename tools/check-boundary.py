#!/usr/bin/env python3
"""Fails if any tracked file, file name or commit names a term that must never appear in this repository.

The terms mark a boundary with an unrelated older project. They are stored only as SHA-256 hashes, so this
file does not name them either. Text is lowercased and split into alphanumeric words. A term matches as any
substring of a word (catching it inside a longer identifier or address) or as two adjacent words joined
(catching it split by a separator). `.gz` files are decompressed and other binary files are read raw. Embedded data
is decoded too: every base64 run of 16 or more characters and every even-length hex run of 32 or more is decoded
(and gunzipped when it is gzip data) and scanned, so a term inside embedded bytes (a test model, a fixture) is found.

    tools/check-boundary.py                  # tracked files and names; commits since the recorded baseline
    tools/check-boundary.py --range A..B     # tracked files and names; the commits of a revision range (a PR)
    tools/check-boundary.py --all-history    # tracked files and names; every commit reachable from HEAD

Files and file names are always scanned in full (the tree at HEAD). Commits (author, committer and message) are
scanned from a baseline on, because a published commit cannot be corrected without rewriting history. The baseline
is the full commit SHA in `tools/check-boundary.baseline` (lines starting with `#` are comments): the commits
reachable from HEAD but not from the baseline are scanned. The baseline commit and its ancestors were scanned when it
was recorded; only the maintainer moves it. Without that file every commit reachable from HEAD is scanned, as before
the baseline existed. `--range` scans the commits `git log` lists for the range instead (for a pull request,
`<base sha>..<head sha>`). The baseline commit must be in the clone (a full-history checkout); a missing baseline or
an invalid range is an error (exit 2), never a pass.

Exit 0 when nothing is found, 1 when a term is found, 2 on bad arguments or a baseline the clone does not hold.

Add a term by appending the SHA-256 of its lowercase alphanumeric form under its length.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN = {
    4: {"f3d3b2bf8e65685428b5dff594bc3fbb013dd5fd3a5e4a7265fc0407202c4bc9"},
    6: {"4838ce42488728518b613f4eace7a76b6fa4e8531c836a4d479a1687fc7b9a8c"},
    9: {
        "6792f98719dd00ddafce75a997d5b3144bd36d89af43b76b7071d9a140bfd513",
        "c6fee12e25f3691684afdc3fb01476feb8f157f37114e84babf975609e93c5f1",
    },
    15: {"b94db545e869d7667df778c958853fbf578df4047f799119d25ac3ea2afdbfc8"},
}
WORD = re.compile(rb"[a-z0-9]+")
BASE64 = re.compile(rb"[A-Za-z0-9+/]{16,}={0,2}")
HEX = re.compile(rb"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32,}(?![0-9A-Fa-f])")
BASELINE_FILE = Path("tools/check-boundary.baseline")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class UsageError(Exception):
    """Bad arguments, a malformed baseline file, or a baseline or range the clone cannot resolve."""


def _decoded(data: bytes) -> list[bytes]:
    """Payloads hidden in base64 or hex runs of the data (gunzipped when they are gzip data)."""
    payloads = []
    for match in BASE64.finditer(data):
        run = match.group()
        try:
            payload = base64.b64decode(run + b"=" * (-len(run) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        payloads.append(payload)
    for match in HEX.finditer(data):
        run = match.group()
        if len(run) % 2 == 0:
            payloads.append(bytes.fromhex(run.decode()))
    for i, payload in enumerate(payloads):
        if payload[:2] == b"\x1f\x8b":
            try:
                payloads[i] = gzip.decompress(payload)
            except (OSError, EOFError):
                pass
    return payloads


def _hit(candidate: bytes) -> bool:
    hashes = FORBIDDEN.get(len(candidate))
    return hashes is not None and hashlib.sha256(candidate).hexdigest() in hashes


def find(data: bytes) -> list[int]:
    """Line numbers (1-based) on which a forbidden term occurs."""
    lines: set[int] = set()
    lowered = data.lower()
    previous: re.Match[bytes] | None = None
    for match in WORD.finditer(lowered):
        word = match.group()
        found = any(
            _hit(word[i : i + length])
            for length in FORBIDDEN
            if length <= len(word)
            for i in range(len(word) - length + 1)
        )
        if not found and previous is not None:
            found = _hit(previous.group() + word)
        if found:
            lines.add(lowered.count(b"\n", 0, match.start()) + 1)
        previous = match
    return sorted(lines)


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True
    ).stdout


def _git_ok(root: Path, *args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False
        ).returncode
        == 0
    )


def read_baseline(root: Path) -> str | None:
    """The recorded baseline commit, or None when the repository records none."""
    path = root / BASELINE_FILE
    if not path.is_file():
        return None
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != 1 or not FULL_SHA.match(entries[0]):
        raise UsageError(
            f"{BASELINE_FILE.as_posix()} must hold exactly one full 40-hex commit SHA (and # comments)"
        )
    return entries[0]


def commit_selection(
    root: Path, revision_range: str | None, all_history: bool
) -> tuple[list[str], str]:
    """The `git log` revision arguments of the commits to scan, and a description for the report."""
    if not _git_ok(root, "rev-parse", "--verify", "-q", "HEAD"):
        return [], "no commits (no HEAD)"
    if revision_range is not None:
        if not revision_range or revision_range.startswith("-"):
            raise UsageError(
                f"--range must be a revision range, not {revision_range!r}"
            )
        if not _git_ok(root, "rev-list", "--quiet", "--end-of-options", revision_range):
            raise UsageError(
                f"--range {revision_range!r} does not resolve in this clone"
            )
        return [revision_range], f"the commits in {revision_range}"
    baseline = None if all_history else read_baseline(root)
    if baseline is None:
        return ["HEAD"], "the commits reachable from HEAD"
    if not _git_ok(root, "cat-file", "-e", f"{baseline}^{{commit}}"):
        raise UsageError(
            f"baseline commit {baseline} ({BASELINE_FILE.as_posix()}) is not in this clone: check out the full "
            "history (actions/checkout fetch-depth: 0) or pass --range"
        )
    return [f"{baseline}..HEAD"], f"the commits since baseline {baseline[:12]}"


def scan_commits(root: Path, revisions: list[str]) -> list[str]:
    """Problems in the author, committer and message of the commits `git log` lists for the revisions."""
    if not revisions:
        return []
    log = git(
        root,
        "log",
        "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x01",
        "--end-of-options",
        *revisions,
    )
    problems = []
    for entry in log.split(b"\x01"):
        fields = entry.strip(b"\n").split(b"\x00")
        if len(fields) == 4 and find(b"\n".join(fields[1:])):
            problems.append(
                f"commit {fields[0][:12].decode()}: author, committer or message"
            )
    return problems


def scan_files(root: Path) -> tuple[int, list[str]]:
    """The number of tracked files, and the problems in their names and contents."""
    problems: list[str] = []
    files = [p for p in git(root, "ls-files", "-z").split(b"\0") if p]
    for raw in files:
        name = raw.decode("utf-8", "surrogateescape")
        if find(raw):
            problems.append(f"file name: {name}")
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        for line in find(data):
            problems.append(f"{name}:{line}")
        if any(find(payload) for payload in _decoded(data)):
            problems.append(f"{name}: inside embedded base64 or hex data")
    return len(files), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Boundary check of tracked files, file names and commits."
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--range",
        dest="revision_range",
        metavar="A..B",
        help="scan the commits of this revision range instead of those since the baseline",
    )
    which.add_argument(
        "--all-history",
        action="store_true",
        help="scan every commit reachable from HEAD, ignoring the baseline",
    )
    args = parser.parse_args(argv)
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    try:
        revisions, scope = commit_selection(root, args.revision_range, args.all_history)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    count, problems = scan_files(root)
    problems += scan_commits(root, revisions)
    if problems:
        print("Boundary check failed: a forbidden term appears in", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"ok:   {count} tracked files, their names, and {scope} are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
