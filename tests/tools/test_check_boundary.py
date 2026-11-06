"""Tests of tools/check-boundary.py: term matching, the full-tree file scan and the commit range selection.

The real terms are known only by their hashes, so each test replaces FORBIDDEN with the hash of a made-up term. The
git repositories are temporary, with a fixed throwaway identity in the environment of their own git calls only.

    uvx --from pytest==8.4.2 pytest -q tests/tools/test_check_boundary.py
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "check-boundary.py"
TERM = "quokkazeta"


def _load():
    spec = importlib.util.spec_from_file_location("check_boundary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FORBIDDEN = {len(TERM): {hashlib.sha256(TERM.encode()).hexdigest()}}
    return module


check = _load()

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.invalid",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


class Repo:
    """A temporary git repository with one commit per call to commit()."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str, env: dict[str, str] | None = None) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={**os.environ, **GIT_ENV, **(env or {})},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def commit(self, message: str, name: str = "file.txt", text: str = "x") -> str:
        path = self.root / name
        path.write_text(path.read_text() + text if path.exists() else text)
        self.git("add", name)
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def baseline(self, sha: str) -> None:
        path = self.root / check.BASELINE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# recorded baseline\n{sha}\n")
        self.git("add", str(check.BASELINE_FILE))
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "record baseline")

    def run(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = check.main(list(argv))
        finally:
            os.chdir(previous)
        return status, out.getvalue(), err.getvalue()


class FindTests(unittest.TestCase):
    def test_term_inside_a_word(self) -> None:
        self.assertEqual(check.find(f"line one\nmy{TERM.upper()}x\n".encode()), [2])

    def test_term_split_by_a_separator(self) -> None:
        self.assertEqual(check.find(b"a quokka-zeta b"), [1])

    def test_clean_text(self) -> None:
        self.assertEqual(check.find(b"quokka and zeta, apart"), [])

    def test_embedded_base64_and_gzip(self) -> None:
        payload = base64.b64encode(gzip.compress(TERM.encode() * 4))
        self.assertTrue(any(check.find(p) for p in check._decoded(payload)))


class CommitRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.repo = Repo(Path(self._dir.name))

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_without_baseline_every_commit_is_scanned(self) -> None:
        self.repo.commit(f"old {TERM} message")
        self.repo.commit("clean")
        status, _, err = self.repo.run()
        self.assertEqual(status, 1)
        self.assertIn("author, committer or message", err)

    def test_commits_up_to_the_baseline_are_not_scanned(self) -> None:
        bad = self.repo.commit(f"old {TERM} message")
        self.repo.baseline(bad)
        self.repo.commit("clean")
        status, out, _ = self.repo.run()
        self.assertEqual(status, 0)
        self.assertIn(f"since baseline {bad[:12]}", out)

    def test_commits_after_the_baseline_are_scanned(self) -> None:
        base = self.repo.commit("clean")
        self.repo.baseline(base)
        bad = self.repo.commit(f"new {TERM} message")
        status, _, err = self.repo.run()
        self.assertEqual(status, 1)
        self.assertIn(f"commit {bad[:12]}", err)

    def test_author_after_the_baseline_is_scanned(self) -> None:
        base = self.repo.commit("clean")
        self.repo.baseline(base)
        (self.repo.root / "file.txt").write_text("changed")
        self.repo.git("add", "file.txt")
        self.repo.git(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "clean message",
            env={"GIT_AUTHOR_NAME": TERM},
        )
        self.assertEqual(self.repo.run()[0], 1)

    def test_all_history_ignores_the_baseline(self) -> None:
        bad = self.repo.commit(f"old {TERM} message")
        self.repo.baseline(bad)
        status, _, err = self.repo.run("--all-history")
        self.assertEqual(status, 1)
        self.assertIn(f"commit {bad[:12]}", err)

    def test_range_scans_only_the_range(self) -> None:
        self.repo.commit(f"old {TERM} message")
        base = self.repo.commit("clean base")
        head = self.repo.commit("clean head")
        status, out, _ = self.repo.run("--range", f"{base}..{head}")
        self.assertEqual(status, 0)
        self.assertIn(f"the commits in {base}..{head}", out)
        bad = self.repo.commit(f"new {TERM} message")
        status, _, err = self.repo.run("--range", f"{base}..{bad}")
        self.assertEqual(status, 1)
        self.assertIn(f"commit {bad[:12]}", err)

    def test_files_are_scanned_in_full_whatever_the_baseline(self) -> None:
        self.repo.commit("clean", name="data.txt", text=f"{TERM}\n")
        head = self.repo.commit("clean")
        self.repo.baseline(head)
        status, _, err = self.repo.run()
        self.assertEqual(status, 1)
        self.assertIn("data.txt:1", err)

    def test_file_names_are_scanned(self) -> None:
        self.repo.commit("clean", name=f"{TERM}.txt")
        status, _, err = self.repo.run("--range", "HEAD~0..HEAD")
        self.assertEqual(status, 1)
        self.assertIn(f"file name: {TERM}.txt", err)

    def test_baseline_missing_from_the_clone_is_an_error(self) -> None:
        self.repo.commit("clean")
        self.repo.baseline("0123456789abcdef0123456789abcdef01234567")
        status, _, err = self.repo.run()
        self.assertEqual(status, 2)
        self.assertIn("is not in this clone", err)

    def test_malformed_baseline_is_an_error(self) -> None:
        self.repo.commit("clean")
        self.repo.baseline("main")
        status, _, err = self.repo.run()
        self.assertEqual(status, 2)
        self.assertIn("exactly one full 40-hex commit SHA", err)

    def test_unresolvable_or_option_like_range_is_an_error(self) -> None:
        self.repo.commit("clean")
        self.assertEqual(self.repo.run("--range", "nosuchref..HEAD")[0], 2)
        self.assertEqual(self.repo.run("--range=--all")[0], 2)

    def test_range_and_all_history_exclude_each_other(self) -> None:
        self.repo.commit("clean")
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            self.repo.run("--all-history", "--range", "HEAD..HEAD")
        self.assertEqual(raised.exception.code, 2)


class RecordedBaselineTests(unittest.TestCase):
    def test_this_repository_records_one_full_sha(self) -> None:
        root = SCRIPT.parents[1]
        if not (root / check.BASELINE_FILE).is_file():
            self.skipTest("this repository records no baseline")
        self.assertRegex(check.read_baseline(root), r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
