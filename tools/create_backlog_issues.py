#!/usr/bin/env python3
"""Turn the build plan's section 10 backlog table into GitHub issues.

The plan document is the single source for the backlog (adapted from starling's tools/create_backlog_issues.py).
This script parses its table and prints the issues it would create. Pass --create to open them with the GitHub CLI
(`gh`), which must be installed and authenticated. Creating issues is the maintainer's call.

    python3 -I tools/create_backlog_issues.py                 # dry run
    python3 -I tools/create_backlog_issues.py --create --repo starlinghq/roost

Labels: `backlog`, one `stream:<id>` per owning stream, `phase:<phase>`, and `starling-pr` for items whose work is a
pull request to starlinghq/starling (the issue then links that pull request).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent / "docs" / "plan" / "build-plan.md"
ROW = re.compile(r"^\| (R-\d{3}) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \|$")
EXPECTED = 109

# Work that lands in starlinghq/starling (build plan section 10, first paragraph).
STARLING_STREAMS = {"B1", "B2", "B3"}
STARLING_IDS = {"R-001", "R-002", "R-059", "R-061"}


def parse_backlog(text: str) -> list[dict[str, str]]:
    section = text.split("## 10. Backlog", 1)[1].split("\n## ", 1)[0]
    items = []
    for line in section.splitlines():
        match = ROW.match(line.strip())
        if match:
            rid, streams, phase, title, deps, accept = (g.strip() for g in match.groups())
            items.append(
                {"id": rid, "streams": streams, "phase": phase, "title": title, "depends": deps, "acceptance": accept}
            )
    return items


def labels_for(item: dict[str, str]) -> list[str]:
    streams = [s.strip() for s in item["streams"].split(",")]
    phases = [p.strip().replace(" ", "-") for p in item["phase"].split(",")]
    labels = ["backlog", *(f"stream:{s}" for s in streams), *(f"phase:{p}" for p in phases)]
    if item["id"] in STARLING_IDS or STARLING_STREAMS & set(streams):
        labels.append("starling-pr")
    return labels


def body_for(item: dict[str, str]) -> str:
    return (
        f"Backlog item **{item['id']}** from `docs/plan/build-plan.md` section 10.\n\n"
        f"**Stream:** {item['streams']} · **Phase:** {item['phase']}\n\n"
        f"**Depends on:** {item['depends']}\n\n"
        f"**Acceptance criteria:** {item['acceptance']}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--create", action="store_true", help="create issues with gh instead of printing them")
    parser.add_argument("--repo", default="starlinghq/roost")
    args = parser.parse_args()

    items = parse_backlog(PLAN.read_text(encoding="utf-8"))
    if len(items) != EXPECTED:
        print(f"expected {EXPECTED} backlog items, parsed {len(items)}", file=sys.stderr)
        return 1

    if args.create:
        for label in sorted({label for item in items for label in labels_for(item)}):
            subprocess.run(["gh", "label", "create", label, "--repo", args.repo, "--force"], check=True)

    for item in items:
        title = f"{item['id']}: {item['title']}"
        labels = labels_for(item)
        if not args.create:
            print(f"{title}\n  labels: {', '.join(labels)}\n  depends: {item['depends']}")
            continue
        cmd = ["gh", "issue", "create", "--repo", args.repo, "--title", title, "--body", body_for(item)]
        for label in labels:
            cmd += ["--label", label]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
