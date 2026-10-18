"""The `roost-evidence` command. Subcommands are added here as the pipeline grows (R-010 adds the evaluator)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from roost_evidence import manifest


def _validate_manifest(args: argparse.Namespace) -> int:
    files = list(manifest.find_manifests(args.paths))
    if not files:
        # An empty directory, such as a registry nothing has been admitted to yet, is valid.
        print("no manifest files found", file=sys.stderr)
        return 0
    problems = [p for file in files for p in manifest.validate_file(file, registry=args.registry)]
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roost-evidence", description="Roost evidence pipeline tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-manifest",
        help="validate profile manifests (.json, .yaml, .yml; directories are searched)",
        description="Exit 0 when every manifest is valid, 1 with one line per problem otherwise.",
    )
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument(
        "--registry", action="store_true", help="also reject synthetic manifests, as evidence/registry/ does"
    )
    validate.set_defaults(handler=_validate_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
