"""The `roost-evidence` command. Subcommands are added here as the pipeline grows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from roost_evidence import evaluate_generation, manifest, reference_runner


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


def _evaluate(args: argparse.Namespace) -> int:
    try:
        report = evaluate_generation.evaluate(args.results, args.reference, args.tolerances, args.reference_vectors)
    except evaluate_generation.InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    for check in report["checks"]:
        suffix = f": {check['detail']}" if check["detail"] else ""
        print(f"{check['status']:<14} {check['id']}{suffix}")
    for d in report["divergences"]:
        verdict = "allowed" if d["allowed"] else "not allowed"
        print(f"divergence     {d['id']} step {d['step']} reference margin {d['reference_margin']} ({verdict})")
    print(f"{report['status']}: {report['profile']}")
    return 0 if report["status"] == evaluate_generation.PASSED else 1


def _reference_run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        path = reference_runner.run_reference(
            args.model,
            args.prompts,
            args.out,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            stop_ids=args.stop_id,
        )
    except (reference_runner.ExtraMissingError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


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

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate stage results (format v3, task generation) against a reference run and tolerances",
        description=(
            "Exit 0 when every check passed, 1 when any failed or was not run, 2 when the inputs cannot be evaluated. "
            "Tolerances are an input; Roost's protected file is maintainer-only."
        ),
    )
    evaluate.add_argument("results", type=Path, help="directory with template.results.json and dotnet.results.json")
    evaluate.add_argument("--reference", type=Path, required=True, help="reference.jsonl of a reference run")
    evaluate.add_argument("--tolerances", type=Path, required=True, help="tolerance file (.yaml, .yml or .json)")
    evaluate.add_argument(
        "--reference-vectors", type=Path, help="directory holding step0/<id>.f32 of the reference run"
    )
    evaluate.add_argument("--report", type=Path, help="write the JSON report here")
    evaluate.set_defaults(handler=_evaluate)

    reference = commands.add_parser(
        "reference-run",
        help="run the transformers fp32 CPU reference (needs the `reference` extra)",
        description="Greedy float32 CPU decoding of a prompts file; writes reference.jsonl and step-0 vectors.",
    )
    reference.add_argument("model", help="local model directory, or a hub id together with --revision")
    reference.add_argument("prompts", type=Path, help="prompts jsonl: id and messages or prompt_ids per line")
    reference.add_argument("--out", type=Path, required=True, help="output directory (run artifacts, not Git)")
    reference.add_argument("--revision", help="full 40-character commit SHA, required for a hub id")
    reference.add_argument("--max-new-tokens", type=int, default=128)
    reference.add_argument("--stop-id", type=int, action="append", help="extra stop token id (repeatable)")
    reference.set_defaults(handler=_reference_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
