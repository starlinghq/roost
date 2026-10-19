#!/usr/bin/env python3
"""Run every registered variant of a kernel library under compute-sanitizer (build plan R-019).

    python3 -I tests/Starlings.Roost.Conformance/sanitize.py --runner <dir> --library libroost_kernels_sm86.so --out <dir> \
        [--tools memcheck,racecheck,synccheck] [--device 3090] [--profile sanitizer] [--goldens <file>] [--variants a,b]

`--runner` is the output directory of tests/Starlings.Roost.Conformance.Cuda (dotnet build -o <dir>). The variants come from the
library's own registration table (the runner's --list-variants), so a variant added to a library is run without a change here.
For each variant and each tool one lease is taken (tools/gpu-lease run, one short lease per run) and the conformance runner is
started under `compute-sanitizer --tool <tool> --error-exitcode 99`, restricted to that variant. Each run leaves

    <out>/<variant>.<tool>.sanitizer.log   what compute-sanitizer printed (--log-file)
    <out>/<variant>.<tool>.out             what the runner printed
    <out>/<variant>.<tool>.results.json    the runner's conformance results for that run

and the driver writes <out>/sanitizer.results.json in the stage-results envelope (task kernel-conformance, stage cuda): one case
per variant and tool with the error count compute-sanitizer reported, and, next to it, whether the conformance cases passed under it (a separate result: a conformance
failure is not a sanitizer report).

compute-sanitizer reports errors without crashing the process, so a run that ends in a signal (SIGSEGV, SIGABRT, SIGBUS, ...) is not
an expected outcome: the driver stops at once, records the command and what it printed, and starts nothing else (the maintainer's
desktop rule: a process that dies by a signal floods it with crash alerts, so nothing is rerun in a loop).

Exit status: 0 when every run reported zero errors, 1 when any reported errors or did not complete, 3 after a signal death.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = ("memcheck", "racecheck", "synccheck")
ERROR_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) error")
SIGNAL_EXIT = {128 + s for s in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGILL, signal.SIGFPE, signal.SIGTRAP)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lease(args: argparse.Namespace, command: list[str]) -> list[str]:
    return [
        str(REPO / "tools" / "gpu-lease"),
        "run",
        "--device",
        args.device,
        "--need-mib",
        str(args.need_mib),
        "--stream",
        "D",
        "--wait",
        str(args.wait),
        "--",
        *command,
    ]


def died_by_signal(returncode: int) -> bool:
    return returncode < 0 or returncode in SIGNAL_EXIT


def run(command: list[str], out: Path) -> tuple[int, str]:
    with out.open("w", encoding="utf-8") as f:
        proc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=False)  # noqa: S603
    return proc.returncode, out.read_text(encoding="utf-8", errors="replace")


def list_variants(args: argparse.Namespace, dll: Path, library: Path, out: Path) -> list[tuple[str, str]]:
    code, text = run(lease(args, ["dotnet", str(dll), str(library), "--device", "0", "--list-variants"]), out / "variants.out")
    if code != 0:
        raise SystemExit(f"listing the variants failed with status {code}:\n{text}")
    rows = [line.split("\t") for line in text.splitlines() if "\t" in line]
    return [(r[0], r[1]) for r in rows]


def conformance_summary(results: Path) -> dict[str, int | bool]:
    if not results.exists():
        return {"passed": 0, "failed": 0, "not_run": 0, "green": False}
    s = json.loads(results.read_text(encoding="utf-8"))["sections"]["summary"]
    return {"passed": s["passed"], "failed": s["failed"] + s["errors"], "not_run": s["not_run"], "green": s["green"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runner", type=Path, required=True, help="output directory of the Conformance.Cuda project")
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory for logs and results (outside the repository)")
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--device", default="3090")
    ap.add_argument("--need-mib", type=int, default=6000)
    ap.add_argument("--wait", type=int, default=3600)
    ap.add_argument("--profile", default="sanitizer")
    ap.add_argument("--goldens", type=Path)
    ap.add_argument("--variants", help="comma-separated variant ids; default every registered variant")
    ap.add_argument("--commit", help="repository commit recorded in the results")
    ap.add_argument("--leak-check", action="store_true", help="add --leak-check full to memcheck")
    args = ap.parse_args(argv)

    dll = args.runner / "Starlings.Roost.Conformance.Cuda.dll"
    if not dll.exists():
        raise SystemExit(f"{dll} does not exist; build tests/Starlings.Roost.Conformance.Cuda with -o {args.runner}")
    args.out.mkdir(parents=True, exist_ok=True)
    variants = list_variants(args, dll, args.library, args.out)
    if args.variants:
        wanted = set(args.variants.split(","))
        variants = [v for v in variants if v[1] in wanted]
    version = subprocess.run(["compute-sanitizer", "--version"], capture_output=True, text=True, check=False).stdout  # noqa: S607
    version = " ".join(version.split())[:200]

    cases: dict[str, dict[str, object]] = {}
    failures = 0
    for tool in args.tools.split(","):
        for op, variant in variants:
            stem = f"{args.out / variant}.{tool}"
            sanitizer_log = Path(f"{stem}.sanitizer.log")
            results = Path(f"{stem}.results.json")
            command = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", "--log-file", str(sanitizer_log)]
            if tool == "memcheck" and args.leak_check:
                command += ["--leak-check", "full"]
            command += ["dotnet", str(dll), str(args.library), "--variants", variant, "--profile", args.profile, "--results", str(results)]
            if args.goldens:
                command += ["--goldens", str(args.goldens)]
            if args.commit:
                command += ["--commit", args.commit]
            code, text = run(lease(args, command), Path(f"{stem}.out"))
            print(f"{tool:10} {variant:34} exit {code}", flush=True)
            if died_by_signal(code):
                print(f"STOP: {' '.join(command)} ended with status {code} (a signal). Nothing else is started.\n{text[-4000:]}")
                (args.out / "SIGNAL-DEATH.txt").write_text(f"{' '.join(command)}\nstatus {code}\n{text}", encoding="utf-8")
                return 3
            log_text = sanitizer_log.read_text(encoding="utf-8", errors="replace") if sanitizer_log.exists() else ""
            m = ERROR_SUMMARY.search(log_text) or ERROR_SUMMARY.search(text)
            errors = int(m.group(1)) if m else -1
            summary = conformance_summary(results)
            # The criterion is compute-sanitizer's: zero reports, from a run that completed. Whether the conformance cases passed
            # under the tool is recorded next to it and is a separate result.
            ok = errors == 0 and results.exists() and code in (0, 1)
            failures += 0 if ok else 1
            cases[f"{op}/{variant}/{tool}"] = {
                "op": op,
                "variant": variant,
                "tool": tool,
                "outcome": "pass" if ok else "fail",
                "sanitizer_errors": errors,
                "exit_code": code,
                "conformance_passed": summary["passed"],
                "conformance_failed": summary["failed"],
                "conformance_green": summary["green"],
                "log": sanitizer_log.name,
                "log_sha256": sha256(sanitizer_log) if sanitizer_log.exists() else None,
            }

    document = {
        "format": "roost.stage-results",
        "format_version": 3,
        "task": "kernel-conformance",
        "profile": "kernels/" + re.sub(r"^lib", "", args.library.stem).replace("_", "-").lower(),
        "stage": "cuda",
        "candidate": f"{args.library.name}@sha256:{sha256(args.library)}",
        "producer": {"name": "tests/Starlings.Roost.Conformance/sanitize.py", "repository_commit": args.commit},
        "environment": {
            "provider": "cuda",
            "provider_options": {"compute_sanitizer": version.strip(), "case_profile": args.profile, "device": args.device},
        },
        "assets": [{"path": args.library.name, "sha256": sha256(args.library)}],
        "cases": cases,
        "sections": {
            "summary": {
                "runs": len(cases),
                "failed": failures,
                "tools": args.tools.split(","),
                "variants": [v for _, v in variants],
                "green": failures == 0,
            }
        },
    }
    (args.out / "sanitizer.results.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{len(cases) - failures} of {len(cases)} runs clean; {args.out / 'sanitizer.results.json'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
