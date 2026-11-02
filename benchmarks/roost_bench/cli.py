"""Command line: ``python3 -I benchmarks/bench.py {run,summarize,validate} ...``."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from . import TRACE_FORMAT, __version__
from .client import Endpoint
from .metrics import RequestRecord, Slo, max_goodput_level, saturation, summarize
from .runner import run_concurrency, run_rate
from .workloads import WorkloadError, load, load_sharegpt


def load_slo(path: str | Path) -> Slo:
    """Read ``slo.toml``: ``ttft_p99_ms``, ``itl_p99_ms`` and ``status``."""
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    return Slo(ttft_s=raw["ttft_p99_ms"] / 1000, itl_s=raw["itl_p99_ms"] / 1000, status=raw.get("status", "proposed"))


def _report(levels: list[tuple[float, dict]]) -> dict:
    return {
        "levels": [{"level": lv, **s} for lv, s in levels],
        "saturation": saturation(levels),
        "max_goodput_level": max_goodput_level(levels),
    }


def cmd_run(args: argparse.Namespace) -> int:
    w = load(args.workload)
    if w.status == "unsupported":
        print(f"{w.id}: status unsupported; not run", file=sys.stderr)
        return 2
    dataset = load_sharegpt(args.dataset) if args.dataset else None
    sessions = w.sessions(seed=args.seed, dataset=dataset)
    if args.max_requests:
        sessions = sessions[: args.max_requests]
    ep = Endpoint(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        embed_model=args.embed_model,
        timeout_s=args.timeout,
        include_usage=not args.no_usage,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
    )
    slo = load_slo(args.slo) if args.slo else None
    levels = [float(x) for x in args.levels.split(",")] if args.levels else w.levels
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = {
        "format": TRACE_FORMAT,
        "note": "interim trace; not the Roost raw-results schema (R-047 follows starlings-bench-raw/1)",
        "harness_version": __version__,
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "workload": w.id,
        "workload_status": w.status,
        "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
        "seed": args.seed,
        "load_mode": w.load_mode,
        "extra_body": ep.extra_body,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "client_python": platform.python_version(),
    }
    summaries: list[tuple[float, dict]] = []
    with (out / "trace.jsonl").open("w", encoding="utf-8") as trace:
        trace.write(json.dumps({"header": header}) + "\n")
        for level in levels:
            if args.warmup:
                warm = sessions[: args.warmup]
                run_concurrency(ep, warm, min(len(warm), max(int(level), 1)))
            if w.load_mode == "concurrency":
                res = run_concurrency(ep, sessions, int(level))
            else:
                res = run_rate(ep, sessions, level, seed=args.seed)
            for r in res.records:
                trace.write(json.dumps({"level": level, **dataclasses.asdict(r)}) + "\n")
            s = summarize(res.records, slo)
            s["max_client_lag_s"] = res.max_client_lag_s
            summaries.append((level, s))
            print(
                f"{w.id} {w.load_mode}={level:g}: {s['completed']}/{s['requests']} ok, "
                f"out {s['output_throughput_tps'] or 0:.1f} tok/s"
                + (f", goodput {s['goodput_rps']:.2f} req/s" if slo else ""),
                file=sys.stderr,
            )
    report = {"header": header, **_report(summaries)}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(s["failed"] == 0 for _, s in summaries) else 1


def read_trace(path: str | Path) -> tuple[dict, dict[float, list[RequestRecord]]]:
    """Read a ``trace.jsonl`` back into its header and records grouped by level."""
    header: dict = {}
    by_level: dict[float, list[RequestRecord]] = {}
    fields = {f.name for f in dataclasses.fields(RequestRecord)}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "header" in row:
            header = row["header"]
            continue
        rec = RequestRecord(**{k: v for k, v in row.items() if k in fields})
        by_level.setdefault(float(row["level"]), []).append(rec)
    return header, by_level


def cmd_summarize(args: argparse.Namespace) -> int:
    header, by_level = read_trace(args.trace)
    slo = load_slo(args.slo) if args.slo else None
    report = {"header": header, **_report([(lv, summarize(recs, slo)) for lv, recs in sorted(by_level.items())])}
    print(json.dumps(report, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bad = 0
    for p in args.files:
        try:
            w = load(p)
            w.sessions(seed=0, dataset=["placeholder prompt"])
            print(f"{p}: ok ({w.id}, {w.status})")
        except (WorkloadError, tomllib.TOMLDecodeError, KeyError) as e:
            print(f"{p}: {e}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive an OpenAI-compatible endpoint with one workload")
    run.add_argument("--workload", required=True)
    run.add_argument("--base-url", required=True, help="API root, e.g. http://127.0.0.1:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--label", required=True, help="backend label recorded in the trace, e.g. roost, vllm-0.31.0")
    run.add_argument("--out", required=True, help="output directory (trace.jsonl, summary.json); keep it outside Git")
    run.add_argument("--levels", help="comma-separated load levels; default from the workload")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--dataset", help="ShareGPT-format JSON for workloads with prompt.source = sharegpt")
    run.add_argument("--slo", help="SLO file, e.g. benchmarks/slo.toml (proposed values, decision 25)")
    run.add_argument("--api-key-env", help="environment variable holding the API key")
    run.add_argument("--embed-model", help="model name for /v1/embeddings requests (W6)")
    run.add_argument("--extra-body", help="JSON object merged into generation requests, e.g. '{\"ignore_eos\": true}'")
    run.add_argument("--no-usage", action="store_true", help="do not send stream_options.include_usage")
    run.add_argument("--max-requests", type=int, help="truncate the session list (smoke runs)")
    run.add_argument("--warmup", type=int, default=0, help="sessions sent before each level and not recorded")
    run.add_argument("--timeout", type=float, default=600.0)
    run.set_defaults(func=cmd_run)

    summ = sub.add_parser("summarize", help="recompute the summary of a trace.jsonl")
    summ.add_argument("trace")
    summ.add_argument("--slo")
    summ.set_defaults(func=cmd_summarize)

    val = sub.add_parser("validate", help="check workload files")
    val.add_argument("files", nargs="+")
    val.set_defaults(func=cmd_validate)

    args = ap.parse_args(argv)
    return args.func(args)
