#!/usr/bin/env python3
"""Roost scorecard harness entry point.

    python3 -I benchmarks/bench.py run --workload benchmarks/workloads/w1-chat.toml \
        --base-url http://127.0.0.1:8000/v1 --model <served name> --label roost --out <dir outside Git> \
        --slo benchmarks/slo.toml
    python3 -I benchmarks/bench.py summarize <dir>/trace.jsonl --slo benchmarks/slo.toml
    python3 -I benchmarks/bench.py validate benchmarks/workloads/*.toml
"""

import sys
from pathlib import Path

# -I leaves the script directory off sys.path; add it so the roost_bench package next to this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roost_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
