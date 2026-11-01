"""Puts benchmarks/ on sys.path so the tests import the roost_bench package as bench.py does."""

import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
