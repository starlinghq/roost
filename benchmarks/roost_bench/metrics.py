"""Serving metrics over per-request traces: TTFT, ITL, latency, throughput, goodput under SLO and saturation.

Definitions (build plan section 14.3; the SLO values themselves are proposals pending decision 25):

- TTFT: time from sending the request to the first streamed chunk that carries output (text, tool-call delta or
  logprob), in seconds.
- ITL: the gap between successive output-bearing chunks. When a chunk carries k > 1 tokens (as reported by the
  client), its gap is split evenly into k samples, so every output token after the first contributes one sample.
  Run-level ITL percentiles pool the samples of every request (as vLLM's ``benchmark_serving`` does).
- TPOT: (last chunk time - first chunk time) / (output tokens - 1), per request with at least two output tokens.
- Latency: end-to-end time from send to the last byte.
- Percentiles use linear interpolation between closest ranks (NumPy's default ``linear`` method):
  rank = q / 100 * (n - 1).
- Throughput: over the wall window from the first send to the last completion of a level.
- Goodput: completed requests per second whose own TTFT is at most the TTFT SLO and whose own p99 ITL is at most the
  ITL SLO (DistServe-style per-request attainment). ``max_goodput_level`` reports the highest load level whose
  run-level p99 TTFT and p99 ITL both meet the SLO.
- Saturation: the highest output-token throughput over a load sweep, and the knee, the first level after which
  throughput grows by less than ``knee_gain`` (default 5 %).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    """One request as the client observed it. Times are seconds on one monotonic clock."""

    request_id: str
    kind: str  # "generate" or "embed"
    start: float
    end: float
    ok: bool
    chunk_times: list[float] = field(default_factory=list)
    chunk_tokens: list[int] | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None
    session_id: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.kind != "generate" or not self.chunk_times:
            return None
        return self.chunk_times[0] - self.start

    @property
    def latency(self) -> float:
        return self.end - self.start

    def itl_samples(self) -> list[float]:
        """Inter-token gaps; a chunk of k tokens contributes k samples of gap / k."""
        times = self.chunk_times
        if len(times) < 2:
            return []
        counts = self.chunk_tokens if self.chunk_tokens is not None else [1] * len(times)
        if len(counts) != len(times):
            raise ValueError(f"{self.request_id}: chunk_tokens has {len(counts)} entries for {len(times)} chunks")
        samples: list[float] = []
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            if gap < 0:
                raise ValueError(f"{self.request_id}: chunk times go backwards at index {i}")
            k = max(counts[i], 1)
            samples.extend([gap / k] * k)
        return samples

    def tokens_out(self) -> int:
        """Output tokens: the server's usage count when given, else the chunk token count."""
        if self.output_tokens is not None:
            return self.output_tokens
        if self.chunk_tokens is not None:
            return sum(self.chunk_tokens)
        return len(self.chunk_times)

    @property
    def tpot(self) -> float | None:
        n = self.tokens_out()
        if self.kind != "generate" or n < 2 or len(self.chunk_times) < 2:
            return None
        return (self.chunk_times[-1] - self.chunk_times[0]) / (n - 1)


@dataclass(frozen=True)
class Slo:
    """Latency SLO for goodput. Values are proposals until decision 25 sets them."""

    ttft_s: float
    itl_s: float
    status: str = "proposed"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (NumPy ``linear``). Raises on an empty sequence."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile {q} outside [0, 100]")
    ordered = sorted(values)
    rank = q / 100 * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _p(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values),
        "n": len(values),
    }


def meets_slo(record: RequestRecord, slo: Slo) -> bool:
    """Per-request SLO attainment. Embeddings meet it when their latency is within the TTFT bound."""
    if not record.ok:
        return False
    if record.kind == "embed":
        return record.latency <= slo.ttft_s
    ttft = record.ttft
    if ttft is None or ttft > slo.ttft_s:
        return False
    itl = record.itl_samples()
    return not itl or percentile(itl, 99) <= slo.itl_s


def summarize(records: Iterable[RequestRecord], slo: Slo | None = None) -> dict:
    """Run-level summary of one load level."""
    records = list(records)
    if not records:
        raise ValueError("no records to summarize")
    ok = [r for r in records if r.ok]
    gen = [r for r in ok if r.kind == "generate"]
    window = max(r.end for r in records) - min(r.start for r in records)
    ttfts = [t for r in gen if (t := r.ttft) is not None]
    itls = [s for r in gen for s in r.itl_samples()]
    tpots = [t for r in gen if (t := r.tpot) is not None]
    out_tokens = sum(r.tokens_out() for r in gen)
    in_tokens = sum(r.prompt_tokens or 0 for r in ok)
    summary: dict = {
        "requests": len(records),
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "window_s": window,
        "ttft_s": _p(ttfts),
        "itl_s": _p(itls),
        "tpot_s": _p(tpots),
        "latency_s": _p([r.latency for r in ok]),
        "output_tokens": out_tokens,
        "prompt_tokens": in_tokens,
        "request_throughput_rps": len(ok) / window if window > 0 else None,
        "output_throughput_tps": out_tokens / window if window > 0 else None,
        "total_throughput_tps": (out_tokens + in_tokens) / window if window > 0 else None,
    }
    if slo is not None:
        good = sum(1 for r in records if meets_slo(r, slo))
        ttft_p99 = summary["ttft_s"]["p99"] if summary["ttft_s"] else None
        itl_p99 = summary["itl_s"]["p99"] if summary["itl_s"] else None
        summary["slo"] = {"ttft_s": slo.ttft_s, "itl_s": slo.itl_s, "status": slo.status}
        summary["good_requests"] = good
        summary["goodput_rps"] = good / window if window > 0 else None
        summary["slo_attainment"] = good / len(records)
        summary["run_meets_slo"] = (
            ttft_p99 is not None
            and ttft_p99 <= slo.ttft_s
            and (itl_p99 is None or itl_p99 <= slo.itl_s)
            and summary["failed"] == 0
        )
    return summary


def saturation(levels: Sequence[tuple[float, dict]], knee_gain: float = 0.05) -> dict:
    """Saturation over a load sweep of ``(level, summary)`` pairs, sorted by level here.

    ``knee_level`` is the first level whose next level adds less than ``knee_gain`` relative output throughput;
    None when throughput still grows at the last level (the sweep did not reach saturation).
    """
    pts = sorted(((lv, s.get("output_throughput_tps") or 0.0) for lv, s in levels), key=lambda p: p[0])
    if not pts:
        raise ValueError("no levels")
    best_level, best_tps = max(pts, key=lambda p: p[1])
    knee = None
    for (lv, tps), (_, nxt) in zip(pts, pts[1:], strict=False):
        if tps > 0 and (nxt - tps) / tps < knee_gain:
            knee = lv
            break
    return {"saturation_output_tps": best_tps, "saturation_level": best_level, "knee_level": knee}


def max_goodput_level(levels: Sequence[tuple[float, dict]]) -> float | None:
    """Highest load level whose run-level p99 TTFT and p99 ITL meet the SLO (summaries made with an SLO)."""
    passing = [lv for lv, s in levels if s.get("run_meets_slo")]
    return max(passing) if passing else None
