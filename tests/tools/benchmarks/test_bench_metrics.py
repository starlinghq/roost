"""Metric math of the scorecard harness on synthetic traces (no server)."""

from __future__ import annotations

import pytest
from roost_bench.metrics import (
    RequestRecord,
    Slo,
    max_goodput_level,
    meets_slo,
    percentile,
    saturation,
    summarize,
)


def gen(rid: str, start: float, chunks: list[float], end: float | None = None, **kw) -> RequestRecord:
    return RequestRecord(
        request_id=rid, kind="generate", start=start, end=end if end is not None else chunks[-1], ok=True,
        chunk_times=chunks, **kw,
    )  # fmt: skip


def test_percentile_matches_linear_interpolation():
    # NumPy's default method: rank = q/100 * (n-1), interpolate between neighbours.
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([1, 2, 3, 4], 99) == pytest.approx(3.97)
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([7.0], 99) == 7.0
    assert percentile(list(range(101)), 99) == pytest.approx(99.0)


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_ttft_itl_tpot_of_one_request():
    r = gen("a", 10.0, [10.5, 10.55, 10.65, 10.70], end=10.71, output_tokens=4)
    assert r.ttft == pytest.approx(0.5)
    assert r.itl_samples() == pytest.approx([0.05, 0.10, 0.05])
    assert r.tpot == pytest.approx(0.2 / 3)
    assert r.latency == pytest.approx(0.71)


def test_multi_token_chunk_splits_its_gap():
    r = gen("a", 0.0, [1.0, 1.3, 1.4], chunk_tokens=[1, 3, 1])
    assert r.itl_samples() == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert r.tokens_out() == 5
    assert r.tpot == pytest.approx(0.4 / 4)


def test_backwards_chunk_times_and_mismatched_counts_raise():
    with pytest.raises(ValueError, match="backwards"):
        gen("a", 0.0, [1.0, 0.9]).itl_samples()
    with pytest.raises(ValueError, match="chunk_tokens"):
        gen("a", 0.0, [1.0, 1.1], chunk_tokens=[1]).itl_samples()


def test_single_chunk_has_no_itl_or_tpot():
    r = gen("a", 0.0, [0.2], output_tokens=1)
    assert r.itl_samples() == []
    assert r.tpot is None


def test_summary_pools_itl_and_measures_throughput_over_the_window():
    recs = [
        gen("a", 0.0, [0.1, 0.2, 0.3], output_tokens=3, prompt_tokens=10),
        gen("b", 1.0, [1.4, 1.6], end=2.0, output_tokens=2, prompt_tokens=20),
    ]
    s = summarize(recs)
    assert s["window_s"] == pytest.approx(2.0)
    assert s["ttft_s"]["p50"] == pytest.approx(0.25)
    assert s["ttft_s"]["n"] == 2
    assert s["itl_s"]["n"] == 3  # 2 gaps from a, 1 from b
    assert s["itl_s"]["p50"] == pytest.approx(0.1)
    assert s["output_tokens"] == 5
    assert s["output_throughput_tps"] == pytest.approx(2.5)
    assert s["total_throughput_tps"] == pytest.approx(17.5)
    assert s["request_throughput_rps"] == pytest.approx(1.0)
    assert "goodput_rps" not in s


def test_failed_requests_count_in_the_window_but_not_in_latency():
    recs = [
        gen("a", 0.0, [0.1, 0.12], output_tokens=2),
        RequestRecord("b", "generate", start=0.0, end=4.0, ok=False, error="HTTP 500"),
    ]
    s = summarize(recs, Slo(1.0, 0.05))
    assert s["failed"] == 1
    assert s["window_s"] == pytest.approx(4.0)
    assert s["latency_s"]["n"] == 1
    assert s["good_requests"] == 1
    assert s["slo_attainment"] == pytest.approx(0.5)
    assert s["run_meets_slo"] is False  # any failure fails the level


def test_goodput_counts_requests_within_both_slos():
    slo = Slo(ttft_s=1.0, itl_s=0.05)
    fast = gen("fast", 0.0, [0.5, 0.54, 0.58])
    slow_ttft = gen("slow_ttft", 0.0, [1.5, 1.52])
    slow_itl = gen("slow_itl", 0.0, [0.2, 0.3, 0.4])
    assert meets_slo(fast, slo)
    assert not meets_slo(slow_ttft, slo)
    assert not meets_slo(slow_itl, slo)
    s = summarize([fast, slow_ttft, slow_itl], slo)
    assert s["good_requests"] == 1
    assert s["goodput_rps"] == pytest.approx(1 / 1.52)
    assert s["run_meets_slo"] is False
    assert s["slo"]["status"] == "proposed"


def test_run_meets_slo_uses_run_level_p99():
    slo = Slo(ttft_s=1.0, itl_s=0.05)
    recs = [gen(str(i), 0.0, [0.3, 0.33, 0.36]) for i in range(10)]
    assert summarize(recs, slo)["run_meets_slo"] is True


def test_embeddings_count_in_latency_and_goodput_only():
    slo = Slo(ttft_s=1.0, itl_s=0.05)
    emb = RequestRecord("e", "embed", start=0.0, end=0.2, ok=True, prompt_tokens=50)
    late = RequestRecord("f", "embed", start=0.0, end=1.2, ok=True, prompt_tokens=50)
    assert emb.ttft is None
    assert meets_slo(emb, slo) and not meets_slo(late, slo)
    s = summarize([emb, late, gen("g", 0.0, [0.1, 0.12])], slo)
    assert s["ttft_s"]["n"] == 1
    assert s["latency_s"]["n"] == 3
    assert s["prompt_tokens"] == 100


def test_saturation_finds_peak_and_knee():
    levels = [
        (lv, {"output_throughput_tps": tps}) for lv, tps in [(8, 1500), (1, 200), (2, 400), (4, 1000), (16, 1520)]
    ]
    sat = saturation(levels)
    assert sat["saturation_output_tps"] == 1520
    assert sat["saturation_level"] == 16
    assert sat["knee_level"] == 8  # 1500 -> 1520 is below 5 %


def test_saturation_without_knee_when_still_growing():
    levels = [(1, {"output_throughput_tps": 100}), (2, {"output_throughput_tps": 190})]
    assert saturation(levels)["knee_level"] is None


def test_max_goodput_level():
    levels = [(1, {"run_meets_slo": True}), (2, {"run_meets_slo": True}), (4, {"run_meets_slo": False})]
    assert max_goodput_level(levels) == 2
    assert max_goodput_level([(1, {"run_meets_slo": False})]) is None
