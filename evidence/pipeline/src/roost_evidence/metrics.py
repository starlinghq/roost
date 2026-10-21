"""Comparison metrics shared by the evaluators: absolute, row-relative and cosine.

For a candidate vector c and a reference vector r of equal length:

    max_abs    max |c - r|
    rel_l2     ||c - r|| / ||r||       row-relative error (L2)
    cosine     (c . r) / (||c|| ||r||)
    max_ref    max |r|                 the denominator of the per-tensor ratio max |c - r| / max |r|

Guards, so that a broken candidate can never look better than a correct one: a non-finite candidate value makes the
errors infinite and the cosine -infinity; a zero reference row has rel_l2 0 when the candidate is exactly zero too and
infinity otherwise; a cosine with exactly one zero vector is 0.0 and with two zero vectors 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RowStats:
    """Metrics of one candidate row against its reference row."""

    max_abs: float
    rel_l2: float
    cosine: float
    max_ref: float


@dataclass(frozen=True)
class Metric:
    """One way to compare a candidate tensor with its reference, and how a bound on it reads."""

    name: str  # report field suffix and tolerance key suffix
    summary: str  # the name of the aggregate in a check's metrics
    lower_is_worse: bool = False
    per_tensor: bool = False  # one value per tensor, so a violation has no position

    def violates(self, value: float, bound: float) -> bool:
        """True when `value` breaks `bound`; a NaN always does."""
        return not (value >= bound if self.lower_is_worse else value <= bound)


ABS = Metric("max_abs", "max_abs")
RELATIVE_METRICS = (
    Metric("rel_l2", "max_rel_l2"),
    Metric("max_abs_rel", "max_abs_rel", per_tensor=True),
    Metric("cosine_min", "min_cosine", lower_is_worse=True),
)
METRICS = (ABS, *RELATIVE_METRICS)


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator for non-negative arguments; 0/0 is 0, x/0 and non-finite arguments are infinity."""
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.inf
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def centered(values: Sequence[float]) -> list[float]:
    """`values` minus their mean. The mean of a non-finite vector is non-finite, which the metrics then report."""
    mean = math.fsum(values) / len(values) if values else 0.0
    return [v - mean for v in values]


def compare(candidate: Sequence[float], reference: Sequence[float], *, center: bool = False) -> RowStats:
    """Metrics of `candidate` against `reference` (equal lengths).

    With `center`, rel_l2 and cosine are taken after subtracting each vector's mean. A log-softmax row differs from the
    logits row only by a constant, so the centred metrics are the logits' metrics and do not depend on the normaliser.
    max_abs and max_ref are always those of the vectors as given.
    """
    if len(candidate) != len(reference):
        raise ValueError(f"vectors of length {len(candidate)} and {len(reference)} cannot be compared")
    if not candidate:
        return RowStats(0.0, 0.0, 1.0, 0.0)
    max_ref = max(map(abs, reference))
    max_abs = max(abs(a - b) for a, b in zip(candidate, reference, strict=True))
    cand = centered(candidate) if center else candidate
    ref = centered(reference) if center else reference
    diff_sq = sum((a - b) * (a - b) for a, b in zip(cand, ref, strict=True))
    cand_sq = sum(a * a for a in cand)
    ref_sq = sum(b * b for b in ref)
    if not (math.isfinite(diff_sq) and math.isfinite(cand_sq) and math.isfinite(ref_sq)):
        return RowStats(math.inf, math.inf, -math.inf, max_ref if math.isfinite(max_ref) else math.inf)
    dot = sum(a * b for a, b in zip(cand, ref, strict=True))
    if cand_sq == 0.0 or ref_sq == 0.0:
        cosine = 1.0 if cand_sq == ref_sq else 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / (math.sqrt(cand_sq) * math.sqrt(ref_sq))))
    return RowStats(max_abs, ratio(math.sqrt(diff_sq), math.sqrt(ref_sq)), cosine, max_ref)
