"""The per-layer evaluator (build plan R-011): candidate captures against a reference run, per case and per layer.

Targets and checks, each with an explicit status (`passed`, `failed`, `not_run`) and the tolerance key that binds it.
Every target can be bound by up to four metrics (definitions in roost_evidence.metrics):

    layers.hidden            every stage output (embeddings, each decoder layer, final norm) at every position of
                             every forward                                              float.layer_hidden_<metric>
    layers.recurrent_state   Gated DeltaNet recurrent state after each captured forward float.recurrent_state_<metric>
    layers.conv_state        Gated DeltaNet convolution state, likewise                 float.conv_state_<metric>
    layers.nan_or_inf        a non-finite candidate value anywhere                      float.nan_or_inf

    <metric>  max_abs        max |c - r|                                  check id layers.<target>
              rel_l2         ||c - r|| / ||r||, per position (a state is one row)     layers.<target>_rel_l2
              max_abs_rel    max |c - r| / max |r| over one whole tensor              layers.<target>_max_abs_rel
              cosine_min     cosine similarity, a LOWER bound                         layers.<target>_cosine_min

A tolerance file chooses its metrics by naming keys. A target none of whose keys is named keeps its absolute check,
reported `not_run` (an absolute-only file therefore behaves as it always did); once a target names any metric key, only
the named metrics are checked, and the report lists the others under `unconfigured_checks`. A check whose input is
absent (no candidate capture for a case, no candidate state tensors) is `not_run`; `not_run` never counts as a pass.
Measurements of every metric are reported whether or not a bound exists.

Attribution: for each case and metric the first divergent stage is the earliest forward in which any stage violates the
bound, and within that forward the stage that comes first in model order, at its first violating position. An error in
layer k reaches every later layer in the same forward, so the earliest stage in model order is where it entered. An
absolute bound must be loose enough for the largest stage (the final norm is about ten times the layers' scale), so it
can miss a defect in an early layer and name a late one; the relative metrics are scale-free and do not. Recurrent and
convolution state are attributed the same way, to the earliest forward and then the lowest layer.

The candidate must have been teacher-forced on the reference's tokens; captures whose prompt or fed tokens differ, or
whose layer types or stages differ, cannot be compared and raise InputError. Tolerances are an input: the evaluator
never reads or writes evidence/protected/.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roost_evidence import generation_reference, layer_captures, metrics, stage_results
from roost_evidence.evaluate_generation import (
    FAILED,
    NOT_APPLICABLE,
    NOT_RUN,
    PASSED,
    Check,
    InputError,
    load_tolerances,
    overall,
)
from roost_evidence.layer_captures import Capture
from roost_evidence.metrics import ABS, METRICS, Metric, RowStats

REPORT_FORMAT = "roost.evaluation-report"
TASK = "layers"
HIDDEN_KEY = "float.layer_hidden_max_abs"
RECURRENT_KEY = "float.recurrent_state_max_abs"
CONV_KEY = "float.conv_state_max_abs"
NAN_KEY = "float.nan_or_inf"
KINDS = ("recurrent_state", "conv_state")


# target -> (tolerance key prefix, check id)
TARGETS = {
    "hidden": ("layer_hidden", "layers.hidden"),
    "recurrent_state": ("recurrent_state", "layers.recurrent_state"),
    "conv_state": ("conv_state", "layers.conv_state"),
}


def tolerance_key(target: str, metric: Metric) -> str:
    return f"float.{TARGETS[target][0]}_{metric.name}"


def check_id(target: str, metric: Metric) -> str:
    return TARGETS[target][1] + ("" if metric is ABS else "_" + metric.name)


def _first_name(target: str, metric: Metric) -> str:
    """Report field of a first divergence: first_divergent[_<state kind>][_<metric>]; absolute hidden is bare."""
    return (
        "first_divergent" + ("" if target == "hidden" else "_" + target) + ("" if metric is ABS else "_" + metric.name)
    )


def row_stats(candidate: array, reference: array, width: int) -> list[RowStats]:
    """Metrics per row of `width` values; a row with any non-finite candidate value reports infinite errors."""
    return [
        metrics.compare(candidate[start : start + width], reference[start : start + width])
        for start in range(0, len(candidate), width)
    ]


def row_max_abs(candidate: array, reference: array, width: int) -> list[float]:
    """Max abs difference per row of `width` values; a row with any non-finite difference reports infinity."""
    return [r.max_abs for r in row_stats(candidate, reference, width)]


def _finite(values: array) -> bool:
    return math.isfinite(sum(values))


@dataclass
class _Case:
    """Measurements for one case."""

    worst: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    first: dict[tuple[str, str], dict[str, Any] | None] = field(default_factory=dict)
    nonfinite: list[str] = field(default_factory=list)
    state_missing: list[str] = field(default_factory=list)

    def observe(
        self,
        target: str,
        metric: Metric,
        stage: str,
        forward: int,
        values: list[float],
        bound: float | None,
    ) -> None:
        """Fold the per-position (or single, per-tensor) `values` of one stage output into the case's tables."""
        pick = min if metric.lower_is_worse else max
        table = self.worst.setdefault((target, metric.name), {})
        worst = pick(values)
        table[stage] = pick(table.get(stage, worst), worst)
        if self.first.get((target, metric.name)) is not None or bound is None:
            self.first.setdefault((target, metric.name), None)
            return
        index = next((i for i, v in enumerate(values) if metric.violates(v, bound)), None)
        if index is None:
            return
        found: dict[str, Any] = {"forward": forward, "stage": stage, "value": values[index]}
        if target == "hidden":
            found["position"] = None if metric.per_tensor else index
        if metric is ABS:
            found["max_abs"] = values[index]
        self.first[(target, metric.name)] = found


def _compatible(name: str, candidate: Capture, reference: Capture) -> None:
    if candidate.layer_types != reference.layer_types:
        raise InputError(
            f"case {name}: candidate has {len(candidate.layer_types)} layers, reference {len(reference.layer_types)}, "
            "or their layer types differ"
        )
    if candidate.hidden_size != reference.hidden_size:
        raise InputError(f"case {name}: hidden size {candidate.hidden_size} vs reference {reference.hidden_size}")
    if candidate.stages != reference.stages:
        raise InputError(f"case {name}: stages {candidate.stages} vs reference {reference.stages}")
    if candidate.prompt_ids != reference.prompt_ids or candidate.decode_ids != reference.decode_ids:
        raise InputError(f"case {name}: the candidate was not run on the reference's prompt and generated tokens")
    if candidate.state_forwards > reference.state_forwards:
        raise InputError(f"case {name}: candidate captures state for forwards the reference did not")


def _observe_all(
    result: _Case,
    target: str,
    stage: str,
    forward: int,
    rows: list[RowStats],
    bounds: dict[tuple[str, str], float | None],
) -> None:
    """All four metrics of one tensor: per row for the row metrics, one ratio of maxima for max_abs_rel."""
    values = {
        "max_abs": [r.max_abs for r in rows],
        "rel_l2": [r.rel_l2 for r in rows],
        "max_abs_rel": [metrics.ratio(max(r.max_abs for r in rows), max(r.max_ref for r in rows))],
        "cosine_min": [r.cosine for r in rows],
    }
    for metric in METRICS:
        result.observe(target, metric, stage, forward, values[metric.name], bounds[(target, metric.name)])


def _compare_case(
    name: str, candidate: Capture, reference: Capture, bounds: dict[tuple[str, str], float | None]
) -> _Case:
    _compatible(name, candidate, reference)
    result = _Case()
    for forward in range(reference.forwards):
        for stage in reference.stages:
            key = layer_captures.hidden_name(forward, stage)
            cand = candidate.values(key)
            if not _finite(cand):
                result.nonfinite.append(key)
            rows = row_stats(cand, reference.values(key), reference.hidden_size)
            _observe_all(result, "hidden", stage, forward, rows, bounds)
    for forward in range(reference.state_forwards):
        for layer in reference.linear_layers:
            for kind in KINDS:
                key = layer_captures.state_name(forward, layer, kind)
                if key not in candidate.shapes:
                    result.state_missing.append(key)
                    continue
                if candidate.shapes[key] != reference.shapes[key]:
                    raise InputError(
                        f"case {name}: {key} has shape {list(candidate.shapes[key])}, "
                        f"reference {list(reference.shapes[key])}"
                    )
                cand = candidate.values(key)
                if not _finite(cand):
                    result.nonfinite.append(key)
                rows = row_stats(cand, reference.values(key), max(len(cand), 1))
                _observe_all(result, kind, layer_captures.layer_stage(layer), forward, rows, bounds)
    return result


class _Checks:
    def __init__(self, tolerances: dict[str, Any]) -> None:
        self.tol = tolerances
        self.checks: list[Check] = []

    def check(self, check_id: str, key: str) -> Check:
        c = Check(check_id, key)
        self.checks.append(c)
        if key not in self.tol:
            c.detail = f"no tolerance for {key}"
        elif self.tol[key] == NOT_APPLICABLE:
            c.status, c.bound, c.detail = NOT_APPLICABLE, NOT_APPLICABLE, "tolerance file marks it not applicable"
        else:
            c.bound = self.tol[key]
        return c

    @staticmethod
    def active(c: Check) -> bool:
        return c.bound is not None and c.status == NOT_RUN

    @staticmethod
    def settle(c: Check, failures: list[str], incomplete: list[str]) -> None:
        """failed beats not_run beats passed."""
        if failures:
            c.status, c.detail = FAILED, "; ".join(failures[:8])
        elif incomplete:
            c.status, c.detail = NOT_RUN, "; ".join(incomplete[:8])
        else:
            c.status = PASSED

    def number(self, c: Check) -> float | None:
        """The check's numeric bound, or None (after explaining on the check) when it cannot run."""
        if not self.active(c):
            return None
        if not isinstance(c.bound, int | float) or isinstance(c.bound, bool) or not math.isfinite(c.bound):
            c.detail = f"bound {c.bound!r} is not a number"
            return None
        return float(c.bound)


def _case_report(r: _Case) -> dict[str, Any]:
    """Per case: the worst value per stage of every metric, and the first divergence of every metric."""
    report: dict[str, Any] = {}
    for target in TARGETS:
        for metric in METRICS:
            report[_first_name(target, metric)] = r.first.get((target, metric.name))
    for target in TARGETS:
        for metric in METRICS:
            report[f"{target}_{metric.name}"] = r.worst.get((target, metric.name), {})
    return report


def evaluate(
    candidate_dir: Path,
    reference_path: Path,
    tolerances_path: Path,
    reference_captures_dir: Path,
) -> dict[str, Any]:
    """Compare `candidate_dir/layers/<id>.safetensors` with the reference run's captures; return the report.

    Raises InputError for inputs that cannot be evaluated: unreadable or malformed captures, captures that are not
    comparable, a reference case without a capture, or candidate captures for cases the reference lacks.
    """
    try:
        cases = generation_reference.load_reference(reference_path)
    except generation_reference.ReferenceFileError as error:
        raise InputError(str(error)) from error
    checks = _Checks(load_tolerances(tolerances_path))
    metric_checks: dict[tuple[str, str], Check] = {}
    unconfigured: list[str] = []
    for target in TARGETS:
        named = [m for m in METRICS if tolerance_key(target, m) in checks.tol]
        for metric in named or [ABS]:
            metric_checks[(target, metric.name)] = checks.check(check_id(target, metric), tolerance_key(target, metric))
        if named:
            unconfigured += [check_id(target, m) for m in METRICS if m not in named]
    nan = checks.check("layers.nan_or_inf", NAN_KEY)
    bounds: dict[tuple[str, str], float | None] = {
        (target, m.name): checks.number(metric_checks[(target, m.name)]) if (target, m.name) in metric_checks else None
        for target in TARGETS
        for m in METRICS
    }

    layers_dir = candidate_dir / layer_captures.LAYERS_DIR
    present = {p.name.removesuffix(".safetensors") for p in layers_dir.glob("*.safetensors")}
    foreign = sorted(present - set(cases))
    if foreign:
        raise InputError(f"candidate captures name cases that are not in the reference: {foreign}")

    results: dict[str, _Case] = {}
    absent: list[str] = []
    digests: dict[str, str] = {}
    try:
        for name, case in cases.items():
            ref_path = layer_captures.capture_path(reference_captures_dir, name)
            if not ref_path.is_file():
                raise InputError(f"{ref_path}: the reference run has no capture for case {name!r}")
            reference = layer_captures.parse_capture(ref_path)
            expected = case.get("layers_sha256")
            if expected is not None and reference.sha256 != expected:
                raise InputError(f"{ref_path}: sha256 does not match layers_sha256 of case {name!r}")
            if name not in present:
                absent.append(name)
                continue
            candidate = layer_captures.parse_capture(layer_captures.capture_path(candidate_dir, name))
            digests[f"layers/{name}.safetensors"] = candidate.sha256
            results[name] = _compare_case(name, candidate, reference, bounds)
    except layer_captures.CaptureError as error:
        raise InputError(str(error)) from error

    lacking = [f"no candidate capture for case {n}" for n in absent]
    for (target, metric_name), c in metric_checks.items():
        metric = next(m for m in METRICS if m.name == metric_name)
        tables = [r.worst.get((target, metric_name), {}) for r in results.values()]
        pick = min if metric.lower_is_worse else max
        c.metrics = {
            metric.summary: pick(
                (v for table in tables for v in table.values()), default=None if metric.lower_is_worse else 0.0
            ),
            "compared": len(results),
        }
        if bounds[(target, metric_name)] is None:
            continue
        failures: list[str] = []
        incomplete = list(lacking)
        for name, r in results.items():
            first = r.first.get((target, metric_name))
            if first is not None:
                where = f"forward {first['forward']} {first['stage']}"
                if first.get("position") is not None:
                    where += f" position {first['position']}"
                failures.append(f"{name}: first divergent {where} ({first['value']:.3g})")
            if target != "hidden":
                incomplete += [f"{name}: no {k}" for k in r.state_missing if k.endswith("/" + target)]
        checks.settle(c, failures, incomplete)
    if checks.active(nan):
        if nan.bound != "fail":
            nan.detail = f"rule {nan.bound!r} is not implemented"
        else:
            failures = [f"{n}: non-finite {r.nonfinite[:3]}" for n, r in results.items() if r.nonfinite]
            checks.settle(nan, failures, lacking)

    return {
        "format": REPORT_FORMAT,
        "format_version": 1,
        "task": TASK,
        "status": overall(checks.checks),
        "checks": [c.as_json() for c in checks.checks],
        "unconfigured_checks": unconfigured,
        "cases": {name: _case_report(r) for name, r in results.items()},
        "missing_cases": absent,
        "inputs": {
            **digests,
            reference_path.name: stage_results.sha256_file(reference_path),
            tolerances_path.name: stage_results.sha256_file(tolerances_path),
        },
    }
