"""The per-layer evaluator (build plan R-011): candidate captures against a reference run, per case and per layer.

Checks, each with an explicit status (`passed`, `failed`, `not_run`) and the tolerance key that binds it:

    layers.hidden            every stage output (embeddings, each decoder layer, final norm) at every position of
                             every forward: max abs difference                          float.layer_hidden_max_abs
    layers.recurrent_state   Gated DeltaNet recurrent state after each captured forward float.recurrent_state_max_abs
    layers.conv_state        Gated DeltaNet convolution state, likewise                 float.conv_state_max_abs
    layers.nan_or_inf        a non-finite candidate value anywhere                      float.nan_or_inf

A check with no tolerance key is `not_run`, as is one whose inputs are absent (no candidate capture for a case, no
candidate state tensors); `not_run` never counts as a pass. Measurements are reported whether or not a bound exists.

Attribution: for each case the first divergent stage is the earliest forward in which any stage exceeds the hidden-state
bound, and within that forward the stage that comes first in model order, at its first exceeding position. An error
in layer k reaches every later layer in the same forward, so the earliest stage in model order is where it entered.
Recurrent and convolution state are attributed the same way, to the earliest forward and then the lowest layer.

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

from roost_evidence import generation_reference, layer_captures, stage_results
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

REPORT_FORMAT = "roost.evaluation-report"
TASK = "layers"
HIDDEN_KEY = "float.layer_hidden_max_abs"
RECURRENT_KEY = "float.recurrent_state_max_abs"
CONV_KEY = "float.conv_state_max_abs"
NAN_KEY = "float.nan_or_inf"
KINDS = ("recurrent_state", "conv_state")


def row_max_abs(candidate: array, reference: array, width: int) -> list[float]:
    """Max abs difference per row of `width` values; a row with any non-finite difference reports infinity."""
    out: list[float] = []
    for start in range(0, len(candidate), width):
        pairs = zip(candidate[start : start + width], reference[start : start + width], strict=True)
        diffs = [abs(a - b) for a, b in pairs]
        out.append(max(diffs) if math.isfinite(sum(diffs)) else math.inf)
    return out


def _finite(values: array) -> bool:
    return math.isfinite(sum(values))


@dataclass
class _Case:
    """Measurements for one case."""

    hidden_max: dict[str, float] = field(default_factory=dict)
    first_hidden: dict[str, Any] | None = None
    state_max: dict[str, dict[str, float]] = field(default_factory=lambda: {k: {} for k in KINDS})
    first_state: dict[str, dict[str, Any] | None] = field(default_factory=lambda: dict.fromkeys(KINDS))
    nonfinite: list[str] = field(default_factory=list)
    state_missing: list[str] = field(default_factory=list)


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


def _compare_case(name: str, candidate: Capture, reference: Capture, bounds: dict[str, float | None]) -> _Case:
    _compatible(name, candidate, reference)
    result = _Case()
    hidden_bound = bounds["hidden"]
    for forward in range(reference.forwards):
        for stage in reference.stages:
            key = layer_captures.hidden_name(forward, stage)
            cand = candidate.values(key)
            if not _finite(cand):
                result.nonfinite.append(key)
            rows = row_max_abs(cand, reference.values(key), reference.hidden_size)
            worst = max(rows)
            result.hidden_max[stage] = max(result.hidden_max.get(stage, 0.0), worst)
            if result.first_hidden is None and hidden_bound is not None and worst > hidden_bound:
                position = next(i for i, v in enumerate(rows) if v > hidden_bound)
                result.first_hidden = {
                    "forward": forward,
                    "stage": stage,
                    "position": position,
                    "max_abs": rows[position],
                }
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
                (worst,) = row_max_abs(cand, reference.values(key), max(len(cand), 1))
                stage = layer_captures.layer_stage(layer)
                table = result.state_max[kind]
                table[stage] = max(table.get(stage, 0.0), worst)
                bound = bounds[kind]
                if result.first_state[kind] is None and bound is not None and worst > bound:
                    result.first_state[kind] = {"forward": forward, "stage": stage, "max_abs": worst}
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
    hidden = checks.check("layers.hidden", HIDDEN_KEY)
    recurrent = checks.check("layers.recurrent_state", RECURRENT_KEY)
    conv = checks.check("layers.conv_state", CONV_KEY)
    nan = checks.check("layers.nan_or_inf", NAN_KEY)
    bounds = {
        "hidden": checks.number(hidden),
        "recurrent_state": checks.number(recurrent),
        "conv_state": checks.number(conv),
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
    for c, kind in ((hidden, "hidden"), (recurrent, "recurrent_state"), (conv, "conv_state")):
        tables = (r.hidden_max if kind == "hidden" else r.state_max[kind] for r in results.values())
        c.metrics = {
            "max_abs": max((v for table in tables for v in table.values()), default=0.0),
            "compared": len(results),
        }
        if bounds[kind] is None:
            continue
        failures: list[str] = []
        incomplete = list(lacking)
        for name, r in results.items():
            first = r.first_hidden if kind == "hidden" else r.first_state[kind]
            if first is not None:
                where = f"forward {first['forward']} {first['stage']}"
                if kind == "hidden":
                    where += f" position {first['position']}"
                failures.append(f"{name}: first divergent {where} ({first['max_abs']:.3g})")
            if kind != "hidden":
                incomplete += [f"{name}: no {k}" for k in r.state_missing if k.endswith("/" + kind)]
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
        "cases": {
            name: {
                "first_divergent": r.first_hidden,
                "first_divergent_recurrent_state": r.first_state["recurrent_state"],
                "first_divergent_conv_state": r.first_state["conv_state"],
                "hidden_max_abs": r.hidden_max,
                "recurrent_state_max_abs": r.state_max["recurrent_state"],
                "conv_state_max_abs": r.state_max["conv_state"],
            }
            for name, r in results.items()
        },
        "missing_cases": absent,
        "inputs": {
            **digests,
            reference_path.name: stage_results.sha256_file(reference_path),
            tolerances_path.name: stage_results.sha256_file(tolerances_path),
        },
    }
