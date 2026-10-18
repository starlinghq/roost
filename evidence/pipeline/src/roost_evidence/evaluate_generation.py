"""The generation evaluator (build plan R-010): format v3 stage results against a reference run and tolerances.

Checks, each with an explicit status (`passed`, `failed`, `not_run`) and the tolerance key that binds it:

    template      rendering, prompt ids                                     exact against the reference     (G1)
    generation    greedy ids (near-tie rule), finish reason, step-0 argmax, step-0 log-probabilities
                  (reference top 5; whole vocabulary), non-finite values, streaming reassembly, usage       (G2)
    sections      cached vs uncached logits, concurrent requests                                            (G2/G4)

A check with no tolerance key is `not_run`, as is one whose inputs are absent (no sidecar, no reference vectors);
`not_run` never counts as a pass. The report lists every greedy divergence with its step and the reference margin,
whether the rule allows it or not. Tolerances are an input: the evaluator never reads or writes evidence/protected/.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roost_evidence import generation_reference, stage_results
from roost_evidence.manifest import load_text

PASSED, FAILED, NOT_RUN = "passed", "failed", "not_run"
NOT_APPLICABLE = "not_applicable"
REPORT_FORMAT = "roost.evaluation-report"
TOP5 = 5

TOLERANCE_KEYS: dict[str, str] = {
    "profile": "<family>/<profile>, equal to the results' profile",
    "provider_options": "mapping, equal to environment.provider_options of the dotnet stage",
    "discrete.template_rendering": "exact",
    "discrete.prompt_ids": "exact",
    "discrete.greedy_ids": (
        "exact | exact_except_listed_margin_below_<m> | equal_until_first_divergence_at_margin_below_<m>"
    ),
    "discrete.finish_reason": "equal | equal_when_ids_equal",
    "discrete.step0_argmax": "equal",
    "discrete.streaming_reassembly": "exact",
    "discrete.usage": "exact",
    "discrete.concurrent_requests": "equal_to_sequential",
    "float.step0_logprob_max_abs": "number",
    "float.step0_top5_logprob_max_abs": "number",
    "float.cached_vs_uncached_logprob_max_abs": "number",
    "float.chunked_vs_one_call_prefill_logprob_max_abs": "number; no check reads chunked_prefill before R-039",
    "float.nan_or_inf": "fail",
}
_GREEDY_RULE = re.compile(
    r"^(?:exact_except_listed_margin_below_|equal_until_first_divergence_at_margin_below_)"
    r"(?P<m>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$"
)


class InputError(ValueError):
    """The inputs cannot be evaluated (unreadable, malformed, or naming cases the reference lacks)."""


@dataclass
class Check:
    id: str
    key: str
    status: str = NOT_RUN
    bound: Any = None
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "status": self.status,
            "bound": self.bound,
            "detail": self.detail,
            "metrics": self.metrics,
        }


def flatten(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted keys from nested mappings; `provider_options` stays one value."""
    out: dict[str, Any] = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and name != "provider_options":
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def load_tolerances(path: Path) -> dict[str, Any]:
    try:
        document = load_text(path.read_text(encoding="utf-8"), path.suffix.lower())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise InputError(f"{path}: cannot load: {error}") from error
    if not isinstance(document, dict):
        raise InputError(f"{path}: expected a mapping")
    tolerances = flatten(document)
    unknown = sorted(set(tolerances) - set(TOLERANCE_KEYS))
    if unknown:
        raise InputError(f"{path}: unknown tolerance keys {unknown}; known keys: {sorted(TOLERANCE_KEYS)}")
    return tolerances


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    """Index of the first position where the sequences differ; the shorter length when one is a prefix."""
    for step, (a, b) in enumerate(zip(reference, candidate, strict=False)):
        if a != b:
            return step
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate))


def _argmax(pairs: list[tuple[int, float]]) -> int:
    return min(pairs, key=lambda p: (-p[1], p[0]))[0]


class _Evaluation:
    def __init__(
        self,
        run: stage_results.Run,
        reference: dict[str, dict[str, Any]],
        tolerances: dict[str, Any],
        reference_vectors: generation_reference.ReferenceVectors | None,
    ) -> None:
        self.run = run
        self.ref = reference
        self.tol = tolerances
        self.ref_vectors = reference_vectors.vectors if reference_vectors else {}
        self.checks: list[Check] = []
        self.divergences: list[dict[str, Any]] = []
        self.template = run.template.cases
        self.dotnet = run.dotnet.cases
        self.missing_template = [n for n in reference if n not in self.template]
        self.missing_dotnet = [n for n in reference if n not in self.dotnet]

    # Plumbing ---------------------------------------------------------------------------------------------

    def check(self, check_id: str, key: str) -> Check:
        """A new check: not_run without a tolerance key; `not_applicable` is recorded and does not block."""
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

    def missing(self, names: list[str], stage: str) -> list[str]:
        return [f"{stage} stage lacks case {n}" for n in names]

    # Template ---------------------------------------------------------------------------------------------

    def _exact_template(self, check_id: str, key: str, field_name: str, ref_field: str) -> None:
        c = self.check(check_id, key)
        if not self.active(c):
            return
        if c.bound != "exact":
            c.detail = f"rule {c.bound!r} is not implemented"
            return
        failures = self.missing(self.missing_template, "template")
        wrong = []
        for name, ref in self.ref.items():
            case = self.template.get(name)
            if case is not None and case[field_name] != ref[ref_field]:
                error = case.get("render_error")
                wrong.append(f"{name}" + (f" (render error: {error})" if error else ""))
        c.metrics = {"exact": len(self.ref) - len(wrong) - len(self.missing_template), "compared": len(self.ref)}
        self.settle(c, failures + ([f"differs: {wrong}"] if wrong else []), [])

    # Generation -------------------------------------------------------------------------------------------

    def greedy(self) -> None:
        c = self.check("generation.greedy_ids", "discrete.greedy_ids")
        rule = c.bound
        limit: float | None = None
        if rule == "exact":
            limit = 0.0
        elif isinstance(rule, str) and (m := _GREEDY_RULE.match(rule)):
            limit = float(m.group("m"))
        failures = self.missing(self.missing_dotnet, "dotnet")
        exact = 0
        for name, ref in self.ref.items():
            case = self.dotnet.get(name)
            if case is None:
                continue
            if case["prompt_ids"] != ref["prompt_ids"]:
                failures.append(f"{name}: generated from prompt ids other than the reference's")
            step = first_divergence(ref["output_ids"], case["output_ids"])
            if step is None:
                exact += 1
                continue
            margins = ref["step_margin"]
            margin = margins[step] if step < len(margins) else None
            allowed = limit is not None and margin is not None and margin < limit
            self.divergences.append(
                {
                    "id": name,
                    "step": step,
                    "reference_margin": margin,
                    "reference_token": ref["output_ids"][step] if step < len(ref["output_ids"]) else None,
                    "candidate_token": case["output_ids"][step] if step < len(case["output_ids"]) else None,
                    "allowed": allowed,
                }
            )
            if not allowed:
                failures.append(f"{name} diverges at step {step} (reference margin {margin})")
        c.metrics = {"exact": exact, "compared": len(self.ref), "divergences": len(self.divergences)}
        if not self.active(c):
            return
        if limit is None:
            c.detail = f"rule {rule!r} is not implemented"
            return
        self.settle(c, failures, [])
        if c.status == PASSED and self.divergences:
            c.detail = "allowed near-tie divergences: " + ", ".join(
                f"{d['id']}@{d['step']} ({d['reference_margin']:.3g})" for d in self.divergences
            )

    def finish_reason(self) -> None:
        c = self.check("generation.finish_reason", "discrete.finish_reason")
        if not self.active(c):
            return
        if c.bound not in ("equal", "equal_when_ids_equal"):
            c.detail = f"rule {c.bound!r} is not implemented"
            return
        diverged = {d["id"] for d in self.divergences}
        failures = self.missing(self.missing_dotnet, "dotnet")
        for name, ref in self.ref.items():
            case = self.dotnet.get(name)
            if case is None or case["finish_reason"] == ref["finish_reason"]:
                continue
            if c.bound == "equal_when_ids_equal" and name in diverged:
                continue
            failures.append(f"{name}: {case['finish_reason']} vs {ref['finish_reason']}")
        self.settle(c, failures, [])

    def step0(self) -> None:
        nan = self.check("generation.nan_or_inf", "float.nan_or_inf")
        argmax = self.check("generation.step0_argmax", "discrete.step0_argmax")
        top5 = self.check("generation.step0_top5_logprob", "float.step0_top5_logprob_max_abs")
        full = self.check("generation.step0_logprob", "float.step0_logprob_max_abs")
        nan_fail: list[str] = []
        argmax_fail: list[str] = []
        top5_fail: list[str] = []
        top5_gap: list[str] = []
        full_fail: list[str] = []
        full_gap: list[str] = []
        top5_max = full_max = 0.0
        for name, ref in self.ref.items():
            case = self.dotnet.get(name)
            if case is None:
                continue
            vector = self.run.step0.get(name)
            pairs = [(int(t), float(lp)) for t, lp in case["step0_top"]]
            values = vector if vector is not None else [lp for _, lp in pairs]
            if not all(math.isfinite(v) for v in values):
                nan_fail.append(name)
                continue
            ref_top = [(int(t), float(lp)) for t, lp in ref["step0_top"]]
            if vector is not None:
                if len(vector) != ref["step0_vocab"]:
                    raise InputError(
                        f"case {name}: sidecar has {len(vector)} values, reference vocab is {ref['step0_vocab']}"
                    )
                cand = dict(enumerate(vector))
                cand_best = _argmax(list(cand.items()))
            else:
                cand = dict(pairs)
                cand_best = _argmax(pairs)
            if cand_best != ref_top[0][0]:
                argmax_fail.append(f"{name}: {cand_best} vs {ref_top[0][0]}")
            diffs = [(t, abs(cand[t] - lp)) for t, lp in ref_top[:TOP5] if t in cand]
            if len(diffs) < min(TOP5, len(ref_top)):
                top5_gap.append(
                    f"{name}: candidate lists only {len(pairs)} step-0 pairs, none for a reference top-5 id"
                )
            if diffs:
                worst = max(d for _, d in diffs)
                top5_max = max(top5_max, worst)
                if self.active(top5) and isinstance(top5.bound, int | float) and worst > top5.bound:
                    top5_fail.append(f"{name}: {worst:.3g}")
            ref_vector = self.ref_vectors.get(name)
            if vector is None:
                full_gap.append(f"{name}: no step-0 sidecar")
            elif ref_vector is not None:
                worst = max(abs(a - b) for a, b in zip(vector, ref_vector, strict=True))
                full_max = max(full_max, worst)
                if self.active(full) and isinstance(full.bound, int | float) and worst > full.bound:
                    full_fail.append(f"{name}: {worst:.3g}")
            else:
                # Without the reference vector only the reference's best ids are comparable. A value above the bound
                # already fails (the full maximum can only be larger); otherwise the full bound is not measured.
                worst = max(abs(vector[t] - lp) for t, lp in ref_top)
                full_max = max(full_max, worst)
                if self.active(full) and isinstance(full.bound, int | float) and worst > full.bound:
                    full_fail.append(f"{name}: {worst:.3g} on the reference top ids")
                full_gap.append(f"{name}: no reference step-0 vector")
        lacking = self.missing(self.missing_dotnet, "dotnet")
        if self.active(nan):
            if nan.bound != "fail":
                nan.detail = f"rule {nan.bound!r} is not implemented"
            else:
                self.settle(nan, [f"non-finite step-0 values in {nan_fail}"] if nan_fail else lacking, [])
        if self.active(argmax):
            self.settle(argmax, argmax_fail + lacking, [])
        for c, fails, gaps, worst in ((top5, top5_fail, top5_gap, top5_max), (full, full_fail, full_gap, full_max)):
            c.metrics = {"max_abs": worst}
            if self.active(c):
                if not isinstance(c.bound, int | float) or isinstance(c.bound, bool):
                    c.detail = f"bound {c.bound!r} is not a number"
                else:
                    self.settle(c, fails + lacking, gaps)

    def streaming_and_usage(self) -> None:
        stream = self.check("generation.streaming_reassembly", "discrete.streaming_reassembly")
        usage = self.check("generation.usage", "discrete.usage")
        flags = ("stream_text_equals_result", "stream_ids_equal_result", "stream_text_equals_decode")
        lacking = self.missing(self.missing_dotnet, "dotnet")
        stream_fail, stream_gap, usage_fail, usage_gap = [], [], [], []
        for name in self.ref:
            case = self.dotnet.get(name)
            if case is None:
                continue
            absent = [f for f in flags if f not in case]
            if absent:
                stream_gap.append(f"{name}: no {absent}")
            stream_fail += [f"{name}: {f} is false" for f in flags if case.get(f) is False]
            reported = case.get("usage")
            if reported is None:
                usage_gap.append(f"{name}: no usage")
                continue
            stop = 1 if case["finish_reason"] == "eos" and case["output_ids"] else 0
            expected = (len(case["prompt_ids"]), len(case["output_ids"]) - stop)
            if (reported["prompt_tokens"], reported["completion_tokens"]) != expected:
                usage_fail.append(
                    f"{name}: usage {reported['prompt_tokens']}/{reported['completion_tokens']}, expected "
                    f"{expected[0]}/{expected[1]}"
                )
        for c, bound, fails, gaps in (
            (stream, "exact", stream_fail, stream_gap),
            (usage, "exact", usage_fail, usage_gap),
        ):
            if self.active(c):
                if c.bound != bound:
                    c.detail = f"rule {c.bound!r} is not implemented"
                else:
                    self.settle(c, fails + lacking, gaps)

    def provider_options(self) -> None:
        c = self.check("environment.provider_options", "provider_options")
        if self.active(c):
            actual = self.run.dotnet.document.get("environment", {}).get("provider_options", {})
            self.settle(c, [] if actual == c.bound else [f"{actual!r} vs {c.bound!r}"], [])

    # Sections ---------------------------------------------------------------------------------------------

    def sections(self) -> None:
        sections = self.run.dotnet.document["sections"]
        cached = self.check("sections.cached_vs_uncached", "float.cached_vs_uncached_logprob_max_abs")
        part = sections.get("cached_vs_uncached")
        if part is None:
            cached.detail = (cached.detail + "; " if cached.detail else "") + "no cached_vs_uncached section"
        else:
            cached.metrics = dict(part)
            if self.active(cached):
                if not isinstance(cached.bound, int | float) or isinstance(cached.bound, bool):
                    cached.detail = f"bound {cached.bound!r} is not a number"
                else:
                    value = part["max_abs_logprob"]
                    failures = []
                    if not math.isfinite(value) or value > cached.bound:
                        failures.append(f"max {value:.3g} vs {cached.bound:g}")
                    if not part["tokens_equal"]:
                        failures.append("cached and uncached tokens differ")
                    self.settle(cached, failures, [])
        conc = self.check("sections.concurrent_requests", "discrete.concurrent_requests")
        part = sections.get("concurrency")
        if part is None:
            conc.detail = (conc.detail + "; " if conc.detail else "") + "no concurrency section"
        else:
            conc.metrics = {"results": dict(part["results"])}
            if self.active(conc):
                if conc.bound != "equal_to_sequential":
                    conc.detail = f"rule {conc.bound!r} is not implemented"
                else:
                    self.settle(
                        conc, [f"{k} differs from the sequential run" for k, v in part["results"].items() if not v], []
                    )
        chunked = self.check("sections.chunked_prefill", "float.chunked_vs_one_call_prefill_logprob_max_abs")
        if self.active(chunked):
            chunked.detail = "no evaluator check reads chunked_prefill before R-039"

    def foreign_cases(self) -> None:
        sections = self.run.dotnet.document["sections"]
        named: set[str] = set(self.template) | set(self.dotnet)
        if "cached_vs_uncached" in sections:
            named.add(sections["cached_vs_uncached"]["case"])
        if "concurrency" in sections:
            named.update(sections["concurrency"]["results"])
        if "chunked_prefill" in sections:
            named.update(str(k).rsplit("/", 1)[0] for k in sections["chunked_prefill"])
        foreign = sorted(named - set(self.ref))
        if foreign:
            raise InputError(f"results name cases that are not in the reference: {foreign}")

    def run_all(self) -> None:
        self.foreign_cases()
        self._exact_template("template.rendering", "discrete.template_rendering", "rendered", "rendered")
        self._exact_template("template.prompt_ids", "discrete.prompt_ids", "prompt_ids", "prompt_ids")
        self.greedy()
        self.finish_reason()
        self.step0()
        self.streaming_and_usage()
        self.provider_options()
        self.sections()


def overall(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    if FAILED in statuses:
        return FAILED
    return NOT_RUN if NOT_RUN in statuses else PASSED


def evaluate(
    results_dir: Path,
    reference_path: Path,
    tolerances_path: Path,
    reference_vectors_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a run directory and return the report. Raises InputError for inputs that cannot be evaluated."""
    try:
        run = stage_results.load_run(results_dir)
        reference = generation_reference.load_reference(reference_path)
        vectors = (
            generation_reference.load_reference_vectors(reference_vectors_dir, reference)
            if reference_vectors_dir
            else None
        )
    except (stage_results.RunError, generation_reference.ReferenceFileError) as error:
        raise InputError(str(error)) from error
    tolerances = load_tolerances(tolerances_path)
    if "profile" in tolerances and tolerances["profile"] != run.profile:
        raise InputError(f"tolerances are for {tolerances['profile']!r}, results are for {run.profile!r}")
    evaluation = _Evaluation(run, reference, tolerances, vectors)
    evaluation.run_all()
    inputs = {
        **run.files,
        reference_path.name: stage_results.sha256_file(reference_path),
        tolerances_path.name: stage_results.sha256_file(tolerances_path),
    }
    return {
        "format": REPORT_FORMAT,
        "format_version": 1,
        "task": "generation",
        "profile": run.profile,
        "status": overall(evaluation.checks),
        "checks": [c.as_json() for c in evaluation.checks],
        "divergences": evaluation.divergences,
        "inputs": inputs,
    }
