"""Row-relative and cosine metrics: definitions, bound directions, back compatibility and attribution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from roost_evidence import cli, evaluate_generation, evaluate_layers, metrics, stage_results
from synthetic import reference_cases, write_reference, write_run, write_tolerances
from synthetic_layers import Pair, Tensors

CASE = "case-0"
# Stage scales grow with depth, as in the real model (the final norm is about ten times the layers).
SCALES = {
    "embeddings": 0.5,
    "layer.0": 0.5,
    "layer.1": 0.5,
    "layer.2": 4.0,
    "layer.3": 8.0,
    "final_norm": 40.0,
}
ABSOLUTE = {"float.layer_hidden_max_abs": 2.0}
HIDDEN_RELATIVE = {
    "float.layer_hidden_rel_l2": 0.1,
    "float.layer_hidden_max_abs_rel": 0.05,
    "float.layer_hidden_cosine_min": 0.99,
}
STATE_RELATIVE = {
    f"float.{kind}_{metric}": bound
    for kind in ("recurrent_state", "conv_state")
    for metric, bound in (("rel_l2", 0.1), ("max_abs_rel", 0.05), ("cosine_min", 0.99))
}
ALL_RELATIVE = {**HIDDEN_RELATIVE, **STATE_RELATIVE}


def disturb(pair: Pair, rel: float, *, from_stage: str | None = None, forward: int | None = None) -> dict[str, Tensors]:
    """Candidate tensors with an alternating-sign error of `rel` times the stage's scale added to every value.

    From `from_stage` onward within `forward` (every forward when None), like a defect that enters at one layer and
    reaches everything after it; the states of the first forwards get an error of `rel` as well.
    """
    out = {n: {k: (shape, list(v)) for k, (shape, v) in t.items()} for n, t in pair.tensors.items()}
    stages = json.loads(pair.metadata[CASE]["stages"])
    start = stages.index(from_stage) if from_stage else 0
    for tensors in out.values():
        for key, (shape, values) in tensors.items():
            parts = key.split("/")
            index = int(parts[0].removeprefix("forward"))
            if forward is not None and index != forward:
                continue
            if "state" in key:
                scale = 1.0
            else:
                if stages.index(parts[1]) < start:
                    continue
                scale = pair.scales.get(parts[1], 1.0)
            tensors[key] = (shape, [v + rel * scale * (1 if i % 2 else -1) for i, v in enumerate(values)])
    return out


def run(pair: Pair, tmp_path: Path, tolerances: dict[str, Any]) -> dict[str, Any]:
    limits = pair.tolerance_file(tmp_path, tolerances)
    return evaluate_layers.evaluate(pair.candidate_dir, pair.reference, limits, pair.reference_dir)


def checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in report["checks"]}


# Definitions -----------------------------------------------------------------------------------------------


def test_identical_vectors_are_perfect() -> None:
    stats = metrics.compare([1.0, -2.0, 3.0], [1.0, -2.0, 3.0])
    assert (stats.max_abs, stats.rel_l2, stats.max_ref) == (0.0, 0.0, 3.0)
    assert stats.cosine == pytest.approx(1.0)


def test_relative_error_and_cosine_values() -> None:
    stats = metrics.compare([3.0, 0.0], [4.0, 0.0])  # a shorter vector in the same direction
    assert stats.rel_l2 == pytest.approx(0.25)
    assert stats.cosine == pytest.approx(1.0)
    assert metrics.compare([1.0, 0.0], [0.0, 2.0]).cosine == pytest.approx(0.0)
    assert metrics.compare([-1.0, -2.0], [1.0, 2.0]).cosine == pytest.approx(-1.0)
    assert metrics.compare([0.0, 3.0], [0.0, 4.0]).max_abs == 1.0


def test_zero_and_non_finite_vectors_cannot_look_good() -> None:
    assert metrics.compare([0.0, 0.0], [0.0, 0.0]).rel_l2 == 0.0
    assert metrics.compare([0.0, 0.0], [0.0, 0.0]).cosine == 1.0
    assert metrics.compare([0.1, 0.0], [0.0, 0.0]).rel_l2 == math.inf
    assert metrics.compare([0.1, 0.0], [0.0, 0.0]).cosine == 0.0
    bad = metrics.compare([math.nan, 1.0], [1.0, 1.0])
    assert (bad.max_abs, bad.rel_l2, bad.cosine) == (math.inf, math.inf, -math.inf)
    assert metrics.compare([math.inf, 1.0], [1.0, 1.0]).cosine == -math.inf
    with pytest.raises(ValueError, match="cannot be compared"):
        metrics.compare([1.0], [1.0, 2.0])


def test_centering_ignores_a_constant_offset() -> None:
    reference = [-20.0, -18.5, -25.0, -17.0]
    shifted = [v + 0.7 for v in reference]  # a different log-softmax normaliser
    plain = metrics.compare(shifted, reference)
    centred = metrics.compare(shifted, reference, center=True)
    assert plain.rel_l2 > 0.0
    assert centred.rel_l2 == pytest.approx(0.0, abs=1e-12)
    assert centred.cosine == pytest.approx(1.0)
    assert centred.max_abs == plain.max_abs  # max abs stays that of the vectors as given


def test_the_bound_direction_and_nan() -> None:
    upper, lower = metrics.METRICS[1], metrics.METRICS[3]
    assert not upper.violates(0.1, 0.1) and upper.violates(0.11, 0.1) and upper.violates(math.nan, 0.1)
    assert not lower.violates(0.99, 0.99) and lower.violates(0.98, 0.99) and lower.violates(math.nan, 0.99)


def test_new_keys_are_known_and_documented() -> None:
    for key in ALL_RELATIVE | {f"float.step0_logprob_{m}": 0 for m in ("rel_l2", "max_abs_rel", "cosine_min")}:
        assert key in evaluate_generation.TOLERANCE_KEYS
    assert "LOWER" in evaluate_generation.TOLERANCE_KEYS["float.layer_hidden_cosine_min"]


# The window-defect shape ----------------------------------------------------------------------------------


def test_a_row_relative_check_attributes_what_the_absolute_check_misattributes(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    # The defect enters at layer.1 (absolute error 0.15) and reaches every later stage; scaled by the later layers'
    # magnitudes it is 1.2 at layer.2, 2.4 at layer.3 and 12 at the final norm; relative to the row, 0.3 throughout.
    pair.write_candidate(disturb(pair, 0.3, from_stage="layer.1", forward=0))
    report = run(pair, tmp_path, {**ABSOLUTE, **HIDDEN_RELATIVE})
    by_id = checks(report)
    assert all(
        by_id[i]["status"] == "failed"
        for i in ("layers.hidden", *(f"layers.hidden_{m}" for m in ("rel_l2", "max_abs_rel", "cosine_min")))
    )
    case = report["cases"][CASE]
    # Absolute bound 2.0: layers 1 and 2 are under it, so the defect is blamed on layer.3.
    assert (case["first_divergent"]["forward"], case["first_divergent"]["stage"]) == (0, "layer.3")
    # Every relative form names the layer where it entered.
    for field in ("first_divergent_rel_l2", "first_divergent_max_abs_rel", "first_divergent_cosine_min"):
        assert (case[field]["forward"], case[field]["stage"]) == (0, "layer.1"), field
    assert case["first_divergent_rel_l2"]["position"] == 0
    assert case["first_divergent_max_abs_rel"]["position"] is None
    assert "forward 0 layer.1 position 0" in by_id["layers.hidden_rel_l2"]["detail"]
    assert "forward 0 layer.1" in by_id["layers.hidden_max_abs_rel"]["detail"]
    # The whole-case measurements are per stage and scale free.
    assert case["hidden_rel_l2"]["layer.1"] == pytest.approx(case["hidden_rel_l2"]["final_norm"], rel=0.5)
    assert case["hidden_rel_l2"]["embeddings"] == 0.0
    assert case["hidden_cosine_min"]["layer.0"] == pytest.approx(1.0)
    assert case["hidden_cosine_min"]["layer.1"] < 0.99
    assert report["cases"]["case-1"]["first_divergent_rel_l2"]["stage"] == "layer.1"  # every case carries the defect


def test_benign_noise_passes_every_relative_bound_and_the_absolute_one(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    pair.write_candidate(disturb(pair, 0.01))
    report = run(pair, tmp_path, {**ABSOLUTE, **ALL_RELATIVE})
    assert report["status"] == "passed", [c for c in report["checks"] if c["status"] != "passed"]
    assert report["unconfigured_checks"] == ["layers.recurrent_state", "layers.conv_state"]
    metrics_ = checks(report)["layers.hidden_rel_l2"]["metrics"]
    assert 0.005 < metrics_["max_rel_l2"] < 0.1
    assert 0.999 < checks(report)["layers.hidden_cosine_min"]["metrics"]["min_cosine"] < 1.0


def test_a_cosine_bound_is_a_lower_bound(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    pair.write_candidate(disturb(pair, 0.3, forward=0))  # cosine about 0.97 everywhere
    loose = run(pair, tmp_path, {"float.layer_hidden_cosine_min": 0.9})
    assert loose["status"] == "not_run"  # the state targets name no key, so their absolute checks are not_run
    assert loose["unconfigured_checks"][:3] == ["layers.hidden", "layers.hidden_rel_l2", "layers.hidden_max_abs_rel"]
    assert checks(loose)["layers.hidden_cosine_min"]["status"] == "passed"
    strict = run(pair, tmp_path, {"float.layer_hidden_cosine_min": 0.99})
    assert checks(strict)["layers.hidden_cosine_min"]["status"] == "failed"
    assert strict["cases"][CASE]["first_divergent_cosine_min"]["stage"] == "embeddings"


def test_a_non_finite_candidate_fails_every_relative_metric(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    pair.write_candidate(pair.perturbed(CASE, "forward0/layer.3", float("nan"), positions=1))
    report = run(pair, tmp_path, {**HIDDEN_RELATIVE})
    for metric in ("rel_l2", "max_abs_rel", "cosine_min"):
        assert report["cases"][CASE][f"first_divergent_{metric}"]["stage"] == "layer.3"
    assert report["cases"][CASE]["first_divergent_rel_l2"]["position"] == 1


# State metrics ---------------------------------------------------------------------------------------------


def test_state_metrics_are_attributed_to_the_first_forward_and_layer(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    tensors = pair.perturbed(CASE, "forward1/layer.2/recurrent_state", 0.5)
    tensors = {**tensors, CASE: {**tensors[CASE]}}
    shape, values = tensors[CASE]["forward0/layer.1/conv_state"]
    tensors[CASE]["forward0/layer.1/conv_state"] = (shape, [v + (0.5 if i % 2 else -0.5) for i, v in enumerate(values)])
    pair.write_candidate(tensors)
    report = run(pair, tmp_path, STATE_RELATIVE)
    by_id = checks(report)
    assert by_id["layers.recurrent_state_rel_l2"]["status"] == "failed"
    assert by_id["layers.conv_state_cosine_min"]["status"] == "failed"
    case = report["cases"][CASE]
    first = case["first_divergent_recurrent_state_rel_l2"]
    assert (first["forward"], first["stage"]) == (1, "layer.2") and "position" not in first
    assert case["first_divergent_conv_state_max_abs_rel"]["stage"] == "layer.1"
    assert case["recurrent_state_rel_l2"]["layer.0"] == 0.0
    assert case["conv_state_cosine_min"]["layer.0"] == pytest.approx(1.0)


# Tolerance files --------------------------------------------------------------------------------------------


def test_an_absolute_only_file_behaves_as_before(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    report = run(
        pair,
        tmp_path,
        {k: 1e-4 for k in ("float.layer_hidden_max_abs", "float.recurrent_state_max_abs", "float.conv_state_max_abs")},
    )
    assert [c["id"] for c in report["checks"]] == [
        "layers.hidden",
        "layers.recurrent_state",
        "layers.conv_state",
        "layers.nan_or_inf",
    ]
    assert report["status"] == "passed"
    assert "layers.hidden_rel_l2" in report["unconfigured_checks"]  # listed, not checked, not blocking


def test_a_relative_only_file_needs_no_absolute_keys(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    report = run(pair, tmp_path, {**ALL_RELATIVE, "float.nan_or_inf": "fail"})
    ids = [c["id"] for c in report["checks"]]
    assert "layers.hidden" not in ids and "layers.conv_state" not in ids
    assert report["status"] == "passed"
    assert report["unconfigured_checks"] == ["layers.hidden", "layers.recurrent_state", "layers.conv_state"]


def test_a_family_with_no_key_is_still_not_run(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    report = run(pair, tmp_path, {**HIDDEN_RELATIVE, "float.nan_or_inf": "fail"})
    by_id = checks(report)
    assert by_id["layers.hidden_rel_l2"]["status"] == "passed"
    assert by_id["layers.recurrent_state"]["status"] == "not_run"  # its absolute check, as before
    assert report["status"] == "not_run"


def test_a_metric_marked_not_applicable_does_not_block(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    report = run(pair, tmp_path, {**HIDDEN_RELATIVE, "float.layer_hidden_max_abs": "not_applicable", **STATE_RELATIVE})
    assert checks(report)["layers.hidden"]["status"] == "not_applicable"
    assert report["status"] == "passed"


def test_a_bound_that_is_not_a_number_is_explained(tmp_path: Path) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    report = run(pair, tmp_path, {"float.layer_hidden_cosine_min": "high"})
    assert checks(report)["layers.hidden_cosine_min"]["status"] == "not_run"
    assert "not a number" in checks(report)["layers.hidden_cosine_min"]["detail"]


def test_the_command_prints_each_attribution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pair = Pair(tmp_path, scales=SCALES)
    pair.write_candidate(disturb(pair, 0.3, from_stage="layer.1", forward=0))
    limits = pair.tolerance_file(tmp_path, {**ABSOLUTE, **HIDDEN_RELATIVE})
    args = ["evaluate-layers", str(pair.candidate_dir), "--reference", str(pair.reference)]
    args += ["--reference-captures", str(pair.reference_dir), "--tolerances", str(limits)]
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert f"divergence     {CASE} first at forward 0 layer.3 position 0 (max_abs " in out
    assert f"divergence     {CASE} first at forward 0 layer.1 position 0 (rel_l2 " in out
    assert f"divergence     {CASE} first at forward 0 layer.1 (max_abs_rel " in out


# Step-0 vectors -----------------------------------------------------------------------------------------------

STEP0_RELATIVE = {
    "float.step0_logprob_rel_l2": 0.05,
    "float.step0_logprob_max_abs_rel": 0.05,
    "float.step0_logprob_cosine_min": 0.99,
}


class Step0:
    def __init__(self, tmp_path: Path, tolerances: dict[str, Any]) -> None:
        self.tmp = tmp_path
        self.cases = reference_cases()
        self.results = tmp_path / "results"
        self.reference = write_reference(tmp_path / "reference", self.cases, vectors=True)
        base = {"float.chunked_vs_one_call_prefill_logprob_max_abs": "not_applicable", **tolerances}
        self.tolerances = write_tolerances(tmp_path / "limits.json", base)
        self.run = write_run(self.results, self.cases)

    def break_vector(self, name: str, index: int, delta: float) -> None:
        vectors = self.run["vectors"]
        vectors[name][index] += delta
        self.run["dotnet"]["tensors"][stage_results.STEP0_TENSOR] = stage_results.write_sidecar(
            self.results, "step0.f32", vectors
        )
        stage_results.write_stage_file(self.results, self.run["dotnet"])

    def checks(self, vectors: bool = True) -> dict[str, dict[str, Any]]:
        directory = self.tmp / "reference" if vectors else None
        report = evaluate_generation.evaluate(self.results, self.reference, self.tolerances, directory)
        return {c["id"]: c for c in report["checks"]} | {"report": report}


def test_step0_relative_metrics_pass_on_a_matching_run_and_replace_the_absolute_check(tmp_path: Path) -> None:
    step0 = Step0(tmp_path, {**STEP0_RELATIVE, "float.step0_logprob_max_abs": None})
    by_id = step0.checks()
    assert "generation.step0_logprob" not in by_id
    for metric in ("rel_l2", "max_abs_rel", "cosine_min"):
        assert by_id[f"generation.step0_logprob_{metric}"]["status"] == "passed"
    assert by_id["report"]["status"] == "passed"
    assert by_id["report"]["unconfigured_checks"] == ["generation.step0_logprob"]
    assert by_id["generation.step0_logprob_cosine_min"]["metrics"]["min_cosine"] == pytest.approx(1.0)


def test_step0_relative_metrics_fail_on_a_wrong_non_top_logprob(tmp_path: Path) -> None:
    step0 = Step0(tmp_path, {**STEP0_RELATIVE, "float.step0_logprob_max_abs": None})
    top = {t for t, _ in step0.cases["case-1"]["step0_top"][:5]}
    index = next(i for i in range(16) if i not in top)
    step0.break_vector("case-1", index, 3.0)
    by_id = step0.checks()
    for metric in ("rel_l2", "max_abs_rel", "cosine_min"):
        check = by_id[f"generation.step0_logprob_{metric}"]
        assert check["status"] == "failed" and "case-1" in check["detail"], metric
    assert by_id["report"]["status"] == "failed"


def test_step0_relative_metrics_need_the_reference_vector(tmp_path: Path) -> None:
    step0 = Step0(tmp_path, STEP0_RELATIVE)
    by_id = step0.checks(vectors=False)
    assert by_id["generation.step0_logprob_rel_l2"]["status"] == "not_run"
    assert "no reference step-0 vector" in by_id["generation.step0_logprob_rel_l2"]["detail"]


def test_step0_without_relative_keys_is_unchanged(tmp_path: Path) -> None:
    by_id = Step0(tmp_path, {}).checks()
    assert by_id["generation.step0_logprob"]["status"] == "passed"
    assert not [i for i in by_id if i.startswith("generation.step0_logprob_")]
    assert by_id["report"]["unconfigured_checks"] == []
