"""The generation evaluator on synthetic references and candidates (no torch, no transformers)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from roost_evidence import cli, evaluate_generation, stage_results
from synthetic import EOS, PROFILE, reference_cases, write_reference, write_run, write_tolerances

Mutate = Callable[[dict[str, Any]], None]


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.cases = reference_cases()
        self.results = tmp_path / "results"
        self.reference = write_reference(tmp_path / "reference", self.cases)
        self.tolerances = write_tolerances(tmp_path / "limits.json")
        self.run = write_run(self.results, self.cases)

    def rewrite(self, stage: str) -> None:
        stage_results.write_stage_file(self.results, self.run[stage])

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return evaluate_generation.evaluate(self.results, self.reference, self.tolerances, **kwargs)

    def checks(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return {c["id"]: c for c in self.evaluate(**kwargs)["checks"]}


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def test_a_matching_run_passes_every_check(setup: Setup) -> None:
    report = setup.evaluate()
    statuses = {c["id"]: c["status"] for c in report["checks"]}
    # Everything passes except the checks that need a reference step-0 vector or the unread chunked section.
    assert statuses["generation.step0_logprob"] == "not_run"
    assert statuses["sections.chunked_prefill"] == "not_run"
    assert {s for i, s in statuses.items() if i not in ("generation.step0_logprob", "sections.chunked_prefill")} == {
        "passed"
    }
    assert report["profile"] == PROFILE
    assert report["status"] == "not_run"
    assert report["divergences"] == []


def test_reference_vectors_settle_the_full_vocabulary_check(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    write_reference(tmp_path / "reference", setup.cases, vectors=True)
    setup.tolerances = write_tolerances(
        tmp_path / "limits.json", {"float.chunked_vs_one_call_prefill_logprob_max_abs": "not_applicable"}
    )
    report = setup.evaluate(reference_vectors_dir=tmp_path / "reference")
    assert report["status"] == "passed"
    full = next(c for c in report["checks"] if c["id"] == "generation.step0_logprob")
    assert full["status"] == "passed"
    assert full["metrics"]["max_abs"] < 1e-6


def test_full_vocabulary_difference_over_the_bound_fails(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    write_reference(tmp_path / "reference", setup.cases, vectors=True)
    vectors = setup.run["vectors"]
    vectors["case-1"][3] += 0.5  # a non-top id, so the top-5 and argmax checks do not see it
    spec = stage_results.write_sidecar(setup.results, "step0.f32", vectors)
    setup.run["dotnet"]["tensors"]["step0_logprobs"] = spec
    setup.rewrite("dotnet")
    checks = setup.checks(reference_vectors_dir=tmp_path / "reference")
    assert checks["generation.step0_logprob"]["status"] == "failed"
    assert "case-1" in checks["generation.step0_logprob"]["detail"]


def test_reference_vector_digest_mismatch_is_an_input_error(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    write_reference(tmp_path / "reference", setup.cases, vectors=True)
    (tmp_path / "reference" / "step0" / "case-0.f32").write_bytes(b"\x00" * 64)
    with pytest.raises(evaluate_generation.InputError, match="sha256"):
        setup.evaluate(reference_vectors_dir=tmp_path / "reference")


# Greedy ids and near-tie divergences ----------------------------------------------------------------------


def diverge(setup: Setup, name: str, step: int) -> None:
    ids = setup.run["dotnet"]["cases"][name]["output_ids"]
    ids[step] = (ids[step] + 1) % EOS
    setup.rewrite("dotnet")


def test_a_divergence_below_the_margin_is_allowed_and_listed(setup: Setup) -> None:
    setup.cases["case-0"]["step_margin"][2] = 5e-4
    write_reference(setup.tmp / "reference", setup.cases)
    diverge(setup, "case-0", 2)
    report = setup.evaluate()
    greedy = next(c for c in report["checks"] if c["id"] == "generation.greedy_ids")
    assert greedy["status"] == "passed"
    assert "case-0@2" in greedy["detail"]
    [entry] = report["divergences"]
    assert entry["id"] == "case-0"
    assert entry["step"] == 2
    assert entry["reference_margin"] == 5e-4
    assert entry["allowed"] is True
    assert entry["candidate_token"] != entry["reference_token"]


def test_a_divergence_at_a_large_margin_fails_with_step_and_margin(setup: Setup) -> None:
    diverge(setup, "case-1", 3)  # reference margin at step 3 is 1.25
    report = setup.evaluate()
    greedy = next(c for c in report["checks"] if c["id"] == "generation.greedy_ids")
    assert greedy["status"] == "failed"
    assert "case-1 diverges at step 3 (reference margin 1.25)" in greedy["detail"]
    assert report["status"] == "failed"
    assert report["divergences"][0]["allowed"] is False


def test_the_margin_bound_is_strict(setup: Setup) -> None:
    setup.cases["case-0"]["step_margin"][2] = 1e-3
    write_reference(setup.tmp / "reference", setup.cases)
    diverge(setup, "case-0", 2)
    assert setup.checks()["generation.greedy_ids"]["status"] == "failed"


def test_exact_rule_allows_no_divergence(setup: Setup) -> None:
    setup.cases["case-0"]["step_margin"][2] = 1e-9
    write_reference(setup.tmp / "reference", setup.cases)
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"discrete.greedy_ids": "exact"})
    diverge(setup, "case-0", 2)
    assert setup.checks()["generation.greedy_ids"]["status"] == "failed"


def test_equal_until_first_divergence_rule(setup: Setup) -> None:
    setup.cases["case-0"]["step_margin"][2] = 0.01
    write_reference(setup.tmp / "reference", setup.cases)
    setup.tolerances = write_tolerances(
        setup.tmp / "limits.json",
        {"discrete.greedy_ids": "equal_until_first_divergence_at_margin_below_0.05"},
    )
    diverge(setup, "case-0", 2)
    assert setup.checks()["generation.greedy_ids"]["status"] == "passed"


def test_a_candidate_that_stops_early_diverges_at_its_length(setup: Setup) -> None:
    del setup.run["dotnet"]["cases"]["case-0"]["output_ids"][-2:]
    setup.rewrite("dotnet")
    report = setup.evaluate()
    [entry] = report["divergences"]
    assert entry["step"] == len(setup.cases["case-0"]["output_ids"]) - 2
    assert entry["candidate_token"] is None
    assert entry["allowed"] is False


def test_unknown_greedy_rule_is_not_run(setup: Setup) -> None:
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"discrete.greedy_ids": "sloppy"})
    check = setup.checks()["generation.greedy_ids"]
    assert check["status"] == "not_run"
    assert "not implemented" in check["detail"]


def test_generating_from_other_prompt_ids_fails(setup: Setup) -> None:
    setup.run["dotnet"]["cases"]["case-2"]["prompt_ids"] = [9, 9]
    setup.rewrite("dotnet")
    check = setup.checks()["generation.greedy_ids"]
    assert check["status"] == "failed"
    assert "case-2" in check["detail"]


# Finish reason, step 0, streaming, usage ----------------------------------------------------------------


def test_finish_reason_difference_fails(setup: Setup) -> None:
    setup.run["dotnet"]["cases"]["case-0"]["finish_reason"] = "length"
    setup.rewrite("dotnet")
    assert setup.checks()["generation.finish_reason"]["status"] == "failed"


def test_finish_reason_may_differ_where_ids_diverged_when_the_rule_says_so(setup: Setup) -> None:
    setup.cases["case-0"]["step_margin"][1] = 1e-4
    write_reference(setup.tmp / "reference", setup.cases)
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"discrete.finish_reason": "equal_when_ids_equal"})
    diverge(setup, "case-0", 1)
    setup.run["dotnet"]["cases"]["case-0"]["finish_reason"] = "length"
    setup.rewrite("dotnet")
    assert setup.checks()["generation.finish_reason"]["status"] == "passed"


def test_step0_argmax_difference_fails(setup: Setup) -> None:
    vectors = setup.run["vectors"]
    best = setup.cases["case-0"]["step0_top"][0][0]
    vectors["case-0"][best] = -30.0
    setup.run["dotnet"]["tensors"]["step0_logprobs"] = stage_results.write_sidecar(setup.results, "step0.f32", vectors)
    setup.rewrite("dotnet")
    checks = setup.checks()
    assert checks["generation.step0_argmax"]["status"] == "failed"
    assert checks["generation.step0_top5_logprob"]["status"] == "failed"


def test_nan_in_the_sidecar_fails(setup: Setup) -> None:
    vectors = setup.run["vectors"]
    vectors["case-1"][4] = float("nan")
    setup.run["dotnet"]["tensors"]["step0_logprobs"] = stage_results.write_sidecar(setup.results, "step0.f32", vectors)
    setup.rewrite("dotnet")
    check = setup.checks()["generation.nan_or_inf"]
    assert check["status"] == "failed"
    assert "case-1" in check["detail"]


def test_without_a_sidecar_the_candidates_best_pairs_are_checked(tmp_path: Path) -> None:
    cases = reference_cases()
    reference = write_reference(tmp_path / "reference", cases)
    tolerances = write_tolerances(tmp_path / "limits.json")
    write_run(tmp_path / "results", cases, sidecar=False)
    report = evaluate_generation.evaluate(tmp_path / "results", reference, tolerances)
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["generation.step0_argmax"]["status"] == "passed"
    assert checks["generation.step0_top5_logprob"]["status"] == "passed"
    assert checks["generation.step0_logprob"]["status"] == "not_run"
    assert "no step-0 sidecar" in checks["generation.step0_logprob"]["detail"]


def test_top5_over_the_bound_fails_without_a_sidecar(tmp_path: Path) -> None:
    cases = reference_cases()
    reference = write_reference(tmp_path / "reference", cases)
    tolerances = write_tolerances(tmp_path / "limits.json")
    run = write_run(tmp_path / "results", cases, sidecar=False)
    run["dotnet"]["cases"]["case-0"]["step0_top"][2][1] -= 0.1
    stage_results.write_stage_file(tmp_path / "results", run["dotnet"])
    report = evaluate_generation.evaluate(tmp_path / "results", reference, tolerances)
    top5 = next(c for c in report["checks"] if c["id"] == "generation.step0_top5_logprob")
    assert top5["status"] == "failed"
    assert report["status"] == "failed"


def test_a_stream_flag_that_is_false_fails(setup: Setup) -> None:
    setup.run["dotnet"]["cases"]["case-1"]["stream_text_equals_decode"] = False
    setup.rewrite("dotnet")
    check = setup.checks()["generation.streaming_reassembly"]
    assert check["status"] == "failed"
    assert "case-1" in check["detail"]


def test_a_missing_stream_flag_is_not_run(setup: Setup) -> None:
    del setup.run["dotnet"]["cases"]["case-1"]["stream_ids_equal_result"]
    setup.rewrite("dotnet")
    assert setup.checks()["generation.streaming_reassembly"]["status"] == "not_run"


def test_usage_excludes_the_stop_token(setup: Setup) -> None:
    case = setup.run["dotnet"]["cases"]["case-0"]
    assert case["finish_reason"] == "eos"
    case["usage"]["completion_tokens"] += 1
    setup.rewrite("dotnet")
    check = setup.checks()["generation.usage"]
    assert check["status"] == "failed"
    assert "case-0" in check["detail"]


def test_provider_options_must_equal_the_tolerance_file(setup: Setup) -> None:
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"provider_options": {"use_tf32": 0}})
    assert setup.checks()["environment.provider_options"]["status"] == "failed"


# Template ----------------------------------------------------------------------------------------------------


def test_rendering_difference_fails(setup: Setup) -> None:
    setup.run["template"]["cases"]["case-2"]["rendered"] += " "
    setup.rewrite("template")
    check = setup.checks()["template.rendering"]
    assert check["status"] == "failed"
    assert "case-2" in check["detail"]


def test_prompt_id_difference_fails(setup: Setup) -> None:
    setup.run["template"]["cases"]["case-2"]["prompt_ids"] = [1]
    setup.rewrite("template")
    assert setup.checks()["template.prompt_ids"]["status"] == "failed"


def test_a_rejected_render_fails_and_names_the_error(setup: Setup) -> None:
    setup.run["template"]["cases"]["case-0"] = {"rendered": None, "render_error": "boom", "prompt_ids": None}
    setup.rewrite("template")
    check = setup.checks()["template.rendering"]
    assert check["status"] == "failed"
    assert "boom" in check["detail"]


# Missing cases and foreign cases ---------------------------------------------------------------------------


def test_a_missing_case_fails_the_checks_that_need_it(setup: Setup) -> None:
    del setup.run["dotnet"]["cases"]["case-2"]
    del setup.run["dotnet"]["sections"]["concurrency"]["results"]["case-2"]
    setup.run["dotnet"]["tensors"]["step0_logprobs"] = stage_results.write_sidecar(
        setup.results, "step0.f32", {k: v for k, v in setup.run["vectors"].items() if k != "case-2"}
    )
    setup.rewrite("dotnet")
    checks = setup.checks()
    for check_id in (
        "generation.greedy_ids",
        "generation.finish_reason",
        "generation.step0_argmax",
        "generation.usage",
    ):
        assert checks[check_id]["status"] == "failed", check_id
        assert "lacks case case-2" in checks[check_id]["detail"]
    assert checks["template.rendering"]["status"] == "passed"


def test_a_case_that_is_not_in_the_reference_is_an_input_error(setup: Setup) -> None:
    setup.run["dotnet"]["cases"]["extra"] = setup.run["dotnet"]["cases"]["case-0"]
    setup.rewrite("dotnet")
    with pytest.raises(evaluate_generation.InputError, match="extra"):
        setup.evaluate()


def test_a_section_naming_a_foreign_case_is_an_input_error(setup: Setup) -> None:
    setup.run["dotnet"]["sections"]["cached_vs_uncached"]["case"] = "elsewhere"
    setup.rewrite("dotnet")
    with pytest.raises(evaluate_generation.InputError, match="elsewhere"):
        setup.evaluate()


# Sections ----------------------------------------------------------------------------------------------------


def test_cached_vs_uncached_over_the_bound_fails(setup: Setup) -> None:
    setup.run["dotnet"]["sections"]["cached_vs_uncached"]["max_abs_logprob"] = 0.01
    setup.rewrite("dotnet")
    check = setup.checks()["sections.cached_vs_uncached"]
    assert check["status"] == "failed"
    assert "0.01" in check["detail"]


def test_cached_vs_uncached_with_different_tokens_fails(setup: Setup) -> None:
    setup.run["dotnet"]["sections"]["cached_vs_uncached"]["tokens_equal"] = False
    setup.rewrite("dotnet")
    assert setup.checks()["sections.cached_vs_uncached"]["status"] == "failed"


def test_a_missing_cached_vs_uncached_section_is_not_run(setup: Setup) -> None:
    del setup.run["dotnet"]["sections"]["cached_vs_uncached"]
    setup.rewrite("dotnet")
    check = setup.checks()["sections.cached_vs_uncached"]
    assert check["status"] == "not_run"
    assert "no cached_vs_uncached section" in check["detail"]


def test_a_concurrent_request_that_differs_fails(setup: Setup) -> None:
    setup.run["dotnet"]["sections"]["concurrency"]["results"]["case-1"] = False
    setup.rewrite("dotnet")
    check = setup.checks()["sections.concurrent_requests"]
    assert check["status"] == "failed"
    assert "case-1" in check["detail"]


# Tolerances --------------------------------------------------------------------------------------------------


def test_a_missing_tolerance_key_is_not_run_and_blocks_the_verdict(setup: Setup) -> None:
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"float.cached_vs_uncached_logprob_max_abs": None})
    check = setup.checks()["sections.cached_vs_uncached"]
    assert check["status"] == "not_run"
    assert "no tolerance for float.cached_vs_uncached_logprob_max_abs" in check["detail"]


def test_not_applicable_is_recorded_and_does_not_block(setup: Setup) -> None:
    overrides = {
        "float.chunked_vs_one_call_prefill_logprob_max_abs": "not_applicable",
        "float.step0_logprob_max_abs": "not_applicable",
    }
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", overrides)
    report = setup.evaluate()
    assert report["status"] == "passed"
    assert next(c for c in report["checks"] if c["id"] == "generation.step0_logprob")["status"] == "not_applicable"


def test_unknown_tolerance_keys_are_an_input_error(setup: Setup) -> None:
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"float.surprise": 1.0})
    with pytest.raises(evaluate_generation.InputError, match="float.surprise"):
        setup.evaluate()


def test_tolerances_for_another_profile_are_an_input_error(setup: Setup) -> None:
    setup.tolerances = write_tolerances(setup.tmp / "limits.json", {"profile": "synthetic/other"})
    with pytest.raises(evaluate_generation.InputError, match="synthetic/other"):
        setup.evaluate()


def test_flat_dotted_tolerance_keys_are_accepted(setup: Setup) -> None:
    setup.tolerances.write_text(json.dumps({"profile": PROFILE, "discrete.prompt_ids": "exact"}), encoding="utf-8")
    checks = setup.checks()
    assert checks["template.prompt_ids"]["status"] == "passed"
    assert checks["template.rendering"]["status"] == "not_run"


def test_yaml_tolerances_are_accepted(setup: Setup) -> None:
    path = setup.tmp / "limits.yaml"
    path.write_text(f"profile: {PROFILE}\ndiscrete:\n  prompt_ids: exact\n", encoding="utf-8")
    setup.tolerances = path
    assert setup.checks()["template.prompt_ids"]["status"] == "passed"


def test_first_divergence() -> None:
    assert evaluate_generation.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert evaluate_generation.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert evaluate_generation.first_divergence([1, 2, 3], [1, 2]) == 2
    assert evaluate_generation.first_divergence([1, 2], [1, 2, 3]) == 2


# Command line -------------------------------------------------------------------------------------------------


def run_cli(setup: Setup, *extra: str) -> int:
    return cli.main(
        [
            "evaluate",
            str(setup.results),
            "--reference",
            str(setup.reference),
            "--tolerances",
            str(setup.tolerances),
            *extra,
        ]
    )


def test_cli_exit_codes_and_report(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    setup.tolerances = write_tolerances(
        setup.tmp / "limits.json",
        {
            "float.chunked_vs_one_call_prefill_logprob_max_abs": "not_applicable",
            "float.step0_logprob_max_abs": "not_applicable",
        },
    )
    report = setup.tmp / "report.json"
    assert run_cli(setup, "--report", str(report)) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "passed"
    assert capsys.readouterr().out.rstrip().endswith(f"passed: {PROFILE}")

    diverge(setup, "case-1", 3)
    assert run_cli(setup) == 1
    out = capsys.readouterr().out
    assert "divergence     case-1 step 3 reference margin 1.25 (not allowed)" in out


def test_cli_not_run_exits_1(setup: Setup) -> None:
    assert run_cli(setup) == 1


def test_cli_input_error_exits_2(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    (setup.results / "dotnet.results.json").unlink()
    assert run_cli(setup) == 2
    assert "dotnet.results.json" in capsys.readouterr().err
