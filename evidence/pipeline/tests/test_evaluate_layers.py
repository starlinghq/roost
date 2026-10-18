"""The per-layer evaluator on synthetic captures: attribution, statuses and unusable inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from roost_evidence import cli, evaluate_generation, evaluate_layers, layer_captures
from synthetic_layers import LAYER_TOLERANCES, Pair, make_capture

CASE = "case-0"


def run(pair: Pair, tmp_path: Path, tolerances: dict | None = None) -> dict:
    limits = pair.tolerance_file(tmp_path, tolerances)
    return evaluate_layers.evaluate(pair.candidate_dir, pair.reference, limits, pair.reference_dir)


def check(report: dict, check_id: str) -> dict:
    return next(c for c in report["checks"] if c["id"] == check_id)


def test_identical_captures_pass(tmp_path: Path) -> None:
    report = run(Pair(tmp_path), tmp_path)
    assert report["status"] == "passed", [c for c in report["checks"] if c["status"] != "passed"]
    assert report["missing_cases"] == []
    assert all(c["first_divergent"] is None for c in report["cases"].values())
    assert report["cases"][CASE]["hidden_max_abs"]["layer.2"] == 0.0


def test_a_perturbation_is_attributed_to_its_layer_forward_and_position(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    # Layer 1 of forward 0 is wrong at prompt position 2 only; the layers after it inherit the error.
    tensors = pair.perturbed(CASE, "forward0/layer.1", 0.5, positions=2)
    for later in ("layer.2", "layer.3", "final_norm"):
        tensors = {**tensors, CASE: {**tensors[CASE]}}
        shape, values = tensors[CASE][f"forward0/{later}"]
        tensors[CASE][f"forward0/{later}"] = (shape, [v + 0.25 for v in values])
    pair.write_candidate(tensors)
    report = run(pair, tmp_path)
    assert report["status"] == "failed"
    first = report["cases"][CASE]["first_divergent"]
    assert (first["forward"], first["stage"], first["position"]) == (0, "layer.1", 2)
    assert first["max_abs"] == pytest.approx(0.5, abs=1e-6)
    assert report["cases"]["case-1"]["first_divergent"] is None
    assert "forward 0 layer.1 position 2" in check(report, "layers.hidden")["detail"]
    assert check(report, "layers.recurrent_state")["status"] == "passed"


def test_the_earliest_forward_wins_over_the_earliest_layer(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    tensors = pair.perturbed(CASE, "forward2/layer.0", 1.0)
    tensors[CASE]["forward1/layer.3"] = (
        tensors[CASE]["forward1/layer.3"][0],
        [v + 1.0 for v in tensors[CASE]["forward1/layer.3"][1]],
    )
    pair.write_candidate(tensors)
    first = run(pair, tmp_path)["cases"][CASE]["first_divergent"]
    assert (first["forward"], first["stage"]) == (1, "layer.3")


def test_the_embeddings_can_be_the_first_divergent_stage(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward0/embeddings", 0.1))
    assert run(pair, tmp_path)["cases"][CASE]["first_divergent"]["stage"] == "embeddings"


def test_a_difference_within_the_bound_passes_and_is_measured(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward1/layer.2", 5e-5))
    report = run(pair, tmp_path)
    assert report["status"] == "passed"
    assert report["cases"][CASE]["hidden_max_abs"]["layer.2"] == pytest.approx(5e-5, rel=1e-2)
    tight = run(pair, tmp_path, {**LAYER_TOLERANCES, "float.layer_hidden_max_abs": 1e-5})
    assert tight["status"] == "failed"


def test_recurrent_state_is_attributed_separately(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward1/layer.2/recurrent_state", 0.01))
    report = run(pair, tmp_path)
    assert check(report, "layers.hidden")["status"] == "passed"
    assert check(report, "layers.recurrent_state")["status"] == "failed"
    assert check(report, "layers.conv_state")["status"] == "passed"
    first = report["cases"][CASE]["first_divergent_recurrent_state"]
    assert (first["forward"], first["stage"]) == (1, "layer.2")
    assert report["cases"][CASE]["recurrent_state_max_abs"]["layer.0"] == 0.0


def test_a_conv_state_difference_fails_its_check(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward0/layer.0/conv_state", 0.01))
    report = run(pair, tmp_path)
    assert check(report, "layers.conv_state")["status"] == "failed"
    assert report["cases"][CASE]["first_divergent_conv_state"]["stage"] == "layer.0"


def test_a_non_finite_candidate_fails(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward0/layer.3", float("nan"), positions=0))
    report = run(pair, tmp_path)
    assert check(report, "layers.nan_or_inf")["status"] == "failed"
    assert check(report, "layers.hidden")["status"] == "failed"
    assert report["cases"][CASE]["first_divergent"]["stage"] == "layer.3"


def test_no_tolerance_means_not_run_but_the_measurements_remain(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.write_candidate(pair.perturbed(CASE, "forward0/layer.0", 0.5))
    limits = pair.tolerances  # the generation tolerances carry no layer keys
    report = evaluate_layers.evaluate(pair.candidate_dir, pair.reference, limits, pair.reference_dir)
    assert report["status"] == "not_run"
    assert check(report, "layers.hidden")["status"] == "not_run"
    assert check(report, "layers.hidden")["metrics"]["max_abs"] == pytest.approx(0.5, abs=1e-6)
    assert report["cases"][CASE]["first_divergent"] is None
    assert check(report, "layers.nan_or_inf")["status"] == "passed"


def test_a_missing_candidate_capture_is_not_run(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    layer_captures.capture_path(pair.candidate_dir, "case-1").unlink()
    report = run(pair, tmp_path)
    assert report["status"] == "not_run"
    assert report["missing_cases"] == ["case-1"]
    assert "case-1" in check(report, "layers.hidden")["detail"]


def test_missing_candidate_states_are_not_run(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    tensors = {n: {k: v for k, v in t.items() if "state" not in k} for n, t in pair.tensors.items()}
    metadata = {n: {**m, "state_forwards": "0"} for n, m in pair.metadata.items()}
    pair.metadata = metadata
    pair.write_candidate(tensors)
    report = run(pair, tmp_path)
    assert check(report, "layers.hidden")["status"] == "passed"
    assert check(report, "layers.recurrent_state")["status"] == "not_run"
    assert report["status"] == "not_run"


def test_a_different_layer_count_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    case = pair.cases[CASE]
    metadata, tensors = make_capture(case, seed=3)
    metadata["layer_types"] = json.dumps(["full_attention"] * 3)
    metadata["stages"] = json.dumps(["embeddings", "layer.0", "layer.1", "layer.2", "final_norm"])
    tensors = {k: v for k, v in tensors.items() if "layer.3" not in k and "state" not in k}
    metadata["state_forwards"] = "0"
    layer_captures.write_capture(layer_captures.capture_path(pair.candidate_dir, CASE), metadata, tensors)
    with pytest.raises(evaluate_generation.InputError, match="layers"):
        run(pair, tmp_path)


def test_captures_of_other_tokens_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    pair.metadata[CASE] = {**pair.metadata[CASE], "prompt_ids": json.dumps([9, 9, 9, 9])}
    pair.write_candidate(only=[CASE])
    with pytest.raises(evaluate_generation.InputError, match="prompt and generated tokens"):
        run(pair, tmp_path)


def test_a_candidate_for_an_unknown_case_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    source = layer_captures.capture_path(pair.candidate_dir, CASE)
    source.with_name("stranger.safetensors").write_bytes(source.read_bytes())
    with pytest.raises(evaluate_generation.InputError, match="stranger"):
        run(pair, tmp_path)


def test_a_reference_capture_that_changed_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    path = layer_captures.capture_path(pair.reference_dir, CASE)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(evaluate_generation.InputError, match="layers_sha256"):
        run(pair, tmp_path)


def test_a_reference_run_without_a_capture_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    layer_captures.capture_path(pair.reference_dir, "case-1").unlink()
    with pytest.raises(evaluate_generation.InputError, match="no capture"):
        run(pair, tmp_path)


def test_a_malformed_candidate_cannot_be_evaluated(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    layer_captures.capture_path(pair.candidate_dir, CASE).write_bytes(b"not safetensors at all")
    with pytest.raises(evaluate_generation.InputError):
        run(pair, tmp_path)


def test_the_generation_evaluator_accepts_the_layer_keys(tmp_path: Path) -> None:
    pair = Pair(tmp_path)
    limits = pair.tolerance_file(tmp_path)
    assert evaluate_generation.load_tolerances(limits)["float.layer_hidden_max_abs"] == 1e-4


def test_the_command_reports_and_sets_the_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pair = Pair(tmp_path)
    limits = pair.tolerance_file(tmp_path)
    args = [
        "evaluate-layers",
        str(pair.candidate_dir),
        "--reference",
        str(pair.reference),
        "--reference-captures",
        str(pair.reference_dir),
        "--tolerances",
        str(limits),
        "--report",
        str(tmp_path / "report.json"),
    ]
    assert cli.main(args) == 0
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["status"] == "passed"
    pair.write_candidate(pair.perturbed(CASE, "forward0/layer.2", 0.5, positions=1))
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert f"divergence     {CASE} first at forward 0 layer.2 position 1" in out
    layer_captures.capture_path(pair.reference_dir, CASE).unlink()
    assert cli.main(args) == 2
