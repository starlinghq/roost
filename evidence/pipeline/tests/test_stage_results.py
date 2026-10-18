"""Stage results format v3: schema, semantic rules and the step-0 sidecar, on synthetic runs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from roost_evidence import stage_results
from synthetic import candidate_documents, reference_cases, write_run


@pytest.fixture
def documents() -> tuple[dict, dict]:
    template, dotnet, _ = candidate_documents(reference_cases())
    return template, dotnet


def messages(document: object) -> list[str]:
    return [f"{p.location}: {p.message}" for p in stage_results.validate_document(document)]


def test_schema_is_a_valid_draft_2020_12_schema() -> None:
    Draft202012Validator.check_schema(stage_results.schema())
    props = stage_results.schema()["properties"]
    assert props["format_version"] == {"const": 3}
    assert props["task"] == {"const": "generation"}


def test_valid_documents_pass(documents: tuple[dict, dict]) -> None:
    assert all(messages(d) == [] for d in documents)


@pytest.mark.parametrize("version", [1, 2, 4])
def test_other_format_versions_are_rejected(documents: tuple[dict, dict], version: int) -> None:
    template, _ = documents
    template["format_version"] = version
    assert any("$.format_version" in m for m in messages(template))


def test_task_must_be_generation(documents: tuple[dict, dict]) -> None:
    template, _ = documents
    template["task"] = "embedding"
    assert any("$.task" in m for m in messages(template))


def test_unknown_stage_is_rejected(documents: tuple[dict, dict]) -> None:
    template, _ = documents
    template["stage"] = "tokenizer"
    assert any("$.stage" in m for m in messages(template))


def test_template_stage_has_no_sections(documents: tuple[dict, dict]) -> None:
    template, dotnet = documents
    template["sections"] = dotnet["sections"]
    assert messages(template)


def test_dotnet_stage_requires_sections(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    del dotnet["sections"]
    assert any("sections" in m for m in messages(dotnet))


def test_dotnet_case_requires_its_fields(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    del dotnet["cases"]["case-0"]["output_ids"]
    assert any("output_ids" in m for m in messages(dotnet))


def test_unknown_finish_reason_is_rejected(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    dotnet["cases"]["case-0"]["finish_reason"] = "stop"
    assert any("finish_reason" in m for m in messages(dotnet))


def test_template_case_may_carry_a_render_error(documents: tuple[dict, dict]) -> None:
    template, _ = documents
    template["cases"]["case-0"] = {"rendered": None, "render_error": "tools are unsupported", "prompt_ids": None}
    assert messages(template) == []


@pytest.mark.parametrize("key", ["tolerances", "Threshold"])
def test_tolerance_keys_are_rejected_anywhere(documents: tuple[dict, dict], key: str) -> None:
    _, dotnet = documents
    dotnet["environment"]["provider_options"] = {key: {"x": 1}}
    assert any("protected tolerance file" in m for m in messages(dotnet))


def test_step0_top_must_be_best_first(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    top = dotnet["cases"]["case-0"]["step0_top"]
    top[0], top[1] = top[1], top[0]
    assert any("best first" in m for m in messages(dotnet))


def test_step0_top_rejects_a_repeated_token(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    top = dotnet["cases"]["case-0"]["step0_top"]
    top[1] = [top[0][0], top[1][1]]
    assert any("more than once" in m for m in messages(dotnet))


def test_concurrency_flags_must_be_booleans(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    dotnet["sections"]["concurrency"]["results"]["case-0"] = "yes"
    assert any("concurrency" in m for m in messages(dotnet))


def test_sections_reject_unknown_names(documents: tuple[dict, dict]) -> None:
    _, dotnet = documents
    dotnet["sections"]["selection"] = {}
    assert messages(dotnet)


def test_tensor_case_must_be_a_case(tmp_path: Path) -> None:
    run = write_run(tmp_path, reference_cases())
    dotnet = copy.deepcopy(run["dotnet"])
    dotnet["tensors"]["step0_logprobs"]["cases"][0]["id"] = "nope"
    assert any("not a case" in m for m in messages(dotnet))


def test_load_run_reads_both_stages_and_the_sidecar(tmp_path: Path) -> None:
    cases = reference_cases()
    write_run(tmp_path, cases)
    run = stage_results.load_run(tmp_path)
    assert run.profile == "synthetic/fp32-cpu"
    assert set(run.step0) == set(cases)
    assert run.step0["case-1"] == pytest.approx(cases["case-1"]["_vector"], abs=1e-6)
    assert set(run.files) == {"template.results.json", "dotnet.results.json", "step0.f32"}


def test_load_run_without_a_sidecar(tmp_path: Path) -> None:
    write_run(tmp_path, reference_cases(), sidecar=False)
    assert stage_results.load_run(tmp_path).step0 == {}


def test_sidecar_digest_mismatch_is_an_error(tmp_path: Path) -> None:
    write_run(tmp_path, reference_cases())
    with (tmp_path / "step0.f32").open("r+b") as handle:
        handle.write(b"\x00\x00\x80\x3f")
    with pytest.raises(stage_results.RunError, match="sha256"):
        stage_results.load_run(tmp_path)


def test_sidecar_size_must_match_the_shapes(tmp_path: Path) -> None:
    run = write_run(tmp_path, reference_cases())
    dotnet = run["dotnet"]
    dotnet["tensors"]["step0_logprobs"]["cases"][0]["shape"] = [5]
    stage_results.write_stage_file(tmp_path, dotnet)
    with pytest.raises(stage_results.RunError, match="bytes"):
        stage_results.load_run(tmp_path)


def test_stage_files_must_name_one_profile(tmp_path: Path) -> None:
    run = write_run(tmp_path, reference_cases())
    run["template"]["profile"] = "synthetic/other"
    stage_results.write_stage_file(tmp_path, run["template"])
    with pytest.raises(stage_results.RunError, match="different profiles"):
        stage_results.load_run(tmp_path)


def test_a_file_in_the_wrong_slot_is_an_error(tmp_path: Path) -> None:
    run = write_run(tmp_path, reference_cases())
    (tmp_path / "template.results.json").write_text(
        (tmp_path / "dotnet.results.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    del run
    with pytest.raises(stage_results.RunError, match="expected 'template'"):
        stage_results.load_run(tmp_path)


def test_duplicate_keys_are_errors(tmp_path: Path) -> None:
    write_run(tmp_path, reference_cases())
    path = tmp_path / "template.results.json"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"task": "generation",', '"task": "generation", "task": "generation",', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(stage_results.RunError, match="duplicate key"):
        stage_results.load_run(tmp_path)
