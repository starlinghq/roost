"""The evaluator on the engine gate rows (R-039): the stage results Starlings.Roost.EngineGates.Tests writes on CpuOps.

data/engine-gates/<model>/ is a sample run of that harness (regenerate it with ROOST_ENGINE_GATES_OUT pointing at
data/engine-gates; a C# test checks the sample still has the harness's shape). The gates compare the engine with
itself, so there is no reference run of the tiny models: each test builds a reference from the sample's own baseline
generation to exercise the sections and the per-case streaming and usage checks. Nothing here is fidelity evidence.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from roost_evidence import evaluate_generation, stage_results

DATA = Path(__file__).parent / "data" / "engine-gates"
MODELS = ("roost-tiny", "roost-tiny-hybrid")


def self_reference(run: stage_results.Run, path: Path) -> Path:
    """A reference.jsonl whose rows repeat the run's own baseline generation (margins are placeholders)."""
    lines = []
    for name, case in run.dotnet.cases.items():
        lines.append(
            json.dumps(
                {
                    "id": name,
                    "rendered": run.template.cases[name]["rendered"],
                    "prompt_ids": case["prompt_ids"],
                    "output_ids": case["output_ids"],
                    "finish_reason": case["finish_reason"],
                    "step_margin": [1.0] * len(case["output_ids"]),
                    "step0_top": case["step0_top"],
                    "step0_vocab": 256,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tolerances(profile: str, path: Path) -> Path:
    document: dict[str, Any] = {
        "profile": profile,
        "provider_options": {},
        "discrete": {
            "template_rendering": "exact",
            "prompt_ids": "exact",
            "greedy_ids": "exact",
            "finish_reason": "equal",
            "step0_argmax": "equal",
            "streaming_reassembly": "exact",
            "usage": "exact",
            "concurrent_requests": "equal_to_sequential",
        },
        "float": {
            "step0_top5_logprob_max_abs": 1e-6,
            "cached_vs_uncached_logprob_max_abs": 1e-3,
            "chunked_vs_one_call_prefill_logprob_max_abs": 1e-3,
            "nan_or_inf": "fail",
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class Gate:
    def __init__(self, model: str, tmp_path: Path) -> None:
        self.results = tmp_path / model
        shutil.copytree(DATA / model, self.results)
        run = stage_results.load_run(self.results)
        self.profile = run.profile
        self.dotnet = copy.deepcopy(run.dotnet.document)
        self.reference = self_reference(run, tmp_path / "reference.jsonl")
        self.tolerances = tolerances(run.profile, tmp_path / "tolerances.json")

    def checks(self) -> dict[str, dict[str, Any]]:
        stage_results.write_stage_file(self.results, self.dotnet)
        report = evaluate_generation.evaluate(self.results, self.reference, self.tolerances)
        return {c["id"]: c for c in report["checks"]}


@pytest.mark.parametrize("model", MODELS)
def test_the_gate_rows_are_valid_format_v3(model: str) -> None:
    run = stage_results.load_run(DATA / model)
    assert run.profile.startswith(f"{model}/")
    sections = run.dotnet.document["sections"]
    assert set(sections) == {"cached_vs_uncached", "chunked_prefill", "concurrency"}
    assert set(sections["concurrency"]["results"]) == set(run.dotnet.cases)


@pytest.mark.parametrize("model", MODELS)
def test_the_evaluator_passes_the_gate_sections_and_streaming(model: str, tmp_path: Path) -> None:
    checks = Gate(model, tmp_path).checks()
    for check in (
        "sections.cached_vs_uncached",
        "sections.concurrent_requests",
        "generation.streaming_reassembly",
        "generation.usage",
        "generation.greedy_ids",
        "generation.finish_reason",
        "generation.step0_argmax",
        "generation.step0_top5_logprob",
        "generation.nan_or_inf",
        "template.rendering",
        "template.prompt_ids",
    ):
        assert checks[check]["status"] == "passed", (check, checks[check]["detail"])
    # The evaluator reads no chunked_prefill check yet (stream B2); it is reported, never passed.
    assert checks["sections.chunked_prefill"]["status"] == "not_run"


@pytest.mark.parametrize("model", MODELS)
def test_a_failed_gate_fails_its_check(model: str, tmp_path: Path) -> None:
    gate = Gate(model, tmp_path)
    gate.dotnet["sections"]["concurrency"]["results"]["long"] = False
    gate.dotnet["sections"]["cached_vs_uncached"]["tokens_equal"] = False
    gate.dotnet["cases"]["short"]["stream_ids_equal_result"] = False
    checks = gate.checks()
    assert checks["sections.concurrent_requests"]["status"] == "failed"
    assert checks["sections.cached_vs_uncached"]["status"] == "failed"
    assert checks["generation.streaming_reassembly"]["status"] == "failed"
