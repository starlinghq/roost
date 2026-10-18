"""Synthetic generation fixtures: a reference run and a candidate run that matches it, built without torch."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from roost_evidence import stage_results

PROFILE = "synthetic/fp32-cpu"
VOCAB = 16
EOS = 15

TOLERANCES: dict[str, Any] = {
    "profile": PROFILE,
    "provider_options": {},
    "discrete": {
        "template_rendering": "exact",
        "prompt_ids": "exact",
        "greedy_ids": "exact_except_listed_margin_below_1e-3",
        "finish_reason": "equal",
        "step0_argmax": "equal",
        "streaming_reassembly": "exact",
        "usage": "exact",
        "concurrent_requests": "equal_to_sequential",
    },
    "float": {
        "step0_logprob_max_abs": 5e-3,
        "step0_top5_logprob_max_abs": 2e-3,
        "cached_vs_uncached_logprob_max_abs": 2e-4,
        "chunked_vs_one_call_prefill_logprob_max_abs": 1e-4,
        "nan_or_inf": "fail",
    },
}


def log_softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    total = math.log(sum(math.exp(v - top) for v in logits))
    return [v - top - total for v in logits]


def reference_cases(count: int = 3) -> dict[str, dict[str, Any]]:
    """Cases whose output ends at the stop token (the last one at the length limit)."""
    rng = random.Random(7)
    cases: dict[str, dict[str, Any]] = {}
    for index in range(count):
        vector = log_softmax([rng.uniform(-3, 3) for _ in range(VOCAB)])
        order = sorted(range(VOCAB), key=lambda i: (-vector[i], i))
        length = 4 + index
        outputs = [order[0], *[rng.randrange(0, EOS) for _ in range(length - 2)]]
        last_is_stop = index != count - 1
        outputs.append(EOS if last_is_stop else rng.randrange(0, EOS))
        name = f"case-{index}"
        cases[name] = {
            "id": name,
            "rendered": f"<user>prompt {index}</user>\n<assistant>",
            "prompt_ids": [1, 2 + index, 3, 4],
            "output_ids": outputs,
            "finish_reason": "eos" if last_is_stop else "length",
            "step_margin": [round(0.5 + 0.25 * s, 6) for s in range(len(outputs))],
            "step0_top": [[i, vector[i]] for i in order[:8]],
            "step0_vocab": VOCAB,
            "_vector": vector,
        }
    return cases


def write_reference(directory: Path, cases: dict[str, dict[str, Any]], *, vectors: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for case in cases.values():
        row = {k: v for k, v in case.items() if not k.startswith("_")}
        if vectors:
            (directory / "step0").mkdir(exist_ok=True)
            path = directory / "step0" / f"{case['id']}.f32"
            stage_results.write_float32(path, case["_vector"])
            row["step0_sha256"] = stage_results.sha256_file(path)
        lines.append(json.dumps(row))
    path = directory / "reference.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def candidate_documents(
    cases: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    """The template and dotnet documents and step-0 vectors of a candidate that equals the reference."""
    head = {
        "format": "roost.stage-results",
        "format_version": 3,
        "task": "generation",
        "profile": PROFILE,
        "candidate": "synthetic-candidate",
        "producer": {"name": "synthetic", "repository_commit": None},
        "environment": {"machine": "synthetic", "provider": "cpu", "provider_options": {}},
    }
    template = {**head, "stage": "template", "cases": {}}
    dotnet = {**head, "stage": "dotnet", "cases": {}}
    vectors: dict[str, list[float]] = {}
    for name, case in cases.items():
        template["cases"][name] = {
            "rendered": case["rendered"],
            "render_error": None,
            "prompt_ids": case["prompt_ids"],
        }
        stop = 1 if case["finish_reason"] == "eos" else 0
        dotnet["cases"][name] = {
            "prompt_ids": list(case["prompt_ids"]),
            "output_ids": list(case["output_ids"]),
            "finish_reason": case["finish_reason"],
            "step0_top": copy.deepcopy(case["step0_top"][:5]),
            "stream_text_equals_result": True,
            "stream_ids_equal_result": True,
            "stream_text_equals_decode": True,
            "usage": {"prompt_tokens": len(case["prompt_ids"]), "completion_tokens": len(case["output_ids"]) - stop},
        }
        vectors[name] = list(case["_vector"])
    first = next(iter(cases))
    dotnet["sections"] = {
        "cached_vs_uncached": {"case": first, "steps": 12, "max_abs_logprob": 3.4e-5, "tokens_equal": True},
        "concurrency": {"results": {name: True for name in cases}},
    }
    return template, dotnet, vectors


def write_run(directory: Path, cases: dict[str, dict[str, Any]], *, sidecar: bool = True) -> dict[str, Any]:
    """Write a matching candidate run; return the documents so a test can break one and rewrite it."""
    directory.mkdir(parents=True, exist_ok=True)
    template, dotnet, vectors = candidate_documents(cases)
    if sidecar:
        dotnet["tensors"] = {stage_results.STEP0_TENSOR: stage_results.write_sidecar(directory, "step0.f32", vectors)}
    stage_results.write_stage_file(directory, template)
    stage_results.write_stage_file(directory, dotnet)
    return {"template": template, "dotnet": dotnet, "vectors": vectors}


def write_tolerances(path: Path, overrides: dict[str, Any] | None = None) -> Path:
    document = copy.deepcopy(TOLERANCES)
    for dotted, value in (overrides or {}).items():
        node = document
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        if value is None:
            node.pop(leaf)
        else:
            node[leaf] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
