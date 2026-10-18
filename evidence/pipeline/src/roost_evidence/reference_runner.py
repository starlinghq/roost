"""Transformers fp32 CPU reference runner (build plan R-010): the optional `reference` extra, tool time only.

Model-agnostic: any causal language model that `AutoModelForCausalLM` loads from a local directory, or from a hub id
pinned by full commit SHA. The runner chooses no model. It renders chat messages with the tokenizer's own template (or
takes prompt ids as given), decodes greedily in float32 on the CPU with a KV cache, and writes the reference format
that `generation_reference` reads, plus the whole step-0 log-softmax vector of every case as a float32 sidecar.

torch and transformers are imported inside `run_reference` so that importing this module needs neither.

Prompts file: one JSON object per line with `id` and either `messages` (optionally `chat_template_kwargs`) or
`prompt_ids`; optionally `max_new_tokens`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from roost_evidence import stage_results
from roost_evidence.generation_reference import CASE_ID_CHARS

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
STEP0_DIR = "step0"
TOP_PAIRS = 64


class ExtraMissingError(RuntimeError):
    """torch or transformers is not installed: install the `reference` extra."""


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        name = record.get("id")
        where = f"{path}:{number}"
        if not isinstance(name, str) or not name or not set(name) <= CASE_ID_CHARS:
            raise ValueError(f"{where}: id must be a non-empty string of letters, digits, '.', '_' or '-'")
        if name in seen:
            raise ValueError(f"{where}: case {name!r} is listed more than once")
        seen.add(name)
        if ("messages" in record) == ("prompt_ids" in record):
            raise ValueError(f"{where}: give exactly one of messages and prompt_ids")
        prompts.append(record)
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def _stop_ids(model: Any, tokenizer: Any, extra: list[int]) -> set[int]:
    ids: set[int] = set(extra)
    for source in (getattr(model, "generation_config", None), tokenizer):
        eos = getattr(source, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, list | tuple):
            ids.update(int(i) for i in eos)
    return ids


def run_reference(
    model: str,
    prompts_path: Path,
    out_dir: Path,
    *,
    revision: str | None = None,
    max_new_tokens: int = 128,
    stop_ids: list[int] | None = None,
) -> Path:
    """Write `reference.jsonl`, `step0/<id>.f32` and `environment.json` into `out_dir`; return the jsonl path."""
    local = Path(model).is_dir()
    if not local and not (revision and FULL_SHA.match(revision)):
        raise ValueError("a hub model needs --revision, the full 40-character commit SHA; a local directory does not")
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ExtraMissingError(
            "the reference runner needs the `reference` extra: uv run --project evidence/pipeline --extra reference"
        ) from error

    prompts = load_prompts(prompts_path)
    kwargs: dict[str, Any] = {} if local else {"revision": revision}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)
    network = transformers.AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, **kwargs)
    network.eval()
    stops = _stop_ids(network, tokenizer, stop_ids or [])
    (out_dir / STEP0_DIR).mkdir(parents=True, exist_ok=True)

    lines = []
    with torch.inference_mode():
        for record in prompts:
            if "messages" in record:
                rendered = tokenizer.apply_chat_template(
                    record["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **(record.get("chat_template_kwargs") or {}),
                )
                prompt_ids = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            else:
                prompt_ids = [int(i) for i in record["prompt_ids"]]
                rendered = tokenizer.decode(prompt_ids)
            budget = int(record.get("max_new_tokens", max_new_tokens))
            outputs, margins, finish, step0 = _greedy(network, torch, prompt_ids, budget, stops)
            vector = step0.tolist()
            path = out_dir / STEP0_DIR / f"{record['id']}.f32"
            stage_results.write_float32(path, vector)
            order = sorted(range(len(vector)), key=lambda i: (-vector[i], i))[:TOP_PAIRS]
            lines.append(
                {
                    "id": record["id"],
                    "rendered": rendered,
                    "prompt_ids": prompt_ids,
                    "output_ids": outputs,
                    "finish_reason": finish,
                    "step_margin": margins,
                    "step0_top": [[i, vector[i]] for i in order],
                    "step0_vocab": len(vector),
                    "step0_sha256": stage_results.sha256_file(path),
                }
            )
    reference = out_dir / "reference.jsonl"
    reference.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    environment = {
        "model": model,
        "revision": revision,
        "dtype": "float32",
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "stop_ids": sorted(stops),
    }
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=1) + "\n", encoding="utf-8")
    return reference


def _greedy(network: Any, torch: Any, prompt_ids: list[int], budget: int, stops: set[int]) -> tuple:
    """Greedy decode with a KV cache. Returns ids (stop token included), margins, finish reason, step-0 vector."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    out = network(input_ids=ids, use_cache=True)
    outputs: list[int] = []
    margins: list[float] = []
    step0 = None
    finish = "length"
    for _ in range(budget):
        logprobs = torch.log_softmax(out.logits[0, -1].to(torch.float32), dim=-1)
        if step0 is None:
            step0 = logprobs.clone()
        best = torch.topk(logprobs, 2)
        token = int(best.indices[0])
        margins.append(float(best.values[0] - best.values[1]))
        outputs.append(token)
        if token in stops:
            finish = "eos"
            break
        out = network(
            input_ids=torch.tensor([[token]], dtype=torch.long), past_key_values=out.past_key_values, use_cache=True
        )
    if step0 is None:
        raise ValueError("max_new_tokens must be at least 1")
    return outputs, margins, finish, step0
