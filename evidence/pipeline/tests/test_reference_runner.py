"""The transformers fp32 CPU reference runner on a tiny randomly initialised model.

Skipped unless the optional `reference` extra (torch, transformers) is installed. No weights are committed: the model
and its tokenizer are built from a config in `tmp_path` and discarded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from roost_evidence import evaluate_generation, generation_reference, reference_runner, stage_results
from synthetic import candidate_documents, write_tolerances

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = ["<pad>", "<eos>", "<user>", "</user>", "<assistant>", "hello", "world", "one", "two", "three", "four", "five"]
VOCAB = {word: index for index, word in enumerate(WORDS)}
CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </{{ m['role'] }}> {% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    directory = tmp_path_factory.mktemp("tiny-model")
    vocab = VOCAB
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", chat_template=CHAT_TEMPLATE
    )
    tokenizer.save_pretrained(directory)
    torch.manual_seed(1234)
    config = transformers.Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(directory)
    return directory


def write_prompts(path: Path) -> Path:
    records = [
        {"id": "chat", "messages": [{"role": "user", "content": "hello world"}], "max_new_tokens": 6},
        {"id": "ids", "prompt_ids": [5, 6, 7], "max_new_tokens": 5},
        {"id": "long", "messages": [{"role": "user", "content": "one two three four five"}], "max_new_tokens": 10},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def run(tiny_model: Path, tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    out.mkdir()
    return reference_runner.run_reference(str(tiny_model), write_prompts(tmp_path / f"{name}.jsonl"), out)


def test_the_reference_is_valid_and_deterministic(tiny_model: Path, tmp_path: Path) -> None:
    first = run(tiny_model, tmp_path, "a")
    second = run(tiny_model, tmp_path, "b")
    cases = generation_reference.load_reference(first)
    assert set(cases) == {"chat", "ids", "long"}
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    for case in cases.values():
        assert len(case["step_margin"]) == len(case["output_ids"])
        assert all(m >= 0 for m in case["step_margin"])
        assert case["finish_reason"] in ("eos", "length")
        assert case["finish_reason"] == ("eos" if case["output_ids"][-1] == 1 else "length")
        assert len(case["output_ids"]) <= 10


def test_the_template_renders_and_ids_follow_the_rendering(tiny_model: Path, tmp_path: Path) -> None:
    cases = generation_reference.load_reference(run(tiny_model, tmp_path, "a"))
    assert cases["chat"]["rendered"] == "<user> hello world </user> <assistant>"
    expected = [VOCAB[w] for w in ("<user>", "hello", "world", "</user>", "<assistant>")]
    assert cases["chat"]["prompt_ids"] == expected
    assert cases["ids"]["prompt_ids"] == [5, 6, 7]


def test_step0_vector_is_a_log_softmax_and_matches_its_top_pairs(tiny_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "a"
    reference = run(tiny_model, tmp_path, "a")
    cases = generation_reference.load_reference(reference)
    vectors = generation_reference.load_reference_vectors(out, cases).vectors
    assert set(vectors) == set(cases)
    for name, vector in vectors.items():
        assert math.isclose(sum(math.exp(v) for v in vector), 1.0, rel_tol=1e-5)
        top = cases[name]["step0_top"]
        assert top[0][0] == max(range(len(vector)), key=lambda i: (vector[i], -i))
        assert top[0][1] == pytest.approx(vector[top[0][0]], abs=1e-6)
        assert cases[name]["step0_vocab"] == len(vector)
        assert cases[name]["step_margin"][0] == pytest.approx(top[0][1] - top[1][1], abs=1e-5)


def test_the_reference_evaluates_against_itself(tiny_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "a"
    reference = run(tiny_model, tmp_path, "a")
    cases = generation_reference.load_reference(reference)
    vectors = generation_reference.load_reference_vectors(out, cases).vectors
    for name, case in cases.items():
        case["_vector"] = vectors[name]
    template, dotnet, sidecar = candidate_documents(cases)
    results = tmp_path / "results"
    results.mkdir()
    dotnet["tensors"] = {stage_results.STEP0_TENSOR: stage_results.write_sidecar(results, "step0.f32", sidecar)}
    for document in (template, dotnet):
        stage_results.write_stage_file(results, document)
    tolerances = write_tolerances(
        tmp_path / "limits.json", {"float.chunked_vs_one_call_prefill_logprob_max_abs": "not_applicable"}
    )
    report = evaluate_generation.evaluate(results, reference, tolerances, out)
    assert report["status"] == "passed", [c for c in report["checks"] if c["status"] != "passed"]


def test_a_hub_model_needs_a_full_sha_revision(tmp_path: Path) -> None:
    prompts = write_prompts(tmp_path / "p.jsonl")
    with pytest.raises(ValueError, match="40-character"):
        reference_runner.run_reference("org/model", prompts, tmp_path, revision="main")


def test_an_extra_stop_id_ends_generation_with_eos(tiny_model: Path, tmp_path: Path) -> None:
    first = generation_reference.load_reference(run(tiny_model, tmp_path, "a"))["chat"]["output_ids"][0]
    out = tmp_path / "stopped"
    out.mkdir()
    reference = reference_runner.run_reference(str(tiny_model), tmp_path / "a.jsonl", out, stop_ids=[first])
    case = generation_reference.load_reference(reference)["chat"]
    assert case["output_ids"] == [first]
    assert case["finish_reason"] == "eos"
