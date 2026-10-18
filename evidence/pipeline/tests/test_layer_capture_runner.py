"""Hook capture on a tiny randomly initialised Qwen3.5-shaped model (three Gated DeltaNet layers, one attention layer).

Skipped unless the optional `reference` extra is installed. No weights are committed: the model is built from a config
in `tmp_path` and discarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from roost_evidence import cli, evaluate_layers, generation_reference, layer_captures, reference_runner
from synthetic_layers import LAYER_TOLERANCES

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from test_reference_runner import CHAT_TEMPLATE, VOCAB, write_prompts  # noqa: E402

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


@pytest.fixture(scope="module")
def hybrid_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    directory = tmp_path_factory.mktemp("tiny-hybrid")
    raw = Tokenizer(models.WordLevel(VOCAB, unk_token="<pad>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", chat_template=CHAT_TEMPLATE
    ).save_pretrained(directory)
    torch.manual_seed(4321)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=LAYER_TYPES,
        eos_token_id=VOCAB["<eos>"],
        pad_token_id=VOCAB["<pad>"],
        tie_word_embeddings=False,
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def captured(hybrid_model: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("capture")
    prompts = write_prompts(out.parent / "hybrid-prompts.jsonl")
    reference = reference_runner.run_reference(
        str(hybrid_model), prompts, out, capture_layers=True, state_forwards=2, max_new_tokens=8
    )
    return out, reference


def load(out: Path, name: str) -> layer_captures.Capture:
    return layer_captures.parse_capture(layer_captures.capture_path(out, name))


def test_every_case_has_a_capture_recorded_in_the_reference(captured: tuple[Path, Path]) -> None:
    out, reference = captured
    cases = generation_reference.load_reference(reference)
    assert set(cases) == {"chat", "ids", "long"}
    for name, case in cases.items():
        capture = load(out, name)
        assert capture.sha256 == case["layers_sha256"]
        assert capture.prompt_ids == case["prompt_ids"]
        assert capture.decode_ids == case["output_ids"][:-1]
        assert capture.forwards == len(case["output_ids"])
        assert capture.layer_types == LAYER_TYPES
        assert capture.stages == [
            "embeddings",
            "layer.0",
            "layer.1",
            "layer.2",
            "layer.3",
            "final_norm",
        ]
    environment = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert environment["transformers"] == "5.20.1"
    assert environment["layer_capture"]["layer_types"] == LAYER_TYPES


def test_recurrent_state_exists_for_exactly_the_recurrent_layers(captured: tuple[Path, Path]) -> None:
    out, _ = captured
    capture = load(out, "long")
    assert capture.state_forwards == 2
    states = {name for name in capture.shapes if name.endswith("_state")}
    expected = {
        layer_captures.state_name(forward, layer, kind)
        for forward in range(2)
        for layer in (0, 1, 2)
        for kind in ("recurrent_state", "conv_state")
    }
    assert states == expected
    assert capture.shapes["forward0/layer.0/recurrent_state"] == (4, 8, 8)
    assert capture.shapes["forward0/layer.0/conv_state"] == (2 * 2 * 8 + 4 * 8, 4)


def test_the_states_are_copies_not_views_of_the_cache(captured: tuple[Path, Path]) -> None:
    out, _ = captured
    capture = load(out, "long")
    first = capture.values("forward0/layer.1/recurrent_state")
    second = capture.values("forward1/layer.1/recurrent_state")
    assert list(first) != list(second)
    assert list(capture.values("forward0/layer.1/conv_state")) != list(capture.values("forward1/layer.1/conv_state"))


def test_the_hooks_agree_with_an_uncached_forward(hybrid_model: Path, captured: tuple[Path, Path]) -> None:
    out, reference = captured
    case = generation_reference.load_reference(reference)["long"]
    capture = load(out, "long")
    network = transformers.AutoModelForCausalLM.from_pretrained(str(hybrid_model), dtype=torch.float32).eval()
    tokens = case["prompt_ids"] + case["output_ids"][:-1]
    with torch.inference_mode():
        result = network(input_ids=torch.tensor([tokens]), output_hidden_states=True, use_cache=True)
    prompt = len(case["prompt_ids"])
    hidden = result.hidden_states  # embeddings, outputs of layers 0..2, then the normalised last output
    for stage, tensor in (("embeddings", hidden[0]), ("layer.0", hidden[1]), ("layer.2", hidden[3])):
        got = torch.tensor(list(capture.values(f"forward0/{stage}"))).reshape(prompt, -1)
        assert torch.allclose(got, tensor[0, :prompt], atol=1e-5), stage
    # A cached decode step equals the same position of the uncached run (chunked prefill rule vs recurrent rule).
    got = torch.tensor(list(capture.values("forward1/layer.2"))).reshape(1, -1)
    full = network(input_ids=torch.tensor([tokens[: prompt + 1]]), output_hidden_states=True).hidden_states
    assert torch.allclose(got, full[3][0, prompt:], atol=1e-4)
    # The recurrent state after the prefill equals the state a fresh uncached prefill leaves in its cache.
    with torch.inference_mode():
        fresh = network(input_ids=torch.tensor([case["prompt_ids"]]), use_cache=True).past_key_values
    expected = fresh.layers[1].recurrent_states[0][0]
    got = torch.tensor(list(capture.values("forward0/layer.1/recurrent_state"))).reshape(expected.shape)
    assert torch.equal(got, expected)


def test_the_capture_is_deterministic(hybrid_model: Path, captured: tuple[Path, Path], tmp_path: Path) -> None:
    out, reference = captured
    again = tmp_path / "again"
    again.mkdir()
    second = reference_runner.run_reference(
        str(hybrid_model),
        tmp_path.parent / "hybrid-prompts.jsonl",
        again,
        capture_layers=True,
        state_forwards=2,
        max_new_tokens=8,
    )
    assert second.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8")


def test_the_reference_evaluates_against_itself_and_a_flipped_layer_is_attributed(
    captured: tuple[Path, Path], tmp_path: Path
) -> None:
    out, reference = captured
    limits = tmp_path / "limits.json"
    limits.write_text(
        json.dumps(
            {"float": {"nan_or_inf": "fail", **{k.removeprefix("float."): v for k, v in LAYER_TOLERANCES.items()}}}
        ),
        encoding="utf-8",
    )
    report = evaluate_layers.evaluate(out, reference, limits, out)
    assert report["status"] == "passed", [c for c in report["checks"] if c["status"] != "passed"]

    cases = generation_reference.load_reference(reference)
    candidate = tmp_path / "candidate"
    for name in cases:
        tensors = safetensors_torch.load_file(str(layer_captures.capture_path(out, name)))
        if name == "long":
            tensors["forward0/layer.2"] = tensors["forward0/layer.2"] + 0.1
        with open(layer_captures.capture_path(out, name), "rb") as handle:
            from safetensors import safe_open

            with safe_open(handle.name, framework="pt") as source:
                metadata = source.metadata()
        target = layer_captures.capture_path(candidate, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        safetensors_torch.save_file(tensors, str(target), metadata=metadata)
    report = evaluate_layers.evaluate(candidate, reference, limits, out)
    assert report["status"] == "failed"
    first = report["cases"]["long"]["first_divergent"]
    assert (first["forward"], first["stage"]) == (0, "layer.2")
    assert report["cases"]["chat"]["first_divergent"] is None


def test_the_stdlib_reader_and_safetensors_read_each_others_files(tmp_path: Path) -> None:
    tensors = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.tensor([1.5, -2.0])}
    path = tmp_path / "x.safetensors"
    safetensors_torch.save_file(tensors, str(path), metadata={"k": "v"})
    data = path.read_bytes()
    import struct

    header = json.loads(data[8 : 8 + struct.unpack("<Q", data[:8])[0]])
    assert header["__metadata__"] == {"k": "v"}
    assert header["a"]["dtype"] == "F32"
    ours = tmp_path / "ours.safetensors"
    layer_captures.write_capture(
        ours, {"k": "v"}, {"a": ((2, 3), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), "b": ((2,), [1.5, -2.0])}
    )
    loaded = safetensors_torch.load_file(str(ours))
    assert torch.equal(loaded["a"], tensors["a"])
    assert torch.equal(loaded["b"], tensors["b"])


def test_the_cli_captures_layers(hybrid_model: Path, tmp_path: Path) -> None:
    prompts = write_prompts(tmp_path / "p.jsonl")
    out = tmp_path / "run"
    code = cli.main(
        [
            "reference-run",
            str(hybrid_model),
            str(prompts),
            "--out",
            str(out),
            "--capture-layers",
            "--state-forwards",
            "1",
        ]
    )
    assert code == 0
    assert load(out, "chat").state_forwards == 1
