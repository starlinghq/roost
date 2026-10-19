"""Writes qwen35-gating-cases.json: the Qwen3.5 gating ops of contract version 3 run through transformers itself.

Test time only (AGENTS.md: no Python under src/). Needs torch 2.9.1 and transformers 5.20.1, the versions
evidence/pipeline/pyproject.toml pins for the reference runner, and checks the sha256 of modeling_qwen3_5.py before it
runs. Every weight is random (seeded); nothing comes from a checkpoint. Run:

    python -I make_qwen35_gating_cases.py qwen35-gating-cases.json

One Qwen3_5GatedDeltaNet and one Qwen3_5Attention module are built in float32 at a small size and their forward is run
on random hidden states. Hooks record the in_proj_a, in_proj_b and in_proj_z outputs, the q, k, v, g and beta the
forward passes to torch_chunk_gated_delta_rule (patched to record them), the delta-rule output, the input and output of
Qwen3_5RMSNormGated, and for the attention module the packed q_proj output and the o_proj input. The script asserts the
packing of q_proj (per head: head_dim query rows, then head_dim gate rows) and that o_proj reads
attn_output * sigmoid(gate), then writes the recorded arrays for CpuOps to be checked against
(tests/Starlings.Roost.Ops.Tests/GatingReferenceTests.cs).
"""

import hashlib
import json
import sys

import torch
import transformers
from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5

MODELING_SHA256 = "b48635f7388fbfd1b214b8c19275f8c9854fd02c4f3a3376a2339b753a2a5457"
SEED = 20261019
TOKENS = 5


def flat(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().to(torch.float32).reshape(-1).tolist()]


def main(out_path: str) -> None:
    with open(modeling_qwen3_5.__file__, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if transformers.__version__ != "5.20.1" or sha != MODELING_SHA256:
        sys.exit(
            f"needs transformers 5.20.1, modeling_qwen3_5.py {MODELING_SHA256}; got {transformers.__version__} {sha}"
        )

    # One thread: threaded BLAS sums in a varying order, which changes the last bits from run to run.
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    config = configuration_qwen3_5.Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "eager"

    # ---- Gated DeltaNet layer ----
    lin = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, 0).to(torch.float32).eval()
    with torch.no_grad():
        for p in lin.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        # Ranges seen in the Qwen3.5-0.8B checkpoint's layer 0 (A_log -6.6..0.26, dt_bias -13..8.4, norm 0.52..1.06).
        lin.A_log.copy_(torch.empty(4).uniform_(-6.6, 0.26))
        lin.dt_bias.copy_(torch.empty(4).uniform_(-13.0, 8.4))
        lin.norm.weight.copy_(torch.empty(16).uniform_(0.5, 1.06))
        hidden = torch.randn(1, TOKENS, 32)
        # Solve head 0's in_proj_a row so that a + dt_bias covers both sides of softplus's threshold of 20 and the
        # region where log(1 + exp(x)) is used.
        target = torch.tensor([30.0, -30.0, 19.5, -2.0, 0.5]) - lin.dt_bias[0]
        h64 = hidden[0].to(torch.float64)
        row = h64.T @ torch.linalg.solve(h64 @ h64.T, target.to(torch.float64))
        lin.in_proj_a.weight[0] = row.to(torch.float32)

    rec = {}

    def keep(name):
        def hook(_m, args, out):
            rec[name] = out
            rec[name + ".args"] = args

        return hook

    lin.in_proj_a.register_forward_hook(keep("a"))
    lin.in_proj_b.register_forward_hook(keep("b"))
    lin.in_proj_z.register_forward_hook(keep("z"))
    lin.norm.register_forward_hook(keep("norm"))
    original = modeling_qwen3_5.torch_chunk_gated_delta_rule

    def recording(query, key, value, g, beta, **kwargs):
        rec.update(q=query, k=key, v=value, g=g, beta=beta, l2=kwargs.get("use_qk_l2norm_in_kernel"))
        out, state = original(query, key, value, g=g, beta=beta, **kwargs)
        rec["core"] = out
        return out, state

    modeling_qwen3_5.torch_chunk_gated_delta_rule = recording
    try:
        with torch.no_grad():
            lin(hidden)
    finally:
        modeling_qwen3_5.torch_chunk_gated_delta_rule = original

    assert rec["l2"] is True, "Qwen3.5 passes use_qk_l2norm_in_kernel=True"
    a_plus = rec["a"][0, :, 0] + lin.dt_bias[0]
    assert (a_plus > 20).any() and (a_plus < -20).any() and ((a_plus > 0) & (a_plus < 20)).any(), a_plus
    group = 4 // 2
    norm_in, norm_gate = rec["norm.args"]
    assert torch.equal(norm_in, rec["core"].reshape(-1, 16)) and torch.equal(norm_gate, rec["z"].reshape(-1, 16))
    linear = {
        "tokens": TOKENS,
        "k_heads": 2,
        "v_heads": 4,
        "k_dim": 16,
        "v_dim": 16,
        "eps": config.rms_norm_eps,
        "a": flat(rec["a"]),
        "b": flat(rec["b"]),
        "a_log": flat(lin.A_log),
        "dt_bias": flat(lin.dt_bias),
        "g": flat(rec["g"]),
        "beta": flat(rec["beta"]),
        # The forward repeats each key head for its value heads (repeat_interleave); keep one copy per key head.
        "q": flat(rec["q"][:, :, ::group]),
        "k": flat(rec["k"][:, :, ::group]),
        "v": flat(rec["v"]),
        "core": flat(rec["core"]),
        "z": flat(rec["z"]),
        "norm_weight": flat(lin.norm.weight),
        "norm_out": flat(rec["norm"]),
    }

    # ---- Full-attention layer ----
    att = modeling_qwen3_5.Qwen3_5Attention(config, 1).to(torch.float32).eval()
    with torch.no_grad():
        for p in att.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    att.q_proj.register_forward_hook(keep("q_proj"))
    att.o_proj.register_forward_hook(keep("o_proj"))
    hidden = torch.randn(1, TOKENS, 32)
    cos = torch.ones(1, TOKENS, 8)
    sin = torch.zeros(1, TOKENS, 8)
    with torch.no_grad():
        att(hidden, (cos, sin), None)
        gated = rec["o_proj.args"][0].clone()
        packed = rec["q_proj"].clone()
        heads, hd = 2, 8
        # Per head h, rows [2*h*hd, 2*h*hd + hd) of q_proj are the query and the next hd rows the gate.
        gate_rows = torch.cat([torch.arange(2 * h * hd + hd, 2 * h * hd + 2 * hd) for h in range(heads)])
        gate = packed.view(1, TOKENS, heads, 2 * hd)[..., hd:].reshape(1, TOKENS, heads * hd)
        assert torch.allclose(gate, hidden @ att.q_proj.weight[gate_rows].T, atol=1e-5)
        # Zero the gate rows: sigmoid(0) = 0.5 exactly, so o_proj then reads attn_output / 2 and the ungated attention
        # output is twice that, with no rounding in either step.
        att.q_proj.weight[gate_rows] = 0.0
        att(hidden, (cos, sin), None)
        attn = rec["o_proj.args"][0] * 2.0
        assert torch.equal(gated, attn * torch.sigmoid(gate)), "o_proj reads attn_output * sigmoid(gate)"
    attention = {
        "tokens": TOKENS,
        "heads": heads,
        "head_dim": hd,
        "attn": flat(attn),
        "gate": flat(gate),
        "gated": flat(gated),
    }

    doc = {
        "generator": "tests/Starlings.Roost.Ops.Tests/Reference/make_qwen35_gating_cases.py",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "modeling_qwen3_5_sha256": sha,
        "seed": SEED,
        "dtype": "float32",
        "linear": linear,
        "attention": attention,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
