"""R-022: is a full-attention layer's difference storage precision or a definition error?

Recomputes one Qwen3.5-0.8B full-attention decoder layer (3, 7, 11, 15, 19 or 23) for the prefill of one case in double
precision from the checkpoint's weights, twice: exactly (as transformers computes it), and with every op output rounded
to bfloat16 and the zero-centred norm weights stored as bfloat16(1 + w), as the generated forward does. Each is fed the
captured input of the layer (the previous layer's output) from the reference run and from the candidate run.

If the exact recomputation from the reference input reproduces the reference output, the recomputation is the
checkpoint's function; if the bf16-rounded recomputation from the candidate input reproduces the candidate output, the
candidate computes that function with bf16 storage, and what remains between candidate and reference is rounding.
Test-time only; needs torch and safetensors. Reads the checkpoint (run it under the heavy-work lock).

    python -I emulate_bf16_attention_layer.py <snapshot dir> <reference run dir> <candidate run dir> <layer> <case>
"""

from __future__ import annotations

import sys
from pathlib import Path

HEADS, KV_HEADS, HEAD_DIM, ROTARY, THETA, EPS = 8, 2, 256, 64, 1e7, 1e-6
TENSORS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def layer_forward(torch, w: dict, x, bf16: bool):
    def r(t):
        return t.to(torch.bfloat16).double() if bf16 else t

    def norm(t, weight):
        scale = (1 + weight).to(torch.bfloat16).double() if bf16 else 1 + weight
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * scale

    x = r(x)
    tokens = x.shape[0]
    h = r(norm(x, w["input_layernorm.weight"]))
    qg = (h @ w["self_attn.q_proj.weight"].T).view(tokens, HEADS, 2 * HEAD_DIM)
    q, gate = r(qg[..., :HEAD_DIM]), r(qg[..., HEAD_DIM:].reshape(tokens, -1))
    k = r((h @ w["self_attn.k_proj.weight"].T).view(tokens, KV_HEADS, HEAD_DIM))
    v = r((h @ w["self_attn.v_proj.weight"].T).view(tokens, KV_HEADS, HEAD_DIM))
    q, k = r(norm(q, w["self_attn.q_norm.weight"])), r(norm(k, w["self_attn.k_norm.weight"]))
    positions = torch.arange(tokens, dtype=torch.float64)
    inv = 1.0 / (THETA ** (torch.arange(0, ROTARY, 2, dtype=torch.float64) / ROTARY))
    angles = positions[:, None] * inv[None]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]

    def rope(t):
        t = t.clone()
        a, b = t[..., : ROTARY // 2].clone(), t[..., ROTARY // 2 : ROTARY].clone()
        t[..., : ROTARY // 2] = a * cos - b * sin
        t[..., ROTARY // 2 : ROTARY] = b * cos + a * sin
        return r(t)

    q, k = rope(q), rope(k)
    out = torch.zeros(tokens, HEADS, HEAD_DIM, dtype=torch.float64)
    mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), 1)
    for head in range(HEADS):
        kv = head // (HEADS // KV_HEADS)
        scores = (q[:, head] @ k[:, kv].T) * HEAD_DIM**-0.5
        out[:, head] = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v[:, kv]
    attn = r(r(out.reshape(tokens, -1)) * torch.sigmoid(gate))
    mixed = r(x + r(attn @ w["self_attn.o_proj.weight"].T))
    y = r(norm(mixed, w["post_attention_layernorm.weight"]))
    up = r(torch.nn.functional.silu(r(y @ w["mlp.gate_proj.weight"].T)) * r(y @ w["mlp.up_proj.weight"].T))
    return r(mixed + r(up @ w["mlp.down_proj.weight"].T))


def main(snapshot: Path, reference: Path, candidate: Path, layer: int, case: str) -> int:
    import torch
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    prefix = f"model.language_model.layers.{layer}."
    with safe_open(str(snapshot / "model.safetensors-00001-of-00001.safetensors"), "pt") as handle:
        w = {name: handle.get_tensor(prefix + name).double() for name in TENSORS}

    def capture(run: Path, stage: str):
        with safe_open(str(run / "layers" / f"{case}.safetensors"), "pt") as handle:
            return handle.get_tensor(f"forward0/{stage}").double()

    def rel(a, b):
        return (a - b).norm(dim=1) / b.norm(dim=1)

    ref_in, ref_out = capture(reference, f"layer.{layer - 1}"), capture(reference, f"layer.{layer}")
    cand_in, cand_out = capture(candidate, f"layer.{layer - 1}"), capture(candidate, f"layer.{layer}")
    exact = layer_forward(torch, w, ref_in, bf16=False)
    rounded = layer_forward(torch, w, ref_in, bf16=True)
    emulated = layer_forward(torch, w, cand_in, bf16=True)
    print(f"layer {layer}, case {case}, prefill of {ref_in.shape[0]} tokens; row-relative L2, position 0 and max:")
    print(f"  exact recomputation from reference input vs reference output:   max {rel(exact, ref_out).max():.2e}")
    print(
        f"  bf16 recomputation from reference input vs reference output:    {rel(rounded, ref_out)[0]:.4f}, "
        f"max {rel(rounded, ref_out).max():.4f}"
    )
    print(
        f"  bf16 recomputation from candidate input vs candidate output:    {rel(emulated, cand_out)[0]:.4f}, "
        f"max {rel(emulated, cand_out).max():.4f}"
    )
    print(
        f"  candidate vs reference, output:                                 {rel(cand_out, ref_out)[0]:.4f}, "
        f"max {rel(cand_out, ref_out).max():.4f}"
    )
    print(
        f"  candidate vs reference, input:                                  {rel(cand_in, ref_in)[0]:.4f}, "
        f"max {rel(cand_in, ref_in).max():.4f}"
    )
    return 0


if __name__ == "__main__":
    a = sys.argv
    raise SystemExit(main(Path(a[1]), Path(a[2]), Path(a[3]), int(a[4]), a[5]))
