"""Shared helpers for the S3 spike (throwaway evidence, not part of Roost.slnx)."""

import torch

# Qwen3.5-0.8B linear-attention layer (text_config of the model's config.json, 18 of 24 layers):
# linear_num_key_heads=16, linear_num_value_heads=16, key/value head dim 128, bf16 activations,
# float32 recurrent state (mamba_ssm_dtype).
H = 16
HV = 16
K = 128
V = 128
BT = 64


def make_inputs(lens, seed=0, device="cuda", varlen=True, with_h0=True):
    """Random inputs for sequences of the given lengths.

    varlen=True packs them into batch 1 with cu_seqlens; otherwise len(lens)==1 and B=1 without cu_seqlens.
    g is the log decay (<= 0) already computed upstream of the kernels, beta the per-head write strength.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(lens)
    t = sum(lens)
    q = torch.randn(1, t, H, K, device=device, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, t, H, K, device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, t, HV, V, device=device, dtype=torch.bfloat16, generator=gen)
    g = -torch.rand(1, t, HV, device=device, dtype=torch.float32, generator=gen) * 0.2
    beta = torch.rand(1, t, HV, device=device, dtype=torch.bfloat16, generator=gen)
    h0 = torch.randn(n, HV, K, V, device=device, dtype=torch.float32, generator=gen) * 0.1 if with_h0 else None
    cu = None
    if varlen:
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=device, dtype=torch.int32)
    return q, k, v, g, beta, h0, cu
