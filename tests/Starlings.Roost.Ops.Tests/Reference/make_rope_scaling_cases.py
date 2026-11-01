"""Writes rope-scaling-cases.json: RoPE scaling frequencies, cos/sin cache rows and rotations for RopeScaled (R-098).

Test time only (AGENTS.md: no Python under src/). Needs torch 2.9.1, transformers 5.20.1 (the versions
evidence/pipeline/pyproject.toml pins for the reference runner) and numpy, and checks the sha256 of
modeling_rope_utils.py and models/llama/modeling_llama.py before it runs. Nothing comes from a checkpoint: every
parameter set below is written out here, and the longrope factors and the rotated vectors are seeded random numbers.
Run:

    python -I make_rope_scaling_cases.py rope-scaling-cases.json

For every case the inverse frequencies and the attention factor are computed twice:

- in float64 with numpy, written out here line by line from the formulas of transformers' `_compute_yarn_parameters`,
  `_compute_longrope_parameters`, `_compute_dynamic_ntk_parameters` and `_compute_llama3_parameters`;
- by calling those transformers functions themselves (which compute in float32), on a stub config that carries only the
  fields they read.

The script asserts that the two agree to float32 accuracy, then writes both: CpuOps is checked against the float64
values tightly and against the transformers values to float32 accuracy
(tests/Starlings.Roost.Ops.Tests/RopeScalingTests.cs).
Cache rows are `[cos(p * f) * af for f] ++ [sin(p * f) * af for f]` in float64, the layout RopeScaledArgs.CosSinCache
documents; transformers multiplies cos and sin by the attention factor the same way (`LlamaRotaryEmbedding.forward`).
Rotation cases apply transformers' own `apply_rotary_pos_emb` (rotate_half, the NeoX halves) in float64 to the first
rotary_dim elements of each head, as its models do for a partial rotary factor, and keep the rest.
"""

import hashlib
import json
import math
import sys

import numpy as np
import torch
import transformers
from transformers import modeling_rope_utils as mru
from transformers.models.llama import modeling_llama

ROPE_UTILS_SHA256 = "6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24"
MODELING_LLAMA_SHA256 = "affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830"
SEED = 20261101


class StubConfig:
    """The fields the transformers rope init functions read, and nothing else."""

    def __init__(self, head_dim: int, max_position_embeddings: int, rope_parameters: dict) -> None:
        self.head_dim = head_dim
        self.hidden_size = head_dim
        self.num_attention_heads = 1
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters

    def standardize_rope_params(self) -> None:
        pass


# ---- float64 formulas, written from modeling_rope_utils.py ---------------------------------------------------------


def np_default(dim: int, theta: float) -> np.ndarray:
    return 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def np_yarn(dim: int, theta: float, rp: dict, max_pos: int) -> tuple[np.ndarray, float]:
    factor = rp.get("factor")
    orig = rp["original_max_position_embeddings"]
    if factor is None:
        factor = max_pos / orig

    def get_mscale(scale: float, m: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    af = rp.get("attention_factor")
    if af is None:
        ms, msa = rp.get("mscale"), rp.get("mscale_all_dim")
        af = get_mscale(factor, ms) / get_mscale(factor, msa) if ms and msa else get_mscale(factor)
    beta_fast = rp.get("beta_fast") or 32
    beta_slow = rp.get("beta_slow") or 1

    def corr_dim(rot: float) -> float:
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(theta))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if rp.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    extrap = 1.0 - ramp
    pos_freqs = theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv = (1.0 / (factor * pos_freqs)) * (1.0 - extrap) + (1.0 / pos_freqs) * extrap
    return inv, float(af)


def np_longrope(dim: int, theta: float, rp: dict, max_pos: int, seq_len: int | None) -> tuple[np.ndarray, float]:
    orig = rp["original_max_position_embeddings"]
    factor = rp.get("factor")
    if factor is None:
        factor = max_pos / orig
    af = rp.get("attention_factor")
    if af is None:
        af = 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(orig))
    ext = np.asarray(rp["long_factor"] if seq_len and seq_len > orig else rp["short_factor"], dtype=np.float64)
    return 1.0 / (ext * theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)), float(af)


def np_dynamic(dim: int, theta: float, rp: dict, max_pos: int, seq_len: int | None) -> tuple[np.ndarray, float]:
    factor = rp["factor"]
    s = max_pos if seq_len is None else max(seq_len, max_pos)
    base = theta * ((factor * s / max_pos) - (factor - 1)) ** (dim / (dim - 2))
    return np_default(dim, base), 1.0


def np_llama3(dim: int, theta: float, rp: dict) -> tuple[np.ndarray, float]:
    inv = np_default(dim, theta)
    factor, lo, hi = rp["factor"], rp["low_freq_factor"], rp["high_freq_factor"]
    old = rp["original_max_position_embeddings"]
    wavelen = 2 * math.pi / inv
    out = np.where(wavelen > old / lo, inv / factor, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = ~(wavelen < old / hi) & ~(wavelen > old / lo)
    return np.where(medium, smoothed, out), 1.0


def np_freqs(kind: str, dim: int, theta: float, rp: dict, max_pos: int, seq_len: int | None):
    if kind == "yarn":
        return np_yarn(dim, theta, rp, max_pos)
    if kind == "longrope":
        return np_longrope(dim, theta, rp, max_pos, seq_len)
    if kind == "dynamic":
        return np_dynamic(dim, theta, rp, max_pos, seq_len)
    if kind == "llama3":
        return np_llama3(dim, theta, rp)
    raise ValueError(kind)


def tf_freqs(kind: str, head_dim: int, rp: dict, max_pos: int, seq_len: int | None):
    cfg = StubConfig(head_dim, max_pos, dict(rp, rope_type=kind))
    inv, af = mru.ROPE_INIT_FUNCTIONS[kind](cfg, torch.device("cpu"), seq_len=seq_len)
    return inv.to(torch.float64).numpy(), float(af)


def rows(inv: np.ndarray, af: float, positions: list[int]) -> list[float]:
    p = np.asarray(positions, dtype=np.float64)[:, None]
    ang = p * inv[None, :]
    return np.concatenate([np.cos(ang) * af, np.sin(ang) * af], axis=1).reshape(-1).tolist()


def rotate(q: np.ndarray, k: np.ndarray, inv: np.ndarray, af: float, positions: list[int], rot: int):
    """transformers' apply_rotary_pos_emb in float64 on the first rot elements of each head; the rest pass through."""
    p = torch.tensor(positions, dtype=torch.float64)[:, None]
    freqs = p * torch.from_numpy(inv)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = (emb.cos() * af)[None], (emb.sin() * af)[None]
    tq = torch.from_numpy(q).permute(1, 0, 2)[None]  # [1, heads, tokens, head_dim]
    tk = torch.from_numpy(k).permute(1, 0, 2)[None]
    rq, rk = modeling_llama.apply_rotary_pos_emb(tq[..., :rot], tk[..., :rot], cos, sin)
    oq = torch.cat((rq, tq[..., rot:]), dim=-1)[0].permute(1, 0, 2)
    ok = torch.cat((rk, tk[..., rot:]), dim=-1)[0].permute(1, 0, 2)
    return oq.reshape(-1).tolist(), ok.reshape(-1).tolist()


def main(out_path: str) -> None:
    with open(mru.__file__, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(modeling_llama.__file__, "rb") as f:
        llama_sha = hashlib.sha256(f.read()).hexdigest()
    if transformers.__version__ != "5.20.1" or sha != ROPE_UTILS_SHA256 or llama_sha != MODELING_LLAMA_SHA256:
        sys.exit(
            f"needs transformers 5.20.1, modeling_rope_utils.py {ROPE_UTILS_SHA256}, modeling_llama.py "
            f"{MODELING_LLAMA_SHA256}; got {transformers.__version__} {sha} {llama_sha}"
        )

    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)

    def lr_factors(n: int, long: bool) -> list[float]:
        # Seeded, increasing with the dimension like published longrope factors; not taken from any checkpoint.
        if long:
            return [float(x) for x in np.sort(rng.uniform(1.0, 48.0, n))]
        return [float(x) for x in np.sort(rng.uniform(1.0, 1.25, n))]

    yarn = "yarn"
    specs = [
        # name, kind, head_dim, rotary_dim, theta, max_position_embeddings, rope parameters, seq_len, positions
        (
            "yarn-f4-orig32k-hd128",
            yarn,
            128,
            128,
            1e6,
            131072,
            {"factor": 4.0, "original_max_position_embeddings": 32768},
            None,
            [0, 1, 7, 1000, 32767, 32768, 65535, 131071],
        ),
        (
            "yarn-f40-mscale-hd64",
            yarn,
            64,
            64,
            1e4,
            163840,
            {
                "factor": 40.0,
                "original_max_position_embeddings": 4096,
                "mscale": 0.707,
                "mscale_all_dim": 1.0,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            None,
            [0, 3, 4095, 4096, 50000, 163839],
        ),
        (
            "yarn-f4-orig256k-rot64-notruncate",
            yarn,
            256,
            64,
            1e7,
            1048576,
            {
                "factor": 4.0,
                "original_max_position_embeddings": 262144,
                "beta_fast": 16,
                "beta_slow": 2,
                "truncate": False,
            },
            None,
            [0, 5, 262143, 262144, 1048575],
        ),
        (
            "yarn-attention-factor-hd48-rot32",
            yarn,
            48,
            32,
            1e4,
            2048,
            {"factor": 8.0, "original_max_position_embeddings": 256, "attention_factor": 1.25},
            None,
            [0, 1, 255, 256, 2047],
        ),
        (
            "yarn-factor-from-max-hd64",
            yarn,
            64,
            64,
            5e5,
            16384,
            {"factor": None, "original_max_position_embeddings": 2048},
            None,
            [0, 2047, 2048, 16383],
        ),
        (
            "longrope-short-hd96",
            "longrope",
            96,
            96,
            1e4,
            131072,
            {
                "original_max_position_embeddings": 4096,
                "short_factor": lr_factors(48, False),
                "long_factor": lr_factors(48, True),
            },
            4096,
            [0, 1, 4095],
        ),
        ("longrope-long-hd96", "longrope", 96, 96, 1e4, 131072, None, 4097, [0, 1, 4095, 4096, 131071]),
        (
            "longrope-explicit-hd64-rot32",
            "longrope",
            64,
            32,
            1e4,
            2048,
            {
                "original_max_position_embeddings": 256,
                "factor": 8.0,
                "attention_factor": 1.1,
                "short_factor": lr_factors(16, False),
                "long_factor": lr_factors(16, True),
            },
            1000,
            [0, 255, 256, 999],
        ),
        ("dynamic-init-hd128", "dynamic", 128, 128, 1e4, 4096, {"factor": 2.0}, None, [0, 1, 4095]),
        ("dynamic-short-hd128", "dynamic", 128, 128, 1e4, 4096, {"factor": 2.0}, 2048, [0, 2047]),
        ("dynamic-10000-hd128", "dynamic", 128, 128, 1e4, 4096, {"factor": 2.0}, 10000, [0, 4096, 9999]),
        ("dynamic-65536-hd128", "dynamic", 128, 128, 1e4, 4096, {"factor": 2.0}, 65536, [0, 65535]),
        ("dynamic-hd256-rot64", "dynamic", 256, 64, 1e7, 1024, {"factor": 4.0}, 3000, [0, 1023, 2999]),
        (
            "llama3-f8",
            "llama3",
            128,
            128,
            5e5,
            131072,
            {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
            None,
            [0, 1, 8191, 8192, 131071],
        ),
        (
            "llama3-f32",
            "llama3",
            64,
            64,
            5e5,
            131072,
            {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
            None,
            [0, 70000],
        ),
    ]

    cases = []
    prev_rp = None
    for name, kind, head_dim, rot, theta, max_pos, rp, seq_len, positions in specs:
        rp = dict(prev_rp if rp is None else rp)
        prev_rp = rp
        rpt = dict(rp, rope_theta=theta, partial_rotary_factor=rot / head_dim)
        inv, af = np_freqs(kind, rot, theta, rp, max_pos, seq_len)
        tinv, taf = tf_freqs(kind, head_dim, rpt, max_pos, seq_len)
        assert tinv.shape == inv.shape, name
        assert np.allclose(tinv, inv, rtol=2e-6, atol=0), (name, np.max(np.abs(tinv / inv - 1)))
        assert math.isclose(taf, af, rel_tol=1e-12), (name, taf, af)
        cases.append(
            {
                "name": name,
                "kind": kind,
                "head_dim": head_dim,
                "rotary_dim": rot,
                "theta": theta,
                "max_position_embeddings": max_pos,
                "params": {k: v for k, v in rp.items()},
                "seq_len": seq_len,
                "inv_freq": inv.tolist(),
                "inv_freq_transformers": tinv.tolist(),
                "attention_factor": af,
                "attention_factor_transformers": taf,
                "positions": positions,
                "cos_sin_rows": rows(inv, af, positions),
            }
        )

    rotations = []
    by_name = {c["name"]: c for c in cases}
    for name, tokens, qh, kvh in [
        ("yarn-attention-factor-hd48-rot32", 6, 4, 2),
        ("longrope-explicit-hd64-rot32", 5, 2, 1),
        ("dynamic-hd256-rot64", 3, 2, 1),
        ("yarn-f40-mscale-hd64", 3, 2, 1),
    ]:
        c = by_name[name]
        hd, rot = c["head_dim"], c["rotary_dim"]
        top = max(c["positions"])
        positions = [int(x) for x in rng.integers(0, top + 1, tokens - 1)] + [top]
        q = rng.standard_normal((tokens, qh, hd))
        k = rng.standard_normal((tokens, kvh, hd))
        inv = np.asarray(c["inv_freq"])
        oq, ok = rotate(q, k, inv, c["attention_factor"], positions, rot)
        rotations.append(
            {
                "case": name,
                "tokens": tokens,
                "q_heads": qh,
                "kv_heads": kvh,
                "positions": positions,
                "q": q.reshape(-1).tolist(),
                "k": k.reshape(-1).tolist(),
                "q_out": oq,
                "k_out": ok,
            }
        )

    doc = {
        "generator": "tests/Starlings.Roost.Ops.Tests/Reference/make_rope_scaling_cases.py",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "modeling_rope_utils_sha256": sha,
        "modeling_llama_sha256": llama_sha,
        "seed": SEED,
        "cases": cases,
        "rotations": rotations,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
