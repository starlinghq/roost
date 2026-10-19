"""Upstream golden tensors for the Roost kernel conformance suite (build plan R-019).

    uv run --project tests/Starlings.Roost.Conformance/goldens --locked roost-conformance-goldens --out <dir> [--lock <file>]
    uv run --project tests/Starlings.Roost.Conformance/goldens --locked roost-conformance-goldens --verify <file> --lock <file>

Each golden case is one op call on seeded random inputs of the Qwen3.5-0.8B full-attention layer's shapes (Qwen/Qwen3.5-0.8B at
2fc06364715b967f1860aea9cf38778875588b17: hidden 1024, 8 query heads over 2 KV heads, head_dim 256, 64 rotary dimensions,
rope_theta 1e7, MLP 3584), plus the result of the *upstream* definition of that op, evaluated in float64:

    RmsNorm, QkNorm    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm._norm, times the weight
                       (the (1 + weight) convention of Qwen3_5RMSNorm.forward is model definition, not the op)
    Rope               modeling_qwen3_5.apply_rotary_pos_emb / rotate_half with cos and sin formed in float64
    PagedAttention     modeling_qwen3_5.eager_attention_forward / repeat_kv (its softmax is float32, as upstream)
    SiluAndMul         modeling_qwen3_5.ACT2FN["silu"] on the gate half times the up half
    Gemm, LmHead       torch.nn.functional.linear; Embedding torch.nn.functional.embedding
    CacheWrite/Copy    torch indexing (bit-exact data movement)
    Residual           torch.add

The transformers (5.20.1) and torch (2.9.1) versions are the ones the evidence pipeline pins for its reference runner. Output is
one safetensors file, a release asset that is never committed; `--lock` writes (or, with `--verify`, checks) the small lock file
that is committed and pins the asset by SHA-256. The same script and seed give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

ASSET_NAME = "roost-conformance-goldens.safetensors"
FORMAT = "roost.conformance-goldens"
BASE_SEED = 20261019
EPS = 1e-6
Q_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 256
TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def seed_of(case_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{BASE_SEED}/{case_id}".encode()).digest()[:8], "little")


class Builder:
    """Collects tensors and the case descriptions that go into the file's metadata."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.cases: list[dict[str, Any]] = []

    def add(
        self,
        case_id: str,
        op: str,
        dtype: str,
        cls: str,
        params: dict[str, float],
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, tuple[str, torch.Tensor]],
        scale_buffer: str | None = None,
        initial: dict[str, int] | None = None,
    ) -> None:
        for name, t in inputs.items():
            self.tensors[f"{case_id}/{name}"] = t.contiguous()
        outs = []
        for name, (out_dtype, expected) in outputs.items():
            self.tensors[f"{case_id}/expected/{name}"] = expected.contiguous()
            entry: dict[str, Any] = {"name": name, "dtype": out_dtype}
            if initial and name in initial:
                entry["initial"] = initial[name]
            outs.append(entry)
        self.cases.append(
            {
                "id": case_id,
                "op": op,
                "dtype": dtype,
                "class": cls,
                "seed": seed_of(case_id),
                "params": params,
                "inputs": list(inputs),
                "outputs": outs,
                "scale_buffer": scale_buffer,
            }
        )


def randn(rng: np.random.Generator, shape: tuple[int, ...], dtype: str, std: float = 1.0, mean: float = 0.0) -> torch.Tensor:
    return torch.from_numpy(mean + std * rng.standard_normal(shape)).to(TORCH_DTYPES[dtype])


def elementwise_class(dtype: str) -> str:
    return {"BF16": "elementwise-bf16", "F16": "elementwise-f16", "F32": "elementwise-f32"}[dtype]


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Qwen3_5RMSNorm._norm is an instance method that reads only self.eps.
    return upstream.Qwen3_5RMSNorm._norm(types.SimpleNamespace(eps=EPS), x.double()) * w.double()


def rms_norm_cases(b: Builder) -> None:
    for dtype in ("BF16", "F16"):
        for rows, dim in ((1, 1024), (7, 1024), (3, 3584)):
            case_id = f"RmsNorm-{dtype}-{rows}x{dim}"
            rng = np.random.default_rng(seed_of(case_id))
            x, w = randn(rng, (rows, dim), dtype, 2.0), randn(rng, (dim,), dtype, 0.2, 1.0)
            b.add(
                case_id,
                "RmsNorm",
                dtype,
                elementwise_class(dtype),
                {"rows": rows, "dim": dim, "eps": EPS, "in_stride": 0, "out_stride": 0},
                {"x": x, "w": w},
                {"out": (dtype, rms_norm(x, w))},
            )


def qk_norm_cases(b: Builder) -> None:
    for dtype in ("BF16", "F16"):
        for tokens in (1, 5):
            case_id = f"QkNorm-{dtype}-t{tokens}"
            rng = np.random.default_rng(seed_of(case_id))
            q = randn(rng, (tokens, Q_HEADS, HEAD_DIM), dtype, 1.5)
            k = randn(rng, (tokens, KV_HEADS, HEAD_DIM), dtype, 1.5)
            qw, kw = randn(rng, (HEAD_DIM,), dtype, 0.2, 1.0), randn(rng, (HEAD_DIM,), dtype, 0.2, 1.0)
            b.add(
                case_id,
                "QkNorm",
                dtype,
                elementwise_class(dtype),
                {
                    "tokens": tokens,
                    "q_heads": Q_HEADS,
                    "kv_heads": KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "eps": EPS,
                    "q_stride": 0,
                    "k_stride": 0,
                },
                {"q": q.reshape(tokens, -1), "k": k.reshape(tokens, -1), "qw": qw, "kw": kw},
                {"q": (dtype, rms_norm(q, qw).reshape(tokens, -1)), "k": (dtype, rms_norm(k, kw).reshape(tokens, -1))},
            )


def rope_reference(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor, rotary: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=torch.float64) / rotary))
    freqs = pos.double()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]  # [1, tokens, rotary]
    qe, ke = upstream.apply_rotary_pos_emb(q.double()[None], k.double()[None], cos, sin, unsqueeze_dim=2)
    return qe[0], ke[0]


def rope_cases(b: Builder) -> None:
    # (label, max position): the M0 model's rope_theta 1e7 with 64 of 256 dimensions rotated.
    for dtype in ("BF16", "F16"):
        for label, max_pos in (("pos4096", 4096), ("pos262144", 262144), ("pos2e31", 2**31 - 1)):
            tokens = 17
            case_id = f"Rope-{dtype}-{label}-t{tokens}"
            rng = np.random.default_rng(seed_of(case_id))
            q = randn(rng, (tokens, Q_HEADS, HEAD_DIM), dtype)
            k = randn(rng, (tokens, KV_HEADS, HEAD_DIM), dtype)
            pos = torch.from_numpy(rng.integers(0, max_pos, size=tokens, dtype=np.int64).astype(np.int32))
            qe, ke = rope_reference(q, k, pos, 64, 1e7)
            b.add(
                case_id,
                "Rope",
                dtype,
                elementwise_class(dtype),
                {
                    "tokens": tokens,
                    "q_heads": Q_HEADS,
                    "kv_heads": KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "rotary_dim": 64,
                    "kind": 0,
                    "interleaved": 0,
                    "theta": 1e7,
                    "llama3_factor": 8,
                    "llama3_low_freq_factor": 1,
                    "llama3_high_freq_factor": 4,
                    "llama3_original_max_position": 8192,
                },
                {"q": q.reshape(tokens, -1), "k": k.reshape(tokens, -1), "pos": pos},
                {"q": (dtype, qe.reshape(tokens, -1)), "k": (dtype, ke.reshape(tokens, -1))},
            )


def attention_cases(b: Builder) -> None:
    module = types.SimpleNamespace(num_key_value_groups=Q_HEADS // KV_HEADS, training=False)
    # (label, mode 0 decode / 1 prefill, [(q_len, kv_len)], page size, layout 2 NHD / 3 HND, causal)
    specs = [
        ("decode-mixed4", 0, [(1, 5), (1, 37), (1, 130), (1, 16)], 16, 2, True),
        ("decode-mixed4-hnd-ps64", 0, [(1, 5), (1, 37), (1, 130), (1, 16)], 64, 3, True),
        ("prefill-fresh-100", 1, [(100, 100)], 16, 2, True),
        ("prefill-chunk-64-of-300-hnd-ps32", 1, [(64, 300)], 32, 3, True),
        ("prefill-multi-seq", 1, [(17, 17), (64, 300), (1, 50)], 16, 2, True),
        ("prefill-non-causal-32-of-200", 1, [(32, 200)], 16, 2, False),
    ]
    for label, mode, seqs, ps, layout, causal in specs:
        case_id = f"PagedAttention-BF16-{label}"
        rng = np.random.default_rng(seed_of(case_id))
        batch, tokens = len(seqs), sum(s[0] for s in seqs)
        pages_per = [(kv + ps - 1) // ps for _, kv in seqs]
        total = sum(pages_per)
        pool_pages = total + 3
        perm = rng.permutation(pool_pages).astype(np.int32)
        q_indptr = np.concatenate([[0], np.cumsum([s[0] for s in seqs])]).astype(np.int32)
        kv_indptr = np.concatenate([[0], np.cumsum(pages_per)]).astype(np.int32)
        last = np.array([kv - (p - 1) * ps for (_, kv), p in zip(seqs, pages_per, strict=True)], dtype=np.int32)
        indices = perm[:total].copy()
        pool_shape = (pool_pages, ps, KV_HEADS, HEAD_DIM) if layout == 2 else (pool_pages, KV_HEADS, ps, HEAD_DIM)
        q = randn(rng, (tokens, Q_HEADS, HEAD_DIM), "BF16")
        kc, vc = randn(rng, pool_shape, "BF16"), randn(rng, pool_shape, "BF16")

        def gather(pool: torch.Tensor, seq: int, kv_len: int) -> torch.Tensor:
            pages = torch.from_numpy(indices[kv_indptr[seq] : kv_indptr[seq + 1]].astype(np.int64))
            seg = pool[pages].double()
            seg = seg if layout == 2 else seg.permute(0, 2, 1, 3)
            return seg.reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len]  # [kv, kv_heads, head_dim]

        out = torch.zeros(tokens, Q_HEADS, HEAD_DIM, dtype=torch.float64)
        for s, (n, kv_len) in enumerate(seqs):
            qs = q[q_indptr[s] : q_indptr[s + 1]].double().permute(1, 0, 2)[None]  # [1, heads, n, hd]
            ks = gather(kc, s, kv_len).permute(1, 0, 2)[None]
            vs = gather(vc, s, kv_len).permute(1, 0, 2)[None]
            mask = None
            if causal and mode == 1:
                visible = torch.arange(kv_len)[None, :] <= (kv_len - n + torch.arange(n))[:, None]
                mask = torch.zeros(n, kv_len, dtype=torch.float64).masked_fill(~visible, float("-inf"))[None, None]
            o, _ = upstream.eager_attention_forward(module, qs, ks, vs, mask, HEAD_DIM**-0.5)
            out[q_indptr[s] : q_indptr[s + 1]] = o[0]
        b.add(
            case_id,
            "PagedAttention",
            "BF16",
            "attention-bf16",
            {
                "mode": mode,
                "batch": batch,
                "tokens": tokens,
                "q_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "page_size": ps,
                "kv_layout": layout,
                "causal": 1 if causal else 0,
            },
            {
                "q": q.reshape(tokens, -1),
                "k_cache": kc.reshape(-1),
                "v_cache": vc.reshape(-1),
                "q_indptr": torch.from_numpy(q_indptr),
                "kv_indptr": torch.from_numpy(kv_indptr),
                "kv_indices": torch.from_numpy(indices),
                "last_page_len": torch.from_numpy(last),
            },
            {"out": ("BF16", out.reshape(tokens, -1))},
            scale_buffer="v_cache",
            initial={"out": 255},
        )


def silu_cases(b: Builder) -> None:
    for dtype in ("BF16", "F16"):
        for rows, dim in ((1, 3584), (8, 3584), (3, 1001)):
            case_id = f"SiluAndMul-{dtype}-{rows}x{dim}"
            rng = np.random.default_rng(seed_of(case_id))
            x = randn(rng, (rows, 2 * dim), dtype, 2.0)
            xd = x.double()
            b.add(
                case_id,
                "SiluAndMul",
                dtype,
                elementwise_class(dtype),
                {"rows": rows, "dim": dim},
                {"x": x},
                {"out": (dtype, upstream.ACT2FN["silu"](xd[:, :dim]) * xd[:, dim:])},
            )


def gemm_cases(b: Builder) -> None:
    specs = [
        # (label, m, n, k, trans_b, bias, alpha, beta, bucket)
        ("o_proj", 1, 1024, 2048, True, False, 1.0, 0.0, 0),
        ("o_proj", 17, 1024, 2048, True, False, 1.0, 0.0, 0),
        ("k_proj", 3, 512, 1024, True, False, 1.0, 0.0, 0),
        ("down-nontransposed", 17, 1024, 3584, False, False, 1.0, 0.0, 0),
        ("bias", 17, 1024, 2048, True, True, 1.0, 0.0, 0),
        ("alpha-beta-c-bias-bucket64", 5, 512, 1024, False, True, 2.0, 1.0, 64),
    ]
    for dtype in ("BF16", "F16"):
        for label, m, n, k, tb, bias, alpha, beta, bucket in specs:
            case_id = f"Gemm-{dtype}-{label}-{m}x{n}x{k}"
            rng = np.random.default_rng(seed_of(case_id))
            a = randn(rng, (m, k), dtype)
            w = randn(rng, (n, k) if tb else (k, n), dtype, 1.0 / np.sqrt(k))
            bi, c = randn(rng, (n,), dtype, 0.5), randn(rng, (m, n), dtype)
            weight = w.double() if tb else w.double().T
            d = alpha * torch.nn.functional.linear(a.double(), weight)
            if beta != 0.0:
                d = d + beta * c.double()
            if bias:
                d = d + bi.double()
            b.add(
                case_id,
                "Gemm",
                dtype,
                "gemm-bf16" if dtype == "BF16" else "gemm-f16",
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "trans_b": 1 if tb else 0,
                    "alpha": alpha,
                    "beta": beta,
                    "bucket": bucket,
                    "use_bias": 1 if bias else 0,
                    "use_c": 1 if beta != 0.0 else 0,
                },
                {"a": a, "b": w, "bias": bi, "c": c},
                {"d": (dtype, d)},
            )


def embedding_cases(b: Builder) -> None:
    for dtype in ("BF16", "F16", "F32"):
        for vocab, dim, tokens in ((2048, 1024, 33),):
            case_id = f"Embedding-{dtype}-v{vocab}-d{dim}-t{tokens}"
            rng = np.random.default_rng(seed_of(case_id))
            table = randn(rng, (vocab, dim), dtype)
            ids = torch.from_numpy(rng.integers(0, vocab, size=tokens).astype(np.int32))
            b.add(
                case_id,
                "Embedding",
                dtype,
                "exact",
                {"tokens": tokens, "dim": dim, "vocab": vocab},
                {"ids": ids, "table": table},
                {"out": (dtype, torch.nn.functional.embedding(ids.long(), table))},
            )


def lm_head_cases(b: Builder) -> None:
    for dtype, vocab, dim, padded, selected, cap in (
        ("BF16", 4096, 1024, 4096, 1, 0.0),
        ("BF16", 4000, 1024, 4096, 5, 30.0),
        ("F16", 4096, 1024, 4128, 2, 0.0),
    ):
        case_id = f"LmHead-{dtype}-v{vocab}-d{dim}-sel{selected}-cap{cap}"
        rng = np.random.default_rng(seed_of(case_id))
        tokens = selected + 3
        weight = randn(rng, (vocab, dim), dtype, 0.05)
        hidden = randn(rng, (tokens, dim), dtype)
        rows = torch.from_numpy(np.array([tokens - 1 - i for i in range(selected)], dtype=np.int32))
        logits = torch.nn.functional.linear(hidden.double()[rows.long()], weight.double())
        if cap != 0.0:
            logits = cap * torch.tanh(logits / cap)
        b.add(
            case_id,
            "LmHead",
            dtype,
            "lm-head-f32",
            {
                "selected": selected,
                "dim": dim,
                "vocab": vocab,
                "vocab_padded": padded,
                "soft_cap": cap,
                "tokens": tokens,
                "bucket": 0,
            },
            {"hidden": hidden, "rows": rows, "weight": weight},
            {"logits": ("F32", logits)},
        )


def residual_cases(b: Builder) -> None:
    for dtype in ("F32", "BF16", "F16"):
        for elements in (1024, 7 * 1024):
            case_id = f"Residual-{dtype}-n{elements}"
            rng = np.random.default_rng(seed_of(case_id))
            a, c = randn(rng, (elements,), dtype), randn(rng, (elements,), dtype)
            b.add(
                case_id,
                "Residual",
                dtype,
                elementwise_class(dtype),
                {"elements": elements},
                {"a": a, "b": c},
                {"out": (dtype, a.double() + c.double())},
            )


def cache_write_cases(b: Builder) -> None:
    kh, hd, pages = KV_HEADS, HEAD_DIM, 40
    for dtype, layout, ps, tokens in (("BF16", 2, 16, 37), ("BF16", 3, 64, 37), ("F32", 2, 16, 9)):
        case_id = f"CacheWrite-{dtype}-layout{layout}-ps{ps}-t{tokens}"
        rng = np.random.default_rng(seed_of(case_id))
        k, v = randn(rng, (tokens, kh, hd), dtype), randn(rng, (tokens, kh, hd), dtype)
        slots = rng.choice(pages * ps, size=tokens, replace=False).astype(np.int64)
        slots[5::11] = -1
        shape = (pages, ps, kh, hd) if layout == 2 else (pages, kh, ps, hd)
        init = torch.full(shape, 0, dtype=TORCH_DTYPES[dtype]).view(torch.uint8).fill_(0xAB).view(TORCH_DTYPES[dtype])
        kc, vc = init.clone(), init.clone()
        for t in range(tokens):
            if slots[t] < 0:
                continue
            page, off = divmod(int(slots[t]), ps)
            if layout == 2:
                kc[page, off], vc[page, off] = k[t], v[t]
            else:
                kc[page, :, off], vc[page, :, off] = k[t], v[t]
        b.add(
            case_id,
            "CacheWrite",
            dtype,
            "exact",
            {"tokens": tokens, "kv_heads": kh, "head_dim": hd, "page_size": ps, "kv_layout": layout},
            {
                "k": k.reshape(tokens, -1),
                "v": v.reshape(tokens, -1),
                "k_cache": init.clone().reshape(-1),
                "v_cache": init.clone().reshape(-1),
                "slots": torch.from_numpy(slots),
            },
            {"k_cache": (dtype, kc.reshape(-1)), "v_cache": (dtype, vc.reshape(-1))},
        )


def cache_copy_cases(b: Builder) -> None:
    for same in (True, False):
        block_bytes, blocks = 4096, 24
        case_id = f"CacheCopy-bb{block_bytes}-{'one-pool' if same else 'two-pools'}"
        rng = np.random.default_rng(seed_of(case_id))
        src = torch.from_numpy(rng.integers(0, 256, size=blocks * block_bytes, dtype=np.uint8))
        dst = src.clone() if same else torch.full((blocks * block_bytes,), 0xAB, dtype=torch.uint8)
        order = rng.permutation(blocks)
        count = 6
        pairs = np.stack([order[:count], order[count : 2 * count]], axis=1).astype(np.int32)
        pairs = np.concatenate([pairs, [[order[2 * count]] * 2]]).astype(np.int32)
        out = dst.clone()
        for s, d in pairs:
            out[d * block_bytes : (d + 1) * block_bytes] = src[s * block_bytes : (s + 1) * block_bytes]
        inputs = {"src": src, "pairs": torch.from_numpy(pairs.reshape(-1))}
        if not same:
            inputs["dst"] = dst
        b.add(
            case_id,
            "CacheCopy",
            "U8",
            "exact",
            {"block_bytes": block_bytes, "count": len(pairs), "same_pool": 1 if same else 0, "blocks": blocks},
            inputs,
            {"dst": ("U8", out)},
        )


def build() -> Builder:
    b = Builder()
    for make in (
        rms_norm_cases,
        qk_norm_cases,
        rope_cases,
        attention_cases,
        silu_cases,
        gemm_cases,
        embedding_cases,
        lm_head_cases,
        residual_cases,
        cache_write_cases,
        cache_copy_cases,
    ):
        make(b)
    return b


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_document(asset: Path, case_count: int) -> dict[str, Any]:
    return {
        "format": "roost.conformance-goldens-lock",
        "format_version": 1,
        "asset": {"name": ASSET_NAME, "sha256": sha256_file(asset), "size_bytes": asset.stat().st_size},
        "release": {
            "repository": "starlinghq/roost",
            "tag": None,
            "note": "A release asset, not committed. The tag is set when the asset is uploaded; the digest above is what binds it.",
        },
        "generator": {"path": "tests/Starlings.Roost.Conformance/goldens/generate_goldens.py", "sha256": sha256_file(Path(__file__))},
        "case_count": case_count,
        "base_seed": BASE_SEED,
        "versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "model": {"id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
    }


SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
}


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """Writes the safetensors layout with tensors sorted by name, so equal input gives equal bytes.

    safetensors.torch.save_file orders the header by an unordered map, which made two runs differ in the header alone.
    """
    header: dict[str, Any] = {"__metadata__": metadata}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {"dtype": SAFETENSORS_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as f:
        f.write(len(encoded).to_bytes(8, "little"))
        f.write(encoded)
        for raw in blobs:
            f.write(raw)


def generate(out_dir: Path) -> tuple[Path, int]:
    torch.manual_seed(BASE_SEED)
    b = build()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ASSET_NAME
    metadata = {
        "format": FORMAT,
        "format_version": "1",
        "base_seed": str(BASE_SEED),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cases": json.dumps(b.cases, sort_keys=True, separators=(",", ":")),
    }
    write_safetensors(path, b.tensors, metadata)
    return path, len(b.cases)


def verify(path: Path, lock_path: Path) -> list[str]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    problems = []
    if sha256_file(path) != lock["asset"]["sha256"]:
        problems.append(f"{path}: sha256 differs from the lock")
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            problems.append(f"{path}: not a {FORMAT} file")
        if len(json.loads(meta.get("cases", "[]"))) != lock["case_count"]:
            problems.append(f"{path}: case count differs from the lock")
    if sha256_file(Path(__file__)) != lock["generator"]["sha256"]:
        problems.append("the generator script differs from the one the lock was written for")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="directory to write the goldens file into")
    ap.add_argument("--lock", type=Path, help="write the lock here (with --out), or check against it (with --verify)")
    ap.add_argument("--verify", type=Path, help="check an existing goldens file against --lock")
    args = ap.parse_args(argv)
    if args.verify:
        if not args.lock:
            ap.error("--verify needs --lock")
        problems = verify(args.verify, args.lock)
        print("\n".join(problems) if problems else "goldens match the lock")
        return 1 if problems else 0
    if not args.out:
        ap.error("give --out <dir> or --verify <file>")
    path, count = generate(args.out)
    print(f"{path}: {count} cases, {path.stat().st_size} bytes, sha256 {sha256_file(path)}")
    if args.lock:
        args.lock.write_text(json.dumps(lock_document(path, count), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"lock written to {args.lock}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
