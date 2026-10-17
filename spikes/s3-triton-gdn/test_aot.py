"""Compare the C-ABI AOT build with Python FLA (same pinned configs) and with an fp32 naive reference.

usage: test_aot.py <libgdn_aot.so> <pins.json>   (run through tools/gpu-lease, one process)
"""

import ctypes
import json
import sys
from pathlib import Path

import torch
import triton
import triton.runtime.autotuner as AT
import triton.runtime.jit as J
from common import K, V, make_inputs
from fla.modules.norm.l2norm import l2norm_fwd
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule


def pin_fla(pins):
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("fla."):
            continue
        for obj in list(vars(mod).values()):
            o = obj
            while o is not None and not isinstance(o, AT.Autotuner):
                o = None if isinstance(o, J.JITFunction) else getattr(o, "fn", None)
            if o is not None and o.base_fn.__name__ in pins:
                p = pins[o.base_fn.__name__]
                o.configs = [triton.Config(p["kwargs"], num_warps=p["num_warps"], num_stages=p["num_stages"])]


pin_fla(json.loads(Path(sys.argv[2]).read_text()))
lib = ctypes.CDLL(sys.argv[1])
P = ctypes.c_void_p
lib.gdn_make_chunk_tables.restype = ctypes.c_int
lib.gdn_chunk_workspace_bytes.restype = ctypes.c_size_t
lib.gdn_chunk_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
lib.gdn_chunk_prefill.argtypes = [P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [P] * 12
lib.gdn_recurrent.argtypes = [P, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [P] * 9
STREAM = torch.cuda.current_stream().cuda_stream
ptr = lambda x: 0 if x is None else x.data_ptr()  # noqa: E731


def cmp(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    print(
        f"  {name:28s} max|diff|={d.max().item():.3e}  mismatched={int((d > 0).sum())}/{d.numel()}  "
        f"ref max|x|={b.abs().max().item():.3e}"
    )
    return d.max().item()


def run_prefill(lens, varlen, with_h0):
    q, k, v, g, beta, h0, cu = make_inputs(lens, varlen=varlen, with_h0=with_h0)
    n, t = len(lens), sum(lens)
    o_ref, ht_ref = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
    )
    if varlen:
        cu_h = cu.cpu()
        ci = torch.zeros(t // 64 + n + 1, 2, dtype=torch.int32)
        co = torch.zeros(n + 1, dtype=torch.int64)
        nt = lib.gdn_make_chunk_tables(
            ctypes.c_void_p(cu_h.data_ptr()), n, ctypes.c_void_p(ci.data_ptr()), ctypes.c_void_p(co.data_ptr())
        )
        ci, co = ci[:nt].cuda(), co.cuda()
    else:
        nt, ci, co = -(-t // 64), None, None
    ws = torch.empty(lib.gdn_chunk_workspace_bytes(t, nt), dtype=torch.uint8, device="cuda")
    ws.fill_(0xFF)  # poison: every bf16/fp32 element is NaN unless a kernel wrote it first
    o = torch.full_like(v, float("nan"))
    ht = torch.full((n, 16, K, V), float("nan"), dtype=torch.float32, device="cuda")
    r = lib.gdn_chunk_prefill(
        STREAM,
        int(varlen),
        n,
        t,
        nt,
        ptr(q),
        ptr(k),
        ptr(v),
        ptr(g),
        ptr(beta),
        ptr(h0),
        ptr(ht),
        ptr(o),
        ptr(cu),
        ptr(ci),
        ptr(co),
        ptr(ws),
    )
    torch.cuda.synchronize()
    print(f"prefill lens={lens} varlen={varlen} h0={with_h0} NT={nt} status={r}")
    assert r == 0
    e1, e2 = cmp("o (AOT vs Python FLA)", o, o_ref), cmp("final state", ht, ht_ref)
    return e1, e2, (q, k, v, g, beta, h0, cu, o, ht, o_ref, ht_ref)


def naive_ref(q, k, v, g, beta, h0, scale):
    qn, _ = l2norm_fwd(q)
    kn, _ = l2norm_fwd(k)
    return naive_recurrent_gated_delta_rule(
        qn.float(),
        kn.float(),
        v.float(),
        beta.float(),
        g.float(),
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )


def run_decode(lens, varlen):
    q, k, v, g, beta, h0, cu = make_inputs(lens, varlen=varlen, with_h0=True)
    n, t = len(lens), sum(lens)
    o_ref, ht_ref = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
    )
    o = torch.full_like(v, float("nan"))
    ht = torch.full((n, 16, K, V), float("nan"), dtype=torch.float32, device="cuda")
    r = lib.gdn_recurrent(
        STREAM, int(varlen), n, t, ptr(q), ptr(k), ptr(v), ptr(g), ptr(beta), ptr(h0), ptr(ht), ptr(o), ptr(cu)
    )
    torch.cuda.synchronize()
    print(f"decode lens={lens} varlen={varlen} status={r}")
    assert r == 0
    return cmp("o (AOT vs Python FLA)", o, o_ref), cmp("final state", ht, ht_ref)


results = {}
results["prefill_varlen_h0"] = run_prefill([300, 77, 1500, 64, 1], True, True)[:2]
results["prefill_varlen_noh0"] = run_prefill([300, 77, 1500], True, False)[:2]
results["prefill_single_h0"] = run_prefill([1024], False, True)[:2]
results["prefill_single_noh0"] = run_prefill([1000], False, False)[:2]
results["decode_varlen_n4"] = run_decode([1, 1, 1, 1], True)
results["decode_varlen_mixed"] = run_decode([5, 1, 1], True)
results["decode_single"] = run_decode([3], False)

# context: both implementations against an fp32 recurrent reference on a 256-token sequence
print("accuracy against fp32 naive recurrent reference (one 256-token sequence, h0):")
e1, e2, (q, k, v, g, beta, h0, cu, o, ht, o_ref, ht_ref) = run_prefill([256], True, True)
on, hn = naive_ref(q, k, v, g, beta, h0, K**-0.5)
on = on.to(torch.float32)
print("  chunk path, AOT vs naive:")
cmp("o", o, on)
cmp("final state", ht, hn)
print("  chunk path, Python FLA vs naive:")
cmp("o", o_ref, on)
cmp("final state", ht_ref, hn)
Path("out/numeric_results_" + sys.argv[1].split("_")[-1].split(".")[0] + ".json").write_text(
    json.dumps(results, indent=1)
)
