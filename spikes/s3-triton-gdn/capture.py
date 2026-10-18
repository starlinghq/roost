"""Run Python FLA's gated delta rule on Qwen3.5-0.8B shapes with a cold Triton cache and record every
production Triton launch (the autotune winner, not the benchmark candidates).

usage: capture.py <out.json> [pins.json]
  without pins: needs TRITON_CACHE_DIR set to an empty directory so autotune actually runs.
  with pins (kernel name -> {tuning kwargs, num_warps, num_stages}): every autotuned FLA kernel is reduced to the one
  pinned config, so no benchmarking happens and the launches are exactly what the AOT build compiles.
"""

import json
import sys
from pathlib import Path

import torch
import triton.runtime.autotuner as AT
import triton.runtime.jit as J
from common import make_inputs
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

specs = {}
bench_depth = [0]
n_bench = {}

orig_bench = AT.Autotuner._bench


def bench(self, *a, **kw):
    bench_depth[0] += 1
    n_bench[self.base_fn.__name__ if hasattr(self, "base_fn") else str(self.fn)] = (
        n_bench.get(self.base_fn.__name__, 0) + 1
    )
    try:
        return orig_bench(self, *a, **kw)
    finally:
        bench_depth[0] -= 1


AT.Autotuner._bench = bench
orig_run = J.JITFunction.run


def run(self, *args, grid, warmup, **kw):
    kern = orig_run(self, *args, grid=grid, warmup=warmup, **kw)
    if warmup or kern is None or bench_depth[0]:
        return kern
    src = kern.src
    md = kern.metadata
    allkw = dict(zip(self.arg_names, args, strict=False))
    allkw.update(kw)
    sig = {k: v for k, v in src.signature.items()}
    consts = {self.arg_names[k[0]]: v for k, v in src.constants.items()}
    attrs = [[list(k), v] for k, v in src.attrs.items()]
    rec = dict(
        module=self.fn.__module__,
        kernel=self.fn.__name__,
        arg_names=list(self.arg_names),
        signature=sig,
        constexprs=consts,
        attrs=attrs,
        num_warps=md.num_warps,
        num_stages=md.num_stages,
        shared=md.shared,
        global_scratch=getattr(md, "global_scratch_size", 0),
        cubin_bytes=len(kern.asm["cubin"]),
        cubin_sha256=__import__("hashlib").sha256(kern.asm["cubin"]).hexdigest(),
        tensors={k: [str(a.dtype), list(a.shape)] for k, a in allkw.items() if hasattr(a, "shape")},
    )
    key = json.dumps(
        [rec["kernel"], rec["signature"], rec["constexprs"], rec["num_warps"], rec["num_stages"]],
        sort_keys=True,
        default=str,
    )
    specs.setdefault(key, rec)
    return kern


J.JITFunction.run = run

if len(sys.argv) > 2:
    import triton

    pins = json.loads(Path(sys.argv[2]).read_text())
    npinned = 0
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("fla."):
            continue
        for obj in list(vars(mod).values()):
            o = obj
            while o is not None and not isinstance(o, AT.Autotuner):
                o = getattr(o, "fn", None) if not isinstance(o, J.JITFunction) else None
            if o is not None and o.base_fn.__name__ in pins:
                p = pins[o.base_fn.__name__]
                o.configs = [triton.Config(p["kwargs"], num_warps=p["num_warps"], num_stages=p["num_stages"])]
                o.cache.clear()
                npinned += 1
    print("pinned autotuners:", npinned)


def chunk(lens, varlen, h0, ht):
    q, k, v, g, beta, h, cu = make_inputs(lens, varlen=varlen, with_h0=h0)
    chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h, output_final_state=ht, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
    )


def rec(lens, varlen, h0, ht):
    q, k, v, g, beta, h, cu = make_inputs(lens, varlen=varlen, with_h0=h0)
    fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=h, output_final_state=ht, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
    )


# prefill: one fresh sequence, a continuation with state, a packed varlen batch
chunk([1024], False, False, True)
chunk([1024], False, True, True)
chunk([300, 77, 1500], True, True, True)
chunk([300, 77, 1500], True, False, True)
# decode: B sequences of one token each (non-varlen B>1 is not used by an engine; varlen is)
rec([1], False, True, True)
rec([1, 1, 1, 1], True, True, True)
rec([5, 1, 1], True, True, True)
torch.cuda.synchronize()
Path(sys.argv[1]).write_text(json.dumps(list(specs.values()), indent=1, default=str))
print(len(specs), "distinct production specialisations; autotune benchmark rounds:", n_bench)
for r in specs.values():
    print(
        r["kernel"],
        {k: v for k, v in r["constexprs"].items() if v is not None and k.isupper()},
        "w",
        r["num_warps"],
        "s",
        r["num_stages"],
        "smem",
        r["shared"],
        "cubin",
        r["cubin_bytes"],
    )
