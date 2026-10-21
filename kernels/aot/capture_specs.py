"""Record the Triton specialisations that R-034's GatedDeltaRule launches, from live flash-linear-attention launches.

usage: capture_specs.py <fla-checkout> <pins.json> <out.json>

Runs FLA's chunked and recurrent gated delta rule on Qwen3.5-0.8B shapes (16 key and value heads, head size 128,
bf16 activations, float32 state, g float32 and beta bf16 per head) with every autotuned kernel reduced to its pinned
config (pins.json, S3's frozen set), and writes the signature, constexprs, divisibility attributes and launch options of
each *varlen* specialisation that kernels/src/ops/gated_delta_rule.cu launches. The engine packs sequences with
cu_seqlens, so the dense (non-varlen) specialisations of S3 are not built. Needs a visible CUDA device and the S3
Python closure (kernels/manifest.toml, [triton_aot]); build time only. This is the S3 capture.py
(docs/m0/roost-s3-triton-aot-gdn.md) reduced to the varlen set. Set TRITON_CACHE_DIR to an empty directory.
"""

import json
import subprocess
import sys
from pathlib import Path

fla_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(fla_dir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.runtime.autotuner as AT  # noqa: E402
import triton.runtime.jit as J  # noqa: E402

H = HV = 16
K = V = 128

specs = {}
orig_run = J.JITFunction.run


def run(self, *args, grid, warmup, **kw):
    kern = orig_run(self, *args, grid=grid, warmup=warmup, **kw)
    if warmup or kern is None:
        return kern
    src = kern.src
    md = kern.metadata
    rec = dict(
        module=self.fn.__module__,
        kernel=self.fn.__name__,
        arg_names=list(self.arg_names),
        signature=dict(src.signature.items()),
        constexprs={self.arg_names[k[0]]: v for k, v in src.constants.items()},
        attrs=[[list(k), v] for k, v in src.attrs.items()],
        num_warps=md.num_warps,
        num_stages=md.num_stages,
        shared=md.shared,
    )
    key = json.dumps([rec["kernel"], rec["signature"], rec["constexprs"]], sort_keys=True, default=str)
    specs.setdefault(key, rec)
    return kern


J.JITFunction.run = run

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: E402

pins = json.loads(Path(sys.argv[2]).read_text())
pinned = 0
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
            pinned += 1


def inputs(lens, with_h0):
    gen = torch.Generator(device="cuda").manual_seed(0)
    t = sum(lens)
    q = torch.randn(1, t, H, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, t, H, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, t, HV, V, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = -torch.rand(1, t, HV, device="cuda", dtype=torch.float32, generator=gen) * 0.2
    beta = torch.rand(1, t, HV, device="cuda", dtype=torch.bfloat16, generator=gen)
    h0 = torch.randn(len(lens), HV, K, V, device="cuda", dtype=torch.float32, generator=gen) * 0.1 if with_h0 else None
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, h0, cu


for with_h0 in (True, False):
    q, k, v, g, beta, h0, cu = inputs([300, 77, 1500], with_h0)
    chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
    )
q, k, v, g, beta, h0, cu = inputs([5, 1, 1], True)
fused_recurrent_gated_delta_rule(
    q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
)
torch.cuda.synchronize()

# The launches this library makes. l2norm is not among them: the library's own kernel normalises the strided views.
wanted = {
    "chunk_local_cumsum_scalar_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "fused_recurrent_gated_delta_rule_fwd_kernel",
}
out = [r for r in specs.values() if r["kernel"] in wanted and r["constexprs"].get("IS_VARLEN")]
out.sort(key=lambda r: (r["kernel"], bool(r["constexprs"].get("USE_INITIAL_STATE"))))
sha = subprocess.run(
    ["git", "-C", str(fla_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
Path(sys.argv[3]).write_text(
    json.dumps(
        dict(fla_sha=sha, triton=triton.__version__, torch=torch.__version__, pinned_autotuners=pinned, specs=out),
        indent=1,
        sort_keys=True,
        default=str,
    )
    + "\n"
)
print(f"{len(out)} varlen specialisations, {pinned} autotuners pinned, FLA {sha}")
