"""Ahead-of-time compile the pinned Qwen3.5-0.8B gated-delta-rule Triton specialisations.

usage: build_aot.py <specs.json> <sm> <outdir>

For every specialisation in specs.json (written by capture.py with pinned configs) this
  1. rebuilds the Triton ASTSource (signature, constexprs, divisibility attrs, num_warps/stages),
  2. compiles it for the requested SM with triton.compile (the same call triton.tools.compile makes),
  3. writes <outdir>/<stub>.cubin, a C stub embedding the cubin with a runtime-grid launcher, and manifest.json.
Triton's own tools/compile.py cannot be used directly: it cannot express None constexprs (cu_seqlens=None) and it
bakes the launch grid into the stub, whereas the FLA grids depend on T and cu_seqlens at run time.
Set TRITON_CACHE_DIR to an empty directory so every compile is cold.
"""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

C_TYPES = {"i32": "int32_t", "i64": "int64_t", "u32": "uint32_t", "fp32": "float"}


def c_type(t):
    return "CUdeviceptr" if t.startswith("*") else C_TYPES[t]


def unwrap(k):
    while not isinstance(k, triton.runtime.jit.JITFunction):
        k = k.fn
    return k


STUB = r"""
/* {doc} */
static CUmodule {n}_mod; static CUfunction {n}_fn;
static const unsigned char {n}_cubin[{size}] = {{ {data} }};
CUresult {n}(CUstream stream, unsigned gx, unsigned gy, unsigned gz{params}) {{
    if (!{n}_fn) {{
        CUresult r = cuModuleLoadData(&{n}_mod, {n}_cubin);
        if (r != CUDA_SUCCESS) return r;
        r = cuModuleGetFunction(&{n}_fn, {n}_mod, "{kname}");
        if (r != CUDA_SUCCESS) return r;
        if ({shared} > 49152) {{
            int optin = 0; CUdevice dev; cuCtxGetDevice(&dev);
            cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
            if (optin < {shared}) return CUDA_ERROR_INVALID_VALUE;
            r = cuFuncSetAttribute({n}_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {shared});
            if (r != CUDA_SUCCESS) return r;
        }}
    }}
    CUdeviceptr global_scratch = 0, profile_scratch = 0;
    void *args[{nargs}] = {{ {ptrs} }};
    if ((unsigned long long)gx * gy * gz == 0) return CUDA_SUCCESS;
    return cuLaunchKernel({n}_fn, gx, gy, gz, {threads}, 1, 1, {shared}, stream, args, NULL);
}}
"""


def main():
    specs = json.loads(Path(sys.argv[1]).read_text())
    sm = int(sys.argv[2])
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    target = GPUTarget("cuda", sm, 32)
    backend = triton.compiler.make_backend(target)
    manifest = []
    c_parts = ["#include <cuda.h>\n#include <stdint.h>\n#include <stddef.h>\n"]
    h_parts = ["#pragma once\n#include <cuda.h>\n#include <stdint.h>\n"]
    t_all = time.perf_counter()
    for i, s in enumerate(specs):
        fn = unwrap(getattr(importlib.import_module(s["module"]), s["kernel"]))
        fn.create_binder()
        constexprs = {k: v for k, v in s["constexprs"].items()}
        attrs = {tuple(k): [tuple(a) for a in v] for k, v in s["attrs"]}
        # the capture stores attrs as [[index], [[name, value], ...]]
        src = fn.ASTSource(fn=fn, constexprs=constexprs, signature=s["signature"], attrs=attrs)
        options = backend.parse_options({"num_warps": s["num_warps"], "num_stages": s["num_stages"]})
        t0 = time.perf_counter()
        ck = triton.compile(src, target=target, options=options.__dict__)
        dt = time.perf_counter() - t0
        cubin = ck.asm["cubin"]
        md = ck.metadata
        if getattr(md, "global_scratch_size", 0) > 0:
            raise RuntimeError(f"{s['kernel']} needs global scratch; the stub does not provide it")
        stub = f"gdn_{i:02d}_{s['kernel']}"
        (out / f"{stub}.cubin").write_bytes(cubin)
        runtime = [(n, s["signature"][n]) for n in s["arg_names"] if s["signature"][n] != "constexpr"]
        params = "".join(f", {c_type(t)} {n}" for n, t in runtime)
        ptrs = ", ".join([f"&{n}" for n, _ in runtime] + ["&global_scratch", "&profile_scratch"])
        const_doc = {k: v for k, v in constexprs.items() if v is not None}
        c_parts.append(
            STUB.format(
                n=stub,
                size=len(cubin),
                data=", ".join(str(b) for b in cubin),
                params=params,
                kname=md.name,
                shared=md.shared,
                nargs=len(runtime) + 2,
                ptrs=ptrs,
                threads=md.num_warps * 32,
                doc=f"{s['kernel']} sm_{sm} warps={md.num_warps} stages={md.num_stages} {const_doc}".replace("*/", "*"),
            )
        )
        h_parts.append(f"CUresult {stub}(CUstream stream, unsigned gx, unsigned gy, unsigned gz{params});\n")
        manifest.append(
            dict(
                stub=stub,
                kernel=s["kernel"],
                module=s["module"],
                sm=sm,
                constexprs=constexprs,
                runtime_args=runtime,
                num_warps=md.num_warps,
                num_stages=md.num_stages,
                shared=md.shared,
                compile_seconds=round(dt, 3),
                cubin_bytes=len(cubin),
                cubin_sha256=hashlib.sha256(cubin).hexdigest(),
                jit_cubin_sha256=s["cubin_sha256"],
            )
        )
        print(f"{stub:70s} sm{sm} {len(cubin):7d} B {dt:6.2f} s", flush=True)
    total = time.perf_counter() - t_all
    (out / "gdn_kernels.c").write_text("".join(c_parts))
    (out / "gdn_kernels.h").write_text("".join(h_parts))
    summary = dict(
        sm=sm,
        triton=triton.__version__,
        total_compile_seconds=round(total, 2),
        total_cubin_bytes=sum(m["cubin_bytes"] for m in manifest),
        kernels=manifest,
    )
    (out / "manifest.json").write_text(json.dumps(summary, indent=1))
    print(f"sm{sm}: {len(manifest)} cubins, {sum(m['cubin_bytes'] for m in manifest)} B, {total:.1f} s wall")


main()
