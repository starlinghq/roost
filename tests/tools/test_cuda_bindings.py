"""Tests for tools/generate-cuda-bindings.py (R-023).

The toolkit-independent tests check the binding spec itself; the drift test needs the pinned CUDA 13.3 headers and a C
compiler, and is skipped where they are absent (CI has no toolkit).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load():
    name = "generate_cuda_bindings"
    loader = importlib.machinery.SourceFileLoader(name, str(TOOLS / "generate-cuda-bindings.py"))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


gen = load()
INCLUDE = Path("/opt/cuda/include")


def test_every_function_parses_and_maps_to_a_csharp_type():
    names = set()
    for lib, proto in gen.FUNCTIONS:
        assert lib in gen.LIBRARIES
        ret, name, params = gen._parse(proto)
        assert name not in names, f"{name} bound twice"
        names.add(name)
        gen._cs_type(ret)
        for ctype, _ in params:
            gen._cs_type(ctype)


def test_every_constant_has_an_owning_library():
    names = [c[0] for c in gen.CONSTANTS]
    assert len(names) == len(set(names))
    for name in names:
        assert gen._owner(name) in gen.LIBRARIES


def test_pointer_and_handle_mapping():
    assert gen._cs_type("cudaStream_t") == "nint"
    assert gen._cs_type("cudaStream_t *") == "nint*"
    assert gen._cs_type("const void *") == "void*"
    assert gen._cs_type("size_t *") == "nuint*"
    assert gen._cs_type("enum cudaMemcpyKind") == "int"
    assert gen._cs_type("cublasLtMatmulHeuristicResult_t []") == "cublasLtMatmulHeuristicResult_t*"
    with pytest.raises(SystemExit):
        gen._cs_type("struct NotBound *")


def test_the_s1_spike_subset_is_bound():
    bound = {gen._parse(proto)[1] for _, proto in gen.FUNCTIONS}
    s1 = {
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaMemset",
        "cudaStreamCreateWithFlags",
        "cudaStreamDestroy",
        "cudaStreamSynchronize",
        "cudaStreamBeginCapture",
        "cudaStreamEndCapture",
        "cudaGraphInstantiate",
        "cudaGraphLaunch",
        "cudaGraphExecDestroy",
        "cudaGraphDestroy",
        "cudaGetLastError",
        "cudaGetDevice",
        "cudaMemGetInfo",
        "cudaRuntimeGetVersion",
        "cudaDeviceReset",
        "cuInit",
        "cuDriverGetVersion",
        "cuDeviceGetCount",
    }
    assert s1 <= bound


def test_a_different_header_is_refused(tmp_path):
    for header in gen.PINNED_HEADERS:
        (tmp_path / header).write_text("// not the pinned header\n")
    with pytest.raises(SystemExit, match="not the pinned header"):
        gen.check_pins(tmp_path)


@pytest.mark.skipif(
    not (INCLUDE / "cuda.h").is_file() or shutil.which("cc") is None,
    reason="needs the CUDA 13.3 toolkit headers under /opt/cuda/include and a C compiler",
)
def test_committed_bindings_match_the_headers(capsys):
    try:
        gen.check_pins(INCLUDE)
    except SystemExit as e:
        pytest.skip(f"installed toolkit is not the pinned one: {e}")
    facts = gen.run_probes(INCLUDE)
    assert gen.OUT.read_text() == gen.render(facts)
