#!/usr/bin/env python3
"""Generates the CUDA bindings of Starlings.Roost.Cuda from the CUDA toolkit's own headers (R-023).

The bindings cover the CUDA runtime, the CUDA driver, cuBLASLt and NVML, from the headers of the pinned CUDA 13.3
toolkit. They follow the pattern of starling's tools/generate-ffmpeg-bindings.py and bind only what Roost calls: the
functions, struct fields and constants declared in SPEC below. The S1 spike's hand-written subset
(spikes/s1-aot-cuda/app/Cuda.cs) is the starting list. Nothing about the C API is written by hand into the C# code;
this script asks the C compiler:

1. A type check (compiled, not run, with mismatches as errors): every function is assigned to a pointer of the declared
   prototype, and every field's C type is asserted with _Static_assert. A declaration that differs from the headers
   fails here.
2. A layout probe (compiled and run): sizeof of each struct, offsetof of each declared field, the value of each
   constant, the exported symbol name of each function (so a versioned macro such as cuDeviceTotalMem -> _v2 is bound
   under the name the library exports) and the version macros of the toolkit.

The headers are pinned twice: the toolkit versions the probe reads (CUDA_VERSION, CUDART_VERSION, CUBLAS_VER_*,
NVML_API_VERSION) must be the ones below, and the SHA-256 of each top-level header must equal PINNED_HEADERS. A
different toolkit is a deliberate change of this script, not a silent regeneration.

It writes src/Starlings.Roost.Cuda/CudaNative.g.cs: explicit-layout structs with only the declared fields at the
compiler's offsets, the functions as [LibraryImport] declarations over logical library names (resolved per OS by
CudaLibraryResolver) and the constants.

    tools/generate-cuda-bindings.py [--include /opt/cuda/include] [--check]

--check regenerates in memory and fails if the committed file differs (CI-style drift check).
"""

# ruff: noqa: E501  (C prototypes and generated C# lines are kept on one line, as in the headers)
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src/Starlings.Roost.Cuda/CudaNative.g.cs"
NAMESPACE = "Starlings.Roost.Cuda"

# The toolkit the committed bindings were generated from: CUDA 13.3 (nvcc 13.3.73) as installed under /opt/cuda on the
# development host. The version macros and the SHA-256 of the top-level headers are checked before anything is
# generated.
SOURCE = "CUDA Toolkit 13.3 (nvcc 13.3.73), include/ of the development host installation"
EXPECTED_VERSIONS = {
    "CUDA_VERSION": 13030,
    "CUDART_VERSION": 13030,
    "CUBLAS_VER_MAJOR": 13,
    "NVML_API_VERSION": 13,
}
PINNED_HEADERS = {
    "cuda.h": "31df84e16179b6d97db4b3c0bae7697392a370b41983f4a8962f0e5a8069b577",
    "cuda_runtime_api.h": "a848e416776a49904e8bace4bd377c0c5b379461e6bb2fb8accb7e6162c3a96d",
    "cublasLt.h": "36e8db9e46c95af62bfcd796aa6091d6922adb592a219f6a30957e3cd5ead495",
    "nvml.h": "31a26e3ce6f0b98a76cea38a3cf28aa112a20cfb37e9057786c768712d6a487f",
}

HEADERS = ["cuda_runtime_api.h", "cuda.h", "cublasLt.h", "nvml.h"]

# Logical library name (the [LibraryImport] name), the C# class that holds its functions, and a description.
# CudaLibraryResolver maps the logical names to the file names of each OS.
LIBRARIES = {
    "cudart": ("CudaRuntime", "CUDA runtime API"),
    "cuda": ("CudaDriver", "CUDA driver API"),
    "cublasLt": ("CublasLt", "cuBLASLt"),
    "nvml": ("Nvml", "NVML"),
}

# Structs whose fields are read or written by value or through a pointer: (C type, [(field, C type, C# type)]).
# C# types: an explicit field type, or "fixed <type> [n]" for a C array.
STRUCTS = [
    ("cublasLtMatmulAlgo_t", [("data", "uint64_t [8]", "fixed ulong [8]")]),
    (
        "cublasLtMatmulHeuristicResult_t",
        [
            ("algo", "cublasLtMatmulAlgo_t", "cublasLtMatmulAlgo_t"),
            ("workspaceSize", "size_t", "nuint"),
            ("state", "cublasStatus_t", "int"),
            ("wavesCount", "float", "float"),
        ],
    ),
    (
        "nvmlMemory_t",
        [
            ("total", "unsigned long long", "ulong"),
            ("free", "unsigned long long", "ulong"),
            ("used", "unsigned long long", "ulong"),
        ],
    ),
    (
        "nvmlMemory_v2_t",
        [
            ("version", "unsigned int", "uint"),
            ("total", "unsigned long long", "ulong"),
            ("reserved", "unsigned long long", "ulong"),
            ("free", "unsigned long long", "ulong"),
            ("used", "unsigned long long", "ulong"),
        ],
    ),
]

# Handle typedefs that are pointers to incomplete types: bound as nint.
HANDLES = [
    "cudaStream_t",
    "cudaEvent_t",
    "cudaGraph_t",
    "cudaGraphExec_t",
    "CUstream",
    "CUcontext",
    "cublasLtHandle_t",
    "cublasLtMatmulDesc_t",
    "cublasLtMatrixLayout_t",
    "cublasLtMatmulPreference_t",
    "nvmlDevice_t",
]
# Status and enum typedefs bound as int.
INT_TYPEDEFS = [
    "cudaError_t",
    "CUresult",
    "CUdevice",
    "cublasStatus_t",
    "nvmlReturn_t",
    "cudaDataType",
    "cudaDataType_t",
    "cublasComputeType_t",
    "cublasLtMatmulDescAttributes_t",
    "cublasLtMatrixLayoutAttribute_t",
    "cublasLtMatmulPreferenceAttributes_t",
]

# Functions: library, C prototype exactly as in the headers (no default arguments).
FUNCTIONS = [
    # Runtime API: the S1 subset (spikes/s1-aot-cuda/app/Cuda.cs) and what the worker adds.
    ("cudart", "cudaError_t cudaMalloc(void **devPtr, size_t size)"),
    ("cudart", "cudaError_t cudaFree(void *devPtr)"),
    ("cudart", "cudaError_t cudaMallocHost(void **ptr, size_t size)"),
    ("cudart", "cudaError_t cudaFreeHost(void *ptr)"),
    ("cudart", "cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)"),
    ("cudart", "cudaError_t cudaHostUnregister(void *ptr)"),
    ("cudart", "cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)"),
    (
        "cudart",
        "cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)",
    ),
    ("cudart", "cudaError_t cudaMemset(void *devPtr, int value, size_t count)"),
    ("cudart", "cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaMemGetInfo(size_t *free, size_t *total)"),
    ("cudart", "cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)"),
    ("cudart", "cudaError_t cudaStreamDestroy(cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaStreamSynchronize(cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaStreamQuery(cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)"),
    ("cudart", "cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)"),
    ("cudart", "cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)"),
    ("cudart", "cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)"),
    ("cudart", "cudaError_t cudaEventDestroy(cudaEvent_t event)"),
    ("cudart", "cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaEventSynchronize(cudaEvent_t event)"),
    ("cudart", "cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)"),
    (
        "cudart",
        "cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags)",
    ),
    ("cudart", "cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)"),
    ("cudart", "cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)"),
    ("cudart", "cudaError_t cudaGraphDestroy(cudaGraph_t graph)"),
    ("cudart", "cudaError_t cudaGetLastError(void)"),
    ("cudart", "cudaError_t cudaPeekAtLastError(void)"),
    ("cudart", "const char *cudaGetErrorName(cudaError_t error)"),
    ("cudart", "const char *cudaGetErrorString(cudaError_t error)"),
    ("cudart", "cudaError_t cudaGetDevice(int *device)"),
    ("cudart", "cudaError_t cudaSetDevice(int device)"),
    ("cudart", "cudaError_t cudaGetDeviceCount(int *count)"),
    ("cudart", "cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)"),
    ("cudart", "cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)"),
    ("cudart", "cudaError_t cudaDeviceSynchronize(void)"),
    ("cudart", "cudaError_t cudaDeviceReset(void)"),
    ("cudart", "cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)"),
    ("cudart", "cudaError_t cudaDriverGetVersion(int *driverVersion)"),
    # Driver API.
    ("cuda", "CUresult cuInit(unsigned int Flags)"),
    ("cuda", "CUresult cuDriverGetVersion(int *driverVersion)"),
    ("cuda", "CUresult cuDeviceGetCount(int *count)"),
    ("cuda", "CUresult cuDeviceGet(CUdevice *device, int ordinal)"),
    ("cuda", "CUresult cuDeviceGetName(char *name, int len, CUdevice dev)"),
    ("cuda", "CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)"),
    ("cuda", "CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)"),
    ("cuda", "CUresult cuGetErrorName(CUresult error, const char **pStr)"),
    ("cuda", "CUresult cuGetErrorString(CUresult error, const char **pStr)"),
    # cuBLASLt.
    ("cublasLt", "cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle)"),
    ("cublasLt", "cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)"),
    ("cublasLt", "size_t cublasLtGetVersion(void)"),
    ("cublasLt", "size_t cublasLtGetCudartVersion(void)"),
    ("cublasLt", "const char *cublasLtGetStatusString(cublasStatus_t status)"),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType)",
    ),
    ("cublasLt", "cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)"),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void *buf, size_t sizeInBytes)",
    ),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)",
    ),
    ("cublasLt", "cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)"),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void *buf, size_t sizeInBytes)",
    ),
    ("cublasLt", "cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)"),
    ("cublasLt", "cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)"),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void *buf, size_t sizeInBytes)",
    ),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount)",
    ),
    (
        "cublasLt",
        "cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream)",
    ),
    # NVML.
    ("nvml", "nvmlReturn_t nvmlInit(void)"),
    ("nvml", "nvmlReturn_t nvmlShutdown(void)"),
    ("nvml", "const char *nvmlErrorString(nvmlReturn_t result)"),
    ("nvml", "nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)"),
    ("nvml", "nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor)"),
]

# Constants: C# name, C expression, C# type.
CONSTANTS = [
    # Runtime.
    ("cudaSuccess", "cudaSuccess", "int"),
    ("cudaErrorNotReady", "cudaErrorNotReady", "int"),
    ("cudaErrorMemoryAllocation", "cudaErrorMemoryAllocation", "int"),
    ("cudaErrorInsufficientDriver", "cudaErrorInsufficientDriver", "int"),
    ("cudaErrorNoDevice", "cudaErrorNoDevice", "int"),
    ("cudaMemcpyHostToHost", "cudaMemcpyHostToHost", "int"),
    ("cudaMemcpyHostToDevice", "cudaMemcpyHostToDevice", "int"),
    ("cudaMemcpyDeviceToHost", "cudaMemcpyDeviceToHost", "int"),
    ("cudaMemcpyDeviceToDevice", "cudaMemcpyDeviceToDevice", "int"),
    ("cudaMemcpyDefault", "cudaMemcpyDefault", "int"),
    ("cudaStreamDefault", "cudaStreamDefault", "uint"),
    ("cudaStreamNonBlocking", "cudaStreamNonBlocking", "uint"),
    ("cudaEventDefault", "cudaEventDefault", "uint"),
    ("cudaEventDisableTiming", "cudaEventDisableTiming", "uint"),
    ("cudaHostRegisterDefault", "cudaHostRegisterDefault", "uint"),
    ("cudaHostRegisterPortable", "cudaHostRegisterPortable", "uint"),
    ("cudaHostRegisterMapped", "cudaHostRegisterMapped", "uint"),
    ("cudaStreamCaptureModeGlobal", "cudaStreamCaptureModeGlobal", "int"),
    ("cudaStreamCaptureModeThreadLocal", "cudaStreamCaptureModeThreadLocal", "int"),
    ("cudaStreamCaptureModeRelaxed", "cudaStreamCaptureModeRelaxed", "int"),
    ("cudaDevAttrComputeCapabilityMajor", "cudaDevAttrComputeCapabilityMajor", "int"),
    ("cudaDevAttrComputeCapabilityMinor", "cudaDevAttrComputeCapabilityMinor", "int"),
    ("cudaDevAttrMultiProcessorCount", "cudaDevAttrMultiProcessorCount", "int"),
    ("cudaDevAttrMaxSharedMemoryPerBlockOptin", "cudaDevAttrMaxSharedMemoryPerBlockOptin", "int"),
    # Driver.
    ("CUDA_SUCCESS", "CUDA_SUCCESS", "int"),
    ("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR", "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR", "int"),
    ("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR", "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR", "int"),
    # cuBLASLt.
    ("CUBLAS_STATUS_SUCCESS", "CUBLAS_STATUS_SUCCESS", "int"),
    ("CUBLAS_COMPUTE_32F", "CUBLAS_COMPUTE_32F", "int"),
    ("CUDA_R_32F", "CUDA_R_32F", "int"),
    ("CUDA_R_16F", "CUDA_R_16F", "int"),
    ("CUDA_R_16BF", "CUDA_R_16BF", "int"),
    ("CUBLAS_OP_N", "CUBLAS_OP_N", "int"),
    ("CUBLAS_OP_T", "CUBLAS_OP_T", "int"),
    ("CUBLASLT_MATMUL_DESC_TRANSA", "CUBLASLT_MATMUL_DESC_TRANSA", "int"),
    ("CUBLASLT_MATMUL_DESC_TRANSB", "CUBLASLT_MATMUL_DESC_TRANSB", "int"),
    ("CUBLASLT_MATRIX_LAYOUT_ORDER", "CUBLASLT_MATRIX_LAYOUT_ORDER", "int"),
    ("CUBLASLT_ORDER_ROW", "CUBLASLT_ORDER_ROW", "int"),
    ("CUBLASLT_ORDER_COL", "CUBLASLT_ORDER_COL", "int"),
    ("CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", "int"),
    # NVML.
    ("NVML_SUCCESS", "NVML_SUCCESS", "int"),
    ("NVML_ERROR_UNINITIALIZED", "NVML_ERROR_UNINITIALIZED", "int"),
    ("NVML_ERROR_NOT_SUPPORTED", "NVML_ERROR_NOT_SUPPORTED", "int"),
    ("NVML_ERROR_NO_PERMISSION", "NVML_ERROR_NO_PERMISSION", "int"),
    ("NVML_ERROR_LIB_RM_VERSION_MISMATCH", "NVML_ERROR_LIB_RM_VERSION_MISMATCH", "int"),
    ("NVML_DEVICE_NAME_V2_BUFFER_SIZE", "NVML_DEVICE_NAME_V2_BUFFER_SIZE", "int"),
    ("NVML_DEVICE_UUID_V2_BUFFER_SIZE", "NVML_DEVICE_UUID_V2_BUFFER_SIZE", "int"),
    ("NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE", "NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE", "int"),
    ("nvmlMemory_v2", "nvmlMemory_v2", "uint"),
]

# Version macros recorded in the generated file, as constants (the toolkit the bindings target).
VERSION_MACROS = [
    "CUDA_VERSION",
    "CUDART_VERSION",
    "CUBLAS_VER_MAJOR",
    "CUBLAS_VER_MINOR",
    "NVML_API_VERSION",
]

CS_SCALAR = {
    "void": "void",
    "char": "byte",
    "int": "int",
    "unsigned": "uint",
    "unsigned int": "uint",
    "unsigned long long": "ulong",
    "long long": "long",
    "float": "float",
    "double": "double",
    "size_t": "nuint",
    "int32_t": "int",
    "int64_t": "long",
    "uint32_t": "uint",
    "uint64_t": "ulong",
}
CS_NAMED = {name: name for name, _ in STRUCTS}
# C parameter names that are C# keywords are bound with a verbatim identifier.
CS_KEYWORDS = {
    "event",
    "params",
    "ref",
    "out",
    "in",
    "object",
    "string",
    "base",
    "checked",
    "lock",
    "operator",
    "fixed",
}


def _cs_name(name: str) -> str:
    return f"@{name}" if name in CS_KEYWORDS else name


def _cs_type(ctype: str) -> str:
    """C type (as written in a prototype) to the C# interop type. Arrays decay to pointers; enums are int."""
    t = ctype.replace("const", " ").replace("struct ", " ").replace("enum ", "enum:")
    stars = t.count("*") + t.count("[")
    base = " ".join(re.sub(r"\[\d*\]|\*", " ", t).split())
    if base in HANDLES:
        core = "nint"
    elif base in INT_TYPEDEFS or base.startswith("enum:") or base in ("CUdevice_attribute",):
        core = "int"
    elif base in CS_SCALAR:
        core = CS_SCALAR[base]
    elif base in CS_NAMED:
        core = CS_NAMED[base]
    else:
        raise SystemExit(f"no C# mapping for C type {ctype!r}")
    return core + "*" * stars


def _parse(prototype: str) -> tuple[str, str, list[tuple[str, str]]]:
    m = re.fullmatch(r"\s*(.+?)\s*\b(\w+)\s*\((.*)\)\s*", prototype)
    if not m:
        raise SystemExit(f"cannot parse {prototype!r}")
    ret, name, params = m.group(1), m.group(2), m.group(3).strip()
    out = []
    if params != "void":
        for p in params.split(","):
            pm = re.fullmatch(r"(.*?)(\w+)\s*((?:\[\d*\])*)", p.strip())
            if not pm:
                raise SystemExit(f"cannot parse parameter {p!r} of {name}")
            out.append(((pm.group(1) + pm.group(3)).strip(), pm.group(2)))
    return ret, name, out


def _includes() -> str:
    return (
        "".join(f"#include <{h}>\n" for h in HEADERS) + "#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n"
    )


def _array_param(t: str, n: str) -> str:
    return f"{t.split('[')[0]} {n}[{t.split('[', 1)[1]}" if t.endswith("]") else f"{t} {n}"


def type_check_source() -> str:
    lines = [_includes()]
    for _, proto in FUNCTIONS:
        ret, name, params = _parse(proto)
        plist = ", ".join(_array_param(t, n) for t, n in params) or "void"
        lines.append(f"static {ret} (*const check_{name})({plist}) = {name};")
    for ctype, fields in STRUCTS:
        for field, ftype, _ in fields:
            c_typename = re.sub(r"\s*\[(\d+)\]$", r"[\1]", ftype)
            lines.append(
                f"_Static_assert(__builtin_types_compatible_p(__typeof__((({ctype} *)0)->{field}), {c_typename}), "
                f'"{ctype}.{field} is not {ftype}");'
            )
    lines.append("int main(void) { return 0; }")
    return "\n".join(lines) + "\n"


def probe_source() -> str:
    lines = [
        _includes(),
        "#define XSTR_(x) #x",
        "#define XSTR(x) XSTR_(x)",
        "int main(void) {",
    ]
    for macro in VERSION_MACROS:
        lines.append(f'  printf("version {macro} %lld\\n", (long long)({macro}));')
    for _, proto in FUNCTIONS:
        _, name, _ = _parse(proto)
        lines.append(f'  printf("symbol {name} %s\\n", XSTR({name}));')
    for ctype, fields in STRUCTS:
        lines.append(f'  printf("size {ctype} %zu\\n", sizeof({ctype}));')
        for field, _, _ in fields:
            lines.append(f'  printf("offset {ctype} {field} %zu\\n", offsetof({ctype}, {field}));')
    for cs, expr, _ in CONSTANTS:
        lines.append(f'  printf("const {cs} %lld\\n", (long long)({expr}));')
    lines.append('  printf("pointer %zu\\n", sizeof(void *));')
    lines.append("  return 0;\n}")
    return "\n".join(lines) + "\n"


def check_pins(include: Path) -> None:
    for header, digest in PINNED_HEADERS.items():
        path = include / header
        if not path.is_file():
            raise SystemExit(f"{path} not found; pass the include directory of the CUDA 13.3 toolkit")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(
                f"{header} is not the pinned header (sha256 {actual}, pinned {digest}); "
                "the bindings are pinned to CUDA 13.3 (nvcc 13.3.73)"
            )


def run_probes(include: Path) -> dict:
    cc = os.environ.get("CC", "cc")
    flags = ["-std=gnu11", "-Wno-deprecated-declarations", f"-I{include}"]
    with tempfile.TemporaryDirectory(prefix="cuda-bindings-") as tmp:
        check = Path(tmp) / "check.c"
        check.write_text(type_check_source())
        subprocess.run(
            [cc, *flags, "-fsyntax-only", "-Werror", "-Wincompatible-pointer-types", str(check)],
            check=True,
        )
        probe = Path(tmp) / "probe.c"
        probe.write_text(probe_source())
        exe = Path(tmp) / "probe"
        subprocess.run([cc, *flags, str(probe), "-o", str(exe)], check=True)
        output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    facts: dict = {"version": {}, "symbol": {}, "size": {}, "offset": {}, "const": {}}
    for line in output.splitlines():
        kind, *rest = line.split()
        if kind == "offset":
            facts["offset"][(rest[0], rest[1])] = int(rest[2])
        elif kind == "pointer":
            facts["pointer"] = int(rest[0])
        elif kind == "symbol":
            facts["symbol"][rest[0]] = rest[1]
        else:
            facts[kind][rest[0]] = int(rest[1])
    return facts


def render(facts: dict) -> str:
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    o: list[str] = []
    w = o.append
    w("// <auto-generated>")
    w("// Generated by tools/generate-cuda-bindings.py from the CUDA toolkit's headers; do not edit by hand.")
    w(f"// Source: {SOURCE}.")
    w("// Header SHA-256: " + ", ".join(f"{h} {d}" for h, d in PINNED_HEADERS.items()) + ".")
    w(
        f"// Layout target: {target}, {facts['pointer'] * 8}-bit pointers. Struct sizes and field offsets are the C compiler's"
    )
    w(
        "// (sizeof/offsetof); function prototypes and field types were type-checked against the headers, and each function"
    )
    w("// is bound under the symbol name the headers resolve it to.")
    w("// </auto-generated>")
    w("")
    w("#pragma warning disable")
    w("#nullable disable")
    w("")
    w("using System.Runtime.InteropServices;")
    w("")
    w(f"namespace {NAMESPACE};")
    w("")
    w("/// <summary>The CUDA toolkit the bindings were generated from.</summary>")
    w("public static class CudaToolkit")
    w("{")
    for macro in VERSION_MACROS:
        w(f"    public const int {macro} = {facts['version'][macro]};")
    w("")
    w(
        "    /// <summary>The CUDA major version: the library names resolve to it unless a manifest names another.</summary>"
    )
    w(f"    public const int Major = {facts['version']['CUDA_VERSION'] // 1000};")
    w("}")
    w("")
    w(
        "/// <summary>The logical library names the bindings import; <c>CudaLibraryResolver</c> maps them to files per OS.</summary>"
    )
    w("public static class CudaLibraries")
    w("{")
    for logical, (_, what) in LIBRARIES.items():
        w(f"    /// <summary>{what}.</summary>")
        w(f'    public const string {logical[0].upper() + logical[1:]} = "{logical}";')
    w("}")
    w("")
    for ctype, fields in STRUCTS:
        size = facts["size"][ctype]
        w(f"/// <summary><c>{ctype}</c> ({size} bytes); only the fields Roost uses, at the headers' offsets.</summary>")
        w(f"[StructLayout(LayoutKind.Explicit, Size = {size})]")
        w(f"public unsafe struct {ctype}")
        w("{")
        for field, _, cstype in fields:
            off = facts["offset"][(ctype, field)]
            if cstype.startswith("fixed "):
                _, elem, count = cstype.split()
                w(f"    [FieldOffset({off})] public fixed {elem} {field}{count};")
            else:
                w(f"    [FieldOffset({off})] public {cstype} {field};")
        w("}")
        w("")
    by_lib: dict[str, list[str]] = {name: [] for name in LIBRARIES}
    for lib, proto in FUNCTIONS:
        by_lib[lib].append(proto)
    for lib, (cls, what) in LIBRARIES.items():
        w(f"/// <summary>{what}: the functions and constants Roost uses.</summary>")
        w(f"public static unsafe partial class {cls}")
        w("{")
        consts = [c for c in CONSTANTS if _owner(c[0]) == lib]
        for cs, _, cstype in consts:
            w(f"    public const {cstype} {cs} = {facts['const'][cs]};")
        if consts:
            w("")
        for proto in by_lib[lib]:
            ret, name, params = _parse(proto)
            symbol = facts["symbol"][name]
            string_params = {n for t, n in params if t.replace(" ", "") == "constchar*"}
            cs_params = ", ".join(
                f"string {_cs_name(n)}" if n in string_params else f"{_cs_type(t)} {_cs_name(n)}" for t, n in params
            )
            w(f"    /// <summary><c>{proto}</c></summary>")
            args = f'"{lib}", EntryPoint = "{symbol}"'
            if string_params:
                args += ", StringMarshalling = StringMarshalling.Utf8"
            w(f"    [LibraryImport({args})]")
            w(f"    public static partial {_cs_type(ret)} {name}({cs_params});")
            w("")
        o[-1:] = []
        w("}")
        w("")
    o[-1:] = []
    return "\n".join(o) + "\n"


def _owner(const_name: str) -> str:
    """The library whose class holds a constant."""
    if const_name.startswith(("cuda", "cudaDev")) and not const_name.startswith("CUDA"):
        return "cudart"
    if const_name.startswith(("CUBLAS", "CUDA_R_")):
        return "cublasLt"
    if const_name.startswith(("CUDA_", "CU_")):
        return "cuda"
    if const_name.startswith(("NVML", "nvml")):
        return "nvml"
    raise SystemExit(f"no library for constant {const_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--include",
        type=Path,
        default=Path(os.environ.get("ROOST_CUDA_INCLUDE", "/opt/cuda/include")),
        help="include directory of the CUDA 13.3 toolkit (default: $ROOST_CUDA_INCLUDE or /opt/cuda/include)",
    )
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args()
    check_pins(args.include)
    facts = run_probes(args.include)
    if facts.get("pointer") != 8:
        raise SystemExit("generate on a 64-bit platform: the bindings are 64-bit layouts only")
    for macro, expected in EXPECTED_VERSIONS.items():
        if facts["version"][macro] != expected:
            raise SystemExit(f"{macro} is {facts['version'][macro]}, expected {expected}")
    text = render(facts)
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} differs from the headers; regenerate it", file=sys.stderr)
            return 1
        print(f"ok:   {OUT.relative_to(ROOT)} matches the headers")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(FUNCTIONS)} functions, {len(STRUCTS)} structs, {len(CONSTANTS)} constants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
