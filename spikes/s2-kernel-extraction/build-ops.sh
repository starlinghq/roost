#!/usr/bin/env bash
# Build the non-attention ops (FlashInfer headers + patched vLLM layernorm) and their C ABI test.
# Usage: build-ops.sh [sm]  (default 86). Outputs go to $ENVS/build/sm<N>, never into the repository.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS="${ROOST_S2_ENVS:-/home/mark/Projects/roost-wave1/envs/R-006}"
SM="${1:-86}"
OUT="$ENVS/build/sm$SM"
W="$ENVS/work/vllm-patched"
mkdir -p "$OUT"
# Copy only the vLLM files the layernorm kernels include, then apply the patch set.
rm -rf "$W" && mkdir -p "$W/csrc/quantization" "$W/csrc/core"
V="$ENVS/src/vllm/csrc"
cp "$V/layernorm_kernels.cu" "$V/pos_encoding_kernels.cu" "$V/dispatch_utils.h" "$V/cub_helpers.h" "$V/type_convert.cuh" "$V/cuda_vec_utils.cuh" "$W/csrc/"
cp "$V/core/utils.hpp" "$V/core/cuda_compat.h" "$W/csrc/core/"
cp "$V/quantization/vectorization_utils.cuh" "$V/quantization/vectorization.cuh" "$W/csrc/quantization/"
for p in "$HERE"/patches/vllm-*.patch; do patch -s -p1 -d "$W" < "$p"; done
echo "patch lines (added+removed): $(cat "$HERE"/patches/vllm-*.patch | grep -cE '^[+-][^+-]')"
GEN="-gencode arch=compute_${SM},code=sm_${SM}"
FLAGS="-std=c++17 -O2 -DNDEBUG --expt-relaxed-constexpr -Xcompiler -fPIC"
s=$(date +%s)
nvcc $FLAGS $GEN -I"$ENVS/src/flashinfer/include" -I"$HERE/src" -I"$HERE/shim" -I"$W/csrc" \
  -c "$HERE/src/roost_s2_ops.cu" -o "$OUT/roost_s2_ops.o"
e=$(date +%s)
# pos_encoding_kernels.cu: compile-only check that the patched file parses without torch (no kernel run).
printf '#include "pos_encoding_kernels.cu"\n' > "$W/pe_probe.cu"
nvcc -std=c++17 $GEN -I"$W/csrc" -I"$HERE/shim" -c "$W/pe_probe.cu" -o "$W/pe_probe.o" && echo "vllm pos_encoding_kernels.cu: compiles torch-free (not run)"
echo "ops compile: $((e-s)) s, $(stat -c %s "$OUT/roost_s2_ops.o") bytes"
nvcc $GEN -shared -o "$OUT/libroost_s2_ops.so" "$OUT/roost_s2_ops.o" -Xcompiler -fPIC
nvcc -std=c++17 -O1 -I"$HERE/src" "$HERE/test/test_ops.cpp" -o "$OUT/test_ops" \
  -L"$OUT" -lroost_s2_ops -lroost_s2_attn -lcudart -Xlinker -rpath,"$OUT"
ls -l "$OUT/libroost_s2_ops.so"
