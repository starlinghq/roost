#!/usr/bin/env bash
# Builds libroost_moe_spike.so and the test driver into $ENVS/build (outside the repository). At most two jobs; SM 86 only.
set -euo pipefail
ENVS="${ENVS:-/home/mark/Projects/roost-wave1/envs/R-063}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CUT="$ENVS/src/cutlass"; V="$ENVS/build/vllm-patched"; O="$ENVS/build/out"; mkdir -p "$O"
NV="nvcc -std=c++17 -arch=sm_86 -O2 --expt-relaxed-constexpr -Xcompiler -fPIC -lineinfo"
$NV -I"$V/csrc" -I"$V/csrc/moe" -c "$V/csrc/moe/topk_softmax_kernels.cu" -o "$O/topk.o"
$NV -I"$CUT/include" -I"$CUT/tools/util/include" -c "$HERE/src/roost_moe.cu" -o "$O/roost_moe.o"
$NV -shared "$O/topk.o" "$O/roost_moe.o" -o "$O/libroost_moe_spike.so"
g++ -std=c++17 -O2 -march=native -fopenmp -I"$HERE/src" -I/opt/cuda/include "$HERE/test/moe_layer_test.cpp" \
    -L"$O" -lroost_moe_spike -L/opt/cuda/lib64 -lcudart -Wl,-rpath,"$O" -Wl,-rpath,/opt/cuda/lib64 -o "$O/moe_layer_test"
echo built "$O"
