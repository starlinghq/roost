#!/usr/bin/env bash
# Compile-only checks (no GPU, nothing run) for the Marlin INT4, Marlin FP8-weight and vLLM permute instantiations, and a
# timed build of the CUTLASS and top-k translation units. Needs fetch-upstream.sh first. One nvcc at a time.
# Writes a log to $LOGDIR (default: results/ next to this script).
set -euo pipefail
ENVS="${ENVS:-/home/mark/Projects/roost-wave1/envs/R-063}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${LOGDIR:-$HERE/results}"; mkdir -p "$LOGDIR"
CUT="$ENVS/src/cutlass"; V="$ENVS/build/vllm-patched"; O="$ENVS/build/smoke"; mkdir -p "$O"
NV="nvcc -std=c++17 -arch=sm_86 -O2 --expt-relaxed-constexpr -Xcompiler -fPIC -lineinfo"
M="python3 $HERE/measure.py"
{
  echo "nvcc: $(nvcc --version | tail -1)"; echo "gcc: $(gcc --version | head -1)"
  $M $NV -I"$V/csrc/moe/marlin_moe_wna16" -I"$V/csrc" -c "$HERE/src/marlin_moe_int4_smoke.cu" -o "$O/marlin_int4.o"
  $M $NV -I"$V/csrc/moe/marlin_moe_wna16" -I"$V/csrc" -c "$HERE/src/marlin_moe_fp8w_smoke.cu" -o "$O/marlin_fp8w.o"
  $M $NV -I"$V/csrc" -I"$CUT/include" -c "$HERE/src/vllm_permute_smoke.cu" -o "$O/vllm_permute.o"
  echo "HMMA instructions in Marlin INT4 object: $(cuobjdump -sass "$O/marlin_int4.o" | grep -c HMMA)"
  echo "HMMA instructions in Marlin FP8-weight object: $(cuobjdump -sass "$O/marlin_fp8w.o" | grep -c HMMA)"
  $M $NV -I"$CUT/include" -I"$CUT/tools/util/include" -c "$HERE/src/roost_moe.cu" -o "$O/roost_moe.o"
  $M $NV -I"$V/csrc" -I"$V/csrc/moe" -c "$V/csrc/moe/topk_softmax_kernels.cu" -o "$O/topk.o"
} 2>&1 | tee "$LOGDIR/build-smokes.log"
