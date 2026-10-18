#!/usr/bin/env bash
# Fetches the pinned upstream sources for spike S6 into $ENVS (outside the repository) and applies the patch set.
set -euo pipefail
ENVS="${ENVS:-/home/mark/Projects/roost-wave1/envs/R-063}"
CUTLASS_SHA=098de2a652cf8f00fd70b2df54051c7eccbb855a   # NVIDIA/cutlass v4.8.0
VLLM_SHA=7879f3687b5b6a4be4754242e70b9c21cf7b4183      # vllm-project/vllm main on 2026-10-17
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ENVS/src"
fetch() { # dir url sha [sparse patterns...]
  local d="$ENVS/src/$1" url="$2" sha="$3"; shift 3
  [ -d "$d/.git" ] || git init -q "$d"; git -C "$d" remote get-url o >/dev/null 2>&1 || git -C "$d" remote add o "$url"
  if [ $# -gt 0 ]; then git -C "$d" config core.sparseCheckout true; git -C "$d" sparse-checkout set --no-cone "$@"; fi
  git -C "$d" fetch -q --depth 1 o "$sha"; git -C "$d" checkout -q FETCH_HEAD
  [ "$(git -C "$d" rev-parse HEAD)" = "$sha" ]
}
fetch cutlass https://github.com/NVIDIA/cutlass "$CUTLASS_SHA"
fetch vllm https://github.com/vllm-project/vllm "$VLLM_SHA" 'csrc/*.h' 'csrc/*.cuh' 'csrc/moe' 'csrc/core' 'csrc/quantization/marlin' 'csrc/quantization/vectorization*'
# Patched copy of the vLLM top-k kernel (torch launcher replaced by a C entry point).
B="$ENVS/build/vllm-patched"; rm -rf "$B"; mkdir -p "$B/csrc/moe"
cp -r "$ENVS/src/vllm/csrc/core" "$B/csrc/"; mkdir -p "$B/csrc/quantization" "$B/csrc/moe/marlin_moe_wna16"
cp -r "$ENVS/src/vllm/csrc/quantization/marlin" "$B/csrc/quantization/"; cp "$ENVS/src/vllm/csrc/moe/marlin_moe_wna16/"*.h "$B/csrc/moe/marlin_moe_wna16/"; cp "$ENVS/src/vllm/csrc/cub_helpers.h" "$B/csrc/"
cp "$ENVS/src/vllm/csrc/moe/topk_softmax_kernels.cu" "$ENVS/src/vllm/csrc/moe/moeTopKFuncs.cuh" "$B/csrc/moe/"
mkdir -p "$B/csrc/moe/permute_unpermute_kernels"
cp "$ENVS/src/vllm/csrc/moe/permute_unpermute_kernels/"* "$B/csrc/moe/permute_unpermute_kernels/"
(cd "$B" && patch -p1 < "$HERE/patches/vllm-topk-softmax.patch" && patch -p1 < "$HERE/patches/vllm-permute-unpermute.patch" && patch -p1 < "$HERE/patches/vllm-scalar-type.patch")
echo "ready: $B"
