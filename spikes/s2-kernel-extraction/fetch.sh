#!/usr/bin/env bash
# Fetch the pinned upstream sources by full commit SHA. Nothing here enters the repository.
set -euo pipefail
ENVS="${ROOST_S2_ENVS:-/home/mark/Projects/roost-wave1/envs/R-006}"
mkdir -p "$ENVS/src"
fetch() { # name repo sha
  local d="$ENVS/src/$1"
  [ -d "$d/.git" ] || { mkdir -p "$d"; git -C "$d" init -q; }
  if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null || true)" != "$3" ]; then
    git -C "$d" fetch -q --depth 1 "https://github.com/$2" "$3"
    git -C "$d" checkout -q FETCH_HEAD
  fi
  test "$(git -C "$d" rev-parse HEAD)" = "$3"
  echo "$1 $3"
}
fetch flashinfer flashinfer-ai/flashinfer d84335d8c54e0c8f9f7e22faa15c7e34a58abf94   # v0.7.2
fetch vllm vllm-project/vllm 7879f3687b5b6a4be4754242e70b9c21cf7b4183                # main on 2026-10-17
fetch cutlass NVIDIA/cutlass 098de2a652cf8f00fd70b2df54051c7eccbb855a                # v4.8.0
