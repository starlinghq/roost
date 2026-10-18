#!/usr/bin/env bash
# Builds and runs the host expert GEMV benchmarks (CPU only). Records host state and the exact invocations in the logs.
# Usage: run-gemv.sh [reps]   (default 7: the report's "best of 7 repetitions"; both programs default to 5 if run bare)
set -euo pipefail
ENVS="${ENVS:-/home/mark/Projects/roost-wave1/envs/R-063}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${LOGDIR:-$HERE/results}"; mkdir -p "$LOGDIR"
REPS="${1:-7}"
export CARGO_BUILD_JOBS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export RUSTFLAGS="-C target-cpu=native"      # the report's "built with -C target-cpu=native"
export CARGO_TARGET_DIR="$ENVS/build/rust"
(cd "$HERE/host-gemv/rust" && cargo build --release --locked)
dotnet build "$HERE/host-gemv/dotnet/GemvBench.csproj" -c Release -o "$ENVS/build/dotnet/bin" -m:1 -nologo -v:q
hdr() { echo "# $(date -Is) host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs) nproc=$(nproc)"
        echo "# rustc: $(rustc --version); RUSTFLAGS='$RUSTFLAGS'; dotnet sdk: $(dotnet --version)"
        echo "# reps=$REPS; load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"; }
{ hdr
  for t in 1 4; do echo "# cmd: dotnet GemvBench.dll $t $REPS"; dotnet "$ENVS/build/dotnet/bin/GemvBench.dll" "$t" "$REPS"; echo "# load average after: $(cut -d' ' -f1-3 /proc/loadavg)"; done
} 2>&1 | tee "$LOGDIR/gemv-dotnet.log"
{ hdr
  for t in 1 4; do echo "# cmd: gemv-bench $t $REPS"; "$ENVS/build/rust/release/gemv-bench" "$t" "$REPS"; echo "# load average after: $(cut -d' ' -f1-3 /proc/loadavg)"; done
} 2>&1 | tee "$LOGDIR/gemv-rust.log"
