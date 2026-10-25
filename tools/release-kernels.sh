#!/usr/bin/env bash
# Kernel release build (R-060, docs/releasing.md): the per-SM kernel libraries of kernels/manifest.toml with the Triton
# AOT GatedDeltaRule cubins, as release assets. The release workflow runs it on the self-hosted runner with the RTX 3090.
#
#   tools/release-kernels.sh <work-dir> <assets-dir>
#
# 1. Creates the build-time Python of the manifest's [triton_aot] closure (exact pins, uv) unless ROOST_TRITON_PYTHON
#    names one.
# 2. Configures and builds every SM of the manifest twice, into <work-dir>/a and <work-dir>/b, with the AOT step. The AOT
#    step needs a visible CUDA device (kernels/aot/build_gdn.py), so each build runs through tools/gpu-lease on the 3090.
# 3. Fails unless the two builds are byte-identical per SM and each library's registration table loads
#    (roost_stub_check --no-gpu).
# 4. Writes the assets: libroost_kernels_sm<N>.so per SM, manifest.toml, kernel-digests.json, SHA256SUMS, toolchain.txt,
#    and the Cargo.lock of every crate under native/ (none yet).
# 5. Checks the digests against kernels/manifest.toml and FAILS on a difference. CI only reports one
#    (kernels/ci/build-all-sm.sh); a release publishes exactly the bytes the manifest records. The assets are written
#    before the check, so a first build at a new path still gives the libraries to record (docs/releasing.md).
#
# Why not the CI container: the AOT step needs the GPU inside the build, and this runner's Docker has no NVIDIA
# container runtime (runc only), so the build runs on the runner's own toolchain (CUDA 13.3, the host gcc). Digests
# depend on the checkout path and the work directory (nvcc hashes source paths into symbol names; kernels/README.md),
# so they reproduce only from the same paths: the workflow uses $GITHUB_WORKSPACE and $RUNNER_TEMP/roost-kernels.
#
# Environment: ROOST_TRITON_PYTHON (skip step 1), ROOST_GPU_DEVICE (default 3090), ROOST_GPU_WAIT (seconds, default
# 3600), CMAKE_BUILD_PARALLEL_LEVEL (default 2).
set -euo pipefail

work="$(mkdir -p "${1:?usage: release-kernels.sh <work-dir> <assets-dir>}" && cd "$1" && pwd)"
assets="$(mkdir -p "${2:?usage: release-kernels.sh <work-dir> <assets-dir>}" && cd "$2" && pwd)"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$root/kernels/manifest.toml"
device="${ROOST_GPU_DEVICE:-3090}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

fail() {
    echo "release-kernels: $*" >&2
    exit 1
}

sms="$(python3 -I "$root/kernels/tools/manifest_cmake.py" "$manifest" --sm-list)"
echo "SM list:   $sms"
echo "checkout:  $root"
echo "work dir:  $work"

# 1. The [triton_aot] closure, exactly as the manifest pins it. FLA itself is not installed: build_gdn.py imports it from
# the checkout CMake fetches at the manifest's SHA.
python="${ROOST_TRITON_PYTHON:-}"
if [[ -z "$python" ]]; then
    uv=(uv)
    command -v uv >/dev/null || uv=(pipx run uv==0.12.19)
    read -r py_version requirements < <(python3 -I - "$manifest" <<'EOF'
import sys, tomllib
aot = tomllib.load(open(sys.argv[1], "rb"))["triton_aot"]
pins = {k: v for k, v in aot["packages"].items() if k != "fla"}
print(aot["python"], " ".join(f"{k}=={v}" for k, v in sorted(pins.items())))
EOF
)
    venv="$work/triton-venv"
    if [[ ! -x "$venv/bin/python" ]]; then
        "${uv[@]}" venv --python "$py_version" "$venv"
        # torch +cu130 wheels come from the PyTorch index; everything else from PyPI.
        # shellcheck disable=SC2086 # requirements is a list of name==version words
        VIRTUAL_ENV="$venv" "${uv[@]}" pip install --python "$venv/bin/python" \
            --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
            --index-strategy unsafe-best-match $requirements
    fi
    python="$venv/bin/python"
fi
"$python" -I -c 'import sys, torch, triton; print("python", sys.version.split()[0], "torch", torch.__version__, "triton", triton.__version__)'

# 2. Two AOT builds of every SM, through the GPU lease (exit 75: the device stayed busy or short of memory; try later).
lease=("$root/tools/gpu-lease" run --device "$device" --need-mib 1500 --stream A --wait "${ROOST_GPU_WAIT:-3600}" --)
for side in a b; do
    rm -rf "${work:?}/$side"
    cmake -S "$root/kernels" -B "$work/$side" -G Ninja -DROOST_SM_LIST="$sms" \
        -DROOST_UPSTREAM_DIR="$work/upstream" -DROOST_TRITON_PYTHON="$python"
    "${lease[@]}" cmake --build "$work/$side"
done

# 3. Byte-identical builds and loadable registration tables.
IFS=';' read -ra list <<<"$sms"
for sm in "${list[@]}"; do
    lib="libroost_kernels_sm${sm}.so"
    [[ -f "$work/a/$lib" ]] || fail "sm_${sm}: $lib was not built"
    "$work/a/roost_stub_check" "$work/a/$lib" --no-gpu
    cmp -s "$work/a/$lib" "$work/b/$lib" || fail "sm_${sm}: two builds of the same sources differ"
    grep -q gated-delta-rule-chunk-aot "$work/a/$lib" || fail "sm_${sm}: no GatedDeltaRule AOT variant in $lib"
done

# 4. Assets.
rm -f "$assets"/libroost_kernels_sm*.so
for sm in "${list[@]}"; do
    cp "$work/a/libroost_kernels_sm${sm}.so" "$assets/"
done
cp "$manifest" "$assets/manifest.toml"
python3 -I "$root/kernels/tools/record_digests.py" "$assets" --json "$assets/kernel-digests.json"
if [[ -d "$root/native" ]]; then
    while read -r lock; do
        crate="$(basename "$(dirname "$lock")")"
        cp "$lock" "$assets/Cargo.lock.$crate"
    done < <(find "$root/native" -name Cargo.lock | sort)
fi
{
    echo "commit $(git -C "$root" rev-parse HEAD)"
    echo "checkout $root"
    echo "work $work"
    echo "sm $sms"
    nvcc --version | tail -2
    gcc --version | head -1
    cmake --version | head -1
    "$python" -I -c 'import sys, torch, triton; print("triton", triton.__version__, "torch", torch.__version__, "python", sys.version.split()[0])'
} >"$assets/toolchain.txt"
(cd "$assets" && sha256sum -- libroost_kernels_sm*.so manifest.toml kernel-digests.json toolchain.txt Cargo.lock.* 2>/dev/null) \
    >"$assets/SHA256SUMS" || true
cat "$assets/SHA256SUMS"

# 5. A release publishes exactly the bytes the manifest records.
if ! python3 -I "$root/kernels/tools/record_digests.py" "$assets" --check; then
    fail "the built libraries differ from kernels/manifest.toml; record these digests in a PR first (docs/releasing.md, 'Kernel digests')"
fi
echo "release-kernels: ${#list[@]} SM libraries match kernels/manifest.toml"
