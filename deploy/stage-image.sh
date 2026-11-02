#!/usr/bin/env bash
# Stages the build context of Roost's container image (R-046, deploy/README.md). It builds nothing into an image and
# pushes nothing: it publishes the two Native AOT executables and places them with the kernel manifest and the per-SM kernel
# libraries in one directory, which deploy/Dockerfile copies as is.
#
#   deploy/stage-image.sh --kernel-assets <dir> [--out <dir>]
#
#   --kernel-assets  a directory holding libroost_kernels_sm<N>.so for every SM of kernels/manifest.toml: the release assets
#                    (R-060) or a local kernel build. Their digests must equal the manifest's (record_digests.py --check),
#                    the same rule the release drill applies.
#   --out            the context directory (default <repo>/artifacts/image-context/, ignored by Git). It is emptied first.
#
# Then, on a machine with Docker (not run by this script):
#   docker build -f deploy/Dockerfile -t roost:local <out>
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
kernel_assets=""
out="$root/artifacts/image-context"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    exit 2
}

while (($#)); do
    case "$1" in
        --kernel-assets) kernel_assets="$(cd "${2:?--kernel-assets needs a directory}" && pwd)"; shift 2 ;;
        --out) out="${2:?--out needs a directory}"; shift 2 ;;
        -h | --help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

fail() {
    echo "stage-image: $*" >&2
    exit 1
}

[[ -n "$kernel_assets" ]] || fail "--kernel-assets is required: the image carries the kernel library"
dotnet="${DOTNET:-dotnet}"

manifest="$root/kernels/manifest.toml"
sms="$(python3 -I "$root/kernels/tools/manifest_cmake.py" "$manifest" --sm-list)"
IFS=';' read -ra sm_list <<<"$sms"
for sm in "${sm_list[@]}"; do
    [[ -f "$kernel_assets/libroost_kernels_sm$sm.so" ]] || fail "no libroost_kernels_sm$sm.so in $kernel_assets"
done
python3 -I "$root/kernels/tools/record_digests.py" "$kernel_assets" --check \
    || fail "the kernel libraries in $kernel_assets differ from kernels/manifest.toml"

rm -rf "$out"
mkdir -p "$out/publish/roost" "$out/publish/worker" "$out/opt/roost"

# Both executables are Native AOT for linux-x64 (S1, R-062). The worker is published from its own project: the server's
# publish does not carry it (the generation layer references the runtime project only for packing).
"$dotnet" publish "$root/src/Starlings.Roost.Server/Starlings.Roost.Server.csproj" -c Release -r linux-x64 \
    -o "$out/publish/roost" -p:DebugType=none
"$dotnet" publish "$root/src/Starlings.Roost.Worker.Runtime/Starlings.Roost.Worker.Runtime.csproj" -c Release -r linux-x64 \
    --self-contained -o "$out/publish/worker" -p:DebugType=none

install -m 0755 "$out/publish/roost/roost" "$out/opt/roost/roost"
install -m 0755 "$out/publish/worker/roost-worker" "$out/opt/roost/roost-worker"
# The worker reads the manifest beside it (RoostWorkerRuntime.ManifestFileName) and loads the libraries from its directory.
install -m 0644 "$manifest" "$out/opt/roost/roost-kernels.manifest.toml"
for sm in "${sm_list[@]}"; do
    install -m 0644 "$kernel_assets/libroost_kernels_sm$sm.so" "$out/opt/roost/"
done
rm -rf "$out/publish"

( cd "$out/opt/roost" && sha256sum -- * >"$out/SHA256SUMS" )
echo "staged $out:"
cat "$out/SHA256SUMS"
du -sb "$out/opt/roost"
