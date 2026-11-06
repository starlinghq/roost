# Roost container image (R-046, stream J): the Native AOT `roost` binary, the Native AOT `roost-worker` and the per-SM kernel
# library with its manifest, on NVIDIA's CUDA runtime image. Build context: the directory deploy/stage-image.sh writes (never
# the repository). Nothing in this repository builds or pushes the image; see deploy/README.md.
#
#   deploy/stage-image.sh --kernel-assets <dir>
#   docker build -f deploy/Dockerfile -t roost:local artifacts/image-context
#   docker run --rm --gpus all roost:local                       # runs `roost doctor`
#
# Out of date: kernels/manifest.toml now pins cuBLASLt 13.8.0.4 (deploy/README.md).
# Base: nvidia/cuda:13.3.1-runtime-ubuntu24.04, pinned by its multi-platform index digest, read from the Docker Hub registry API
# on 2026-11-02 (GET registry-1.docker.io/v2/nvidia/cuda/manifests/13.3.1-runtime-ubuntu24.04, Docker-Content-Digest header).
# Its linux/amd64 config sets CUDA_VERSION=13.3.1 and NV_LIBCUBLAS_VERSION=13.6.0.2-1: the cuBLASLt version kernels/manifest.toml
# pins ([[external_libraries]] cublaslt 13.6.0.2), which the kernel library links and does not redistribute. The image does
# carry it, so publishing an image is subject to the decision 5 licence review of the CUDA EULA (deploy/README.md).
FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:63da350831208559df18c7b8f3e0d5d1c984eaa815cdde690aacd6606cd0cb11

# The runtime image already sets NVIDIA_VISIBLE_DEVICES=all and NVIDIA_DRIVER_CAPABILITIES=compute,utility; utility mounts
# nvidia-smi and NVML, which `roost doctor` uses. DOTNET_ variables are inert for Native AOT binaries and are not set.
COPY --chown=root:root opt/roost/ /opt/roost/
COPY --chown=root:root SHA256SUMS /opt/roost/SHA256SUMS

# The staged files must be the ones stage-image.sh listed; then the kernel libraries are checked against the manifest by the
# binary itself (no GPU needed with --no-gpu). A mismatch fails the build.
RUN cd /opt/roost && sha256sum --check --strict SHA256SUMS \
    && /opt/roost/roost doctor --no-gpu --min-free-disk-gib 0 --min-free-memory-gib 0 \
    && useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin roost

USER 10001
WORKDIR /opt/roost
# The server listens on loopback unless told otherwise and has no authentication (R-037). In a container it needs
# `--host 0.0.0.0 --allow-non-loopback`, which the operator passes explicitly with the model arguments, behind their own
# network policy. vLLM's port.
EXPOSE 8000
ENTRYPOINT ["/opt/roost/roost"]
CMD ["doctor"]
