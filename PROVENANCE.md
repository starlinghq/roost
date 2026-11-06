# Provenance

Every upstream source Roost builds from, pinned by full commit SHA: origin, paths taken, revision, licence and local changes (patch sets under `kernels/upstream/<name>.patches/`). A SHA bump changes this file and `THIRD-PARTY-NOTICES.md` in the same pull request; R-016 adds a check that enforces it.

| Upstream | Repository | Revision | Paths | Licence | Local changes |
| --- | --- | --- | --- | --- | --- |
| starling `tools/check-boundary.py` | https://github.com/starlinghq/starling | `9a5c143367e3368b9b1cd531887a0f0cb3245b0b` | `tools/check-boundary.py` | MIT OR Apache-2.0 | None (byte-identical) |
| starling `tools/create_backlog_issues.py` | https://github.com/starlinghq/starling | `9a5c143367e3368b9b1cd531887a0f0cb3245b0b` | `tools/create_backlog_issues.py` | MIT OR Apache-2.0 | Adapted to the build plan's section 10 table and labels |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94` | `include/**` (tag v0.7.2; headers used by R-017: `norm`, `activation`, paged batch decode and batch prefill attention with their scheduler) | Apache-2.0 | None (no patch set); fetched at build time, not committed |
| NVIDIA CUTLASS | https://github.com/NVIDIA/cutlass | `098de2a652cf8f00fd70b2df54051c7eccbb855a` | `include/**` (tag v4.8.0; Gemm and grouped Gemm headers) | BSD-3-Clause | None (no patch set); fetched at build time, not committed |
| vLLM | https://github.com/vllm-project/vllm | `7879f3687b5b6a4be4754242e70b9c21cf7b4183` | `csrc/**` (fallback for norm, rotary, MoE routing, Marlin) | Apache-2.0 | None yet: the torch-free patches of S2 and S6 move to `kernels/upstream/vllm.patches/` with the op that adopts them |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | `329b534245d3d7bb27cc03a99e87bdc51fb28d06` | `fla/**` (Triton kernels for Gated DeltaNet, compiled ahead of time; fla 0.6.0) | MIT | None (no patch set); inputs of the Triton AOT build (R-034) |
| llguidance (`llguidance`, `toktrie`, `toktrie_hf_tokenizers`) | https://github.com/guidance-ai/llguidance | `6cef6ed9c973c2a155a85f2bb1996320af28b289` | the three crates (crate 1.9.1), as Cargo git dependencies of `native/roost-grammar` (R-051); features `lark`, `rayon`, `referencing`, `ahash` | MIT | None; fetched by Cargo at build time, not committed. Its 106 crates.io dependencies are pinned by version and checksum in `native/roost-grammar/Cargo.lock` and listed in `THIRD-PARTY-NOTICES.md` |

The kernel pins above are the ones in `kernels/manifest.toml`, which the CMake build reads; nothing from them is committed
in this repository. `kernels/tools/check_provenance.py` fails a pin or patch-set change that does not also change this file
and `THIRD-PARTY-NOTICES.md`. Digests of the built libraries are in the manifest; the libraries are release assets.

cuBLASLt is not a source Roost adopts: the kernel library links `libcublasLt.so.13` of the installed CUDA toolkit (13.8.0.4 with CUDA 13.4.92 on the development host; its digest and the Gemm algorithms pinned for it are in `kernels/manifest.toml`, `external_libraries` and `gemm_algos`). Its licence is the CUDA toolkit EULA; whether the library may be redistributed is decision 5's licence review.

The llguidance pin is the one in `native/roost-grammar/Cargo.toml` and `Cargo.lock` (decision 22), the revision spike S10 measured (`docs/m0/roost-s10-llguidance-abi.md`). The built `roost_grammar` library is a release asset; nothing built is committed.
