# Changelog

Notable changes to Roost's packages and kernel library releases. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow SemVer 2. A release moves the `Unreleased` entries under `## [<version>] - <YYYY-MM-DD>`; the release workflow refuses a tag without that section and uses it as the release notes ([docs/releasing.md](docs/releasing.md)).

## [Unreleased]

Nothing has been released yet. The first release is Preview 1 (build plan R-062).

### Added

- Release workflow (`.github/workflows/release.yml`), kernel release build (`tools/release-kernels.sh`) and release drill (`tools/release-drill.sh`) (R-060).
- `Starlings.Roost.Generation`: `Qwen3RoostChatTemplate`, the Qwen3 template over Starling's published `QwenChatTemplate` (`QwenChatProfile.Qwen3`, text only, `enable_thinking=false`; experimental), with `Starlings.Models.QwenChat` pinned exactly at `[0.3.0]` (host side only, decision 30). Completes R-009.
- `Starlings.Roost.Ops`: `CpuOps` implements the MoE ops MoeRouteTopK, MoePermute, MoeUnpermute, GroupedGemm and FusedMoe (R-080, CPU half): double-precision references with deterministic tie order (lower expert first), a stable permutation and FusedMoe as the composition of the unfused ops; bf16, fp16 and fp32 weights, FP8 and INT4 `unsupported`. No GPU variant exists yet (`untested`).
- `roost-tiny-moe` (`Starlings.Roost.Models.Tiny`, test and CI only): a Qwen3-MoE-shaped tiny model with seeded weights, run on `CpuOps` in CI.
- `Starlings.Roost.Kernels`: `KernelLibraryResolver` fetches the per-SM kernel library of the package version from the GitHub release through `Starlings.Acquisition`, verified by size and SHA-256 against the embedded manifest (decision 12); `KernelManifest.Embedded()`.
- `Starlings.Roost.Server` and the `roost` executable (R-037, partial): `POST /v1/chat/completions` and `POST /v1/completions` (whole or SSE with `data: [DONE]`) and `GET /v1/models` over `RoostChatClient`; one error body for every failure; unknown and unsupported options are a 400 naming the option; loopback only unless `--allow-non-loopback`; client disconnect cancels the generation. CPU-tested only; every route is `untested` (`src/Starlings.Roost.Server/STATUS.md`).
- `Starlings.Generation.Roost`: `RoostChatClient.GetCompletionAsync` and `GetStreamingCompletionAsync` complete a raw prompt without the chat template (capability `raw_completion`, `untested`).

### Changed

- `Starlings.Tokenizers` is a direct, exact pin (`[0.3.0]`, the version the Starling 0.3.0 packages require) in `Starlings.Generation.Roost` and `Starlings.Roost.Generation`, because `Starlings.Tokenizers.Tokenizer` is in the public API (`RoostModelProfile`); it was only transitive before, so `tools/check-roost-boundaries.py` never saw it.
- `Starlings.Roost.Generation`: `Qwen35RoostChatTemplate` builds a `Qwen35ToolCall` from the cloned arguments element directly instead of enumerating and re-serialising its properties. Same rendering.
- Documentation: the generation layer's status, build plan section 3.1 and its section 12 risk, and the kickoff log record that Starling published `Starlings.Models.QwenChat`, `Starlings.Models.Kev` and the public `StopSequenceMatcher` in ai-dotnet 0.3.0 (starling#124), and that `RoostCapabilities` carries `stop_sequences` as `untested`; `evidence/registry/README.md` states what the Qwen3.5-0.8B admission records for the tokenizer asset (`report_to_starling.status = "not needed"`).
