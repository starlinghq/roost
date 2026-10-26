# Changelog

Notable changes to Roost's packages and kernel library releases. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow SemVer 2. A release moves the `Unreleased` entries under `## [<version>] - <YYYY-MM-DD>`; the release workflow refuses a tag without that section and uses it as the release notes ([docs/releasing.md](docs/releasing.md)).

## [Unreleased]

Nothing has been released yet. The first release is Preview 1 (build plan R-062).

### Added

- Release workflow (`.github/workflows/release.yml`), kernel release build (`tools/release-kernels.sh`) and release drill (`tools/release-drill.sh`) (R-060).
- `Starlings.Roost.Kernels`: `KernelLibraryResolver` fetches the per-SM kernel library of the package version from the GitHub release through `Starlings.Acquisition`, verified by size and SHA-256 against the embedded manifest (decision 12); `KernelManifest.Embedded()`.

### Changed

- `Starlings.Tokenizers` is a direct, exact pin (`[0.3.0]`, the version the Starling 0.3.0 packages require) in `Starlings.Generation.Roost` and `Starlings.Roost.Generation`, because `Starlings.Tokenizers.Tokenizer` is in the public API (`RoostModelProfile`); it was only transitive before, so `tools/check-roost-boundaries.py` never saw it.
- `Starlings.Roost.Generation`: `Qwen35RoostChatTemplate` builds a `Qwen35ToolCall` from the cloned arguments element directly instead of enumerating and re-serialising its properties. Same rendering.
- Documentation: the generation layer's status, build plan section 3.1 and its section 12 risk, and the kickoff log record that Starling published `Starlings.Models.QwenChat`, `Starlings.Models.Kev` and the public `StopSequenceMatcher` in ai-dotnet 0.3.0 (starling#124), and that `RoostCapabilities` carries `stop_sequences` as `untested`; `evidence/registry/README.md` states what the Qwen3.5-0.8B admission records for the tokenizer asset (`report_to_starling.status = "not needed"`).
