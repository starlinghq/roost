# Changelog

Notable changes to Roost's packages and kernel library releases. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow SemVer 2. A release moves the `Unreleased` entries under `## [<version>] - <YYYY-MM-DD>`; the release workflow refuses a tag without that section and uses it as the release notes ([docs/releasing.md](docs/releasing.md)).

## [Unreleased]

Nothing has been released yet. The first release is Preview 1 (build plan R-062).

### Added

- Release workflow (`.github/workflows/release.yml`), kernel release build (`tools/release-kernels.sh`) and release drill (`tools/release-drill.sh`) (R-060).
- `Starlings.Roost.Kernels`: `KernelLibraryResolver` fetches the per-SM kernel library of the package version from the GitHub release through `Starlings.Acquisition`, verified by size and SHA-256 against the embedded manifest (decision 12); `KernelManifest.Embedded()`.
