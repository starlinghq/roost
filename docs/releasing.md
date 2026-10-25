# Releasing

A Roost release is one tag, `v<version>`, and everything it publishes is built from that tag's commit: the NuGet packages, and the per-SM kernel libraries as GitHub release assets (build plan R-060, decision 12). Pushing the tag runs [`release.yml`](../.github/workflows/release.yml). It builds the kernel libraries on the self-hosted runner's RTX 3090, checks them against `kernels/manifest.toml`, runs the release drill, waits for a maintainer's approval, and then publishes the exact bytes it checked.

Who may release: the maintainer, and the orchestrating agent after a passing drill (kickoff, "Releases", 2026-10-20). Release kernel builds run on this machine's RTX 3090 because the Triton AOT step needs a GPU; RunPod is the fallback.

## Versions

- **Scheme.** SemVer 2. The version comes from the tag: `v0.1.0-preview.1` builds `0.1.0-preview.1`. The workflow passes it to MSBuild as `VersionPrefix` and `VersionSuffix`; `Directory.Build.props` keeps `0.0.0-dev`, so nothing is edited per release and a local build is never mistaken for a released one.
- **One set.** Every Roost package of a release has the same version, and every `Starlings.Roost.*` dependency is an exact `[version]` pin (`Directory.Build.targets`). Starling packages are exact pins too (`Directory.Packages.props`).
- **Immutability.** A published version is never replaced. The workflow pushes without `--skip-duplicate`, so publishing an existing version fails.
- **Kernel libraries belong to the version.** `Starlings.Roost.Kernels` embeds the manifest it was built with, and fetches `libroost_kernels_sm<N>.so` from the GitHub release `v<version>` of the commit in its informational version (`<version>+<commit>`), verified against that manifest's size and SHA-256 (decision 12). A development build (`0.0.0-dev`) records no release and never downloads. Configured search directories are the only source when set (no download fallback), releases publish Linux libraries only (another platform is refused with `ProfileUnsupported`), and the loader maps the bytes it hashed: on Linux a sealed in-memory copy loaded through `/proc/self/fd`, so the file cannot be swapped between the check and the load.

## Which packages ship

The drill packs every project under `src/` whose `IsPackable` is `true`, plus every `Starlings.Roost.*` project a packed package depends on. A `ProjectReference` becomes a package dependency, so without the second rule a consumer could not restore; the drill prints which packages it added that way (`package <id> closure` in `drill.txt`). A new packable project, for example `Starlings.Generation.Roost` (R-036) or a worker executable package (R-062), joins the next release without a change to the workflow or the drill.

Today that is `Starlings.Roost.ModelCompiler` and `Starlings.Roost.Models.Qwen35` (marked), with `Starlings.Roost.Abstractions`, `Starlings.Roost.Graph` and `Starlings.Roost.Ops` (closure).

## Cutting a release

1. **Kernel digests** (below) must match what the release build produces. If the toolchain, the sources or the runner's paths changed since they were recorded, record them first.
2. In a pull request, move the `Unreleased` entries in [`CHANGELOG.md`](../CHANGELOG.md) under `## [<version>] - <YYYY-MM-DD>`. Merge it.
3. Tag the merge commit on the default branch and push the tag:

   ```sh
   git switch plan/build-plan && git pull     # main, once the maintainer creates it
   git tag v0.1.0-preview.1
   git push origin v0.1.0-preview.1
   ```

4. The workflow runs:
   - **candidate** (GitHub-hosted): the boundary check, then refuses a tag that is not `v<SemVer>`, a commit that is not on the default branch, or a version without a `CHANGELOG.md` section.
   - **kernels** (self-hosted, the only job there): `tools/release-kernels.sh` installs the `[triton_aot]` Python from the hash-locked `tools/release-kernels/uv.lock` (its pins must equal the manifest's; torch only from the PyTorch cu130 index), builds every SM of the manifest twice with the Triton AOT cubins (each build through `tools/gpu-lease` on the 3090, stream A), fails unless the two builds are byte-identical, the registration tables load and the GatedDeltaRule AOT variant is present, writes the assets, and **fails unless the digests equal the manifest's**. CI only reports a difference; a release publishes only bytes the manifest records. The assets are the `kernel-assets` artifact.
   - **packages** (GitHub-hosted): the release drill (below) with `--kernel-assets`; it checks the downloaded kernel assets and builds no kernels. The verified packages and logs are the `packages` artifact.
   - **publish** (GitHub-hosted): waits for a reviewer on the `release` environment and checks that the artifacts are the drilled ones (`drill.txt` says `RESULT: PASS` for this commit and version, checksums hold, digests equal the manifest). Then, in this order: (1) creates the GitHub release **as a draft** with the kernel libraries, `manifest.toml`, `kernel-digests.json`, `SHA256SUMS`, `toolchain.txt` and the notes (changelog section, kernel digest table, toolchain, package checksums); (2) gets a short-lived nuget.org key through trusted publishing and pushes every package (each `.snupkg` goes with its `.nupkg`); (3) attaches the packages and publishes the release. A version with a suffix is a pre-release. Assets of a draft release are not public, so the kernel assets become downloadable only at step 3, seconds after the last push; nuget.org validates a pushed package for some minutes before anyone can restore it, so in practice no consumer can install a package before the assets it fetches exist. If the job stops between steps 2 and 3, the packages may become restorable while the release is still a draft; finish step 3 (below) promptly.
5. Approve the deployment in the workflow run.

Running the workflow by hand (Actions > release > Run workflow, with a version) builds the candidate and never publishes. Turn `kernels` off to drill the packages without the GPU.

## A failed publish

Packages are pushed one at a time, and a pushed version is immutable: nuget.org never accepts the same id and version again, and the workflow pushes without `--skip-duplicate`. If the publish job fails part of the way:

- **Before the push** (artifact check, draft release): nothing is public. Fix the cause and rerun the failed job (Actions > the run > Re-run failed jobs). The draft release is reused and its assets replaced.
- **During the push**: the packages pushed so far are published and stay published. Rerun the failed job in the same workflow run. On a rerun (run attempt 2 or later) the job skips a package whose exact version nuget.org already lists only after checking that the published package holds exactly the drilled bytes (every entry equal, ignoring the repository signature nuget.org adds), and pushes the rest. A mismatch fails the job: that is what happens after "Re-run all jobs", which repacks the candidate with new bytes (a nupkg is not byte-reproducible); rerun only the failed publish job of the original run instead. An already-published GitHub release is never modified by a rerun. nuget.org lists a version only after its validation, a few minutes after the push; a rerun before that fails with 409 Conflict on that package, so wait until `https://api.nuget.org/v3-flatcontainer/<id lower-case>/index.json` lists the version and rerun again. Never start a new run for the same tag to resume: its drill would pack new bytes, which must not mix with the published ones.
- **After the push** (attaching the packages, publishing the release): rerun the failed job; every package is skipped as already published, and the draft is completed. Or finish by hand: `gh release upload v<version> --clobber <the packages artifact>/*.nupkg && gh release edit v<version> --draft=false`.
- **A package that must not stay public** (wrong content): unlist it on nuget.org (it stays restorable by exact version), fix, and release the next version. A version is never reused.

## Kernel digests

The kernel library digests are not portable: nvcc hashes the source path into internal symbol names, so a build reproduces only from the same checkout path, work directory and toolchain (kernels/README.md, "Reproducibility and digests"). The release build runs at `$GITHUB_WORKSPACE` (the runner's `_work/roost/roost`) with the work directory `$RUNNER_TEMP/roost-kernels`, on the runner's CUDA 13.3 and host gcc. The digests in `kernels/manifest.toml` today come from a development checkout, so the first release build will not match them. The workflow never records and checks in the same run, because a check against digests it just wrote would prove nothing. To record them:

0. Apply the self-hosted runner restriction first ([branch-protection.md](branch-protection.md), "Release (R-060)", item 5): the recorded digests are the trust anchor for every later release, and they come from this runner, so it must be restricted before the first recording run, not only before the first tag.
1. Run the workflow by hand with `kernels` on, from the commit to be released. The `kernels` job fails at the digest check but still uploads `kernel-assets`, with `toolchain.txt` naming the paths and versions.
2. Download the artifact and, in a stream C pull request, record it: `python3 -I kernels/tools/record_digests.py <artifact dir> --write`. Review the diff (only `[[ops.variants.binaries]]` blocks change) and the update of the comment above them.
3. Merge, then tag. The tag's build must reproduce those bytes; if it does not, the build is not reproducible at that path and the release stops.

## Release drill

```sh
tools/release-drill.sh --version 0.0.1-drill.1 --no-kernels          # any CPU host
tools/release-drill.sh --version 0.1.0-preview.1 --kernel-assets <dir>  # what the workflow runs
DRILL_KEEP=1 DRILL_MSBUILD=-m:2 tools/release-drill.sh ...              # keep the work directory; limit MSBuild nodes
```

The drill builds the committed `HEAD` from a clean clone, with an empty private NuGet cache and without the shared compiler server. It then:

1. Restores `Roost.slnx` in locked mode, builds it, runs every test project of the solution (each xunit v3 executable directly, requiring a passing summary with at least one test) and `tools/check-roost-boundaries.py`.
2. Packs the release set (above) at the given version.
3. Checks every nuspec: file name and version, the `MIT OR Apache-2.0` license expression, the repository URL `https://github.com/starlinghq/roost` and commit, every `Starlings.*` dependency an exact `[x.y.z]` pin, every `Starlings.Roost.*` dependency one of the packed packages at this version, no native binaries or model assets.
4. With `--kernel-assets`: one library per SM of the manifest, the manifest asset equal to the committed file, and the digests equal to the manifest's.
5. Pushes the packages to a temporary folder feed. It never pushes to nuget.org.
6. Writes a consumer outside the checkout (`tools/release-drill/consumer/`) that references every packed package at `[version]`, with a `NuGet.config` listing only that feed and nuget.org (`Starlings.Roost.*` mapped to the feed) and its own empty package cache. It fails if a restored package differs from the packed bytes, if a worker-side package reaches `Microsoft.ML.OnnxRuntime*` (decision 30), or if any consumer file names the repository path. The consumer loads every packaged assembly, makes a typed call into the packages it knows, and prints `RESULT: PASS`.

The packages, `drill.txt` (commit, version, package list, checksums, `RESULT: PASS`) and logs land in `artifacts/release-drill/` (ignored by Git), or in `--out`.

## One-time setup

These are repository and account settings; agents cannot apply them. [branch-protection.md](branch-protection.md), "Release (R-060)", lists them.
