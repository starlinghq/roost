# Repository settings for the maintainer

Agents do not change repository settings, permissions or secrets (AGENTS.md). These are the settings R-003 asks the maintainer to apply by hand in starlinghq/roost.

## Default branch

The default branch today is `plan/build-plan`, and there is no `main`. Create `main` from the merged plan and the R-003 skeleton, and make it the default. `ci.yml` runs on pushes to `main` and `plan/**` and on every pull request.

## Branch protection on `main`

- Require a pull request before merging, with one approving review (the maintainer).
- Require these status checks to pass, and require branches to be up to date:
  - `boundary`
  - `dotnet (ubuntu-latest)`
  - `dotnet (windows-latest)`
  - `tools (pytest, ruff)`
- Require linear history, or allow merge commits only for integration PRs (as in starling).
- Block force pushes and deletion.
- Do not allow bypassing the above settings.

## Actions

- Workflow permissions: read repository contents by default; do not allow GitHub Actions to approve pull requests.
- The repository is public (decision 5, maintainer, 2026-10-20). Set "Require approval for all external contributors" (Settings > Actions > General > Approval for running fork pull request workflows), because pull-request CI runs on the persistent self-hosted runner.
- Pin actions by full SHA (`ci.yml` does).

## Release (R-060)

`.github/workflows/release.yml` runs on `v*` tags and by hand, never on pull requests. Only its `publish` job asks for `id-token: write` and `contents: write`, and it holds no secret: the nuget.org key is minted per run by trusted publishing. `candidate`, `packages` and `publish` run on fresh GitHub-hosted runners (`ubuntu-latest`); only `kernels`, which needs the RTX 3090, runs on the self-hosted runner, holds no token beyond read, and its output is checked against the committed manifest by the hosted jobs. Procedure: [releasing.md](releasing.md). The maintainer applies:

1. **`release` environment** (Settings > Environments > New environment, name `release`):
   - Required reviewers: the maintainer. Do not allow self-review to be skipped by administrators.
   - Deployment branches and tags: selected tags only, pattern `v*`.
   - Environment variable `NUGET_USER`: the nuget.org **user name of the person who created the trusted publishing policy** below, not the organization's name (with the organization's name the token exchange fails with "No matching trust policy owned by user ... was found"; starlinghq/tokenizers' lesson).
   - No secrets.
2. **nuget.org trusted publishing policy**, created by that user (an administrator of the `starlinghq` organization on nuget.org, whose organization has a confirmed email address):

   | Field | Value |
   | --- | --- |
   | Package owner | `starlinghq` |
   | Repository owner | `starlinghq` |
   | Repository | `roost` |
   | Workflow file | `release.yml` |
   | Environment | `release` |

   The packages use the `Starlings.Roost.` prefix; the `Starlings` prefix reservation requested for starlinghq/starling covers it.
3. **Tag protection for `v*`** (a repository ruleset targeting tags `v*`): restrict creation, update and deletion to the maintainer and the release agent's account; block force pushes. A published tag is never moved.
4. **Default branch.** The candidate job requires the tagged commit to be on the repository's default branch (`github.event.repository.default_branch`), today `plan/build-plan`; when `main` is created and made the default, tags must be on `main`.
5. **Self-hosted runner** (`[self-hosted, omarchy]`), for the `kernels` job:
   - **Restrict it, before the first digest-recording run** (item 6), not only before the first tag: the digests recorded in `kernels/manifest.toml` come from a `kernels` artifact built on this runner, and every later release is checked against them, so they are only as trustworthy as the runner was when they were built. The same persistent runner runs pull-request CI from a public repository, and a fork pull request, once approved, can leave changes behind on it (tool caches, `~/.dotnet`, uv or pipx environments, shell start-up files) for the release build to pick up. Either put a dedicated runner (its own user and work directory) in a runner group limited to the `release.yml` workflow (Settings > Actions > Runner groups > selected workflows) under a label only `release.yml` uses, and change `kernels`' `runs-on` to that label in the same PR; or, at the least, keep "Require approval for all external contributors" (Actions, above) and review every fork workflow run before approving it. The release still only publishes kernel bytes whose digests a reviewed PR recorded in `kernels/manifest.toml`, and the hosted `packages` and `publish` jobs check them again.
   - **Tools:** the RTX 3090 visible to `nvidia-smi` and `tools/gpu-lease` under the runner's user, CUDA 13.3 (`nvcc`), gcc, cmake 3.24 or later, ninja, Python 3.11 or later, git, and `uv` or `pipx` (the job installs the `[triton_aot]` Python 3.12.14 environment from the hash-locked `tools/release-kernels/uv.lock`, torch from the PyTorch cu130 index only). Optional: installing the NVIDIA container toolkit would let the kernel build move into the pinned CUDA container that `ci.yml` uses; today the runner's Docker has the `runc` runtime only, and the Triton AOT step needs the GPU during the build.
6. **Kernel digests.** After item 5 is applied and before the first tag, record the release build's digests in `kernels/manifest.toml` through a pull request (releasing.md, "Kernel digests"). Not a setting, but the release fails until it is done.
7. **Packaging check.** `.github/workflows/packaging.yml` (job `packaging drill (advisory)`) runs the drill without tests or kernels on pull requests that touch packaging files, on GitHub-hosted runners. Leave it out of the required checks until it has run green for a while; then consider adding it.
