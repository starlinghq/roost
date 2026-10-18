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
- Require approval for workflows from outside collaborators (the repository is private until the licence review; decision 5).
- Pin actions by full SHA (`ci.yml` does).

## Later, with R-060

- A `release` environment with required reviewers (the maintainer) and NuGet trusted publishing, as in starlinghq/tokenizers. No secrets reachable from pull request jobs.
- Tag protection for `v*`.
