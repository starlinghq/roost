# Provenance

Every upstream source Roost builds from, pinned by full commit SHA: origin, paths taken, revision, licence and local changes (patch sets under `kernels/upstream/<name>.patches/`). A SHA bump changes this file and `THIRD-PARTY-NOTICES.md` in the same pull request; R-016 adds a check that enforces it.

| Upstream | Repository | Revision | Paths | Licence | Local changes |
| --- | --- | --- | --- | --- | --- |
| starling `tools/check-boundary.py` | https://github.com/starlinghq/starling | `9a5c143367e3368b9b1cd531887a0f0cb3245b0b` | `tools/check-boundary.py` | MIT OR Apache-2.0 | None (byte-identical) |
| starling `tools/create_backlog_issues.py` | https://github.com/starlinghq/starling | `9a5c143367e3368b9b1cd531887a0f0cb3245b0b` | `tools/create_backlog_issues.py` | MIT OR Apache-2.0 | Adapted to the build plan's section 10 table and labels |

No kernel source is vendored yet (R-006, R-016).
