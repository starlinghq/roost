# Kickoff decisions

Roost's decision log (R-002). Decided rows record who decided and when; proposed rows wait for the maintainer. Open decisions are kept in [build plan section 11](plan/build-plan.md#11-decisions-for-the-maintainer) and not copied here, so the two lists cannot drift.

| Decision | Status | Value | Recorded |
| --- | --- | --- | --- |
| Goal | Decided (maintainer, 2026-10-17; confirmed 2026-10-18) | A whole and complete replacement for vLLM without compromises, faster and better in some categories (plan decision 18, [charter](../.github/CHARTER.md)) | 2026-10-18 |
| Relationship to Starling | Decided (maintainer, 2026-10-18) | Starling works out the models; Roost runs on top of it in its own repository and consumes Starling's published NuGet packages, pinned exactly, and nothing else (plan section 2.2). Settles plan decisions 2, 3 and 4 | 2026-10-18 |
| Names and repository | Proposed | `starlinghq/roost`; packages `Starlings.Roost.*` and `Starlings.Generation.Roost`. The repository is private; making it public after the vendored-kernel and CUDA-redistribution licence review, and no later than Preview 1, is plan decision 5 | |
| Schedule basis | Proposed, values open | Recorded capacity is one maintainer at 20 hours a week with agents implementing; at that capacity the plan is schedulable to M3 only. The plan proposes four roles and a hiring order (decision 19) | |
| GPU device roles | Proposed | RTX 4060 Ti (SM 89, 16 GB): development device. RTX 3090 (SM 86, 24 GB): evidence device. An SM 120 card from M2 if bought (decision 24). SM 80, 90 and 100 on rented pods (plan section 8) | |
| Compute budget | Proposed, values open | Three lines per milestone, as in the charter: local GPU hours, rented single-GPU hours and rented multi-GPU hours (decisions 16 and 20) | |
| Open decisions | Open | Plan section 11. Decided so far: 2, 3, 4 and 18 | |
