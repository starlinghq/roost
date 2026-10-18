# R-009 status: partial

Delivered: Roost chat types, the Qwen3.5 template over Starling's published `Qwen35ChatTemplate`, the incremental detokenizer
over Starling's published `IncrementalDetokenizer`, and the `IStopMatcher` seam with a pass-through for an empty stop list.

Still open (do not mark R-009 done):

- Public stop-matching API: not published (starling#124). `PendingStopMatcherFactory` refuses non-empty stop lists. Hold-back and
  truncation semantics are untested against Starling's matcher.
- Qwen3 template from `Starlings.Models.QwenChat`: not on nuget.org (starling#124). Not needed for the M0 Qwen3.5 model.
- Adapter to Roost's engine: not built; belongs to R-036.

Unmet acceptance criteria (R-009 stays open): stop-matching public API (starling#124), Qwen3 template from
`Starlings.Models.QwenChat` (starling#124), and the engine adapter (R-036). The `IStopMatcher` shape is Roost's own design and is
unverified against Starling's matcher; only the empty-stop-list pass-through is tested.

Maintainer ruling needed: `tests/Starlings.Roost.Generation.Tests` is outside the single path plan section 4 lists for B1. It
follows the existing `Abstractions.Tests` convention.
