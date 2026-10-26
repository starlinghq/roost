# R-009 status: partial

Delivered: Roost chat types, the Qwen3.5 template over Starling's published `Qwen35ChatTemplate`, the incremental detokenizer
over Starling's published `IncrementalDetokenizer`, and stop matching through `StarlingStopMatcherFactory` over Starling's
`StopSequenceMatcher`, public since `Starlings.Generation.OnnxGenAI` 0.3.0 (starling#124). Hold-back, release, flush, earliest-match
and longest-at-same-start behaviour are tested through the seam.

Still open (do not mark R-009 done):

- Qwen3 template from `Starlings.Models.QwenChat`: on nuget.org since ai-dotnet 0.3.0 (starling#124, Experimental profiles), not
  yet pinned or consumed by Roost. Not needed for the M0 Qwen3.5 model; the pin comes with the M1 Qwen3 profile (R-015).
- Adapter to Roost's engine: not built; belongs to R-036.

Unmet acceptance criteria (R-009 stays open): Qwen3 template from `Starlings.Models.QwenChat` (published in 0.3.0, not yet consumed)
and the engine adapter (R-036).

Maintainer ruling needed: `tests/Starlings.Roost.Generation.Tests` is outside the single path plan section 4 lists for B1. It
follows the existing `Abstractions.Tests` convention.
