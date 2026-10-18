# R-009 status: partial

Delivered: Roost chat types, the Qwen3.5 template over Starling's published `Qwen35ChatTemplate`, the incremental detokenizer
over Starling's published `IncrementalDetokenizer`, and the `IStopMatcher` seam with a pass-through for an empty stop list.

Still open (do not mark R-009 done):

- Public stop-matching API: not published (starling#124). `PendingStopMatcherFactory` refuses non-empty stop lists. Hold-back and
  truncation semantics are untested against Starling's matcher.
- Qwen3 template from `Starlings.Models.QwenChat`: not on nuget.org (starling#124). Not needed for the M0 Qwen3.5 model.
- Adapter to Roost's engine: not built; belongs to R-036.
