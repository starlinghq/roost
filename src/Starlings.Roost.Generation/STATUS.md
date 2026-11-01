# R-009 status: delivered

Delivered, against the section 10 acceptance criteria:

- Roost chat types and `IRoostChatTemplate`.
- The Qwen3.5 template, `Qwen35RoostChatTemplate`, over Starling's published `Qwen35ChatTemplate` (`Starlings.Models.Qwen35`
  0.3.0).
- The Qwen3 template, `Qwen3RoostChatTemplate`, over Starling's published `QwenChatTemplate` with `QwenChatProfile.Qwen3`
  (`Starlings.Models.QwenChat` 0.3.0, pinned exactly in `Directory.Packages.props`). Starling publishes it as an Experimental,
  text-only profile with `enable_thinking=false`; tool definitions, tool calls, tool messages and thinking mode are unsupported
  and throw `NotSupportedException`, which `RoostChatClient` reports as `InvalidOption`. Status: experimental. No Roost model
  profile uses it yet (the first is Qwen3-1.7B, R-015), and there is no GPU evidence for a Qwen3 model.
- The incremental detokenizer, `RoostIncrementalDetokenizer`, over Starling's published `IncrementalDetokenizer`.
- Stop matching through `StarlingStopMatcherFactory` over Starling's `StopSequenceMatcher`, public since
  `Starlings.Generation.OnnxGenAI` 0.3.0 (starling#124). Hold-back, release, flush, earliest-match and longest-at-same-start
  behaviour are tested through the seam.
- The adapter to Roost's engine: `Starlings.Generation.Roost` (R-036, #58, merged in #133). `RoostChatClient` renders through
  `RoostModelProfile.ChatTemplate` (an `IRoostChatTemplate`), detokenizes with `RoostIncrementalDetokenizer` and matches stops
  through `IStopMatcherFactory` (default `StarlingStopMatcherFactory`).
- No ONNX Runtime package reaches a worker-side package: `Starlings.Models.QwenChat` and `Starlings.Generation.OnnxGenAI` are
  referenced only by this host-side project, and reach `Starlings.Generation.Roost` and test projects through it (decision 30).
  `tools/check-roost-boundaries.py` passes.

Tests: `tests/Starlings.Roost.Generation.Tests` (`ChatTemplateTests` for Qwen3.5, `Qwen3ChatTemplateTests` for Qwen3, the
detokenizer and the stop-matching seam). They compare Roost's rendering with Starling's direct rendering; they are CPU only.

Starling consumption (plan section 3.1): `Starlings.Generation.OnnxGenAI`, `Starlings.Models.Qwen35`, `Starlings.Models.QwenChat`
and `Starlings.Tokenizers`, all `[0.3.0]`. `Starlings.Models.Kev` is still not pinned (R-086).

Maintainer ruling needed: `tests/Starlings.Roost.Generation.Tests` is outside the single path plan section 4 lists for B1. It
follows the existing `Abstractions.Tests` convention.
