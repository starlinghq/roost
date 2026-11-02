# Starlings.Roost.Native status

Loader and bindings for the `native/` Rust libraries (stream F, decision 22). Today that is `roost-grammar` (R-051,
structured output). State on 2026-11-02; nothing here has run on a GPU, and no row is `supported`.

| Part | Status | Notes |
| --- | --- | --- |
| `RoostGrammarLibrary` | untested | Loads `libroost_grammar.so` / `roost_grammar.dll` / `libroost_grammar.dylib` by path (once per path), from `ROOST_GRAMMAR_LIBRARY`, beside the application, or by the default search; resolves the 13 exports to function pointers; refuses a C ABI whose major is not 1. Linux only has been run. |
| `GrammarTokenizer`, `GrammarMatcher` | untested | JSON schema, any JSON object, regex and Lark (`GrammarKind`); compact JSON by default; stop ids replace the tokenizer's EOS (S10 finding 1); a stopped matcher accepts a stop id without consuming it (finding 2); an invalid or unsupported grammar is `GrammarStatus.Grammar` (`IsInvalidGrammar`, finding 3). |
| `GrammarSlots` | untested | The masks behind contract C2's `grammar_slot`: one matcher and one row of a 64-byte-aligned native table per slot; stale masks are recomputed on `ComputeMasks`, several slots in parallel on thread-pool threads. |
| `GrammarSampler` | untested | The CPU sampler path: copies each row's slot mask (all ones for slot -1), calls `CpuOps.Sample` with it and commits every sampled token; a token the matcher then refuses throws instead of being emitted. |

## Evidence

- `native/roost-grammar/tests/abi.rs`: 11 tests of the C ABI (`cargo test --release --locked`).
- `tests/Starlings.Roost.Native.Tests/`: 33 tests, including the soundness tests in `SamplerSoundnessTests`:
  - under adversarial logits (every forbidden token outscores every allowed one), at temperature 0, 1 and 4, no masked token
    was sampled while a row with no grammar in the same call took a forbidden token;
  - roost-tiny and roost-tiny-moe on `CpuOps`, under three schemas (`Schemas/s1-flat.json`, `s2-nested.json` with a `pattern`,
    a bounded array and a nullable type, `s3-refs.json` with `$defs`, `$ref` and `anyOf`), greedy and at three seeds: every
    sampled token was in its mask, every output stopped within 400 tokens, parsed, and satisfied its schema.

These tests skip, with the reason, when the library is not built; CI does not build it (`native/roost-grammar/README.md`).

## Not done

- **Serving.** No contract carries a mask (or a grammar) from the host to the worker: C2 has `grammar_slot` but no mask table,
  the planner (stream G) sets it to -1 and refuses grammar requests, and the eager worker refuses a plan that sets it. Until a
  stream A contract change, `RoostChatClient` and the server compile response formats and refuse them.
- **Device sampler.** Masks are applied on the CPU path only; the fused sampler takes them in R-042.
- **Latency.** Not measured here. S10 measured mask p50 17 to 27 us and p99 0.18 to 0.46 ms per token on Qwen3's vocabulary
  (`docs/m0/roost-s10-llguidance-abi.md`); it is to be re-taken against a measured Roost step.
- **AOT.** The project sets `IsAotCompatible` and uses function pointers only; no AOT publish of it has been run.
