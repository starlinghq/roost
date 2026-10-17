# S10 (R-067): llguidance behind a C ABI from Native AOT .NET

Status: **go** for R-051 on the evidence below. CPU only; no GPU run was made, so no `gpu-lease` was taken.

## Question

Can llguidance be built as a library behind a C ABI, loaded from AOT .NET, and compute a token mask for a JSON schema within the budget of a decode step (plan section 6, S10)?

## Answer

Yes. A 240-line Rust crate wraps llguidance's `Matcher` in 13 `roost_grammar_*` functions. A Native AOT linux-x64 binary calls it through `DllImport` and replays valid JSON token by token. Every instance token was allowed by the mask (0 violations in 68,600 mask computations), and each matcher ended accepting and stopped. Per-token mask latency is p50 35 to 53 us and p99 0.36 to 0.84 ms on Qwen3's vocabulary. That is below the derived lower bound for a batch-1 decode step even with no overlap; with overlap it is hidden. The tail (max 2 to 12 ms) was measured on a box with load average 13, so it is an upper estimate.

## Setup

| Item | Value |
| --- | --- |
| llguidance | `guidance-ai/llguidance` at `6cef6ed9c973c2a155a85f2bb1996320af28b289` (crate 1.9.1), by git `rev` in `Cargo.toml`; `toktrie_hf_tokenizers` at the same SHA; `Cargo.lock` committed |
| Tokenizer | `Qwen/Qwen3-1.7B` `tokenizer.json` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`), padded to the model's `vocab_size` 151936 (`config.json`); mask is 4748 u32 words, 18,992 bytes |
| Toolchain | rustc/cargo 1.98.1, .NET SDK 10.0.401 with `PublishAot`, clang |
| Host | the shared development box, 12 cores, load average about 13 during the runs; one thread, no pinning |
| Method | the harness tokenizes one valid instance per schema (llguidance's greedy `tokenize_bytes`), appends EOS, then for 200 fresh matchers per schema times `compute_mask` before each `consume`. Time is `Stopwatch` around the P/Invoke, so it includes the call and the 18,992-byte mask copy. Rep 0 of each schema is the cold case for that matcher |
| Schemas | `s1-flat`: 4 scalar properties (25 steps). `s2-nested`: nested objects, regex `pattern`, `enum`, bounded `array`, `minimum`/`maximum`, nullable type (120 steps). `s3-complex`: `$defs`/`$ref`, recursive `node`, `oneOf` of three object shapes, `anyOf` additionalProperties, `enum` arrays, `exclusiveMinimum` (198 steps) |

Files: `spikes/s10-llguidance/{schemas,results,aot,src}`.

## Mask latency per token (microseconds)

Second of two identical runs (`results/run-2026-10-17-aot.txt` holds both). 200 matchers per schema.

| Schema | Samples | p50 | p90 | p99 | p99.9 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1-flat | 5,000 | 53.3 | 153.4 | 364.6 | 447.1 | 2,179.9 | 70.7 |
| s2-nested | 24,000 | 35.4 | 111.4 | 387.0 | 1,023.1 | 3,530.2 | 57.0 |
| s3-complex | 39,600 | 37.7 | 197.2 | 739.6 | 1,260.0 | 2,720.3 | 88.4 |

First run, same binary: p50 51.9 / 37.8 / 43.7 and p99 358.8 / 801.8 / 841.1 us (s1/s2/s3). Run to run, p99 moved by up to 2x on s2 under load, so quote p99 as "under 1 ms", not as a precise figure.

Other costs from the same runs:

| Cost | s1 | s2 | s3 |
| --- | --- | --- | --- |
| `consume_token` p50 / p99 (us) | 3.3 / 12.3 | 2.2 / 12.3 | 2.7 / 15.9 |
| `matcher_new` first / p50 (ms), includes schema compile | 1.18 / 1.06 | 1.72 / 1.67 | 2.21 / 2.14 |
| Cold (rep 0) worst mask (us) | 333.7 | 974.7 | 1,143.6 |
| `tokenizer_new` (parse tokenizer.json, build trie and slicer) | 1.6 to 1.9 s once per model, not per request | | |

## Comparison with the decode step

No batch-1 decode step has been measured in this repository yet (S5 and the Roost graphed path do not exist), so there is no measured figure to cite.

A derived lower bound, labelled as a calculation and not a measurement: a Qwen3-1.7B bf16 decode step at batch 1 reads all weights once. Qwen3-1.7B has about 1.7B parameters (model card), so about 3.4 GB in bf16, against the RTX 3090's 936 GB/s peak memory bandwidth (NVIDIA specification). That gives at least about 3.6 ms per step; real steps are slower. Against that floor:

- worst-schema p99 (0.84 ms) is about 23% of the floor, and p50 (0.05 ms) about 1%;
- the S10 stop condition (mask above the decode step with no overlap) is not met at p50, p99 or p99.9. Only isolated maxima (2 to 12 ms, under load average 13) exceed the floor, and overlap would hide those.

R-051 overlaps the mask for step N+1 with the GPU forward of step N+1 (the mask depends only on the token sampled at step N), so the mask is on the critical path only when it takes longer than the forward. The numbers should be re-taken against a measured Roost step once S1/S9 give one.

## AOT load

- `dotnet publish -r linux-x64` with `PublishAot=true`, `TreatWarningsAsErrors`, and the trim analyzer produced no warnings. The binary is 1.46 MB stripped and links only libm and libc; the Rust library is resolved at run time by `dlopen` (`DllImport("roost_grammar")`, found through `LD_LIBRARY_PATH` here; the real layout would put it beside the executable or use an rpath).
- Blittable signatures only (pointers, `nuint`, `uint`, `int`); no marshalling stubs were needed.
- Not exercised: static linking of `libroost_grammar.a` into the AOT binary (`DirectPInvoke`); the 30 MB `.a` builds but was not linked. The shared-library route works, so R-051 does not depend on it.
- Not exercised: Windows or macOS; the Rust panic guard (`catch_unwind`) was not triggered by a test.

## ABI shape used

`abi_version`, `last_error` (thread-local), `tokenizer_new(json, len, n_vocab)`, `tokenizer_free`, `mask_words`, `tokenize`, `eos_token`, `matcher_new(tok, kind, data)`, `matcher_free`, `matcher_compute_mask(m, dest, words)`, `matcher_consume(m, token)`, `is_accepting`, `is_stopped`. Handles are opaque, errors are return codes plus a message buffer, and panics are caught at the boundary. I wrapped the Rust API instead of linking llguidance's own `llg_*` C API (`llguidance.h`) so the surface is Roost's to version; the upstream C API has `llg_matcher_compute_mask_into`, `llg_par_compute_mask` and cancellation handles that R-051 should evaluate before adding its own.

## Findings for R-051

1. **Stop token.** `eos_token` came back as 151643 (`<|endoftext|>`), but Qwen3's generation EOS in `config.json` is 151645 (`<|im_end|>`). After a closed JSON value the mask permits only the tokenizer's EOS, so R-051 must pass the model's stop tokens (`ByteTokenizer::set_eos_tokens`) or the grammar would force a token the model never stops on.
2. **Matcher stops itself.** After the last `}` the matcher is `stopped`, and `consume_token(EOS)` then returns an error. The caller must check `is_stopped` before consuming EOS (the harness skips the consume).
3. **Unsupported keyword.** The first `s3-complex` draft used `uniqueItems`; matcher construction failed with `Unimplemented keys: ["uniqueItems"]`. The keyword was removed for the benchmark. R-051 must surface this error to the API caller as a 400, not a 500.
4. **Per-request cost.** Matcher creation (schema compile) is 1 to 2.5 ms for these schemas, and the tokenizer environment is a 1.6 to 1.9 s one-time cost per model; share one `RgTokenizer` across requests.
5. **Mask size.** 18,992 bytes per token per sequence at 151,936 rows; at batch 8 that is about 152 KB per step to move to the sampler.
6. **Not measured:** many concurrent matchers, `llg_par_compute_mask`, regex and Lark grammars, schemas that blow up the lexer, and mask-compute contention with the decode threads on the same cores.

## Build and licence notes

- Cold `cargo build --release --locked -j2`: 2 min 19 s on the loaded box. Outputs: `libroost_grammar.so` 8.8 MB (depends on libgcc_s and libc), `libroost_grammar.a` 30 MB. Nothing built is committed.
- Dependency closure: 110 crates in `Cargo.lock` (113 lines in `cargo tree`, normal edges). Licences: llguidance, toktrie and toktrie_hf_tokenizers are MIT (Microsoft Corporation). Every other crate is MIT, Apache-2.0, BSD-2-Clause, Zlib, Unlicense, MIT-0 or Unicode-3.0, alone or as an "OR" choice. One crate offers `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; choose MIT. No crate lacks a licence field. R-051 must still add these to `THIRD-PARTY-NOTICES.md` and `PROVENANCE.md` when the crate is adopted; this spike adopts nothing into Roost.
- `Cargo.toml` here uses default-features off with `lark`, `rayon`, `referencing`, `ahash`. The `jsonschema_validation` feature (about 2.5 MB) was not enabled.
- Rust 1.95.0 is what upstream pins in `rust-toolchain.toml`; this build used the machine's 1.98.1 and built cleanly. R-051 should pin the toolchain in the release image.

## Estimate for R-051

The wrapper exists and works; the remaining work is the real `native/roost-grammar` crate with stop-token handling, regex and Lark kinds, a batch mask call, error taxonomy, a header, and the managed binding plus tests: roughly 3 to 5 days, plus the measured-step re-check above.
