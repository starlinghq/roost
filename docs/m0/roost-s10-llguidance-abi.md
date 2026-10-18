# S10 (R-067): llguidance behind a C ABI from Native AOT .NET

Status: **go** for R-051 on the evidence below. The work is CPU only; the runs were still taken inside a short `tools/gpu-lease` on the RTX 3090 (stream `spike`) (the fix-task rule for re-runs).

## Question

Can llguidance be built as a library behind a C ABI, loaded from AOT .NET, and compute a token mask for a JSON schema within the budget of a decode step (plan section 6, S10)?

## Answer

Yes. A 240-line Rust crate wraps llguidance's `Matcher` in 13 `roost_grammar_*` functions. A Native AOT linux-x64 binary calls it through `DllImport` and replays valid JSON token by token. Every instance token was allowed by the mask (0 violations in 68,600 mask computations per run), and each matcher ended accepting and stopped (the `accepting=1 stopped=1` lines are in both committed logs). Per-token mask latency in the two runs below is p50 17 to 27 us and p99 0.18 to 0.46 ms on Qwen3's vocabulary. That is below the derived lower bound for a batch-1 decode step even with no overlap; with overlap it is hidden. Maxima were 0.3 to 1.0 ms in these runs.

What the zero-violation count shows: completeness only. It checks that each token of a valid instance is allowed by the mask. It does not check soundness (that tokens the grammar forbids are masked), which is the R-051 criterion "masked tokens never sampled" and is not established here.

## Setup

| Item | Value |
| --- | --- |
| llguidance | `guidance-ai/llguidance` at `6cef6ed9c973c2a155a85f2bb1996320af28b289` (crate 1.9.1), by git `rev` in `Cargo.toml`; `toktrie_hf_tokenizers` at the same SHA; `Cargo.lock` committed |
| Tokenizer | `Qwen/Qwen3-1.7B` `tokenizer.json` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`), padded to the model's `vocab_size` 151936 (`config.json`); mask is 4748 u32 words, 18,992 bytes |
| Toolchain | rustc/cargo 1.98.1, .NET SDK 10.0.401 with `PublishAot`, clang |
| Host | the shared development box, 12 cores; one thread, no pinning. `uptime` at the start and end of each run is the header of its log (1-minute load 1.9 to 4.9) |
| Method | the harness tokenizes one valid instance per schema (llguidance's greedy `tokenize_bytes`), appends EOS, then for 200 fresh matchers per schema times `compute_mask` before each `consume`. Time is `Stopwatch` around the P/Invoke, so it includes the call and the 18,992-byte mask copy. Rep 0 of each schema is the cold case for that matcher |
| Schemas | `s1-flat`: 4 scalar properties (25 steps). `s2-nested`: nested objects, regex `pattern`, `enum`, bounded `array`, `minimum`/`maximum`, nullable type (120 steps). `s3-complex`: `$defs`/`$ref`, recursive `node`, `oneOf` of three object shapes, `anyOf` additionalProperties, `enum` arrays, `exclusiveMinimum` (198 steps) |

Files: `spikes/s10-llguidance/{schemas,results,aot,src}`. The `spikes/` directory is not in the plan's layout (section 3) or stream paths (section 4); see the PR for the layout question.

## Mask latency per token (microseconds)

Run 2 of two identical runs (`results/run-2026-10-17-aot-2.txt`; run 1 is `run-2026-10-17-aot-1.txt`). Both logs are the unfiltered program output. 200 matchers per schema. These runs replace an earlier pair taken at a higher load. Its committed log is kept as `run-2026-10-17-aot-0-superseded-filtered.txt`; the `accepting/stopped` lines were filtered out of it, so it is not raw output and no latency number here comes from it.

| Schema | Samples | p50 | p90 | p99 | p99.9 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1-flat | 5,000 | 22.6 | 48.5 | 181.4 | 219.7 | 286.1 | 28.7 |
| s2-nested | 24,000 | 17.6 | 57.1 | 188.3 | 498.2 | 834.4 | 28.1 |
| s3-complex | 39,600 | 16.8 | 101.2 | 435.1 | 585.4 | 1,015.8 | 44.9 |

Run 1, same binary: p50 27.3 / 18.3 / 18.0 and p99 276.4 / 192.4 / 455.6 us (s1/s2/s3). Across both runs p99 is 0.18 to 0.46 ms and p50 is 17 to 27 us. p99 moved by up to 1.5x between runs on s1, so quote p99 as "under 0.5 ms on a quiet box", not as a precise figure.

Other costs from the same runs:

| Cost | s1 | s2 | s3 |
| --- | --- | --- | --- |
| `consume_token` p50 / p99 (us), run 2 | 1.4 / 3.6 | 1.1 / 3.9 | 1.2 / 4.0 |
| `matcher_new` first / p50 (ms), run 2, includes schema compile | 0.65 / 0.47 | 0.85 / 0.83 | 1.05 / 1.04 |
| Cold (rep 0) worst mask (us), run 2 | 198.5 | 486.6 | 542.0 |
| `tokenizer_new` (parse tokenizer.json, build trie and slicer) | 0.8 to 1.0 s once per model (1.6 to 1.9 s in the superseded, busier pair, `results/run-2026-10-17-aot-0-superseded-filtered.txt`), not per request | | |

## Comparison with the decode step

No batch-1 decode step has been measured in this repository yet (S5 and the Roost graphed path do not exist), so there is no measured figure to cite.

A derived lower bound, labelled as a calculation and not a measurement. From the Qwen3-1.7B `config.json` at the pinned revision (committed as `results/qwen3-1.7b-config.json`): hidden 2048, 28 layers, 16 heads and 8 KV heads of dim 128, MLP 6144, vocab 151936, tied embeddings. Per layer: attention 2048x2048 (q) + 2 x 2048x1024 (k, v) + 2048x2048 (o) = 12.58 M, MLP 3 x 2048 x 6144 = 37.75 M, norms 4,352, so 50.34 M; x 28 = 1,409.4 M. Embedding (shared with the output head) 151936 x 2048 = 311.2 M. Total about 1.72 B parameters, 3.44 GB in bf16. A batch-1 decode step reads all of it once (the tied matrix is read in full by the output projection). The RTX 3090 peaks at 936 GB/s (NVIDIA specification), so the floor is at least about 3.7 ms per step; real steps are slower. Against that floor:

- the worst p99 in either run (0.46 ms, s3) is about 12% of the floor, and p50 (0.02 to 0.03 ms) under 1%;
- the S10 stop condition (mask above the decode step with no overlap) is not met at p50, p99 or p99.9. The largest sample in either run (1.0 ms) is also below the floor.

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
4. **Per-request cost.** Matcher creation (schema compile) is about 0.5 to 1.1 ms at p50 for these schemas (worst single creation 4.0 ms), and the tokenizer environment is a 0.8 to 1.0 s one-time cost per model (see the table above); share one `RgTokenizer` across requests.
5. **Mask size.** 18,992 bytes per token per sequence at 151,936 rows; at batch 8 that is about 152 KB per step to move to the sampler.
6. **Not measured:** many concurrent matchers, `llg_par_compute_mask`, regex and Lark grammars, schemas that blow up the lexer, and mask-compute contention with the decode threads on the same cores.

## Build and licence notes

- Clean-target `cargo build --release --locked -j2`: 1 min 48 s at load 1.2 to 2.3 with the crate sources already downloaded (`results/cargo-cold-build-2026-10-17.txt`). Outputs: `libroost_grammar.so` 8.8 MB (depends on libgcc_s and libc), `libroost_grammar.a` 30 MB. Nothing built is committed.
- Dependency closure: 110 packages in `Cargo.lock`, counting the spike crate itself. Licences: llguidance, toktrie and toktrie_hf_tokenizers are MIT (Microsoft Corporation). Every other crate is MIT, Apache-2.0, BSD-2-Clause, BSL-1.0, Zlib, Unlicense, MIT-0 or Unicode-3.0, alone or as an "OR" choice, including "Apache-2.0 OR BSL-1.0" and "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT" (all permissive; per `cargo metadata`, which also shows a few "/"-separated MIT/Apache-2.0 spellings). One crate offers `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; choose MIT. No crate lacks a licence field. R-051 must still add these to `THIRD-PARTY-NOTICES.md` and `PROVENANCE.md` when the crate is adopted; this spike adopts nothing into Roost.
- `Cargo.toml` here uses default-features off with `lark`, `rayon`, `referencing`, `ahash`. The `jsonschema_validation` feature (about 2.5 MB) was not enabled.
- Rust 1.95.0 is what upstream pins in `rust-toolchain.toml`; this build used the machine's 1.98.1 and built cleanly. R-051 should pin the toolchain in the release image.

## Estimate for R-051

The wrapper exists and works; the remaining work is the real `native/roost-grammar` crate with stop-token handling, regex and Lark kinds, a batch mask call, error taxonomy, a header, and the managed binding plus tests: roughly 3 to 5 days, plus the measured-step re-check above.
