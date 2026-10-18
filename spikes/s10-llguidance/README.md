# Spike S10 (R-067): llguidance behind a C ABI

Throwaway evidence outside `Roost.slnx`. The Rust crate becomes `native/roost-grammar/` only in R-051.
Report: `docs/m0/roost-s10-llguidance-abi.md`.

- `src/lib.rs`, `Cargo.toml`, `Cargo.lock`: the C ABI (`roost_grammar_*`), llguidance pinned by full SHA.
- `aot/`: a Native AOT .NET harness that P/Invokes the library and times each mask.
- `schemas/`: three JSON schemas and one valid instance each.
- `results/`: the unfiltered program output of the two runs quoted in the report (`run-2026-10-17-aot-{1,2}.txt`, with `uptime` headers; `-0-superseded-filtered` is an older filtered log kept only as a record), the clean-build log, and the Qwen3-1.7B `config.json` the floor calculation uses.

Location: `spikes/` is outside the plan's layout (sections 3 and 4). It holds throwaway code only, outside `Roost.slnx`, and moves to `native/roost-grammar/` in R-051. Whether to keep it there is a maintainer or stream A decision, pending.

Build and run (outputs go outside the repository):

```sh
export CARGO_TARGET_DIR=$ENVS/target CARGO_BUILD_JOBS=2
cargo build --release --locked -j2
dotnet publish aot -c Release -r linux-x64 -o $ENVS/aot-out -p:BaseIntermediateOutputPath=$ENVS/aot-obj/
# tokenizer.json: Qwen/Qwen3-1.7B at 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
LD_LIBRARY_PATH=$ENVS/target/release $ENVS/aot-out/MaskBench tokenizer.json schemas 200 151936
```
