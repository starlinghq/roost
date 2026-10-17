# Spike S10 (R-067): llguidance behind a C ABI

Throwaway evidence outside `Roost.slnx`. The Rust crate becomes `native/roost-grammar/` only in R-051.
Report: `docs/m0/roost-s10-llguidance-abi.md`.

- `src/lib.rs`, `Cargo.toml`, `Cargo.lock`: the C ABI (`roost_grammar_*`), llguidance pinned by full SHA.
- `aot/`: a Native AOT .NET harness that P/Invokes the library and times each mask.
- `schemas/`: three JSON schemas and one valid instance each.
- `results/`: the raw output of the run quoted in the report.

Build and run (outputs go outside the repository):

```sh
export CARGO_TARGET_DIR=$ENVS/target CARGO_BUILD_JOBS=2
cargo build --release --locked -j2
dotnet publish aot -c Release -r linux-x64 -o $ENVS/aot-out -p:BaseIntermediateOutputPath=$ENVS/aot-obj/
# tokenizer.json: Qwen/Qwen3-1.7B at 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
LD_LIBRARY_PATH=$ENVS/target/release $ENVS/aot-out/MaskBench tokenizer.json schemas 200 151936
```
