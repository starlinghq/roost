# roost-grammar

llguidance token masks for structured output behind Roost's C ABI (R-051, decision 22). Promoted from spike S10
(`spikes/s10-llguidance`, report `docs/m0/roost-s10-llguidance-abi.md`); llguidance is pinned there and here at
`6cef6ed9c973c2a155a85f2bb1996320af28b289` (`PROVENANCE.md`, `THIRD-PARTY-NOTICES.md`).

- `include/roost_grammar.h`: the C ABI, version 1.0. Create a tokenizer environment from `tokenizer.json` (with the model's
  stop ids and logits width), compile a JSON schema, any JSON object, a regular expression or a Lark grammar into a matcher,
  compute the next token's mask, consume a token, check accept and stop states, free.
- `src/lib.rs`: the implementation. Panics are caught at the boundary; errors are status codes plus a per-thread message.
- `tests/abi.rs`: the ABI through the rlib, on the tiny byte-level tokenizer the .NET tests use.
- `Starlings.Roost.Native` (`src/Starlings.Roost.Native/`) loads the built library and binds it for .NET; its tests
  (`tests/Starlings.Roost.Native.Tests/`) include the soundness tests on the tiny models.

Only the source and `Cargo.lock` are committed. The built library (`libroost_grammar.so`, `roost_grammar.dll`,
`libroost_grammar.dylib`) is a release asset, like the kernel library; nothing built is committed.

## Build and test

```sh
cd native/roost-grammar
cargo build --release --locked     # target/release/libroost_grammar.so
cargo test --release --locked
```

A clean build took 1 min 38 s with `-j2` on the shared development box (2026-11-02; S10 measured 1 min 48 s). Rust 1.97.0
built it here; S10 used 1.98.1, and upstream pins 1.95.0 in its `rust-toolchain.toml`. The release image should pin one.

## .NET tests and CI

`dotnet test` does not build this crate. The tests that need the library find it through `ROOST_GRAMMAR_LIBRARY` (the path
of the library file) or at `native/roost-grammar/target/{release,debug}/`, and skip with the reason when it is absent, the way
the kernel tests skip without a kernel library:

```sh
export ROOST_GRAMMAR_LIBRARY=$PWD/native/roost-grammar/target/release/libroost_grammar.so
dotnet test --project tests/Starlings.Roost.Native.Tests
```

The CI workflow builds no Rust crate today, so in CI these tests skip. Building the crate there (and shipping it as a release
asset with its notices) is a change to `.github/` for stream A.
