# S4 (R-008): source generator shape

Status: spike report. Code is in `spikes/s4-generator/`, which is throwaway evidence outside `Roost.slnx`. CPU only; no GPU was used.
Base: `ff072f8` (branch `r003/skeleton`). Toolchain: .NET SDK 10.0.401, Roslyn `Microsoft.CodeAnalysis.CSharp` 4.14.0, ILCompiler 10.0.12, clang, linux-x64.

## Question and answer

> Do static-abstract backend dispatch plus a Roslyn incremental generator over the C3 op vocabulary produce allocation-free code under both JIT and AOT, and a compile error naming a missing op? Can the generator build a C7 graph and lower it without losing either property?

**Yes to all four, for the toy.** A `[RoostModel]` definition is read by an incremental generator, turned into a typed op graph, passed through a graph pass, and lowered twice (identity and fused) into `Forward<TOps>() where TOps : struct, IRoostOps`. Over 20 000 calls per variant, the generated code allocated 0 bytes and triggered 0 gen-0 collections under JIT (default tiering, `TieredCompilation=0`) and under Native AOT, with no IL2xxx/IL3xxx warnings (`IsAotCompatible` plus warnings as errors). The analyzer is live: it rejected an `Assembly.Location` call in the toy as IL3000 during the spike, and the code was changed. A negative control (an allocating loop) shows the counter does see allocations (8 800 bytes for 100 `new float[16]`).

Recommendation: **go** for R-020 on this shape. The "stop and escalate" row (generated code allocates or fails AOT analysis) did not trigger. Caveats are under "Findings that change R-020".

## What was built

| Piece | Where | Notes |
| --- | --- | --- |
| Op vocabulary | `Ops/IRoostOps.cs` | Spike-local subset of C3: Embedding, RmsNorm, Gemm, SiluAndMul, Residual, LmHead as static abstract members, plus the optional fused AddRmsNorm. Args structs carry `DeviceSpan` (pointer, length, dtype), no tensor type |
| CPU backend | `Ops/CpuOps.cs` | `CpuOps` in float, and `FusedCpuOps` (provides the fused op). Stand-ins for stream D, not the real reference |
| Generator | `Generator/` (netstandard2.0) | `ForAttributeWithMetadataName` over `[RoostModel]`, `[Node]` attributes read as the model definition; equatable pipeline model; op vocabulary read from the `IRoostOps` symbol in the compilation (nothing hard-coded) |
| C7 graph | `Generator/Graph.cs` | `GNode`/`Graph`, `IGraphPass`, one pass (`fuse-add-rmsnorm`: Residual followed by RmsNorm of its result becomes AddRmsNorm) |
| Lowering | `Generator/ModelGenerator.cs` | Emits `Workspace` (all buffers allocated once), `WeightSlots` (loader slot order), `Forward<TOps>` (identity lowering of the unfused graph) and `ForwardFused<TOps>` (lowering after the pass) |
| Toy model | `Toy/` | Two layers of RmsNorm, Gemm (gate and up), SiluAndMul, Gemm (down), Residual, then final RmsNorm and LmHead; 16 hidden, 32 intermediate, 64 vocab, 4 tokens. No attention: the spike is about generator, graph and dispatch |
| Missing op | `MissingOp/`, `MissingMember/` | See below |
| Incrementality | `IncrementalCheck/` | Drives the generator with step tracking |
| Build time | `measure-build.sh` | Repeated builds, min and median |

## Results

### Allocation and AOT (`Toy/Program.cs`)

The same executable runs both ways. Commands (from `spikes/s4-generator/Toy`):

```
dotnet build -m:2                                              # JIT:  dotnet ../out/Toy/bin/Debug/net10.0/Toy.dll
DOTNET_TieredCompilation=0 dotnet ../out/Toy/bin/Debug/net10.0/Toy.dll
dotnet publish -c Release -r linux-x64 -m:2 -o ../out/Toy/publish   # AOT: ../out/Toy/publish/Toy   (1.5 MB)
```

Each variant is warmed up with 2 000 calls, then 20 000 calls are measured with `GC.GetAllocatedBytesForCurrentThread()` and `GC.CollectionCount(0)`. Every row of every run: **allocatedBytes=0, gen0GCs=0**. Runs: JIT default tiering, JIT with `TieredCompilation=0`, Native AOT.

| Variant | What it exercises | Result (all three runtimes) |
| --- | --- | --- |
| `Forward<CpuOps>` | identity lowering of the unfused graph | 0 B, 0 GC |
| `ForwardFused<CpuOps>` | fused lowering, backend lacks AddRmsNorm, falls back to unfused parts | 0 B, 0 GC; output bit-identical to unfused |
| `ForwardFused<FusedCpuOps>` | fused lowering, backend provides AddRmsNorm | 0 B, 0 GC; output bit-identical to unfused (max abs diff 0) |

Correctness: the generated forward matches an independent managed double-precision transcription of the toy graph with max relative difference 1.97e-5 (tolerance 1e-3), under JIT and AOT. This check exists because the first version of the toy had a wiring bug (the Embedding output width was inferred as 1 from the token-id pseudo-node) that the fused-versus-unfused check could not see, since both lowerings were wrong in the same way. Found by reading the generated file; now a generator error (ROOST006) and covered by the reference check.

Time per call (about 11 to 22 us AOT, Release; 36 to 93 us JIT, Debug build, so unoptimised code) is not a claim: the machine was running other agents' builds, and the toy is tiny. Not measured: whether the JIT folds `TOps.SupportsAddRmsNorm` and devirtualises every call (inspected through allocation and behaviour only, not disassembly).

### Missing-op diagnostic

```
cd spikes/s4-generator/MissingOp && dotnet build -m:2
Model.cs(8,2): error ROOST001: Op 'FlashDecode' used by node 'a0' is not a member of IRoostOps (known ops: AddRmsNorm, Embedding, Gemm, LmHead, Residual, RmsNorm, SiluAndMul)
```

Located on the attribute naming the op; downstream nodes that read the failed node are not reported again. The generator also reports ROOST002 (unknown input), 003 (args field the generator cannot fill), 004 (input count), 005 (`IRoostOps` not found) and 006 (output width not inferable).

The second sense, a backend missing an op, is the compiler's own error:

```
cd spikes/s4-generator/MissingMember && dotnet build -m:2
BrokenOps.cs(6,27): error CS0535: 'BrokenOps' does not implement interface member 'IRoostOps.Gemm(in GemmArgs)'
```

### Incrementality (`IncrementalCheck`, `dotnet run` after `dotnet build`)

Driver-level result of the generator output step after each edit to the compilation (Roslyn `IncrementalStepRunReason`):

| Edit | Output step | Generator time |
| --- | --- | --- |
| baseline | New | 1 221 to 1 939 ms (includes JIT of the generator and the first semantic model; two valid runs; a third, 1 171 ms, ran with a definition error and is excluded) |
| edit an unrelated file | **Cached** | 44 to 70 ms |
| edit the definition file below the attributes (method body in the model class) | **Cached** | 13 to 22 ms |
| change a node (weight slot name) | Modified | 4 to 9 ms |
| insert a comment line above the attributes | Modified | 3 to 4 ms |

The last row is a cost of carrying source locations in the pipeline model: any line shift in the definition file re-emits. Harmless for pre-generated model packages; for the analyzer-package case, keep definitions in their own file, or store only an attribute index and resolve the location at report time.

### Build time (`measure-build.sh 5`, `-m:2`, Release, wall min and median of 5; one run at load average about 4)

| Case | Wall min | Wall median | CPU median |
| --- | --- | --- | --- |
| cold (clean `out/`, Ops + Generator + Toy, N=1) | 3.16 s | n/a | 4.03 s |
| no-op rebuild | 2.65 s | 2.72 s | 3.44 s |
| edit an unrelated Toy file | 2.72 s | 2.82 s | 3.59 s |
| edit the model definition (rename a node) | 2.73 s | 2.80 s | 3.56 s |
| edit generator source (rebuilds Generator and Toy) | 5.12 s | 5.49 s | 10.05 s |

The generator's own time inside the compiler, from `-p:ReportAnalyzer=true`: **9 ms** (50% of 0.634 s total analyzer time, the rest being the other analyzers). Editing the model definition costs the same as a no-op build to within noise, because the 2.7 s floor is MSBuild evaluation and the compiler server, not the generator. An earlier run of the same script under load average 15 gave 5 to 7 s for these cases with min and median spread over 1 s, so absolute build times from this box are not usable beyond the ordering. The bounded claim: the generator adds single-digit milliseconds to a compile for a 13-node model, and re-emits only when the definition changes.

## Findings that change R-020

1. **A static virtual default cannot fall back to sibling ops.** The plan says optional fused ops are "defined as their unfused parts so a backend that lacks it falls back". Inside `IRoostOps` a `static virtual` body cannot call the interface's other static abstract members without a self type parameter. Two ways: make the interface `IRoostOps<TSelf>` (CRTP), or keep `IRoostOps` non-generic and have the generated forward test a `static virtual bool SupportsX` and emit both paths. The spike built and measured only the second (zero allocations); the CRTP form was not built, and is expected, not shown, to behave the same. Decide in R-020 and record it in C3. The cost of the second is two code paths per fused op in generated code; the cost of the first is `where TOps : struct, IRoostOps<TOps>` everywhere.
2. **Generated forward size.** 377 lines of C# for 13 nodes and both lowerings. Large real models (Qwen3 at 28 to 36 layers, MoE) unroll to thousands of lines per lowering if layers are unrolled; the spike did not test a loop-based emission for repeated layers. JIT time for such a method (tier-0 then tier-1) is unmeasured.
3. **Graph and generator packaging.** The plan puts the graph in `Starlings.Roost.Graph` and the generator in `Starlings.Roost.ModelCompiler`, both netstandard2.0. An analyzer cannot take normal package dependencies, so the generator must embed or bundle the Graph assembly. The spike kept the graph inside the generator project. R-020 must plan the bundling (and C7's rule that every pass output is checked against the unfused graph on `CpuOps` runs at test time, outside the generator).
4. **The vocabulary read from the symbol works and stays cached.** Combining the model with a `Compilation.Select` that yields an equatable vocabulary did not defeat caching (rows 2 and 3 above). That is what lets the diagnostic name the op without a hard-coded list and without a dependency on `kernels/manifest.toml`.
5. **Args filling is by field name.** The generator fills `Input`, `Input2`, `Weight`, `Output`, `Output2`, `Rows`, `InDim`, `OutDim` and rejects other names with ROOST003. Real C3 args (attention metadata, cache ops, sharding, dtype and layout) need a richer mapping; that is the main unestimated part.

## Not done, and the estimate

Not covered: attention, KV cache ops, state groups, weight-loader emission beyond slot names, sharding annotations, the graph checker, variant selection, a generated forward for a real model, trimming warnings from a full host, and any GPU backend. The generator does not yet emit the plan input layout or state-group descriptors.

Estimate for R-020 (judgement from this spike, not a measurement): the shape needs no redesign. The C7 types, the identity lowering, the checker and diagnostics are about 1 to 2 engineer-weeks over this spike once C3 args are frozen. A hybrid or MoE model definition (non-linear graph, per-layer variation, loop emission) is where the risk sits, and the first thing to try is emitting a loop over layers with a per-layer weight table, then measuring allocation and AOT again.

## Reproduce

```
git clone https://github.com/starlinghq/roost && cd roost && git checkout spike/s4-generator && git submodule update --init
cd spikes/s4-generator
(cd Toy && dotnet build -m:2 && dotnet ../out/Toy/bin/Debug/net10.0/Toy.dll)
(cd Toy && dotnet publish -c Release -r linux-x64 -m:2 -o ../out/Toy/publish && ../out/Toy/publish/Toy)
(cd MissingOp && dotnet build -m:2); (cd MissingMember && dotnet build -m:2)
(cd IncrementalCheck && dotnet build -m:2 && dotnet ../out/IncrementalCheck/bin/Debug/net10.0/IncrementalCheck.dll)
./measure-build.sh 5
```

`spikes/s4-generator/Directory.Build.props` and `Directory.Packages.props` keep the spike standalone (no inheritance from the repository conventions); all build output goes to the ignored `out/`.
