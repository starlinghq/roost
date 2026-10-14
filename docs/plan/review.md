# Adversarial review of the Roost PRD and build plan

Written 2026-10-13 against `docs/plan/build-plan.md` (commit `3c8a9c3`), the Roost PRD (2026-10-12) and `starlinghq/starling` `main` at `fb27658`. `starlinghq/roost` holds no code yet, so this reviews the plan and the PRD assumptions it inherits.

The question asked: will what we build first still be the right thing in two years, and will people other than us use it? The findings are ranked by how likely each is to make Roost fail as a product, not as a project. Each one names its evidence and the change it makes. Changes marked **applied** are already in the build plan. Changes marked **decision** are new rows in the plan's section 11, because they change product direction and belong to the maintainer.

## Verdict

The engineering design holds up: typed state groups, the step plan as the hot-loop contract, a C ABI kernel layer, a CPU reference for every op, and fidelity gates as the product. Its weakness is product sequencing. Following the plan, nobody outside the project can use anything until M2 (week 32 to 48 at recorded capacity), and the first public claim is a throughput race against vLLM on rented hardware. The plan also never says how a .NET developer would install Roost. For an audience of .NET developers, the most useful and defensible product is the in-process one: `dotnet add package`, then a Starling model runs on their NVIDIA card behind `IChatClient`, with published fidelity evidence and no Python. Nobody ships that today. The PRD already lists it as advantage 4 and falls back to it in its first risk row. The plan should build toward it first and treat the server and the vLLM comparison as later layers on the same engine.

## Findings

### 1. No one can use Roost until M2, and the first claim is the weakest one (decision)

**Evidence.**
- The in-process adapter `Starlings.Generation.Roost` is R-036 in M1. It sits behind async scheduling (R-032) and CUDA graphs (R-033) on the M1 gate.
- The first release workflow is R-060 in M2.
- G2 (0.97x vLLM) needs rented H100 hours that the recorded capacity says do not exist ("no paid cloud GPU").
- The PRD's own risk row expects the host-overhead gap to be small, and its response is "lead with fidelity, operations, the in-process API and the unified task host".

**Why it matters long term.** A library with users gets bug reports, model requests and contributors. A plan that stays private for a year gets none of those, and the one maintainer carries all of the motivation. Throughput parity with vLLM is a moving target that a team of hundreds keeps moving. Fidelity evidence and .NET integration are things vLLM structurally does not do.

**Change.** Add a named ship point, **Preview 1**, straight after the M0 gate:
- a preview NuGet set that runs Qwen3 in-process on an RTX card from C# through `IChatClient`;
- eager worker, greedy and seeded sampling (CPU sampler is acceptable), one sequence at a time;
- every capability not yet built marked `unsupported` in the capability table.

To make that possible, move R-036 (`Starlings.Generation.Roost`) to M0 against the eager worker. The concurrency, graphs and server work in M1 then improve a product that already exists instead of blocking its birth. G2 remains the M2 gate, unchanged. Decision row 11 asks the maintainer to adopt Preview 1.

### 2. The plan never says how a .NET developer installs Roost (decision, partly applied)

**Evidence.**
- Kernel binaries are "published as release assets only" (R-016), and a NuGet consumer cannot restore a release asset.
- The PRD's deployable is one container image for linux-x64.
- `libcublasLt` in CUDA 12 is several hundred megabytes, while nuget.org caps a package at 250 MB. A NuGet package therefore cannot carry it.
- The generator in S4 and R-020 reads `kernels/manifest.toml`. That ties every model package's build to one kernel manifest revision and to the SM list.

**Why it matters long term.** The packaging shape decides who can use Roost and how upgrades work. It is also expensive to change once packages are published.

**Change.**
- **Applied: compile time knows only the op vocabulary.** The generator (R-020, S4) checks a definition against C3, the op vocabulary, only. Whether a specialisation exists for a given dtype, head size or SM is checked at startup (R-023), where it already is. `Starlings.Roost.Models.*` ship pre-generated forwards, so a model-package consumer never runs the generator. Someone defining a new model, the G6 onboarding case, references the generator as an analyzer package that depends on C3 only. No build reads a kernel manifest.
- **Applied: the library packages do not require AOT.** Struct-constrained generics are devirtualised under JIT as well as AOT. AOT is a property of the `roost` binary, not a requirement on the packages, because most ASP.NET services that would host the library are not AOT-published.
- **Decision: how native code reaches a NuGet consumer.** There are two options:
  - (a) RID-specific native runtime packages per SM set, following the `Microsoft.ML.OnnxRuntime.Gpu.Linux` pattern, with CUDA libraries taken from the system install;
  - (b) a small managed package that fetches the kernel library and CUDA libraries at first run through `Starlings.Acquisition`, pinned by digest exactly as weights are.

  The recommendation is (b), with a pre-fetch command for air-gapped hosts. It removes the 250 MB problem, reuses the provenance machinery Starling already has, and keeps one verification path for weights and kernels. S2 should weigh the Gemm choice with this in mind: vendored CUTLASS GEMMs cost build time, while cuBLASLt costs distribution size and an EULA review.

### 3. The R-009 extraction is larger than written, and it is the biggest P0 risk (applied)

**Evidence.**
- `Qwen35ChatTemplate` lives in `Starlings.Models.Qwen35`, and `QwenChatTemplate` lives in `Starlings.Models.QwenChat`.
- Both projects reference `Starlings.Generation.OnnxGenAI`, which brings `Microsoft.ML.OnnxRuntimeGenAI.Managed`.
- Roost needs the templates. If R-009 only extracts the decode loop, Roost's dependency closure still contains ONNX Runtime GenAI.

**Change.**
- R-009's acceptance now requires the template renderers and chat profiles to land in a package whose dependency closure contains no `Microsoft.ML.OnnxRuntime*`.
- R-003 adds a Roost boundary check that fails on any `Microsoft.ML.OnnxRuntime*` or Python reference in an engine package's closure, mirroring starling's `tools/check-boundary.py`. The server is exempt once the M3 pooling routes (R-052) bring Starling's ONNX models into it, because one host for every task is PRD advantage 3.
- The risk table names this as a refactor of two released packages with three consumers.

### 4. A diamond dependency makes the submodule interim fragile (applied)

**Evidence.**
- Starling's `ai-dotnet/Directory.Packages.props` pins `Starlings.Tokenizers` `[0.3.0-preview.1]`, and every Starling project carries a committed `packages.lock.json`.
- Roost project references into `external/starling/ai-dotnet/src/...` pick up Starling's `Directory.Build.props` and `Directory.Packages.props`, not Roost's, because MSBuild finds the nearest one first.
- As soon as Roost and Starling pin different versions of any shared package, restore fails with a downgrade error under `TreatWarningsAsErrors`.

**Change.**
- R-061 (publish `Starlings.Acquisition`, `Starlings.Generation` and the template package) moves from M1 to P0, directly after R-009. Preview packages on nuget.org are cheap, and Starling already publishes `Starlings.Tokenizers` under the same prefix.
- The submodule stays for `model-pipeline`, which is Python test tooling, and for the short window before R-061 lands. Section 3.1 lists the diamond as the reason.

### 5. SM coverage drops a V1 target and has no plan for the GPUs users are buying (applied, plus decision)

**Evidence.**
- The PRD lists SM 90 (H100) as a V1 build target, and M2 runs on an H100. The plan's R-016 builds SM 86 and 89 only.
- By 2026 new consumer NVIDIA cards are Blackwell (SM 120), and none of the plan's targets cover them.

**Change.**
- **Applied.** The architecture list is a field in `kernels/manifest.toml`. SM 90 is built from R-016 onward by a build-only CI job, since nvcc needs no GPU to compile for it. Hardware validation for SM 90 lands with R-047 and R-048.
- **Applied.** An SM the project cannot test is listed with the status `untested`, never `supported`.
- **Decision: what happens on an SM with no cubin.** Either embed PTX for JIT compilation by the driver, labelled with its own fidelity status, or refuse at startup with a typed `ProfileUnsupported` error. The recommendation is to refuse by default and allow PTX behind an explicit opt-in that marks every output `untested`. That keeps the evidence promise and still gives an RTX 50-series owner a path.

### 6. Windows is cheap to keep open now and expensive to retrofit later (applied)

**Evidence.**
- The PRD marks Windows x64 "untested in V1", and its capability table already has a Windows row.
- .NET developers work heavily on Windows. WSL2 runs the Linux CUDA stack today.
- Nothing in the plan stops POSIX-only code, hard-coded `.so` names or a GCC-only CMake build from creeping into the C ABI and the loader.

**Change.**
- R-016's acceptance: the kernel library's CMake build compiles with MSVC in CI (build only).
- R-023's acceptance: the loader resolves library names per OS through a `NativeLibrary` resolver (`libcudart.so.12` or `cudart64_12.dll`, `libroost_kernels.so` or `roost_kernels.dll`).
- C1 forbids POSIX types in the ABI.
- WSL2 is documented in Preview 1 but stays `untested`, as native Windows does, until someone with a Windows host runs the release drill. The project has no Windows host on record.

### 7. The ABI and plan structs have no forward-compatibility mechanism (applied)

**Evidence.**
- C1 versions ops by name suffix (`_v1`). C2 freezes every future field now and asserts V1-unused fields are zero.
- Neither contract says how an older kernel library or an older worker meets a newer struct.
- Kernels ship as separately versioned binaries, and with finding 2(b) they are fetched separately from the managed packages. Version skew between the two is therefore the normal case, not an edge case.

**Change.**
- Every C1 parameter struct begins with `uint32_t struct_size` and `uint32_t abi_minor`, following the Vulkan and Win32 pattern. A newer caller may pass a larger struct; a library rejects a size it cannot read.
- `roost_abi_version()` returns major and minor.
- The loader's compatibility rule is: major equal, and library minor greater than or equal to the caller's minor. A contract test checks that rule.
- C2 keeps its reserved fields, but the `StepPlan` header carries a layout version, so a field can be added later without breaking the pinned double buffers.

### 8. Quantisation is the feature that decides which models fit a consumer GPU, and it is in V2 (decision)

**Evidence.**
- The PRD puts INT4 and INT8 weight quantisation (AWQ, GPTQ) in V2 and FP8 in M2.
- FP8 Tensor Core paths need SM 89 or later, so they do not help the 3090.
- On 16 to 24 GB cards, a 7B to 14B model at useful context needs about 4-bit weights. bf16-only Roost on consumer cards is limited to roughly 8B and below.

**Change (decision).** Move one INT4 weight-only format, AWQ through vendored Marlin-style kernels, into M2 beside FP8 or in place of it, with its own fidelity profile. FP8 matters on H100. INT4 is what makes Roost useful on the cards the target audience owns. The tolerance rules already handle a lossy profile: the profile's reference is the dequantised model, not the bf16 one.

### 9. The M0 model choice puts all of stream B3 on the critical path (decision)

**Evidence.**
- Qwen3-0.6B is already admitted, with locks, a template profile (`qwen3-text-v1`) and a candidate plugin.
- Qwen3-1.7B has none of these, so R-012, R-013 and R-014 all gate M0.

**Change (decision).** Gate M0 on Qwen3-0.6B, which reuses the existing admission and needs only per-layer fixtures (R-011, then R-014 for 0.6B). Qwen3-1.7B becomes the second M0 profile, closed before Preview 1. Both models share their architecture, so the definition, kernels and specialisations are the same apart from shapes. The PRD chose 1.7B, so this is the maintainer's call. The saving is roughly two weeks of B3 work off the M0 gate.

### 10. The plan says what to do when capacity runs short but not when to do it (applied)

**Evidence.**
- The recorded-capacity schedule reaches M3 at week 68.
- The PRD's risk row says "narrow V1 to M0 to M2" when weekly hours stay below 20 for a month. The plan gives no checkpoint that would trigger this, and does not say what the narrowed product is.

**Change.**
- A named checkpoint at week 12 (recorded) or week 6 (filled). If the CUDA role is unfilled, or the GPU ledger shows the M0 burn above 2x estimate, V1 narrows to Preview 1 plus the M1 server, with no vLLM claim.
- M2's throughput comparison becomes a V2 item with its H100 budget decision attached.
- The narrowed product is still complete and honest: fidelity-gated in-process inference and an OpenAI-compatible server on consumer NVIDIA cards.

### 11. Premise check: why not a backend over an existing runtime? (recorded, no change)

**The challenge.** llama.cpp and ONNX Runtime GenAI already run on CUDA, Windows and consumer GPUs with 4-bit weights. A .NET host over either would ship in months.

**The answer**, which the plan should state rather than leave implicit:
- Roost's product is the fidelity evidence and the host-owned scheduler.
- Neither runtime exposes paged, typed state groups or a step plan the host controls.
- llama.cpp's numerics are not referenced against transformers fp32.
- Starling already ships ONNX Runtime GenAI for the in-process path, and its fidelity work recorded where that path diverges.
- The op-level `IRoostOps` boundary keeps a second backend possible later, for example a Vulkan or ROCm `IRoostOps`.

Recorded at the top of the plan's section 3, so the next reviewer does not reopen it without new evidence.

## What this review does not change

- The five PRD design rules, the state-group design, the evaluator-first sequencing and the protected-tolerance rules.
- The non-goals for V1 (training, non-NVIDIA backends, parallelism).
- The stream structure, path ownership and work-in-progress limits.
