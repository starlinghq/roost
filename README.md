# Roost

Roost is a C# serving engine for NVIDIA GPUs, built to replace vLLM, that serves the Starling model library behind one host. Kernels are CUDA C++ behind a C ABI, with Rust for a few host-side native libraries; the scheduler, KV cache, step loop, CUDA graphs, parallelism and OpenAI-compatible API are .NET. Each GPU is driven by its own worker process, so a GPU fault never takes down the host. There is no Python or libtorch at runtime.

[Starling](https://github.com/starlinghq/starling) works out the models; Roost runs on top of it and consumes Starling's published NuGet packages, and nothing else (build plan section 2.2). Roost owns its evidence pipeline in `evidence/`, and every served profile passes Roost's fidelity gates before any throughput number is published. The charter is [.github/CHARTER.md](.github/CHARTER.md) and the decision log is [docs/kickoff.md](docs/kickoff.md).

Status: repository skeleton (R-003). Nothing is usable yet. The build plan is [docs/plan/build-plan.md](docs/plan/build-plan.md); what "replacement" covers is its section 13, and how Roost aims to be faster is its section 14. Agent and contributor rules are in [AGENTS.md](AGENTS.md).

## Building

```sh
git clone --recurse-submodules https://github.com/starlinghq/roost
dotnet test --solution Roost.slnx
```

The SDK is pinned in `global.json`. `external/starling` is a submodule pinned by full SHA (build plan section 3.1).
