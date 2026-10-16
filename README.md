# Roost

Roost is the serving engine of the Starling project: a single-process C# engine for NVIDIA GPUs that serves the Starling model library behind one host. Vendored kernels sit behind a C ABI; the scheduler, KV cache, step loop, CUDA graphs and OpenAI-compatible API are .NET. There is no Python or libtorch at runtime.

Roost consumes the [Starling](https://github.com/starlinghq/starling) packages and evidence pipeline. Every served profile passes Starling's fidelity gates before any throughput number is published.

Status: repository skeleton (R-003). Nothing is usable yet. The build plan is [docs/plan/build-plan.md](docs/plan/build-plan.md); agent and contributor rules are in [AGENTS.md](AGENTS.md).

## Building

```sh
git clone --recurse-submodules https://github.com/starlinghq/roost
dotnet test --solution Roost.slnx
```

The SDK is pinned in `global.json`. `external/starling` is a submodule pinned by full SHA (build plan section 3.1).
