# Roost

Roost is the serving engine of the Starling project: a single-process C# engine for NVIDIA GPUs that serves the Starling model library behind one host. Vendored kernels sit behind a C ABI; the scheduler, KV cache, step loop, CUDA graphs and OpenAI-compatible API are .NET. There is no Python or libtorch at runtime.

Roost consumes the [Starling](https://github.com/starlinghq/starling) packages and evidence pipeline. Every served profile passes Starling's fidelity gates before any throughput number is published.

Status: planning. The build plan is [docs/plan/build-plan.md](docs/plan/build-plan.md).
