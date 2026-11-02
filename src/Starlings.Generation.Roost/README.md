# Starlings.Generation.Roost

Roost's chat API for .NET: `RoostChatClient`, an `IChatClient` (Microsoft.Extensions.AI) that runs a model on an NVIDIA GPU through Roost's worker. This is a preview. Read the status sections below before you build on it.

## Worker process by default

`RoostChatClient.StartAsync` starts the GPU worker, `roost-worker`, as a separate process by default (build plan decision 21). CUDA, the kernel library and the model weights load in that process, not in your application's. A sticky CUDA fault ends the worker, and your application gets a typed error (`RoostChatException`) instead of crashing.

The worker executable comes from the `Starlings.Roost.Worker.Runtime.linux-x64` package, which this package depends on. NuGet copies it into your build output. `RoostWorkerRuntime.FindExecutable` finds it there. The in-process transport (`TransportKind.InProcess`) is for tests and development only.

This package is host-side: it does not reference CUDA, the kernels or the worker's code (decision 30).

## Native code

Decision 12, as recorded in [docs/kickoff.md](https://github.com/starlinghq/roost/blob/main/docs/kickoff.md) (maintainer, 2026-10-20), sets how native code is delivered:

- **Roost's kernel library** (`libroost_kernels_sm<N>.so`, one per GPU architecture) is published as an asset of the GitHub release of the same version. It is fetched by digest through `Starlings.Acquisition` and checked against the kernel manifest the package was built with. No kernel binary is in any NuGet package.
- **The CUDA runtime and cuBLASLt** come from the system's CUDA install. Roost does not ship them.

Status today: the release fetch exists in `Starlings.Roost.Kernels`, but the packaged worker does not use it yet. You must set `RoostWorkerLaunch.KernelDirectory` to a directory that holds the library for your GPU. If that directory has no library, `StartAsync` refuses before it starts the worker. Releases publish Linux x64 libraries only.

## Qwen3.5-0.8B

The first model is Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17`). The weights file must be the locked one: 1,746,942,600 bytes, SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (from `evidence/registry/qwen3.5-0.8b.lock.toml`). The worker checks the digest as it reads the file.

Prompts are rendered with Starling's published `Qwen35ChatTemplate` through `Qwen35RoostChatTemplate.Instance`. Generation stops on `<|im_end|>` and `<|endoftext|>`.

```csharp
using System.Collections.Immutable;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Qwen35; // package Starlings.Roost.Models.Qwen35
using Starlings.Tokenizers;

var tokenizer = Tokenizer.FromFile("/models/qwen3.5-0.8b/tokenizer.json");
var profile = new RoostModelProfile
{
    ModelId = "qwen3.5-0.8b",
    Tokenizer = tokenizer,
    ChatTemplate = Qwen35RoostChatTemplate.Instance,
    StopTokenIds = [tokenizer.TokenToId("<|im_end|>")!.Value, tokenizer.TokenToId("<|endoftext|>")!.Value],
    StateGroups = Qwen35Model.CreateStateGroups(),
    AttentionBlocks = 128,
    ManifestDigest = ImmutableArray.Create(RoostWorkerRuntime.ManifestDigest()),
    WeightsDigest = ImmutableArray.Create(Convert.FromHexString("04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696")),
};

await using var roost = await RoostChatClient.StartAsync(new RoostChatClientOptions
{
    Model = profile,
    Worker = new RoostWorkerOptions
    {
        Launch = new RoostWorkerLaunch
        {
            WeightsPath = "/models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors",
            WeightsSize = 1_746_942_600,
            KernelDirectory = "/opt/roost/kernels", // required until the worker fetches the library (above)
        },
    },
});

IChatClient client = roost;
var response = await client.GetResponseAsync("What is the capital of France?");
Console.WriteLine(response.Text);
```

A complete console app is [samples/ChatConsole](https://github.com/starlinghq/roost/tree/main/samples/ChatConsole).

## Capabilities

Every capability has an explicit status. A capability that is not built is `unsupported`, and a request that asks for it is refused with a typed error naming the option. A capability that is missing from the table is also unsupported. `RoostChatClient.Capabilities` holds the table at run time.

- **`untested`**: built and tested on the CPU path, without GPU evidence. This covers chat, raw completion, streaming, cancellation, usage counts, greedy decoding, stop token ids, `MaxOutputTokens`, instructions and stop sequences.
- **`unsupported`**: not built. This covers sampling (temperature, top-p, top-k), penalties, tools, response formats, log probabilities, reasoning output, multimodal input, conversation state, continuous batching and worker recovery.

The full table is [docs/capabilities.md](https://github.com/starlinghq/roost/blob/main/docs/capabilities.md). The M0 evidence (spike reports, the CPU parity run, the GPU captures) is in [docs/m0/](https://github.com/starlinghq/roost/tree/main/docs/m0).

Platforms: linux-x64 only. Windows and WSL2 are `untested`.

## License

MIT OR Apache-2.0.
