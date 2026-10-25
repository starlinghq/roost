# Starlings.Roost.Worker.Runtime.linux-x64

The Roost GPU worker process, `roost-worker`, compiled with Native AOT for linux-x64 (R-062).

`Starlings.Generation.Roost` depends on this package. Its `RoostChatClient` starts `roost-worker` as a separate process by default (decision 21). CUDA, the kernel library and the model weights load in that process, so a sticky GPU fault can't take the application down, and none of the worker's native code enters the application's process.

## What the package contains

| Path | What it is |
| --- | --- |
| `runtimes/linux-x64/native/roost-worker` | The worker executable (Native AOT, self-contained) |
| `runtimes/linux-x64/native/roost-kernels.manifest.toml` | The kernel manifest the worker was built against (C5) |

NuGet copies native runtime assets into the build output:
- a RID-specific build or publish (`-r linux-x64`) puts them beside the application;
- a portable build puts them under `runtimes/linux-x64/native/`.

`RoostWorkerRuntime.FindExecutable` looks in both places. NuGet doesn't always keep the execute bit when it extracts a package, so the host restores it before starting the worker (only where read access is already granted).

## Using it

```csharp
var options = new RoostChatClientOptions
{
    Model = profile, // ManifestDigest = RoostWorkerRuntime.ManifestDigest(), WeightsDigest = the lock's SHA-256
    Worker = new RoostWorkerOptions
    {
        Launch = new RoostWorkerLaunch
        {
            WeightsPath = "/models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors",
            WeightsSize = 1_746_942_600,
            KernelDirectory = "/opt/roost/kernels", // libroost_kernels_sm<N>.so; until R-060's acquisition by digest lands
        },
    },
};
```

The host passes the profile's step and pool capacities to the worker, so the worker always allocates exactly what the host's plans need. The worker verifies the weights against the profile's digest as it reads them (R-024), and the C8 handshake checks the manifest digest.

## Not yet

- **Kernel libraries:** fetching them by digest at first run (decision 12) comes with R-060 (#140). Until then, `KernelDirectory` names a local build.
- **Other platforms:** linux-x64 only. Windows would be a second runtime package, once the kernel library builds with MSVC (R-114).
