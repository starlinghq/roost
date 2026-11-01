# ChatConsole

A console chat with Qwen3.5-0.8B through Microsoft.Extensions.AI's `IChatClient`, served by `RoostChatClient` (R-062). The client starts the packaged `roost-worker` in a separate process, the default transport (decision 21).

The project references `Starlings.Generation.Roost` and `Starlings.Roost.Models.Qwen35` by version, as a consumer does after `dotnet add package`. It is not in `Roost.slnx`. `samples/Directory.Build.props` and the other two files in `samples/` keep the repository's build settings out.

## Status

- **Build:** the release drill (`tools/release-drill.sh`, step 6) builds the sample against its local feed of the packed packages. This is a CPU build only.
- **Run:** `untested`. Nobody has run the sample on a GPU yet. The R-036 GPU run (`tests/Starlings.Generation.Roost.Tests.Cuda`) covers the same API with a smaller context.
- Nothing is on nuget.org until Preview 1 is released, so `dotnet build` here fails to restore unless a local feed is given (below).

## Build against a local feed

The project pins both packages exactly at the Preview 1 version, `[0.1.0-preview.1]`. The release drill builds a copy of it with both pins changed to the drilled version. To do the same by hand, run the drill with `DRILL_KEEP=1`. It prints the work directory, and the packed packages are in `<work>/feed`. Change both `Version="[...]"` pins in `ChatConsole.csproj` to the drilled version, then build:

```sh
DRILL_KEEP=1 tools/release-drill.sh --version 0.0.1-drill.1 --no-kernels --skip-tests
dotnet build samples/ChatConsole --source <work>/feed --source https://api.nuget.org/v3/index.json
```

## Run (linux-x64, NVIDIA GPU)

You need these:

- the CUDA runtime and cuBLASLt from a system CUDA install (decision 12);
- the locked checkpoint `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (`tokenizer.json` and `model.safetensors-00001-of-00001.safetensors`; the digest is in `evidence/registry/qwen3.5-0.8b.lock.toml`);
- a directory with `libroost_kernels_sm<N>.so` for your GPU. The worker does not fetch the library yet (see the package README, "Native code").

```sh
dotnet run --project samples/ChatConsole -- --model /models/qwen3.5-0.8b --kernels /opt/roost/kernels
```

Pass `--device <ordinal>` to choose a GPU. Pass `--allow-untested-sm` for a GPU whose SM the kernel manifest marks `untested`; a run on such a GPU is evidence for that SM, not a support claim. Decoding is greedy. Sampling and the other capabilities that are not built are `unsupported` (see [docs/capabilities.md](../../docs/capabilities.md)).
