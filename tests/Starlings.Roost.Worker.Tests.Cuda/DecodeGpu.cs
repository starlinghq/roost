using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;
using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;
using Starlings.Roost.Worker.Weights;

// R-025 GPU phase: greedy decode of Qwen3.5-0.8B on the eager worker through the executor path (C2 plan in, C2 result out),
// then a dump of tokens, per-step logits and the op outputs of chosen steps for the CPU phase to compare.
//   usage: gpu --ckpt <dir> --lock <lock.toml> --manifest <manifest.toml> --kernels <dir> --out <dump> [--new-tokens N] [--trace-prompt 0] [--trace-decode 2] [--allow-untested]
// Run it through tools/gpu-lease (CUDA_VISIBLE_DEVICES selects the leased device; ordinal 0 here). Everything returns
// statuses or throws managed exceptions; nothing in it raises a signal.
internal static unsafe class DecodeGpu
{
    private const string Weights = "model.safetensors-00001-of-00001.safetensors";

    public static int Run(string[] args)
    {
        string Opt(string name, string? fallback = null) =>
            args.SkipWhile(a => a != name).Skip(1).FirstOrDefault() ?? fallback ?? throw new ArgumentException($"missing {name}");
        string ckpt = Opt("--ckpt");
        string lockPath = Opt("--lock");
        string manifestPath = Opt("--manifest");
        string kernels = Opt("--kernels");
        string outPath = Opt("--out");
        int newTokens = int.Parse(Opt("--new-tokens", "8"), CultureInfo.InvariantCulture);
        int tracePrompt = int.Parse(Opt("--trace-prompt", "0"), CultureInfo.InvariantCulture);
        int traceDecode = int.Parse(Opt("--trace-decode", "2"), CultureInfo.InvariantCulture);
        bool allowUntested = args.Contains("--allow-untested");

        var manifest = KernelManifest.Load(manifestPath);
        var lockFile = WeightLock.Load(lockPath);
        var locked = lockFile.Get(Weights);
        var model = new ProbedQwen35EagerModel();
        var options = new EagerWorkerOptions
        {
            Device = 0,
            MaxTokens = 64,
            MaxSequences = 1,
            MaxKvBlocks = 8,
            AttentionBlocks = 8,
            RecurrentSlots = 1,
            KernelDirectory = kernels,
            AllowUntested = allowUntested,
            KeepLogits = true,
        };

        using var executor = new EagerStepExecutor(model, options, manifest, Path.Combine(ckpt, Weights), locked);
        var identity = executor.Identity;
        Console.WriteLine($"identity: abi {identity.AbiVersion >> 16}.{identity.AbiVersion & 0xFFFF}, manifest {Convert.ToHexStringLower(identity.ManifestDigest.AsSpan())[..16]}, weights {Convert.ToHexStringLower(identity.WeightsDigest.AsSpan())[..16]}");
        var uuid = new byte[16];
        var reject = executor.OpenDevice(0, uuid);
        Console.WriteLine($"OpenDevice: {reject}; device id {Convert.ToHexStringLower(uuid)}");
        if (reject != Starlings.Roost.Transport.HandshakeReject.None)
        {
            return 3;
        }

        var worker = executor.Worker!;
        var clock = Stopwatch.StartNew();
        reject = executor.LoadWeights();
        Console.WriteLine($"LoadWeights: {reject}, {clock.Elapsed.TotalSeconds:F1} s; sha256 {worker.Weights?.Verification.Sha256}");
        if (reject != Starlings.Roost.Transport.HandshakeReject.None)
        {
            return 4;
        }

        int sm = worker.Sm;
        Console.WriteLine($"device SM {sm}; library {worker.Library.Name}, abi {worker.Library.AbiVersion >> 16}.{worker.Library.AbiVersion & 0xFFFF}, SM status {worker.Library.SmStatus}");
        Console.WriteLine($"device memory: {worker.Allocator.AllocatedBytes / 1048576.0:F0} MiB allocated, {worker.Allocator.PinnedBytes / 1048576.0:F1} MiB pinned");
        foreach (var (purpose, bytes) in worker.Allocator.Report())
        {
            Console.WriteLine($"  {purpose}: {bytes / 1048576.0:F1} MiB");
        }

        string libraryPath = KernelLibraryNames.Find(sm, [kernels], KernelLibraryNames.Current())!;
        var dump = new Dump
        {
            Device = $"SM {sm}",
            Sm = sm,
            LibrarySha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(libraryPath))),
        };
        Console.WriteLine($"library file {Path.GetFileName(libraryPath)} sha256 {dump.LibrarySha256}");

        Span<byte> resultBuffer = new byte[4096];
        int failures = 0;
        for (int pi = 0; pi < Prompts.All.Length; pi++)
        {
            var (name, ids) = Prompts.All[pi];
            var tokens = new List<int>();
            var logits = new List<float[]>();
            var prompt = new PromptDump { Name = name, Ids = ids, Tokens = [] };
            var generated = new List<int>();
            for (int step = 0; step < newTokens; step++)
            {
                int[] stepTokens = step == 0 ? ids : [generated[^1]];
                int offset = step == 0 ? 0 : ids.Length + step - 1;
                int pages = (offset + stepTokens.Length + 15) / 16;
                var blocks = Enumerable.Range(0, pages).ToArray();
                var bytes = PlanWriter.Write((ulong)(step + 1), 2, new TestSequence(1, stepTokens, offset, blocks, 0) { MaxNewTokensLeft = (uint)(newTokens - step) });
                bool traced = pi == tracePrompt && step <= traceDecode;
                if (traced)
                {
                    ProbeState.Recorder = new Recorder { Read = ReadDevice(worker) };
                }

                var outcome = executor.Execute(bytes, resultBuffer);
                var recorder = ProbeState.Recorder;
                ProbeState.Recorder = null;
                if (outcome.Fault != FaultClass.None)
                {
                    Console.WriteLine($"FAIL prompt {name} step {step}: fault {outcome.Fault} code {outcome.ErrorCode}");
                    failures++;
                    break;
                }

                var result = resultBuffer[..outcome.ResultBytes];
                var h = MemoryMarshal.Read<StepResultHeader>(result);
                int token = MemoryMarshal.Read<int>(result[(int)h.TokenIds.Offset..]);
                var rs = MemoryMarshal.Read<ResultSequence>(result[(int)h.Sequences.Offset..]);
                int viaWorker = worker.LastOutput!.Tokens[0];
                if (token != viaWorker || h.StepId != (ulong)(step + 1) || rs.SequenceHandle != 1 || rs.TokenCount != 1)
                {
                    Console.WriteLine($"FAIL prompt {name} step {step}: result header disagrees with the worker's output ({token} vs {viaWorker})");
                    failures++;
                }

                generated.Add(token);
                logits.Add(worker.LastOutput.Logits!);
                var t = worker.LastOutput.Timings;
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  {name} step {step}: {stepTokens.Length} token(s) -> {token}; upload {t.UploadNs / 1e3:F0} us, forward {t.ForwardNs / 1e6:F2} ms (eager, synchronised), argmax {t.SampleNs / 1e3:F0} us; finish {rs.Finish}"));
                if (traced)
                {
                    prompt.Traced[step] = recorder!.Snapshots;
                }
            }

            var final = new PromptDump { Name = name, Ids = ids, Tokens = [.. generated] };
            foreach (var l in logits)
            {
                final.Logits.Add(l);
            }

            foreach (var kv in prompt.Traced)
            {
                final.Traced[kv.Key] = kv.Value;
            }

            dump.Prompts.Add(final);
            Console.WriteLine($"prompt {name}: ids [{string.Join(' ', ids)}] -> greedy tokens [{string.Join(' ', generated)}]");
            tokens.Clear();
        }

        // Determinism: the first prompt again, bit for bit (every variant is flagged deterministic and batch invariant).
        {
            var (name, ids) = Prompts.All[0];
            var again = new List<float[]>();
            var generated = new List<int>();
            for (int step = 0; step < newTokens; step++)
            {
                int[] stepTokens = step == 0 ? ids : [generated[^1]];
                int offset = step == 0 ? 0 : ids.Length + step - 1;
                var blocks = Enumerable.Range(0, (offset + stepTokens.Length + 15) / 16).ToArray();
                var bytes = PlanWriter.Write((ulong)(100 + step), 2, new TestSequence(1, stepTokens, offset, blocks, 0));
                var outcome = executor.Execute(bytes, resultBuffer);
                generated.Add(worker.LastOutput!.Tokens[0]);
                again.Add(worker.LastOutput.Logits!);
                if (outcome.Fault != FaultClass.None)
                {
                    failures++;
                }
            }

            int differing = 0;
            for (int i = 0; i < again.Count; i++)
            {
                if (!again[i].AsSpan().SequenceEqual(dump.Prompts[0].Logits[i]))
                {
                    differing++;
                }
            }

            Console.WriteLine($"determinism: second run of '{name}' has {differing} of {again.Count} steps with logits that differ bitwise; tokens {(generated.SequenceEqual(dump.Prompts[0].Tokens) ? "identical" : "DIFFER")}");
            if (differing != 0)
            {
                failures++;
            }
        }

        // The plan reader refuses what the worker cannot run, on the host, before anything is launched.
        {
            var bad = PlanWriter.Write(200, 2, new TestSequence(1, [model.VocabSize], 0, [0], 0));
            var outcome = executor.Execute(bad, resultBuffer);
            Console.WriteLine($"plan with an out-of-vocabulary token id: {outcome.Fault}, code {outcome.ErrorCode} ({(PlanRejection)outcome.ErrorCode})");
            if (outcome.Fault != FaultClass.Recoverable || (PlanRejection)outcome.ErrorCode != PlanRejection.OutOfRange)
            {
                failures++;
            }

            var sampled = PlanWriter.Write(201, 2, new TestSequence(1, [760], 0, [0], 0) { Sampling = new PlanSampling { Temperature = 0.7f, TopP = 0.9f, TopK = 40 } });
            outcome = executor.Execute(sampled, resultBuffer);
            Console.WriteLine($"plan asking for temperature sampling: {outcome.Fault}, code {outcome.ErrorCode} ({(PlanRejection)outcome.ErrorCode})");
            if (outcome.Fault != FaultClass.Recoverable || (PlanRejection)outcome.ErrorCode != PlanRejection.Unsupported)
            {
                failures++;
            }
        }

        Console.WriteLine("variants chosen: " + string.Join(", ", worker.Session.ChosenVariants.OrderBy(static v => v.Op).Select(static v => $"{v.Op}={v.Variant}")));
        Console.WriteLine("kernel launches: " + string.Join(", ", worker.Session.Launches.OrderBy(static v => v.Key).Select(static v => $"{v.Key}={v.Value}")));
        dump.Save(outPath);
        Console.WriteLine($"dump written to {outPath}");
        Console.WriteLine(failures == 0 ? "PASS" : $"FAIL ({failures})");
        return failures == 0 ? 0 : 1;
    }

    private static Func<Starlings.Roost.DeviceSpan, long, float[]> ReadDevice(EagerWorker worker) => (span, elements) =>
    {
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(worker.Session.Stream), "cudaStreamSynchronize");
        long bytes = elements * (span.Dtype == Starlings.Roost.Abi.DType.F32 ? 4 : 2);
        var host = new byte[bytes];
        fixed (byte* p = host)
        {
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(p, (void*)span.Address, (nuint)bytes, CudaRuntime.cudaMemcpyDeviceToHost), "cudaMemcpy");
            return HostReader.Convert(p, span.Dtype, elements);
        }
    };
}
