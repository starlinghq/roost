using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;
using Starlings.Roost.Models.Qwen35.Tests.Parity;
using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;
using Starlings.Roost.Worker.Weights;

// R-014 GPU check: the eager worker teacher-forced on the reference run's cases (prompt, then the reference's first N-1 generated
// tokens, one per forward), captured per layer in the R-011 layout so that `roost-evidence evaluate-layers` compares it with the
// transformers fp32 reference. Writes layers/<id>.safetensors, step0/<id>.f32 (log-softmax of the prompt forward), candidate.jsonl
// (argmax and top-2 margin per forward) and run.json under --out, outside the repository.
//   usage: capture --ckpt <dir> --lock <lock.toml> --manifest <manifest.toml> --kernels <dir> --reference <reference run dir> --out <dir> [--allow-untested] [--cases a,b]
// Run it through tools/gpu-lease (ordinal 0 is the leased device).
//
// Stages are taken from the probe's op records in call order, as R-022's CPU recorder takes them: the Embedding output; every second
// Residual output (layer.0 .. layer.23); the 49th RmsNorm output (the final norm); for each Gated DeltaNet layer the recurrent state of
// slot 0 after GatedDeltaRule and the convolution state. Roost's conv slot keeps the last width - 1 = 3 inputs per channel and the
// reference keeps 4, so columns 1..3 are the slot after the CausalConv1d and column 0 is the input before them, taken from the conv
// input (the Gemm output just before the CausalConv1d) when the forward has 4 or more tokens, else from column 0 of the slot before
// the call (the previous forward's column 1). The harness checks that columns 1..3 equal the last rows of the conv input it took.
internal static unsafe class CaptureGpu
{
    private const string Weights = "model.safetensors-00001-of-00001.safetensors";
    private const int Layers = 24;
    private const int Hidden = 1024;
    private const int ConvChannels = 6144;
    private const int ConvWidth = 4;

    public static int Run(string[] args)
    {
        string Opt(string name, string? fallback = null) =>
            args.SkipWhile(a => a != name).Skip(1).FirstOrDefault() ?? fallback ?? throw new ArgumentException($"missing {name}");
        string ckpt = Opt("--ckpt");
        string kernels = Opt("--kernels");
        string referenceDir = Opt("--reference");
        string outDir = Opt("--out");
        var only = args.Contains("--cases") ? Opt("--cases").Split(',').ToHashSet(StringComparer.Ordinal) : null;

        var cases = new List<(string Id, int[] Prompt, int[] Output)>();
        foreach (string line in File.ReadLines(Path.Combine(referenceDir, "reference.jsonl")))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            string id = root.GetProperty("id").GetString()!;
            if (only is null || only.Contains(id))
            {
                cases.Add((id, [.. root.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())], [.. root.GetProperty("output_ids").EnumerateArray().Select(e => e.GetInt32())]));
            }
        }

        int maxPrompt = cases.Max(c => c.Prompt.Length);
        int maxContext = cases.Max(c => c.Prompt.Length + c.Output.Length);
        var manifest = KernelManifest.Load(Opt("--manifest"));
        var locked = WeightLock.Load(Opt("--lock")).Get(Weights);
        var model = new Qwen35EagerModel();
        var options = new EagerWorkerOptions
        {
            Device = 0,
            MaxTokens = Math.Max(64, maxPrompt),
            MaxSequences = 1,
            MaxKvBlocks = (maxContext + 15) / 16,
            AttentionBlocks = (maxContext + 15) / 16,
            RecurrentSlots = 1,
            KernelDirectory = kernels,
            AllowUntested = args.Contains("--allow-untested"),
            KeepLogits = true,
        };

        using var executor = new EagerStepExecutor(model, options, manifest, Path.Combine(ckpt, Weights), locked);
        var uuid = new byte[16];
        if (executor.OpenDevice(0, uuid) != Starlings.Roost.Transport.HandshakeReject.None || executor.LoadWeights() != Starlings.Roost.Transport.HandshakeReject.None)
        {
            Console.WriteLine("FAIL: the worker did not open the device or load the weights");
            return 3;
        }

        var worker = executor.Worker!;
        int sm = worker.Sm;
        string libraryPath = KernelLibraryNames.Find(sm, [kernels], KernelLibraryNames.Current())!;
        string librarySha = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(libraryPath)));
        Console.WriteLine($"device SM {sm}; library {Path.GetFileName(libraryPath)} sha256 {librarySha}; SM status {worker.Library.SmStatus}");
        Directory.CreateDirectory(Path.Combine(outDir, "step0"));
        Span<byte> resultBuffer = new byte[4096];
        var summaries = new List<string>();
        var digests = new JsonObject();
        int failures = 0;
        foreach (var (id, prompt, output) in cases)
        {
            var metadata = LayerCaptureFile.ReadMetadata(Path.Combine(referenceDir, "layers", id + ".safetensors"));
            int[] decode = LayerCaptureFile.Ints(metadata["decode_ids"]);
            int stateForwards = int.Parse(metadata["state_forwards"], CultureInfo.InvariantCulture);
            string[] layerTypes = LayerCaptureFile.Strings(metadata["layer_types"]);
            int[] linear = [.. Enumerable.Range(0, Layers).Where(i => layerTypes[i] == "linear_attention")];
            if (!prompt.SequenceEqual(LayerCaptureFile.Ints(metadata["prompt_ids"])) || !output.AsSpan(0, output.Length - 1).SequenceEqual(decode))
            {
                throw new InvalidDataException($"{id}: capture metadata disagrees with reference.jsonl");
            }

            var tensors = new Dictionary<string, (int[] Shape, float[] Values)>(StringComparer.Ordinal);
            var argmax = new JsonArray();
            var margins = new JsonArray();
            var previousConv = new float[linear.Length][];
            var clock = Stopwatch.StartNew();
            for (int forward = 0; forward <= decode.Length; forward++)
            {
                int[] ids = forward == 0 ? prompt : [decode[forward - 1]];
                int offset = forward == 0 ? 0 : prompt.Length + forward - 1;
                int[] blocks = [.. Enumerable.Range(0, (offset + ids.Length + 15) / 16)];
                var plan = PlanWriter.Write((ulong)(forward + 1), 2, new TestSequence(1, ids, offset, blocks, 0) { MaxNewTokensLeft = (uint)(decode.Length + 1 - forward) });
                var recorder = new Recorder { Read = ReadDevice(worker) };
                ProbeState.Recorder = recorder;
                var outcome = executor.Execute(plan, resultBuffer);
                ProbeState.Recorder = null;
                if (outcome.Fault != FaultClass.None)
                {
                    Console.WriteLine($"FAIL {id} forward {forward}: fault {outcome.Fault} code {outcome.ErrorCode}");
                    failures++;
                    break;
                }

                failures += Collect(recorder.Snapshots, id, forward, ids.Length, forward < stateForwards, linear, previousConv, tensors);
                var (best, margin, logprobs) = LogSoftmax(worker.LastOutput!.Logits!);
                argmax.Add(best);
                margins.Add(margin);
                if (forward == 0)
                {
                    using var step0 = File.Create(Path.Combine(outDir, "step0", id + ".f32"));
                    LayerCaptureFile.WriteFloats(step0, logprobs);
                }
            }

            string digest = LayerCaptureFile.Write(Path.Combine(outDir, "layers", id + ".safetensors"), metadata, tensors);
            digests[$"layers/{id}.safetensors"] = digest;
            int agree = argmax.Select((a, i) => (int)a! == output[i] ? 1 : 0).Sum();
            Console.WriteLine($"{id}: {decode.Length + 1} forwards, {prompt.Length} prompt tokens, argmax equals the reference's greedy token at {agree} of {argmax.Count}; {clock.Elapsed.TotalSeconds:F1} s");
            summaries.Add(new JsonObject
            {
                ["id"] = id,
                ["prompt_tokens"] = prompt.Length,
                ["forwards"] = decode.Length + 1,
                ["reference_output_ids"] = new JsonArray([.. output.Select(i => (JsonNode)i)]),
                ["argmax"] = argmax,
                ["margin"] = margins,
                ["layers_sha256"] = digest,
            }.ToJsonString());
        }

        File.WriteAllLines(Path.Combine(outDir, "candidate.jsonl"), summaries, LayerCaptureFile.Utf8);
        var run = new JsonObject
        {
            ["candidate"] = "eager worker (EagerStepExecutor, Qwen35EagerModel over CudaOps through the probe), teacher-forced",
            ["device_sm"] = sm,
            ["library"] = Path.GetFileName(libraryPath),
            ["library_sha256"] = librarySha,
            ["reference"] = referenceDir,
            ["dotnet"] = Environment.Version.ToString(),
            ["captures"] = digests,
        };
        File.WriteAllText(Path.Combine(outDir, "run.json"), run.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n", LayerCaptureFile.Utf8);
        Console.WriteLine(failures == 0 ? "PASS" : $"FAIL ({failures})");
        return failures == 0 ? 0 : 1;
    }

    /// <summary>Maps one forward's op records to the R-011 stage names; returns the number of self-check failures.</summary>
    private static int Collect(List<Snapshot> snaps, string id, int forward, int tokens, bool states, int[] linear, float[][] previousConv, Dictionary<string, (int[] Shape, float[] Values)> tensors)
    {
        string F(string stage) => string.Create(CultureInfo.InvariantCulture, $"forward{forward}/{stage}");
        int failures = 0;
        var embeddings = snaps.Where(s => s.Op == "Embedding").ToList();
        var layerOuts = snaps.Where(s => s.Op == "Residual" && s.What == "layer").ToList();
        var norms = snaps.Where(s => s.Op == "RmsNorm").ToList();
        var deltas = snaps.Where(s => s.Op == "GatedDeltaRule" && s.What == "state").ToList();
        var convStates = snaps.Where(s => s.Op == "CausalConv1d" && s.What == "state").ToList();
        if (embeddings.Count != 1 || layerOuts.Count != Layers || norms.Count != (2 * Layers) + 1 || deltas.Count != linear.Length || convStates.Count != linear.Length)
        {
            throw new InvalidOperationException($"{id} forward {forward}: {embeddings.Count} embeddings, {layerOuts.Count} layer outputs, {norms.Count} norms, {deltas.Count} delta rules, {convStates.Count} conv states");
        }

        tensors[F("embeddings")] = ([tokens, Hidden], embeddings[0].Values);
        for (int i = 0; i < Layers; i++)
        {
            tensors[F($"layer.{i}")] = ([tokens, Hidden], layerOuts[i].Values);
        }

        tensors[F("final_norm")] = ([tokens, Hidden], norms[^1].Values);
        for (int n = 0; n < linear.Length; n++)
        {
            int layer = linear[n];
            float[] slot = convStates[n].Values; // [channels, 3] after the call
            float[] conv = new float[ConvChannels * ConvWidth];
            float[]? input = null;
            if (tokens >= ConvWidth)
            {
                // The conv input is the Gemm output recorded just before this CausalConv1d ([tokens, 6144]).
                int at = snaps.IndexOf(convStates[n]);
                input = snaps.Take(at).Last(s => s.Op == "Gemm" && s.Values.Length == tokens * ConvChannels).Values;
            }

            bool consistent = true;
            for (int c = 0; c < ConvChannels; c++)
            {
                for (int j = 1; j < ConvWidth; j++)
                {
                    conv[(c * ConvWidth) + j] = slot[(c * (ConvWidth - 1)) + j - 1];
                }

                conv[c * ConvWidth] = input is not null
                    ? input[((tokens - ConvWidth) * ConvChannels) + c]
                    : previousConv[n] is { } prev ? prev[(c * ConvWidth) + 1] : 0f;
                if (input is not null)
                {
                    for (int j = 1; j < ConvWidth; j++)
                    {
                        consistent &= conv[(c * ConvWidth) + j] == input[((tokens - ConvWidth + j) * ConvChannels) + c];
                    }
                }
            }

            if (!consistent)
            {
                Console.WriteLine($"FAIL {id} forward {forward} layer {layer}: the conv slot is not the last rows of the Gemm output taken as the conv input");
                failures++;
            }

            previousConv[n] = conv;
            if (states)
            {
                tensors[F($"layer.{layer}/recurrent_state")] = ([16, 128, 128], deltas[n].Values);
                tensors[F($"layer.{layer}/conv_state")] = ([ConvChannels, ConvWidth], conv);
            }
        }

        return failures;
    }

    private static (int Best, double Margin, float[] LogProbs) LogSoftmax(float[] logits)
    {
        double max = double.NegativeInfinity;
        int best = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
                best = i;
            }
        }

        double second = double.NegativeInfinity;
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += Math.Exp(logits[i] - max);
            if (i != best && logits[i] > second)
            {
                second = logits[i];
            }
        }

        double log = max + Math.Log(sum);
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = (float)(logits[i] - log);
        }

        return (best, max - second, result);
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
