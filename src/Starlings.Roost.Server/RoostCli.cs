using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Server.Doctor;
using Starlings.Roost.Server.Pooling;
using Starlings.Tokenizers;

namespace Starlings.Roost.Server;

/// <summary>The arguments of <c>roost serve</c>.</summary>
/// <param name="Model">The model profile name.</param>
/// <param name="Weights">The safetensors weights file.</param>
/// <param name="WeightsSha256">The weights' SHA-256 from the profile lock, as hex.</param>
/// <param name="Tokenizer">The model's <c>tokenizer.json</c>.</param>
/// <param name="KernelDirectory">The kernel library directory, or null for the worker's directory.</param>
/// <param name="Worker">The <c>roost-worker</c> executable, or null for the packaged one.</param>
/// <param name="Manifest">The kernel manifest, or null for the one shipped beside the worker.</param>
/// <param name="AllowUntestedSm">Whether a GPU whose SM the manifest marks untested may serve.</param>
/// <param name="Listen">Where to listen.</param>
/// <param name="Pooling">The pooling model directories (R-052), or null for none.</param>
internal sealed record ServeCommand(
    string Model,
    string Weights,
    string WeightsSha256,
    string Tokenizer,
    string? KernelDirectory,
    string? Worker,
    string? Manifest,
    bool AllowUntestedSm,
    RoostServerOptions Listen,
    PoolingPaths? Pooling = null);

/// <summary>
/// The local snapshot directories of the pooling models <c>roost serve</c> loads (R-052), each verified against its Starling
/// package's asset lock; nothing is downloaded. Null means that model is not served.
/// </summary>
/// <param name="E5">multilingual-e5-small (<c>--pooling-e5</c>).</param>
/// <param name="EmbeddingGemma2">EmbeddingGemma 2 (<c>--pooling-embeddinggemma2</c>).</param>
/// <param name="EmbeddingGemma2Dimensions">EmbeddingGemma 2's vector length (<c>--pooling-embeddinggemma2-dimensions</c>).</param>
/// <param name="Gliner2">GLiNER2 (<c>--pooling-gliner2</c>).</param>
internal sealed record PoolingPaths(string? E5, string? EmbeddingGemma2, int EmbeddingGemma2Dimensions, string? Gliner2)
{
    /// <summary>Whether any pooling model is configured.</summary>
    public bool Any => E5 is not null || EmbeddingGemma2 is not null || Gliner2 is not null;
}

/// <summary>
/// The <c>roost</c> command line: <c>roost serve</c> (R-037) and <c>roost doctor</c> (R-046); vLLM's flags and environment
/// variables come with R-075. The one model profile is Qwen3.5-0.8B on the eager worker, with the capacities of the R-036 GPU run.
/// </summary>
internal static class RoostCli
{
    /// <summary>The profile name of Qwen3.5-0.8B, the M0 model (decision 15).</summary>
    public const string Qwen35Profile = "qwen3.5-0.8b";

    public const string Usage = """
        usage: roost serve --model qwen3.5-0.8b --weights <model.safetensors> --weights-sha256 <hex> --tokenizer <tokenizer.json>
                           [--kernels <dir>] [--worker <roost-worker>] [--manifest <manifest.toml>] [--allow-untested-sm]
                           [--host <ip>] [--port <n>] [--allow-non-loopback] [--enable-sleep-mode]
                           [--pooling-e5 <dir>] [--pooling-embeddinggemma2 <dir> [--pooling-embeddinggemma2-dimensions 768|512|256|128]]
                           [--pooling-gliner2 <dir>]

               roost doctor [--model qwen3.5-0.8b] [--weights <model.safetensors> --weights-sha256 <hex>] [--tokenizer <tokenizer.json>]
                            [--kernels <dir>] [--worker <roost-worker>] [--manifest <manifest.toml>] [--allow-untested-sm] [--no-gpu]
                            [--min-free-disk-gib <n>] [--min-free-memory-gib <n>]

        serve: serves the OpenAI routes /v1/chat/completions, /v1/completions and /v1/models on 127.0.0.1:8000 by default.
        A non-loopback --host needs --allow-non-loopback: the server has no authentication. Each start stage's duration is
        logged and exported on /metrics. The --pooling-* flags load Starling's ONNX models on the CPU from local snapshot
        directories verified against their asset locks (never downloaded) for /v1/embeddings and /v1/extract.
        --enable-sleep-mode adds POST /sleep, POST /wake_up, GET /is_sleeping and POST /update_weights, for loopback
        clients only (the packaged CUDA worker refuses them today).
        doctor: checks the driver, NVML, the GPUs' SMs, the kernel manifest and library digests, the weights lock, the
        tokenizer, free disk and host memory, one line per check; exits 1 when a check fails.
        """;

    private static readonly HashSet<string> Flags = new(StringComparer.Ordinal) { "--allow-untested-sm", "--allow-non-loopback", "--enable-sleep-mode" };

    private static readonly HashSet<string> Valued = new(StringComparer.Ordinal)
    {
        "--model", "--weights", "--weights-sha256", "--tokenizer", "--kernels", "--worker", "--manifest", "--host", "--port",
        "--pooling-e5", "--pooling-embeddinggemma2", "--pooling-embeddinggemma2-dimensions", "--pooling-gliner2",
    };

    private static readonly HashSet<string> DoctorFlags = new(StringComparer.Ordinal) { "--allow-untested-sm", "--no-gpu" };

    private static readonly HashSet<string> DoctorValued = new(StringComparer.Ordinal)
    {
        "--model", "--weights", "--weights-sha256", "--tokenizer", "--kernels", "--worker", "--manifest", "--min-free-disk-gib",
        "--min-free-memory-gib",
    };

    /// <summary>Runs the command line, with <c>roost doctor</c>'s report on standard output.</summary>
    /// <param name="args">The arguments.</param>
    /// <param name="error">Where usage errors and the cold-start log go.</param>
    /// <returns>The exit code: 0 after a clean shutdown or a doctor run with no failed check, 1 when the worker does not start
    /// or a doctor check fails, 2 for a usage error.</returns>
    public static Task<int> RunAsync(string[] args, TextWriter error) => RunAsync(args, Console.Out, error);

    /// <summary>Runs the command line.</summary>
    /// <param name="args">The arguments.</param>
    /// <param name="output">Where <c>roost doctor</c>'s report goes.</param>
    /// <param name="error">Where usage errors and the cold-start log go.</param>
    /// <param name="probes">The environment <c>roost doctor</c> reads; null means the real machine (tests pass fakes).</param>
    /// <param name="sinceProcessStart">The time from process creation to <c>Main</c> (the cold start's first stage), or null.</param>
    /// <returns>The exit code, as above.</returns>
    public static async Task<int> RunAsync(string[] args, TextWriter output, TextWriter error, IDoctorProbes? probes = null, TimeSpan? sinceProcessStart = null)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        return args.Length > 0 && args[0] == "doctor"
            ? await DoctorAsync(args, output, error, probes ?? new SystemDoctorProbes()).ConfigureAwait(false)
            : await ServeAsync(args, error, sinceProcessStart).ConfigureAwait(false);
    }

    private static async Task<int> DoctorAsync(string[] args, TextWriter output, TextWriter error, IDoctorProbes probes)
    {
        DoctorCommand command;
        try
        {
            command = ParseDoctor(args);
        }
        catch (ArgumentException e)
        {
            await error.WriteLineAsync("roost: " + e.Message).ConfigureAwait(false);
            await error.WriteLineAsync(Usage).ConfigureAwait(false);
            return 2;
        }

        var report = RoostDoctor.Run(command, probes);
        report.WriteTo(output);
        await output.FlushAsync().ConfigureAwait(false);
        return report.Healthy ? 0 : 1;
    }

    private static async Task<int> ServeAsync(string[] args, TextWriter error, TimeSpan? sinceProcessStart)
    {
        var clock = TimeProvider.System;
        var coldStart = new ColdStart(clock, error);
        long parsing = clock.GetTimestamp();
        ServeCommand command;
        try
        {
            command = Parse(args);
            command.Listen.ResolveAddress();
        }
        catch (ArgumentException e)
        {
            await error.WriteLineAsync("roost: " + e.Message).ConfigureAwait(false);
            await error.WriteLineAsync(Usage).ConfigureAwait(false);
            return 2;
        }

        if (sinceProcessStart is { } runtime)
        {
            coldStart.Record(ColdStart.Runtime, runtime);
        }

        coldStart.Record(ColdStart.Arguments, clock.GetElapsedTime(parsing));

        RoostPoolingModels pooling;
        IReadOnlyList<IDisposable> poolingOwned;
        try
        {
            (pooling, poolingOwned) = command.Pooling is null
                ? (RoostPoolingModels.None, [])
                : coldStart.Measure(ColdStart.Pooling, () => OpenPooling(command.Pooling));
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            // Missing or corrupt snapshot files (Starlings.Acquisition), a missing ONNX Runtime native library or a bad graph
            // (Starlings.Runtime.Onnx): the message names the file or the cause, never request text.
            await error.WriteLineAsync("roost: a pooling model did not load: " + e.Message).ConfigureAwait(false);
            return 1;
        }

        try
        {
            return await ServeWorkerAsync(command, error, coldStart, clock, pooling).ConfigureAwait(false);
        }
        finally
        {
            foreach (var d in poolingOwned)
            {
                d.Dispose();
            }
        }
    }

    private static async Task<int> ServeWorkerAsync(ServeCommand command, TextWriter error, ColdStart coldStart, TimeProvider clock, RoostPoolingModels pooling)
    {
        RoostChatClient client;
        try
        {
            var options = coldStart.Measure(ColdStart.Profile, () => Options(command));
            client = await coldStart.MeasureAsync(ColdStart.Worker, () => RoostChatClient.StartAsync(options)).ConfigureAwait(false);
        }
        catch (Exception e) when (e is ArgumentException or IOException or UnauthorizedAccessException or RoostChatException or JsonException or FormatException)
        {
            await error.WriteLineAsync("roost: the worker did not start: " + e.Message).ConfigureAwait(false);
            return 1;
        }

        await using (client.ConfigureAwait(false))
        {
            long listening = clock.GetTimestamp();
            var app = RoostServer.Create(client, command.Listen with { ColdStart = coldStart, Pooling = pooling });
            await using (app.ConfigureAwait(false))
            {
                try
                {
                    await app.StartAsync().ConfigureAwait(false);
                }
                catch (IOException e)
                {
                    await error.WriteLineAsync("roost: could not listen: " + e.Message).ConfigureAwait(false);
                    return 1;
                }

                coldStart.Record(ColdStart.Listen, clock.GetElapsedTime(listening));
                coldStart.WriteTotal();
                await app.WaitForShutdownAsync().ConfigureAwait(false);
            }
        }

        return 0;
    }

    /// <summary>Parses <c>serve</c> and its flags; refuses anything else by name.</summary>
    /// <exception cref="ArgumentException">An unknown command or flag, a missing value, or a missing required flag.</exception>
    public static ServeCommand Parse(string[] args)
    {
        if (args.Length == 0 || args[0] != "serve")
        {
            throw new ArgumentException(args.Length == 0 ? "no command given." : $"unknown command '{args[0]}'.");
        }

        var (values, flags) = ReadOptions(args, Flags, Valued);

        string Required(string name) => values.TryGetValue(name, out var v) ? v : throw new ArgumentException($"{name} is required.");
        string? Optional(string name) => values.GetValueOrDefault(name);

        var model = Required("--model");
        if (model != Qwen35Profile)
        {
            throw new ArgumentException($"unknown model profile '{model}'; the profiles are: {Qwen35Profile}.");
        }

        var port = RoostServerOptions.DefaultPort;
        if (Optional("--port") is { } p && !int.TryParse(p, NumberStyles.None, CultureInfo.InvariantCulture, out port))
        {
            throw new ArgumentException($"--port '{p}' is not a port number.");
        }

        var sha = Required("--weights-sha256");
        if (sha.Length != 64 || !sha.All(Uri.IsHexDigit))
        {
            throw new ArgumentException("--weights-sha256 must be 64 hex digits.");
        }

        var gemmaDimensions = 768;
        if (Optional("--pooling-embeddinggemma2-dimensions") is { } dims)
        {
            if (Optional("--pooling-embeddinggemma2") is null)
            {
                throw new ArgumentException("--pooling-embeddinggemma2-dimensions needs --pooling-embeddinggemma2.");
            }

            if (!int.TryParse(dims, NumberStyles.None, CultureInfo.InvariantCulture, out gemmaDimensions)
                || !EmbeddingGemma2EmbeddingModel.SupportedDimensions.Contains(gemmaDimensions))
            {
                throw new ArgumentException($"--pooling-embeddinggemma2-dimensions '{dims}' is not one of 768, 512, 256 or 128.");
            }
        }

        var pooling = new PoolingPaths(Optional("--pooling-e5"), Optional("--pooling-embeddinggemma2"), gemmaDimensions, Optional("--pooling-gliner2"));
        return new ServeCommand(
            model,
            Required("--weights"),
            sha,
            Required("--tokenizer"),
            Optional("--kernels"),
            Optional("--worker"),
            Optional("--manifest"),
            flags.Contains("--allow-untested-sm"),
            new RoostServerOptions
            {
                Host = Optional("--host") ?? RoostServerOptions.DefaultHost,
                Port = port,
                AllowNonLoopback = flags.Contains("--allow-non-loopback"),
                EnableSleepMode = flags.Contains("--enable-sleep-mode"),
            },
            pooling.Any ? pooling : null);
    }

    /// <summary>
    /// Opens the configured pooling models on ONNX Runtime's CPU execution provider, each verified against its package's asset lock.
    /// On a failure the models already opened are disposed.
    /// </summary>
    /// <returns>The models for the server, and the objects to dispose after it stops.</returns>
    public static (RoostPoolingModels Models, IReadOnlyList<IDisposable> Owned) OpenPooling(PoolingPaths? paths)
    {
        if (paths is null)
        {
            return (RoostPoolingModels.None, []);
        }

        var owned = new List<IDisposable>();
        var embeddings = new List<IRoostEmbeddingModel>();
        var extraction = new List<IRoostExtractionModel>();
        try
        {
            if (paths.E5 is { } e5)
            {
                var m = E5EmbeddingModel.Open(e5);
                owned.Add(m);
                embeddings.Add(m);
            }

            if (paths.EmbeddingGemma2 is { } gemma)
            {
                var m = EmbeddingGemma2EmbeddingModel.Open(gemma, paths.EmbeddingGemma2Dimensions);
                owned.Add(m);
                embeddings.Add(m);
            }

            if (paths.Gliner2 is { } gliner)
            {
                var m = Gliner2ExtractionModel.Open(gliner);
                owned.Add(m);
                extraction.Add(m);
            }
        }
        catch
        {
            foreach (var d in owned)
            {
                d.Dispose();
            }

            throw;
        }

        return (new RoostPoolingModels { Embeddings = embeddings, Extraction = extraction }, owned);
    }

    /// <summary>Parses <c>doctor</c> and its flags; refuses anything else by name.</summary>
    /// <exception cref="ArgumentException">An unknown flag, a missing value, an unknown profile, or a weights file without its digest.</exception>
    public static DoctorCommand ParseDoctor(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);
        if (args.Length == 0 || args[0] != "doctor")
        {
            throw new ArgumentException(args.Length == 0 ? "no command given." : $"unknown command '{args[0]}'.");
        }

        var (values, flags) = ReadOptions(args, DoctorFlags, DoctorValued);
        string? Optional(string name) => values.GetValueOrDefault(name);

        var model = Optional("--model") ?? Qwen35Profile;
        if (model != Qwen35Profile)
        {
            throw new ArgumentException($"unknown model profile '{model}'; the profiles are: {Qwen35Profile}.");
        }

        var weights = Optional("--weights");
        var sha = Optional("--weights-sha256");
        if ((weights is null) != (sha is null))
        {
            throw new ArgumentException("--weights and --weights-sha256 go together: the check compares the file with its lock.");
        }

        if (sha is not null && (sha.Length != 64 || !sha.All(Uri.IsHexDigit)))
        {
            throw new ArgumentException("--weights-sha256 must be 64 hex digits.");
        }

        double Gib(string name, double fallback) => Optional(name) is not { } text
            ? fallback
            : double.TryParse(text, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v)
                ? v
                : throw new ArgumentException($"{name} '{text}' is not a non-negative number of GiB.");

        return new DoctorCommand
        {
            Model = model,
            Weights = weights,
            WeightsSha256 = sha,
            Tokenizer = Optional("--tokenizer"),
            KernelDirectory = Optional("--kernels"),
            Worker = Optional("--worker"),
            Manifest = Optional("--manifest"),
            AllowUntestedSm = flags.Contains("--allow-untested-sm"),
            NoGpu = flags.Contains("--no-gpu"),
            MinFreeDiskGib = Gib("--min-free-disk-gib", DoctorCommand.DefaultMinFreeDiskGib),
            MinFreeMemoryGib = Gib("--min-free-memory-gib", DoctorCommand.DefaultMinFreeMemoryGib),
        };
    }

    // Reads the flags after the command word: each flag at most once, each valued flag followed by its value.
    private static (Dictionary<string, string> Values, HashSet<string> Flags) ReadOptions(string[] args, HashSet<string> known, HashSet<string> valued)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var flags = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 1; i < args.Length; i++)
        {
            var a = args[i];
            if (known.Contains(a))
            {
                flags.Add(a);
            }
            else if (valued.Contains(a))
            {
                if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    throw new ArgumentException($"{a} needs a value.");
                }

                if (!values.TryAdd(a, args[++i]))
                {
                    throw new ArgumentException($"{a} is given more than once.");
                }
            }
            else
            {
                throw new ArgumentException($"unknown option '{a}'.");
            }
        }

        return (values, flags);
    }

    /// <summary>
    /// The chat client settings for a serve command: Qwen3.5-0.8B on the packaged eager worker in its own process
    /// (decision 21), sized as the R-036 GPU run (32 attention blocks of 16 tokens, 64 tokens and one sequence per step).
    /// </summary>
    public static RoostChatClientOptions Options(ServeCommand command)
    {
        var tokenizer = Tokenizers.Tokenizer.FromFile(command.Tokenizer);
        int StopId(string token) => tokenizer.TokenToId(token) ?? throw new ArgumentException($"{token} is not in the tokenizer {command.Tokenizer}.");
        const int Blocks = 32;
        var profile = new RoostModelProfile
        {
            ModelId = Qwen35Profile,
            Tokenizer = tokenizer,
            ChatTemplate = Qwen35RoostChatTemplate.Instance,
            StopTokenIds = [StopId("<|im_end|>"), StopId("<|endoftext|>")],
            StateGroups = Qwen35Model.CreateStateGroups(),
            AttentionBlocks = Blocks,
            RecurrentSlots = 1,
            MaxTokensPerStep = 64,
            MaxSequencesPerStep = 1,
            MaxKvBlocksPerStep = Blocks,
            // The handshake checks the worker's manifest against this digest: the --manifest file, else the one shipped beside
            // the --worker executable, else the one beside the packaged worker.
            ManifestDigest = ImmutableArray.Create(RoostWorkerRuntime.ManifestDigest(
                command.Manifest ?? (command.Worker is { } worker
                    ? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(worker))!, RoostWorkerRuntime.ManifestFileName)
                    : null))),
            WeightsDigest = ImmutableArray.Create(Convert.FromHexString(command.WeightsSha256)),
        };
        return new RoostChatClientOptions
        {
            Model = profile,
            Worker = new RoostWorkerOptions
            {
                Launch = new RoostWorkerLaunch
                {
                    Model = Qwen35Model.ModelName,
                    WeightsPath = command.Weights,
                    WeightsSize = new FileInfo(command.Weights).Length,
                    Executable = command.Worker,
                    KernelDirectory = command.KernelDirectory,
                    ManifestPath = command.Manifest,
                    AllowUntestedSm = command.AllowUntestedSm,
                },
            },
        };
    }
}
