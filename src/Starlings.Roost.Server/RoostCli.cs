using System.Collections.Immutable;
using System.Globalization;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Qwen35;
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
internal sealed record ServeCommand(
    string Model,
    string Weights,
    string WeightsSha256,
    string Tokenizer,
    string? KernelDirectory,
    string? Worker,
    string? Manifest,
    bool AllowUntestedSm,
    RoostServerOptions Listen);

/// <summary>
/// The <c>roost</c> command line. Only <c>roost serve</c> exists (R-037); vLLM's flags and environment variables come with
/// R-075. The one model profile is Qwen3.5-0.8B on the eager worker, with the capacities of the R-036 GPU run.
/// </summary>
internal static class RoostCli
{
    /// <summary>The profile name of Qwen3.5-0.8B, the M0 model (decision 15).</summary>
    public const string Qwen35Profile = "qwen3.5-0.8b";

    public const string Usage = """
        usage: roost serve --model qwen3.5-0.8b --weights <model.safetensors> --weights-sha256 <hex> --tokenizer <tokenizer.json>
                           [--kernels <dir>] [--worker <roost-worker>] [--manifest <manifest.toml>] [--allow-untested-sm]
                           [--host <ip>] [--port <n>] [--allow-non-loopback] [--enable-sleep-mode]

        Serves the OpenAI routes /v1/chat/completions, /v1/completions and /v1/models on 127.0.0.1:8000 by default.
        A non-loopback --host needs --allow-non-loopback: the server has no authentication.
        --enable-sleep-mode adds POST /sleep, POST /wake_up, GET /is_sleeping and POST /update_weights, for loopback
        clients only (the packaged CUDA worker refuses them today).
        """;

    private static readonly HashSet<string> Flags = new(StringComparer.Ordinal) { "--allow-untested-sm", "--allow-non-loopback", "--enable-sleep-mode" };

    private static readonly HashSet<string> Valued = new(StringComparer.Ordinal)
    {
        "--model", "--weights", "--weights-sha256", "--tokenizer", "--kernels", "--worker", "--manifest", "--host", "--port",
    };

    /// <summary>Runs the command line.</summary>
    /// <param name="args">The arguments.</param>
    /// <param name="error">Where usage errors go.</param>
    /// <returns>The exit code: 0 after a clean shutdown, 1 when the worker does not start, 2 for a usage error.</returns>
    public static async Task<int> RunAsync(string[] args, TextWriter error)
    {
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

        RoostChatClient client;
        try
        {
            client = await RoostChatClient.StartAsync(Options(command)).ConfigureAwait(false);
        }
        catch (Exception e) when (e is ArgumentException or IOException or UnauthorizedAccessException or RoostChatException)
        {
            await error.WriteLineAsync("roost: the worker did not start: " + e.Message).ConfigureAwait(false);
            return 1;
        }

        await using (client.ConfigureAwait(false))
        {
            var app = RoostServer.Create(client, command.Listen);
            await using (app.ConfigureAwait(false))
            {
                await app.RunAsync().ConfigureAwait(false);
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

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var flags = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 1; i < args.Length; i++)
        {
            var a = args[i];
            if (Flags.Contains(a))
            {
                flags.Add(a);
            }
            else if (Valued.Contains(a))
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
            });
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
